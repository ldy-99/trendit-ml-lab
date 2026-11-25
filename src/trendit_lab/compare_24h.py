"""같은 이벤트의 T0 입력과 T0 + 24시간 조회수 증가량을 비교한다."""

import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .data import KEY, TARGET, load_events, require
from .experiment import build_manifest, write_json
from .model import GROWTH_24H, build_forest, evaluate, positive_probability
from .split import split_events


def select_comparison_rows(events, config):
    # 전체 데이터에서 먼저 분할해야 기존 시간 경계와 이전 기간 영상 규칙을 유지한다.
    splits, assignments = split_events(events, config)
    exact_time = events.actual_time_at_24h.eq(events.T0 + pd.Timedelta(hours=24))
    require(events.has_exact_24h_snapshot.eq(exact_time).all(), "24h snapshot timestamp mismatch")
    require(events.loc[exact_time, "has_24h_observation"].eq(True).all(),
            "Exact snapshot must be a 24h observation")
    require(np.isfinite(events.loc[exact_time, GROWTH_24H]).all(), "Missing or invalid exact 24h growth")

    assignments["has_exact_24h_snapshot"] = exact_time
    assignments["included_in_comparison"] = assignments.split.ne("excluded") & exact_time
    cohort, summary = {}, {}
    for name, frame in splits.items():
        shared = frame.loc[exact_time.loc[frame.index]]
        require(shared[TARGET].nunique() == 2, f"Shared {name} rows require both target classes")
        cohort[name] = shared
        summary[name] = {"baseline_rows": len(frame), "shared_rows": len(shared),
                         "shared_videos": int(shared.video_id.nunique()),
                         "shared_positive_rate": float(shared[TARGET].mean())}
    return cohort, assignments, summary


def prediction_changes(frame, threshold):
    actual = frame[TARGET]
    t0_correct = frame.p_t0.ge(threshold).eq(actual)
    h24_correct = frame.p_t0_plus_24h.ge(threshold).eq(actual)
    return {
        "wrong_to_right": int((~t0_correct & h24_correct).sum()),
        "right_to_wrong": int((t0_correct & ~h24_correct).sum()),
    }


def render_report(metrics, summary, manifest):
    lines = [
        "# T0와 T0+24h 조회수 증가량 비교", "",
        "두 랜덤포레스트 모두 트렌딩 진입부터 48시간 이상 지속했는지 예측했습니다.",
        "기존 분할을 그대로 적용한 뒤 정확히 진입 24시간 후에 조회수를 관측한 이벤트만 남겼습니다.",
        "T0 모델과 T0+24h 모델은 같은 이벤트로 다시 학습했습니다. 두 번째 모델에만 `view_growth_24h_log`를 추가했습니다.",
        f"시드 `{manifest['config']['seed']}`, 임계값 `{metrics['threshold']}`와 기준 실험의 포레스트 설정을 사용했습니다.",
        f"트리가 분기마다 먼저 고려하는 피처 후보 수도 두 모델 모두 `{manifest['effective_max_features']}`개로 맞췄습니다.",
        "",
        "| 세트 | 기존 이벤트 | 공통 이벤트 | 공통 영상 | 공통 이벤트의 양성 비율 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, counts in summary.items():
        lines.append(f"| {name} | {counts['baseline_rows']:,} | {counts['shared_rows']:,} | "
                     f"{counts['shared_videos']:,} | {counts['shared_positive_rate']:.2%} |")
    lines += ["", "| 세트 | 모델 | Balanced accuracy | ROC AUC | Log loss | FP | FN |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for name in ("validation", "test"):
        result = metrics[name]
        for model in ("t0", "t0_plus_24h"):
            score = result[model]
            (_, fp), (fn, _) = score["confusion_matrix"]
            lines.append(f"| {name} | {model} | {score['balanced_accuracy']:.4f} | "
                         f"{score['roc_auc']:.4f} | {score['log_loss']:.4f} | {fp:,} | {fn:,} |")
    lines.append("")
    for name, label in (("validation", "검증"), ("test", "시험")):
        result = metrics[name]
        delta = result["t0_plus_24h"]["balanced_accuracy"] - result["t0"]["balanced_accuracy"]
        change = result["paired_changes"]
        lines.append(f"- {label} balanced accuracy 변화 **{delta * 100:+.2f}%p**. "
                     f"오답에서 정답으로 바뀐 이벤트 {change['wrong_to_right']:,}건, "
                     f"정답에서 오답으로 바뀐 이벤트 {change['right_to_wrong']:,}건.")
    lines += ["", "FP는 48시간 미만 이벤트를 오래 지속한다고 판정한 건수입니다. "
              "FN은 48시간 이상 이벤트를 놓친 건수입니다.", "",
              "이 비교는 24시간 시점에도 관측된 이벤트에 적용됩니다. "
              "2025년 시험 기간은 기존 실험에서 이미 확인했으므로 이 결과는 탐색에 해당합니다. "
              "새로운 기간으로 성능을 다시 확인해야 일반화 여부를 판단할 수 있습니다.", "",
              "검증·시험 예측 CSV에서 두 모형의 확률을 같은 행에서 비교할 수 있습니다. "
              "분할 CSV의 `included_in_comparison` 열은 공통 이벤트 포함 여부를 나타냅니다.", ""]
    return "\n".join(lines)


def run_comparison(root: Path, config_path: Path, output: Path):
    require(not output.exists(), "Output already exists; choose a new run directory")
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    threshold = config["threshold"]
    require(threshold == 0.5, "This comparison fixes the threshold at 0.5")
    cohort, assignments, summary = select_comparison_rows(load_events(root / "data"), config)
    train = cohort["train"]
    t0_features = config["numeric_features"] + config["categorical_features"]
    h24_features = [*t0_features, GROWTH_24H]
    print(f"Fitting both forests on {len(train):,} shared training events", flush=True)
    t0 = build_forest(config).fit(train[t0_features], train[TARGET])

    # sqrt(max features) can change when the encoded width moves from 15 to 16.
    # Match the actual T0 candidate count so the extra input is the comparison variable.
    effective_max_features = t0.named_steps["model"].estimators_[0].max_features_
    h24 = build_forest(config, include_24h_growth=True)
    h24.set_params(model__max_features=effective_max_features)
    h24.fit(train[h24_features], train[TARGET])

    metrics = {"threshold": threshold}
    predictions = {}
    for name in ("validation", "test"):
        frame = cohort[name]
        predicted = frame[KEY + ["T0", TARGET]].copy()
        predicted["p_t0"] = positive_probability(t0, frame[t0_features])
        predicted["p_t0_plus_24h"] = positive_probability(h24, frame[h24_features])
        metrics[name] = {
            "t0": evaluate(frame[TARGET], predicted.p_t0, threshold),
            "t0_plus_24h": evaluate(frame[TARGET], predicted.p_t0_plus_24h, threshold),
            "paired_changes": prediction_changes(predicted, threshold),
        }
        predictions[name] = predicted

    manifest = build_manifest(root, config_path, config)
    manifest["effective_max_features"] = effective_max_features
    manifest["added_feature"] = GROWTH_24H
    report = render_report(metrics, summary, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary_dir:
        staged = Path(temporary_dir) / "result"
        staged.mkdir()
        (staged / "config.toml").write_bytes(config_path.read_bytes())
        (staged / "report.md").write_text(report, encoding="utf-8")
        write_json(staged / "metrics.json", metrics)
        write_json(staged / "audit.json", summary)
        write_json(staged / "manifest.json", manifest)
        assignments.to_csv(staged / "split_assignments.csv", index=False)
        for name, frame in predictions.items():
            frame.to_csv(staged / f"{name}_predictions.csv", index=False)
        staged.rename(output)
    print(f"Saved: {output}")
