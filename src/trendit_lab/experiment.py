"""한 번의 실행에 데이터 지문, 분할, 설정, 예측, 평가 결과를 묶어 저장한다."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import tomllib
import warnings
from datetime import datetime, timezone
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning

from .data import KEY, TARGET, audit_files, load_events, require
from .model import build_models, evaluate, positive_probability
from .split import split_events


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def git_info(root):
    def read(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": read("rev-parse", "HEAD"), "status": read("status", "--porcelain")}


def render_report(metrics, audit, manifest):
    lines = [
        "# T0에서 48시간 이상 지속 여부 예측: 첫 기준 실험", "",
        "이 결과는 고정된 과거 데이터에 대한 후속 검증이며, 기존 프로젝트 성능의 동일 조건 재현은 아닙니다.", "",
        "## 실험 규칙", "",
        f"- 입력: 진입 시점의 {len(manifest['config']['numeric_features']) + len(manifest['config']['categorical_features'])}개 피처. 정답: 관측된 이벤트 길이 48시간 이상 여부.",
        f"- 검증 시작: {manifest['config']['validation_start']}; 시험 시작: {manifest['config']['test_start']} (KST).",
        f"- 기간 경계 및 수집 종료까지 {48 + manifest['config']['observation_buffer_hours']}시간보다 긴 관측 여유가 있는 이벤트만 사용합니다.",
        "- 첫 수집 스냅샷과 이전 기간에 나타난 영상의 후속 기간 이벤트는 제외합니다.",
        "- 전처리와 모델은 학습 세트만 fit합니다. 모델 선택 후 재학습하지 않습니다.",
        "- 임계값 0.5 고정. 검증 balanced accuracy로 선택하고 동률이면 단순한 모델을 유지합니다.",
        "- 시험에서는 선택 모델과 Dummy만 평가합니다. 시험 점수로 모델을 다시 고르지 않습니다.", "",
        "## 데이터 분할", "",
        "| 세트 | 이벤트 | 영상 | 48h 이상 비율 | 시작 | 마지막 진입 |",
        "|---|---:|---:|---:|---|---|",
    ]
    for name, s in audit["splits"].items():
        lines.append(f"| {name} | {s['rows']:,} | {s['videos']:,} | {s['positive_rate']:.2%} | {s['start']} | {s['end']} |")
    lines += ["", f"제외 사유(중복 집계 없음): `{audit['exclusion_counts']}`", "",
              "## 검증 결과: 모델을 고르는 자료", "",
              "| 모델 | Balanced accuracy | ROC AUC | AP | F1 | Log loss |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, score in metrics["validation"].items():
        lines.append(f"| {name} | {score['balanced_accuracy']:.4f} | {score['roc_auc']:.4f} | {score['average_precision']:.4f} | {score['f1']:.4f} | {score['log_loss']:.4f} |")
    lines += ["", f"검증에서 선택한 모델: **{metrics['selected_model']}**", "",
              "## 최종 시험 결과", "",
              "| 모델 | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC AUC | AP |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, score in metrics["test"].items():
        lines.append(f"| {name} | {score['accuracy']:.4f} | {score['balanced_accuracy']:.4f} | {score['precision']:.4f} | {score['recall']:.4f} | {score['f1']:.4f} | {score['roc_auc']:.4f} | {score['average_precision']:.4f} |")
    selected = metrics["test"][metrics["selected_model"]]
    delta = selected["balanced_accuracy"] - metrics["test"]["dummy"]["balanced_accuracy"]
    lines += ["", f"선택 모델의 Dummy 대비 시험 balanced accuracy 차이: **{delta:+.4f}**.",
              f"선택 모델의 혼동행렬 `[[TN, FP], [FN, TP]]`: `{selected['confusion_matrix']}`.", "",
              "정확도는 전체 정답 비율입니다. Balanced accuracy는 0과 1 각각의 재현율 평균이며,",
              "한쪽 답만 하는 모델은 두 클래스가 있는 데이터에서 0.5입니다. F1은 양성 판정의",
              "정밀도와 재현율을 함께 봅니다. AP는 확률 순위의 지표로, 사다리꼴 PR AUC와 같지 않습니다.",
              "Log loss는 틀린 답에 높은 확신을 줄수록 커지며 작을수록 좋습니다.", "",
              "## 해석 범위와 남은 한계", "",
              "- 이미 트렌딩에 진입한 영상의 이벤트만 대상으로 합니다. 모든 업로드의 성공 예측이 아닙니다.",
              "- 같은 영상은 세트 사이에서 분리했지만 같은 채널은 허용합니다. 새 채널 일반화는 검증하지 않았습니다.",
              "- 한 기간 안에 같은 영상의 여러 이벤트가 남습니다. 지표는 이벤트 단위이며 독립 표본으로 간주한 신뢰구간은 제시하지 않습니다.",
              "- 지속시간은 실제 연속 노출 시간이 아니라 6시간 수집과 이벤트 분리 규칙으로 정의한 관측값입니다.",
              "- 종료 시점에 계속 관측 중인 이벤트라도 48시간을 이미 넘었다면 이진 정답은 확정됩니다.",
              "- 원시 스냅샷이 없어 내부 수집 장애, 삭제/수정 이력, 최초 관측값의 진위까지 검증하지는 못했습니다.",
              "- 포화도는 예측일 전날까지의 이벤트 수로 재계산했습니다. 시험 기간에도 그 시점까지 관측한 입력 이력을 사용할 수 있다는 가정입니다.",
              "- 24h 모델은 24시간 관측 가능한 동일 이벤트 집합에서 따로 비교해야 합니다.",
              "- 이 시험 결과를 본 뒤의 변경은 탐색입니다. 새로운 확증에는 별도의 미래 기간이 필요합니다.", "",
              "## 재현 기록", "",
              f"실행 UTC: `{manifest['created_at_utc']}`. Python: `{manifest['python']}`.",
              "`config.toml`, `manifest.json`, `audit.json`, `schema.csv`, `split_assignments.csv`,",
              "`validation_predictions.csv`, `test_predictions.csv`, `metrics.json`을 함께 보관합니다.",
              "분할 CSV에는 제외된 이벤트와 제외 이유도 남깁니다. 소스 해시는 커밋 전 코드 변경도 식별합니다.", ""]
    return "\n".join(lines)


def run(root: Path, config_path: Path, output: Path):
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    require(config["threshold"] == 0.5, "First baseline fixes threshold at 0.5")
    require(config["selection_metric"] == "balanced_accuracy", "Unsupported selection metric")
    models = build_models(config)
    require(not output.exists(), "Output already exists; choose a new run directory")
    events = load_events(root / "data")
    splits, assignments = split_events(events, config)
    files, schema = audit_files(root / "data")
    features = config["numeric_features"] + config["categorical_features"]
    train, valid, test = (splits[n] for n in ["train", "validation", "test"])
    metrics = {"validation": {}, "test": {}}
    valid_predictions = valid[KEY + ["T0", TARGET]].copy()
    for name, model in models.items():
        print(f"Fitting {name}: train={len(train)}, validation={len(valid)}", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(train[features], train[TARGET])
        probability = positive_probability(model, valid[features])
        metrics["validation"][name] = evaluate(valid[TARGET], probability, config["threshold"])
        valid_predictions[f"p_{name}"] = probability

    selected = max(models, key=lambda name: metrics["validation"][name][config["selection_metric"]])
    metrics["selected_model"] = selected
    metrics["threshold"] = config["threshold"]
    test_predictions = test[KEY + ["T0", TARGET]].copy()
    for name in dict.fromkeys(["dummy", selected]):
        probability = positive_probability(models[name], test[features])
        metrics["test"][name] = evaluate(test[TARGET], probability, config["threshold"])
        test_predictions[f"p_{name}"] = probability

    audit = {
        "files": files,
        "observation_start": str(events.T0.min()), "observation_end": str(events.T_end.max()),
        "exclusion_counts": assignments.loc[assignments.split.eq("excluded"), "reason"].value_counts().to_dict(),
        "splits": {name: {"rows": len(frame), "videos": int(frame.video_id.nunique()),
                          "positive_rate": float(frame[TARGET].mean()),
                          "start": str(frame.T0.min()), "end": str(frame.T0.max())}
                   for name, frame in splits.items()},
        "between_split_video_overlap": 0,
        "initial_snapshot_rows": int(events.T0.eq(events.T0.min()).sum()),
        "last_snapshot_rows": int(events.T_end.eq(events.T_end.max()).sum()),
        "single_snapshot_rows": int(events.trending_duration_h_raw.eq(0).sum()),
        "full_data_q75_hours": float(events.trending_duration_h.quantile(.75)),
    }
    tracked_inputs = sorted((root / "data").glob("*.parquet"))
    sources = sorted((root / "src").rglob("*.py")) + sorted((root / "experiments").glob("*.py"))
    sources += sorted((root / "tests").glob("*.py"))
    sources += [root / p for p in ["requirements.txt", "requirements-lock.txt", "pyproject.toml", ".python-version"]]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "platform": platform.platform(), "git": git_info(root),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "data_sha256": {p.relative_to(root).as_posix(): sha256(p) for p in tracked_inputs},
        "source_sha256": {p.relative_to(root).as_posix(): sha256(p) for p in sources},
        "config_sha256": sha256(config_path), "config": config,
    }
    output.mkdir(parents=True)
    (output / "config.toml").write_bytes(config_path.read_bytes())
    write_json(output / "metrics.json", metrics)
    write_json(output / "audit.json", audit)
    write_json(output / "manifest.json", manifest)
    schema.to_csv(output / "schema.csv", index=False)
    assignments.to_csv(output / "split_assignments.csv", index=False)
    valid_predictions.to_csv(output / "validation_predictions.csv", index=False)
    test_predictions.to_csv(output / "test_predictions.csv", index=False)
    (output / "report.md").write_text(render_report(metrics, audit, manifest), encoding="utf-8")
    print(f"Selected: {selected}; test balanced accuracy: {metrics['test'][selected]['balanced_accuracy']:.4f}")
    print(f"Saved: {output}")
    return metrics
