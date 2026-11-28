"""2024년 분기별 확장 학습. 저장소 루트에서 python experiments/run_quarterly.py."""
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trendit_lab.compare_24h import select_comparison_rows
from trendit_lab.data import KEY, TARGET, load_events
from trendit_lab.experiment import build_manifest, write_json
from trendit_lab.model import GROWTH_24H, build_forest, evaluate, positive_probability


def main():
    path = ROOT / "configs/baseline.toml"
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    events = load_events(ROOT / "data")
    features = config["numeric_features"] + config["categorical_features"]
    rows, predictions = [], []
    for quarter in pd.period_range("2024Q1", "2024Q4", freq="Q"):
        start, end = quarter.start_time, (quarter + 1).start_time
        fold = dict(config, validation_start=str(start), test_start=str(end))
        cohort, _, _ = select_comparison_rows(events, fold)
        train, valid = cohort["train"], cohort["validation"]
        assert train.T0.max() + pd.Timedelta(hours=54) < start
        assert valid.T0.min() >= start and valid.T0.max() + pd.Timedelta(hours=54) < end
        assert not set(train.video_id).intersection(valid.video_id)
        pred = valid[KEY + ["T0", TARGET]].copy()
        pred["quarter"] = str(quarter)
        t0 = build_forest(config).fit(train[features], train[TARGET])
        h24 = build_forest(config, include_24h_growth=True)
        h24.set_params(model__max_features=t0.named_steps["model"].estimators_[0].max_features_)
        extended = features + [GROWTH_24H]
        h24.fit(train[extended], train[TARGET])
        for name, model, cols in [("T0", t0, features), ("T24", h24, extended)]:
            prob = positive_probability(model, valid[cols])
            pred[f"p_{name}"] = prob
            score = evaluate(valid[TARGET], prob, config["threshold"])
            (_, fp), (fn, _) = score["confusion_matrix"]
            rows.append(dict(quarter=str(quarter), model=name, train=len(train), valid=len(valid),
                             balanced_accuracy=score["balanced_accuracy"], auc=score["roc_auc"], fp=fp, fn=fn))
        predictions.append(pred)
        print(quarter, flush=True)
    runs = ROOT / "reports/runs"
    runs.mkdir(parents=True, exist_ok=True)
    out = runs / datetime.now(timezone.utc).strftime("quarterly_%Y%m%dT%H%M%S%fZ")
    summary = pd.DataFrame(rows)
    with TemporaryDirectory(dir=runs) as tmp:
        stage = Path(tmp) / "result"
        stage.mkdir()
        summary.to_csv(stage / "quarterly.csv", index=False)
        pd.concat(predictions).to_csv(stage / "predictions.csv", index=False)
        write_json(stage / "manifest.json", build_manifest(ROOT, path, config))
        stage.rename(out)
    print(summary.to_string(index=False), out, sep="\n")


if __name__ == "__main__":
    main()
