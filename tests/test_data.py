from pathlib import Path
import tomllib

import numpy as np
import pandas as pd
import pytest

from trendit_lab.compare_24h import select_comparison_rows
from trendit_lab.data import (CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_events,
                              prior_saturation, validate_events, validate_features)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def events():
    return load_events(ROOT / "data")


def test_current_data_contract(events):
    assert len(events) > 0
    assert not events.duplicated(["video_id", "event_id"]).any()


def test_comparison_rejects_late_snapshot_marked_exact(events):
    config = tomllib.loads((ROOT / "configs/baseline.toml").read_text(encoding="utf-8"))
    frame = events.copy()
    exact_event = frame.index[frame.has_exact_24h_snapshot][0]
    frame.loc[exact_event, "actual_time_at_24h"] += pd.Timedelta(hours=6)
    with pytest.raises(ValueError, match="snapshot timestamp mismatch"):
        select_comparison_rows(frame, config)


@pytest.mark.parametrize("column", ["T_end", "trending_duration_h", "tdi_label", "video_id", "view_growth_24h_log"])
def test_leaking_features_rejected(column):
    with pytest.raises(ValueError, match="Unapproved"):
        validate_features(NUMERIC_FEATURES + [column], CATEGORICAL_FEATURES)


def test_duplicate_events_rejected(events):
    with pytest.raises(ValueError, match="Duplicate"):
        validate_events(pd.concat([events, events.iloc[[0]]], ignore_index=True))


def test_corrupt_target_rejected(events):
    frame = events.copy()
    frame.loc[0, "long_label_48h"] = 1 - frame.loc[0, "long_label_48h"]
    with pytest.raises(ValueError, match="48h label"):
        validate_events(frame)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_missing_and_infinite_features_rejected(events, value):
    frame = events.copy()
    frame.loc[0, "T0_view_log"] = value
    with pytest.raises(ValueError, match="Missing|Non-finite"):
        validate_events(frame)


def test_incorrect_saturation_rejected(events):
    frame = events.copy()
    frame.loc[0, "saturation_index_30d_mean_prev"] += .1
    with pytest.raises(ValueError, match="Feature formula mismatch"):
        validate_events(frame)


def test_saturation_does_not_read_future_or_same_day_events():
    frame = pd.DataFrame({"T0": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
                          "category_group": ["Music"] * 3})
    original = prior_saturation(frame)
    extra = pd.DataFrame({"T0": pd.to_datetime(["2023-01-03"] * 20 + ["2023-01-04"] * 30),
                          "category_group": ["Music"] * 50})
    augmented = prior_saturation(pd.concat([frame, extra], ignore_index=True))
    np.testing.assert_allclose(original, augmented[:3])
    np.testing.assert_allclose(original, [0, 1, 1])
