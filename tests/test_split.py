import pandas as pd
import pytest

from trendit_lab.split import split_events


def sample():
    rows = [
        ("initial", "2023-01-01", 0),
        ("train0", "2023-02-01", 0), ("train1", "2023-03-01", 1),
        ("edge_train", "2023-12-29 18:00:00", 1),
        ("before_edge", "2023-12-29 17:59:59", 0),
        ("train1", "2024-02-01", 1),
        ("edge_train", "2024-02-02", 1),
        ("valid0", "2024-03-01", 0), ("valid1", "2024-04-01", 1),
        ("edge_valid", "2024-12-29 18:00:00", 0),
        ("valid1", "2025-02-01", 1), ("train0", "2025-02-02", 0),
        ("test0", "2025-03-01", 0), ("test1", "2025-04-01", 1),
        ("end", "2025-06-30", 0),
    ]
    frame = pd.DataFrame(rows, columns=["video_id", "T0", "long_label_48h"])
    frame.T0 = pd.to_datetime(frame.T0, format="mixed")
    frame["T_end"] = frame.T0 + pd.to_timedelta(frame.long_label_48h * 60 + 6, unit="h")
    frame["event_id"] = frame.groupby("video_id").cumcount() + 1
    config = {"validation_start": "2024-01-01", "test_start": "2025-01-01",
              "label_horizon_hours": 48, "observation_buffer_hours": 6}
    return frame, config


def test_time_group_and_maturity_rules():
    events, config = sample()
    splits, ledger = split_events(events, config)
    assert set(splits["train"].video_id) == {"train0", "train1", "before_edge"}
    assert set(splits["validation"].video_id) == {"valid0", "valid1"}
    assert set(splits["test"].video_id) == {"test0", "test1"}
    assert len(ledger) == len(events)
    assert ledger.loc[ledger.video_id.eq("initial"), "reason"].item() == "initial_snapshot"
    # 경계에서 제외된 영상도 이후 기간에는 재등장 영상으로 처리한다.
    row = ledger[(ledger.video_id == "edge_train") & (ledger.period == "validation")]
    assert row.reason.item() == "video_seen_in_earlier_period"


def test_incomplete_labels_excluded_independently_of_class():
    events, config = sample()
    _, a = split_events(events, config)
    events.loc[events.video_id.eq("edge_train"), "long_label_48h"] = 0
    _, b = split_events(events, config)
    pd.testing.assert_frame_equal(a, b)


def test_positive_does_not_need_to_end_before_boundary():
    events, config = sample()
    events.loc[events.video_id.eq("train1") & (events.event_id == 1), "T_end"] = pd.Timestamp("2024-01-02")
    splits, _ = split_events(events, config)
    assert "train1" in set(splits["train"].video_id)


def test_invalid_boundaries_fail():
    events, config = sample()
    config["test_start"] = "2023-01-01"
    with pytest.raises(ValueError, match="boundaries"):
        split_events(events, config)
