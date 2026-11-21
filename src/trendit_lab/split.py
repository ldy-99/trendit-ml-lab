"""시간 순서, 정답 관측 여유, 영상 중복을 함께 통제한다."""

import pandas as pd

from .data import KEY, TARGET, require


def split_events(events, config):
    valid_start = pd.Timestamp(config["validation_start"])
    test_start = pd.Timestamp(config["test_start"])
    observation_start, observation_end = events.T0.min(), events.T_end.max()
    require(observation_start < valid_start < test_start < observation_end,
            "Split boundaries must be ordered inside the observed period")
    require(config["label_horizon_hours"] == 48, "This experiment only supports the 48h target")
    require(config["observation_buffer_hours"] >= 6, "At least one 6h observation buffer is required")
    delay = pd.Timedelta(hours=48 + config["observation_buffer_hours"])
    assignment = events[KEY + ["T0"]].copy()
    assignment["period"] = "train"
    assignment.loc[events.T0 >= valid_start, "period"] = "validation"
    assignment.loc[events.T0 >= test_start, "period"] = "test"
    assignment["split"] = "excluded"
    assignment["reason"] = ""
    seen_videos = set()
    splits = {}
    for name, end in [("train", valid_start), ("validation", test_start), ("test", observation_end)]:
        in_period = assignment.period.eq(name)
        # 데이터 첫 스냅샷 영상은 진짜 최초 진입을 관측하지 못했을 수 있다.
        left_censored = in_period & events.T0.eq(observation_start)
        immature = in_period & ((events.T0 + delay) >= end)
        repeated = in_period & events.video_id.isin(seen_videos)
        assignment.loc[left_censored, "reason"] = "initial_snapshot"
        assignment.loc[immature & assignment.reason.eq(""), "reason"] = "insufficient_followup"
        assignment.loc[repeated & assignment.reason.eq(""), "reason"] = "video_seen_in_earlier_period"
        kept = in_period & assignment.reason.eq("")
        assignment.loc[kept, "split"] = name
        splits[name] = events.loc[kept].copy()
        # 제외된 행도 실제로 이전 기간에 나타난 영상이므로 seen에 포함한다.
        seen_videos.update(events.loc[in_period, "video_id"])
        require(len(splits[name]) > 0, f"Empty {name} split")
        require(splits[name][TARGET].nunique() == 2, f"{name} requires both target classes")
        require((splits[name].T0 + delay < end).all(), f"Immature labels in {name}")

    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        require(not (set(splits[a].video_id) & set(splits[b].video_id)), "Video leakage between splits")
        require(splits[a].T0.max() < splits[b].T0.min(), "Time order violation")
    return splits, assignment
