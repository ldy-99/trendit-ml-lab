"""정답과 입력을 분리하고, 저장된 피처의 계산 규칙을 검증한다."""

import numpy as np
import pandas as pd

KEY = ["video_id", "event_id"]
TARGET = "long_label_48h"
NUMERIC_FEATURES = [
    "entry_rank_log", "T0_view_log", "T0_comment_log",
    "T0_engagement_ratio_log", "latency_to_trend_log",
    "pretrend_view_velocity_log", "published_weekday", "hour_sin", "hour_cos",
    "saturation_index_30d_mean_prev",
]
CATEGORICAL_FEATURES = ["category_group"]


def require(condition, message):
    # assert는 python -O에서 사라지므로 데이터 검증에는 명시적 예외를 쓴다.
    if not condition:
        raise ValueError(message)


def validate_features(numeric, categorical):
    require(bool(numeric) and bool(categorical), "Feature lists must not be empty")
    require(len(numeric + categorical) == len(set(numeric + categorical)),
            "Duplicate features")
    require(set(numeric) <= set(NUMERIC_FEATURES), "Unapproved numeric feature (possible leakage)")
    require(set(categorical) <= set(CATEGORICAL_FEATURES), "Unapproved categorical feature")


def prior_saturation(events):
    """전날까지의 30일 평균 이벤트 수 / 30일 최대 이벤트 수.

    라벨은 읽지 않는다. 미래 행이 추가되어도 과거의 결과가 바뀌지 않는다.
    """
    frame = events.assign(day=events.T0.dt.normalize())
    counts = frame.groupby(["day", "category_group"]).size()
    index = pd.MultiIndex.from_product(
        [pd.date_range(frame.day.min(), frame.day.max()),
         sorted(frame.category_group.unique())], names=["day", "category_group"])
    daily = counts.reindex(index, fill_value=0).rename("count").reset_index()
    daily = daily.sort_values(["category_group", "day"])
    grouped = daily.groupby("category_group")["count"]
    mean = grouped.transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean())
    maximum = grouped.transform(lambda s: s.shift(1).rolling(30, min_periods=1).max())
    daily["recomputed"] = (mean / maximum.replace(0, np.nan)).clip(0, 1).fillna(0)
    merged = frame[["day", "category_group"]].merge(
        daily[["day", "category_group", "recomputed"]],
        on=["day", "category_group"], how="left", validate="many_to_one", sort=False)
    return merged.recomputed.to_numpy()


def validate_events(events):
    required = KEY + [TARGET, "T0", "T_end", "published_at", "trending_duration_h",
                      "trending_duration_h_raw", "entry_rank", "T0_view", "T0_comment"]
    required += NUMERIC_FEATURES + CATEGORICAL_FEATURES
    require(set(required) <= set(events), "Missing required data columns")
    require(len(events) > 0, "Empty event dataset")
    require(not events[required].isna().any().any(), "Missing required data values")
    require(not events.duplicated(KEY).any(), "Duplicate (video_id, event_id)")
    for col in ["T0", "T_end", "published_at"]:
        require(pd.api.types.is_datetime64_any_dtype(events[col]), f"Invalid datetime: {col}")
        require(events[col].dt.tz is None, f"Expected naive KST: {col}")
    require((events.T_end >= events.T0).all(), "T_end precedes T0")
    span = (events.T_end - events.T0).dt.total_seconds() / 3600
    require(np.allclose(span, events.trending_duration_h_raw), "Raw duration mismatch")
    require(np.allclose(span.clip(lower=6), events.trending_duration_h), "6h duration rule mismatch")
    require(events[TARGET].isin([0, 1]).all(), "Target must be binary")
    require((events[TARGET] == (span >= 48).astype(int)).all(), "48h label mismatch")
    require(np.isfinite(events[NUMERIC_FEATURES].to_numpy(dtype=float)).all(),
            "Non-finite model inputs")
    require(events.entry_rank.between(1, 200).all(), "Invalid entry rank")
    require((events.T0_view > 0).all() and (events.T0_comment >= 0).all(),
            "Invalid view/comment count")

    latency = ((events.T0 - events.published_at).dt.total_seconds() / 3600).clip(lower=0)
    hour = events.published_at.dt.hour
    expected = {
        "entry_rank_log": np.log1p(events.entry_rank),
        "T0_view_log": np.log1p(events.T0_view),
        "T0_comment_log": np.log1p(events.T0_comment),
        "T0_engagement_ratio_log": np.log1p(events.T0_comment / events.T0_view),
        "latency_to_trend_log": np.log1p(latency),
        "pretrend_view_velocity_log": np.log1p(events.T0_view / latency.clip(lower=1)),
        "published_weekday": events.published_at.dt.dayofweek,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "saturation_index_30d_mean_prev": prior_saturation(events),
    }
    for col, values in expected.items():
        require(np.allclose(events[col], values, rtol=1e-8, atol=1e-10),
                f"Feature formula mismatch: {col}")


def load_events(data_dir):
    events = pd.read_parquet(data_dir / "video_trending_events_analysis.parquet")
    validate_events(events)
    # T0용 파일과 분석용 파일이 같은 이벤트/값인지 확인한다.
    t0 = pd.read_parquet(data_dir / "video_trending_events_T0_model.parquet")
    require(not t0.duplicated(KEY).any(), "Duplicate T0 keys")
    left = events.set_index(KEY).sort_index()
    right = t0.set_index(KEY).sort_index()
    require(left.index.equals(right.index), "T0 and analysis event keys differ")
    cols = [TARGET, "T0"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    pd.testing.assert_frame_equal(left[cols], right[cols], check_dtype=False)
    h24 = pd.read_parquet(data_dir / "video_trending_events_24h_model.parquet")
    growth = pd.read_parquet(data_dir / "view_growth_KR.parquet")
    require(not h24.duplicated(KEY).any() and not growth.duplicated(KEY).any(),
            "Duplicate 24h/growth keys")
    observed = events.T_end >= events.T0 + pd.Timedelta(hours=24)
    require((events.has_24h_observation == observed).all(), "24h observation flag mismatch")
    require(events.view_growth_24h.notna().equals(observed), "24h missingness mismatch")
    expected_growth = (events.view_at_24h.astype(float) - events.T0_view).clip(lower=0)
    require(np.allclose(events.loc[observed, "view_growth_24h"], expected_growth[observed]),
            "24h growth formula mismatch")
    require(np.allclose(events.loc[observed, "view_growth_24h_log"],
                        np.log1p(expected_growth[observed])), "24h log growth mismatch")
    expected24 = events.loc[observed].set_index(KEY).sort_index()
    pd.testing.assert_frame_equal(expected24[cols + ["view_growth_24h_log"]],
                                  h24.set_index(KEY).sort_index()[cols + ["view_growth_24h_log"]],
                                  check_dtype=False)
    growth_cols = [c for c in growth if c not in KEY]
    pd.testing.assert_frame_equal(left[growth_cols], growth.set_index(KEY).sort_index()[growth_cols],
                                  check_dtype=False)
    # 저장된 정답을 무조건 신뢰하지 않고 검증된 시간 차이로 다시 만든다.
    events[TARGET] = ((events.T_end - events.T0) >= pd.Timedelta(hours=48)).astype(int)
    return events.sort_values(["T0"] + KEY).reset_index(drop=True)


def audit_files(data_dir):
    """모델 성능과 무관한 파일별 품질/스키마 기록."""
    summary, schema = {}, []
    for path in sorted(data_dir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        summary[path.name] = {
            "rows": len(frame), "columns": len(frame.columns),
            "duplicate_keys": int(frame.duplicated(KEY).sum()),
            "missing": {c: int(n) for c, n in frame.isna().sum().items() if n},
        }
        require(summary[path.name]["duplicate_keys"] == 0, f"Duplicate keys in {path.name}")
        for col in frame:
            schema.append({"file": path.name, "column": col, "dtype": str(frame[col].dtype),
                           "missing": int(frame[col].isna().sum())})
    return summary, pd.DataFrame(schema)
