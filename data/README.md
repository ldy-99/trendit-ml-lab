# 데이터 계약

데이터 계약은 “이 열은 무슨 뜻이고, 언제 알 수 있으며, 어떤 조건을 만족해야 하는가”를 정한 약속입니다.
코드가 돌아가는 것과 올바른 문제를 푸는 것은 다르므로 모델 학습 전에 확인합니다.

## 출처와 확인 범위

이 저장소에는 집계된 Parquet만 있고 원시 수집 스냅샷은 없습니다.
원본 프로젝트의 아래 전처리 코드를 고정 커밋에서 읽고, 현재 Parquet의 계산값과 대조했습니다.
이 커밋이 실제 파일을 생성한 정확한 실행 커밋이라는 증거는 없으므로, **계산 규칙의 출처**로 기록합니다.

기준 저장소: `ldy-99/SKN29-2nd-1Team`, 커밋 `f06c8175c0a2a6d611e21c9e72a9e0ca9e789a1c`.

| 원본 파일 | 확인한 내용 |
|---|---|
| [clean.py](https://github.com/ldy-99/SKN29-2nd-1Team/blob/f06c8175c0a2a6d611e21c9e72a9e0ca9e789a1c/backend/preprocessing/clean.py) | UTC → KST 변환 후 시간대 정보 제거, 기본 정제 |
| [event_split.py](https://github.com/ldy-99/SKN29-2nd-1Team/blob/f06c8175c0a2a6d611e21c9e72a9e0ca9e789a1c/backend/preprocessing/event_split.py) | 같은 영상의 관측 간격이 6시간을 초과하면 새 이벤트 |
| [aggregate.py](https://github.com/ldy-99/SKN29-2nd-1Team/blob/f06c8175c0a2a6d611e21c9e72a9e0ca9e789a1c/backend/preprocessing/aggregate.py) | 최초·마지막 관측, T0 피처 집계, 지속시간 하한 6시간 |
| [features.py](https://github.com/ldy-99/SKN29-2nd-1Team/blob/f06c8175c0a2a6d611e21c9e72a9e0ca9e789a1c/backend/preprocessing/features.py) | 로그·반응률·진입 지연·시간 피처 |
| [target.py](https://github.com/ldy-99/SKN29-2nd-1Team/blob/f06c8175c0a2a6d611e21c9e72a9e0ca9e789a1c/backend/preprocessing/target.py) | 48h, 전체·카테고리 q75, TDI 정답 |
| [saturation.py](https://github.com/ldy-99/SKN29-2nd-1Team/blob/f06c8175c0a2a6d611e21c9e72a9e0ca9e789a1c/backend/preprocessing/saturation.py) | 전날까지 30일 통계로 포화도 생성 |
| [view_growth.py](https://github.com/ldy-99/SKN29-2nd-1Team/blob/f06c8175c0a2a6d611e21c9e72a9e0ca9e789a1c/backend/preprocessing/view_growth.py) | 24시간 관측과 조회수 성장량 |
| [save.py](https://github.com/ldy-99/SKN29-2nd-1Team/blob/f06c8175c0a2a6d611e21c9e72a9e0ca9e789a1c/backend/preprocessing/save.py) | 분석/T0/24h 파일 구성 및 24h 행 필터 |

원시 스냅샷에서 Parquet를 다시 만드는 end-to-end 재현은 아직 지원하지 않습니다.
현재 실험은 저장된 Parquet를 시작점으로 합니다. 매 실행마다 SHA-256 지문을 남겨 파일 변경 여부를 식별합니다.

## 파일과 한 행의 의미

| 파일 | 행 × 열 | 용도 |
|---|---:|---|
| `video_trending_events_analysis.parquet` | 34,964 × 64 | 전체 이벤트, 정답·입력 계산 검증 및 시간 분할 |
| `video_trending_events_T0_model.parquet` | 34,964 × 25 | T0 입력·정답·ID 묶음; 분석 파일과 대조 |
| `video_trending_events_24h_model.parquet` | 27,234 × 26 | 24시간 이상 관측된 이벤트만 포함 |
| `view_growth_KR.parquet` | 34,964 × 9 | 24시간 조회수 성장 및 관측 상태 |

한 행은 영상 한 개가 아니라 **영상의 한 번의 트렌딩 이벤트**입니다.
기본키는 `(video_id, event_id)`이며 `event_id`만으로는 유일하지 않습니다.
영상 17,884개 중 12,098개는 이벤트가 여러 번 있습니다. 현재 네 파일의 복합키 중복은 없습니다.

- `T0`: 이벤트의 첫 관측 시각. 실제 순위 진입 시각과 완전히 같다는 보장은 없습니다.
- `T_end`: 이벤트의 마지막 관측 시각. 실제 이탈 시각을 직접 측정한 값은 아닙니다.
- `published_at`: 업로드 시각.
- 시간은 **KST를 나타내는 timezone-naive datetime**입니다. UTC로 재해석하면 안 됩니다.
- T0 범위: `2022-07-01 09:00:00` ~ `2025-07-01 03:00:00`.

## 정답 정의와 관측 한계

```text
raw_duration = (T_end - T0)를 시간으로 환산
trending_duration_h = max(raw_duration, 6)
long_label_48h = 1 if raw_duration >= 48 else 0
```

단일 스냅샷 2,586개는 raw 0시간, 보정 후 6시간입니다. 48시간 기준에서는 보정 전후 정답이 같습니다.
첫 수집 시각에 나타난 200개 이벤트는 이미 진행 중이었을 수 있어 첫 실험에서 제외합니다.
수집 종료에 걸친 200개 이벤트는 지속시간이 잘렸을 수 있지만, 이미 48시간 이상 관측됐다면 이진 정답 1은 알 수 있습니다.
모든 클래스에 같은 54시간 관측 여유 조건을 적용해 늦게 시작한 이벤트를 제외합니다.
이는 관측창 끝의 불확실성을 줄이는 규칙이며, 내부 수집 장애까지 해결하지는 못합니다.

`long_label_global_q75`는 전체 데이터의 75분위수 240시간을 사용합니다.
`long_label_cat_q75`도 전체 기간의 카테고리별 분위수를 사용합니다.
TDI는 카테고리별 q95와 이벤트의 최고 순위를 사용합니다.
따라서 첫 실험에서는 이 라벨들을 타깃이나 입력으로 사용하지 않습니다.
나중에 분위수 타깃을 사용하려면 학습 기간에서 기준을 구하고 미래 기간에 고정 적용해야 합니다.

## 허용 입력: 진입 시점에 알 수 있는 11개

`log1p(x)`는 `log(1+x)`입니다. 조회수처럼 값의 범위가 큰 수를 압축하며 0에도 적용할 수 있습니다.

| 입력 | 계산 규칙 / 의미 |
|---|---|
| `category_group` | 카테고리 그룹; One-hot encoding으로 변환 |
| `entry_rank_log` | `log1p(첫 순위)` |
| `T0_view_log` | `log1p(첫 조회수)` |
| `T0_comment_log` | `log1p(첫 댓글 수)` |
| `T0_engagement_ratio_log` | `log1p(첫 댓글 수 / 첫 조회수)`; 좋아요 포함 지표가 아님 |
| `latency_to_trend_log` | `log1p(max(T0 - 업로드 시각, 0시간))` |
| `pretrend_view_velocity_log` | `log1p(첫 조회수 / max(진입 지연, 1시간))` |
| `published_weekday` | 업로드 요일, 월요일 0 ~ 일요일 6 |
| `hour_sin`, `hour_cos` | 업로드 시각을 24시간 주기로 변환해 23시와 0시를 가깝게 표현 |
| `saturation_index_30d_mean_prev` | 카테고리의 전날까지 30일 평균 이벤트 수 / 최대 이벤트 수 |

포화도 계산에는 이벤트가 0개인 날짜도 포함합니다. 이전 기록이 없거나 분모가 0이면 0입니다.
현재 날짜·미래 날짜의 이벤트 수나 정답은 사용하지 않습니다. 현재 파일과 재계산 결과가 일치하는지 자동 검사합니다.
기존 q75/TDI, 종료 시각·최고 순위·최종 조회수·관측 수·당일 집계·24h 성장량·ID·제목은 T0 입력으로 허용하지 않습니다.

## 결측과 24h 비교

24h 성장량 7,730개(22.11%)의 결측은 24시간 관측이 없는 이벤트입니다.
이를 0으로 채우면 “성장하지 않음”과 “관측하지 못함”을 혼동합니다.
`view_at_24h`는 24시간 이전의 마지막 스냅샷일 수도 있으므로 관측 플래그 없이 사용하면 안 됩니다.
현재 파일의 24h 관측 가능 이벤트는 모두 정확한 24h 스냅샷을 갖습니다.

첫 T0 실험은 24h 결측 때문에 행을 버리지 않습니다. 선택한 T0 피처에는 결측이 없으며,
앞으로 필수 입력에 결측·무한대가 들어오면 조용히 채우지 않고 실행을 중단합니다.
24h 비교는 **같은 이벤트, 같은 분할, 같은 타깃**으로 T0 모델과 24h 모델을 다시 평가해야 합니다.
그 결과의 적용 대상은 24시간 관측 가능한 이벤트로 좁아집니다.

전체 열 이름·타입·결측 수는 실행 결과의 `schema.csv`에 저장됩니다.
파일 간 키·선택 입력·정답·24h 성장량의 일치도 자동으로 검사합니다.
