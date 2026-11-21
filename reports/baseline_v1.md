# T0에서 48시간 이상 지속 여부 예측: 첫 기준 실험

이 결과는 고정된 과거 데이터에 대한 후속 검증이며, 기존 프로젝트 성능의 동일 조건 재현은 아닙니다.

## 실험 규칙

- 입력: 진입 시점의 11개 피처. 정답: 관측된 이벤트 길이 48시간 이상 여부.
- 검증 시작: 2024-01-01 00:00:00; 시험 시작: 2025-01-01 00:00:00 (KST).
- 기간 경계 및 수집 종료까지 54시간보다 긴 관측 여유가 있는 이벤트만 사용합니다.
- 첫 수집 스냅샷과 이전 기간에 나타난 영상의 후속 기간 이벤트는 제외합니다.
- 전처리와 모델은 학습 세트만 fit합니다. 모델 선택 후 재학습하지 않습니다.
- 임계값 0.5 고정. 검증 balanced accuracy로 선택하고 동률이면 단순한 모델을 유지합니다.
- 시험에서는 선택 모델과 Dummy만 평가합니다. 시험 점수로 모델을 다시 고르지 않습니다.

## 데이터 분할

| 세트 | 이벤트 | 영상 | 48h 이상 비율 | 시작 | 마지막 진입 |
|---|---:|---:|---:|---|---|
| train | 18,628 | 9,781 | 66.67% | 2022-07-01 15:00:00 | 2023-12-29 15:00:00 |
| validation | 10,528 | 5,408 | 68.61% | 2024-01-01 09:00:00 | 2024-12-29 15:00:00 |
| test | 5,299 | 2,484 | 67.88% | 2025-01-01 03:00:00 | 2025-06-28 15:00:00 |

제외 사유(중복 집계 없음): `{'initial_snapshot': 200, 'video_seen_in_earlier_period': 159, 'insufficient_followup': 150}`

## 검증 결과: 모델을 고르는 자료

| 모델 | Balanced accuracy | ROC AUC | AP | F1 | Log loss |
|---|---:|---:|---:|---:|---:|
| dummy | 0.5000 | 0.5000 | 0.6861 | 0.8138 | 0.6231 |
| logistic | 0.5607 | 0.7141 | 0.8375 | 0.8195 | 0.5616 |
| random_forest | 0.7746 | 0.8838 | 0.9325 | 0.8871 | 0.3916 |

검증에서 선택한 모델: **random_forest**

## 최종 시험 결과

| 모델 | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| dummy | 0.6788 | 0.5000 | 0.6788 | 1.0000 | 0.8087 | 0.5000 | 0.6788 |
| random_forest | 0.8315 | 0.7719 | 0.8340 | 0.9386 | 0.8832 | 0.8782 | 0.9244 |

선택 모델의 Dummy 대비 시험 balanced accuracy 차이: **+0.2719**.
선택 모델의 혼동행렬 `[[TN, FP], [FN, TP]]`: `[[1030, 672], [221, 3376]]`.

정확도는 전체 정답 비율입니다. Balanced accuracy는 0과 1 각각의 재현율 평균이며,
한쪽 답만 하는 모델은 두 클래스가 있는 데이터에서 0.5입니다. F1은 양성 판정의
정밀도와 재현율을 함께 봅니다. AP는 확률 순위의 지표로, 사다리꼴 PR AUC와 같지 않습니다.
Log loss는 틀린 답에 높은 확신을 줄수록 커지며 작을수록 좋습니다.

## 해석 범위와 남은 한계

- 이미 트렌딩에 진입한 영상의 이벤트만 대상으로 합니다. 모든 업로드의 성공 예측이 아닙니다.
- 같은 영상은 세트 사이에서 분리했지만 같은 채널은 허용합니다. 새 채널 일반화는 검증하지 않았습니다.
- 한 기간 안에 같은 영상의 여러 이벤트가 남습니다. 지표는 이벤트 단위이며 독립 표본으로 간주한 신뢰구간은 제시하지 않습니다.
- 지속시간은 실제 연속 노출 시간이 아니라 6시간 수집과 이벤트 분리 규칙으로 정의한 관측값입니다.
- 종료 시점에 계속 관측 중인 이벤트라도 48시간을 이미 넘었다면 이진 정답은 확정됩니다.
- 원시 스냅샷이 없어 내부 수집 장애, 삭제/수정 이력, 최초 관측값의 진위까지 검증하지는 못했습니다.
- 포화도는 예측일 전날까지의 이벤트 수로 재계산했습니다. 시험 기간에도 그 시점까지 관측한 입력 이력을 사용할 수 있다는 가정입니다.
- 24h 모델은 24시간 관측 가능한 동일 이벤트 집합에서 따로 비교해야 합니다.
- 이 시험 결과를 본 뒤의 변경은 탐색입니다. 새로운 확증에는 별도의 미래 기간이 필요합니다.

## 재현 기록

실행 UTC: `2026-11-21T10:49:32.016133+00:00`. Python: `3.13.12`.
`config.toml`, `manifest.json`, `audit.json`, `schema.csv`, `split_assignments.csv`,
`validation_predictions.csv`, `test_predictions.csv`, `metrics.json`을 함께 보관합니다.
분할 CSV에는 제외된 이벤트와 제외 이유도 남깁니다. 소스 해시는 커밋 전 코드 변경도 식별합니다.

## Verification

- `python -m pytest -q`: 18 passed.
- `python -m pip check`: no broken requirements.
- Two runs produced byte-identical metrics, audit, schema, split assignments, predictions and configuration.
- Data and source SHA-256 values matched the saved manifest.
- Full local artifacts: `reports/runs/baseline_v1/`. Generated runs are ignored by Git.
- Review snapshots: `baseline_v1_metrics.json`, `baseline_v1_audit.json`, `baseline_v1_manifest.json`.
