# trendit-ml-lab

[TrendIt](https://github.com/ldy-99/SKN29-2nd-1Team)의 후속 재현·검증 실험 저장소입니다.
담당 영역은 데이터베이스 설계와 머신러닝 모델링입니다.

첫 과제는 **한국 트렌딩에 진입한 이벤트가 관측 기준 48시간 이상 지속할지, 진입 시점 정보로 예측**하는 것입니다.
모든 YouTube 업로드의 성공 여부를 예측하는 모델은 아닙니다.
기존 프로젝트와 목표·분할이 다르므로 기존 발표 성능을 그대로 재현한 결과로 해석하지 않습니다.

## 처음 읽는 순서

1. [데이터 설명](data/README.md): 한 행과 정답이 무엇인지 이해합니다.
2. [실험 따라가기](experiments/README.md): 분할 → 전처리 → 학습 → 평가 흐름을 읽습니다.
3. [첫 실험 결과](reports/baseline_v1.md): Dummy 대비 개선과 한계를 확인합니다.
4. [실험 설정](configs/baseline.toml): 날짜, 입력 열, 모델 설정을 확인합니다.

## 설치와 실행 (Windows PowerShell)

검증 환경은 **CPython 3.13.12, Windows**입니다. 먼저 `python --version`을 확인하세요.
`.python-version`은 사용할 버전의 기록이며 Python을 자동으로 설치하지 않습니다.
아래 명령은 저장소 최상위 디렉터리에서 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe experiments/run_baseline.py
```

가상환경은 이 프로젝트 전용 라이브러리 보관함입니다. 다른 프로젝트의 버전을 건드리지 않습니다.
활성화 스크립트 실행 없이 가상환경의 Python 경로를 직접 사용합니다.
이미 `.venv`가 준비돼 있다면 생성·설치를 반복할 필요 없이 테스트와 실험 명령부터 실행합니다.
macOS/Linux에서는 `.venv/Scripts/python.exe` 대신 `.venv/bin/python`을 사용하되, 이 저장소의 실제 검증 플랫폼은 Windows입니다.

실행 결과는 `reports/runs/<실행 UTC>/`에 저장됩니다. 기존 결과를 덮어쓰지 않습니다.
특정 디렉터리를 지정하려면 다음처럼 실행합니다. 디렉터리는 아직 존재하지 않아야 합니다.

```powershell
.\.venv\Scripts\python.exe experiments/run_baseline.py --output reports/runs/my_first_run
```

`requirements.txt`는 주요 의존성, `requirements-lock.txt`는 검증 당시의 전체 의존성 버전입니다.
재현할 때는 lock 파일을 사용합니다. PyTorch/Jupyter는 이번 실험에 필요하지 않습니다.
`fundamentals/`의 학습용 노트북은 자체 안내에 따라 별도 환경에서 실행하세요.

## 실험에서 고정한 규칙

| 항목 | 규칙 | 이유 |
|---|---|---|
| 정답 | `T_end - T0 >= 48시간` | 전체 데이터의 분위수 없이 의미를 고정할 수 있음 |
| 입력 | T0에 이용 가능한 수치 10개 + 카테고리 1개 | 미래 결과가 입력에 섞이지 않게 함 |
| 학습 / 검증 / 시험 | 2022~2023 / 2024 / 2025 | 과거로 학습해 미래를 평가 |
| 경계 처리 | 다음 기간 시작·수집 종료까지 54시간 초과 여유 | 정답을 확인할 시간을 확보 |
| 영상 처리 | 이전 기간에 나타난 영상은 다음 기간에서 제외 | 같은 영상으로 시험 점수가 부풀 위험을 줄임 |
| 모델 후보 | Dummy, 로지스틱 회귀, 랜덤포레스트 | 단순 기준부터 개선 여부 확인 |
| 모델 선택 | 검증 balanced accuracy, 동률이면 앞선 단순 모델 | 시험 결과에 맞춰 모델을 고르지 않음 |
| 임계값 | 0.5 고정 | 첫 실험에서 튜닝 변수를 줄임 |

이는 **이전 기간에 없던 영상의 미래 이벤트**를 평가하는 정책입니다. 동일 채널은 허용합니다.
각 기간 안에서는 같은 영상의 여러 이벤트가 남으므로 영상당 한 행인 데이터와도 다릅니다.

## 폴더 역할

| 경로 | 역할 |
|---|---|
| `data/` | 원본으로 보관하는 파생 Parquet 4개와 데이터 계약 |
| `configs/` | 실행 전 고정하는 실험 설정 |
| `src/trendit_lab/data.py` | 키·라벨·피처 계산·파일 간 일치 검증 |
| `src/trendit_lab/split.py` | 시간, 정답 관측 여유, 영상 중복에 따른 분할 |
| `src/trendit_lab/model.py` | 전처리 Pipeline, 모델, 평가 지표 |
| `src/trendit_lab/experiment.py` | 실행 순서와 결과 저장 |
| `experiments/run_baseline.py` | 명령줄 진입점 |
| `tests/` | 누수·경계·피처 오류를 잡는 테스트 |
| `reports/` | 검토용 결과와 실행 기록 |
| `fundamentals/` | 프로젝트 실험과 구분한 기초 학습 자료 |

다음 단계는 24시간 관측이 가능한 **같은 이벤트 집합**에서 T0 피처와 24h 피처를 비교하는 것입니다.
현재 시험 점수를 보고 바꾸는 실험은 탐색으로 표시하고, 확증에는 별도의 미래 기간을 확보해야 합니다.
