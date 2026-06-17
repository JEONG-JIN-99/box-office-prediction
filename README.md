# Box Office Prediction

KOFIC 일별 박스오피스 데이터를 기반으로 영화 개봉 첫 7일간의 지표를 입력받아 최종 누적 관객수를 예측하는 PyTorch 기반 MLP 프로젝트입니다.

## 주요 기능

- KOFIC Open API를 이용한 일별 박스오피스 데이터 수집
- 개봉 첫 7일 데이터를 영화 단위 피처로 전처리
- 최종 누적 관객수 예측용 MLP 모델 학습
- 1단계 Random Search, 2단계 Grid Search 기반 하이퍼파라미터 탐색
- 학습 결과 그래프 생성
- CLI 기반 예측 실행

## 프로젝트 구조

```text
box-office-prediction/
  train.py
  predict.py
  requirements.txt
  project_plan.md

  data/
    __init__.py
    collect_data.py
    data_config.py
    raw/
    processed/

  experiments/
    experiment_configs.py
    hyperparameter_search.py
    results/

  models/
    model.py
    best_model.pth
    scaler_config.json

  utils/
    feature_engineering.py
    metrics.py
    preprocessing.py

  visualization/
    plot_results.py
    figures/
```

## 설치

Python 3.11 기준으로 사용했습니다.

```powershell
pip install -r requirements.txt
```

그래프의 일부 기능은 `scikit-learn`을 사용합니다. 현재 환경에서 그래프 생성 중 `RandomForestRegressor` 관련 오류가 나면 추가 설치가 필요합니다.

```powershell
pip install scikit-learn
```

## 설정

데이터 수집 및 경로 설정은 [data/data_config.py](data/data_config.py)에 있습니다.

주요 설정:

- `API_KEY`: KOFIC Open API 키
- `COLLECT_START_DATE`: 수집 시작일
- `COLLECT_END_DATE`: 수집 종료일
- `RAW_CSV_PATH`: 원본 일별 박스오피스 CSV 경로
- `FINAL_CSV_PATH`: 최종 학습용 피처 CSV 경로
- `FEATURE_DAYS`: 개봉 후 사용할 일수, 현재 7일

`PROJECT_ROOT`는 프로젝트 최상위 폴더를 가리킵니다.

## 데이터 처리 흐름

1. 원본 데이터 수집

```powershell
python data/collect_data.py
```

2. 원본 데이터를 영화별 개봉 첫 7일 데이터로 전처리

```powershell
python utils/preprocessing.py
```

3. 최종 학습용 파생 피처 생성

```powershell
python utils/feature_engineering.py
```

최종 학습 데이터는 기본적으로 아래 파일에 저장됩니다.

```text
data/processed/movie_features_final.csv
```

## 사용 피처

모델 입력은 총 16개입니다.

| 순서 | 피처명 | 의미 |
|---:|---|---|
| 1 | `day1_audience` | 개봉 1일차 관객수 |
| 2 | `day2_audience` | 개봉 2일차 관객수 |
| 3 | `day3_audience` | 개봉 3일차 관객수 |
| 4 | `day4_audience` | 개봉 4일차 관객수 |
| 5 | `day5_audience` | 개봉 5일차 관객수 |
| 6 | `day6_audience` | 개봉 6일차 관객수 |
| 7 | `day7_audience` | 개봉 7일차 관객수 |
| 8 | `open_year` | 개봉 연도 |
| 9 | `open_month` | 개봉 월 |
| 10 | `week1_total` | 첫 7일 관객수 합 |
| 11 | `avg_screen_count` | 첫 7일 평균 스크린 수 |
| 12 | `avg_show_count` | 첫 7일 평균 상영 횟수 |
| 13 | `avg_rank` | 첫 7일 평균 박스오피스 순위 |
| 14 | `rank_trend` | 첫 7일 순위 변화 평균 |
| 15 | `day_over_day_ratio` | `day7_audience / day1_audience` |
| 16 | `weekend_ratio` | 첫 7일 중 주말 관객수 비율 |

`predict.py`에서는 입력받은 7일 관객수와 개봉일을 기준으로 `week1_total`, `day_over_day_ratio`, `weekend_ratio`를 자동 계산합니다.

## 학습

기본 학습 실행:

```powershell
python train.py
```

현재 코드 기준으로 `train.py`의 `main()`은 하이퍼파라미터 탐색 단계를 자동 실행하지 않고, 코드 안에 지정된 `best_params`로 최종 학습을 수행합니다.

최종 학습이 완료되면 아래 파일이 생성 또는 갱신됩니다.

```text
models/best_model.pth
models/scaler_config.json
visualization/figures/G3_learning_curve.png
visualization/figures/G5_pred_vs_actual.png
```

## 하이퍼파라미터 탐색

탐색 로직은 [experiments/hyperparameter_search.py](experiments/hyperparameter_search.py)에 정의되어 있습니다.

현재 구조:

- 1단계: 전체 후보 조합 중 중복 없는 Random Search 200회
- 2단계: 1단계 상위 3개 조합 주변 Grid Search

탐색 결과는 아래 위치에 저장됩니다.

```text
experiments/results/stage1_random_search.csv
experiments/results/stage1_best.csv
experiments/results/stage2_grid_candidates.csv
experiments/results/stage2_grid_search.csv
```

최근 실험 기준 best 조합:

```text
learning_rate = 0.005
num_layers = 5
hidden_units = 512
dropout_rate = 0.0
batch_size = 128
```

검증 성능:

```text
val_mae  = 234,195.66
val_mape = 25.86%
val_r2   = 0.8317
```

최종 테스트 성능:

```text
test_mae  = 355,924.29
test_mape = 39.74%
test_r2   = 0.7901
```

## 예측 실행

예측에는 아래 두 파일이 필요합니다.

```text
models/best_model.pth
models/scaler_config.json
```

예시:

```powershell
python predict.py `
  --day1 117783 --day2 91731 --day3 126625 `
  --day4 327112 --day5 308083 --day6 98379 --day7 95582 `
  --open_date 2026-02-04 `
  --avg_screen 1632.86 `
  --avg_show 6980.57 `
  --avg_rank 1.0 `
  --rank_trend 1.43
```

Windows 콘솔에서 출력 인코딩 문제가 생기면 아래 환경변수를 설정한 뒤 실행합니다.

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

## 그래프 생성

Stage 1, Stage 2 탐색 결과 그래프는 `visualization/plot_results.py`의 `generate_all_plots()`에서 생성합니다.

그래프 출력 위치:

```text
visualization/figures/stage1/
visualization/figures/stage2/
```

현재 환경에서 matplotlib이 Tk GUI backend를 잡으면서 오류가 날 수 있습니다. 그 경우 아래 환경변수를 설정합니다.

```powershell
$env:MPLBACKEND = "Agg"
```

## GitHub 업로드 정책

현재 `.gitignore` 기준으로 아래 파일들은 GitHub에 올리지 않습니다.

- `__pycache__/`
- `__cache__/`
- 가상환경 폴더
- `data/raw/`
- `data/processed/`
- `experiments/results/`
- `visualization/figures/`
- `models/*.pt`
- `models/*.ckpt`

`predict.py`를 바로 실행할 수 있도록 아래 파일들은 GitHub에 포함합니다.

```text
models/best_model.pth
models/scaler_config.json
```

## 주의사항

- `train.py`는 현재 최신 `stage2_grid_search.csv`의 best 조합을 자동으로 읽어오지 않습니다.
- `predict.py`는 `models/best_model.pth`와 `models/scaler_config.json`이 모두 있어야 실행됩니다.
- 데이터 파일은 GitHub에 올리지 않도록 설정되어 있으므로, 새 환경에서는 데이터 수집과 전처리를 다시 수행해야 합니다.
- 실행 결과 CSV, 그래프, 모델 weight는 실행할 때마다 최신 결과로 덮어써지는 구조입니다.
