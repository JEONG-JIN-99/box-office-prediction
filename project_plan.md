# 한국 영화 최종 관객수 예측 프로젝트 계획서

> 목적: KOFIC 박스오피스 데이터를 기반으로 영화의 개봉 첫 7일 관객수와 상영 지표를 입력하면 최종 누적 관객수를 예측하는 회귀 모델을 만든다.

---

## 1. 프로젝트 개요

### 문제 정의

| 항목 | 내용 |
|---|---|
| 입력 | 개봉 첫 7일 일별 관객수, 개봉일, 평균 스크린 수, 평균 상영 횟수, 평균 순위, 순위 변동 |
| 출력 | 해당 영화의 최종 누적 관객수 예측값 |
| 모델 유형 | 회귀 모델 |
| 주 모델 | PyTorch 기반 MLP |
| 학습 방식 | 로그 변환된 최종 관객수를 예측하고, 예측 후 원 단위 관객수로 복원 |

### 현재 구현 상태

- 데이터 수집: `data/collect_data.py`
- 피처 엔지니어링: `utils/feature_engineering.py`
- 모델 정의: `models/model.py`
- 하이퍼파라미터 탐색: `experiments/hyperparameter_search.py`
- 최종 학습 및 저장: `train.py`
- 예측 실행: `predict.py`
- 결과 시각화: `visualization/plot_results.py`

---

## 2. 데이터와 피처

### 데이터 출처

- KOFIC Open API의 일별 박스오피스 데이터를 사용한다.
- KOFIC 일별 박스오피스는 Top 10 중심 데이터이므로, 각 영화의 최종 관객수는 수집된 기간 중 `audiAcc`의 최댓값을 proxy target으로 사용한다.

### 최종 학습 피처 16개

최종 피처는 `utils/feature_engineering.py`와 `experiments/hyperparameter_search.py` 기준으로 아래 순서를 사용한다.

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
| 10 | `week1_total` | 첫 7일 관객수 합계 |
| 11 | `avg_screen_count` | 7일 평균 스크린 수 |
| 12 | `avg_show_count` | 7일 평균 상영 횟수 |
| 13 | `avg_rank` | 7일 평균 박스오피스 순위 |
| 14 | `rank_trend` | 7일 평균 순위 변동값 |
| 15 | `day_over_day_ratio` | `day7_audience / day1_audience` |
| 16 | `weekend_ratio` | 첫 7일 중 토요일/일요일 관객수 비율 |

### 주요 파생 피처 계산 방식

```python
week1_total = day1 + day2 + ... + day7
avg_screen_count = mean(day1_scrnCnt ... day7_scrnCnt)
avg_show_count = mean(day1_showCnt ... day7_showCnt)
avg_rank = mean(day1_rank ... day7_rank)
rank_trend = mean(day1_rankInten ... day7_rankInten)
day_over_day_ratio = day7_audience / day1_audience
weekend_ratio = weekend_audience / week1_total
```

`day1_audience`가 0이면 `day_over_day_ratio`는 0으로 처리한다. `week1_total`이 0이면 `weekend_ratio`는 0으로 처리한다.

---

## 3. 모델 구조

### MLP 모델

모델은 `models/model.py`의 `BoxOfficeMLP`를 사용한다.

구조는 다음과 같다.

```text
Input: 16개 피처
반복 hidden block:
  Linear
  BatchNorm1d
  ReLU
  Dropout
Output:
  Linear -> 1개 값
```

예측 대상은 최종 관객수의 `log1p` 값이다. 평가 시에는 `expm1`으로 실제 관객수 스케일로 복원한다.

### 학습 지표

| 지표 | 용도 |
|---|---|
| MSELoss | 학습 손실 |
| MAE | 평균 절대 오차, 관객수 단위 해석에 사용 |
| MAPE | 평균 절대 백분율 오차 |
| R2 | 결정계수 |

---

## 4. 하이퍼파라미터 탐색

### 탐색 대상

`experiments/experiment_configs.py` 기준 현재 탐색 범위는 다음과 같다.

| 하이퍼파라미터 | 후보 |
|---|---|
| `learning_rate` | `[0.0001, 0.0005, 0.001, 0.005, 0.01]` |
| `num_layers` | `[2, 3, 4, 5]` |
| `hidden_units` | `[64, 128, 256, 512]` |
| `dropout_rate` | `[0.0, 0.1, 0.2, 0.3, 0.5]` |
| `batch_size` | `[16, 32, 64, 128]` |

전체 가능한 조합 수는 `5 * 4 * 4 * 5 * 4 = 1,600`개다.

### 1단계: Random Search

`run_stage1_random_search()`가 수행한다.

- 전체 1,600개 조합 중 중복 없는 무작위 조합을 선택한다.
- 현재 설정은 `STAGE1_NUM_TRIALS = 200`이다.
- `set_seed(42)`로 샘플링 순서를 재현 가능하게 만든다.
- 각 trial은 하이퍼파라미터 조합 1개를 학습/검증한 1회 실험을 뜻한다.

저장 파일:

```text
experiments/results/stage1_random_search.csv
experiments/results/stage1_best.csv
```

`stage1_best.csv`에는 `val_mae` 기준 상위 3개 조합이 저장된다.

### 2단계: Grid Search

`run_stage2_grid_search()`가 수행한다.

흐름:

1. 1단계 결과에서 `val_mae`가 가장 낮은 상위 3개 조합을 고른다.
2. 상위 3개 조합의 값을 기준으로 주변 후보 범위를 만든다.
3. 만들어진 후보 조합 전체를 Grid Search로 실행한다.

Learning rate는 상위 값마다 `0.5배`, `원래 값`, `2배`를 포함하되 `0.00009 <= lr <= 0.011` 범위만 유지한다.

이산형 값인 `num_layers`, `hidden_units`, `dropout_rate`, `batch_size`는 기존 후보 리스트에서 선택된 값의 이전 값, 자기 자신, 다음 값을 포함한다.

저장 파일:

```text
experiments/results/stage2_grid_candidates.csv
experiments/results/stage2_grid_search.csv
```

`run_stage2_grid_search()`는 다음 세 값을 반환한다.

```python
df_grid_results, best_params, best_model_weights
```

현재 파이프라인에서 실제로 최종 학습에 필요한 값은 `best_params`다. `best_model_weights`는 탐색 중 최고 조합의 가중치이지만, 최종 모델 저장 단계에서는 `best_params`로 다시 학습한다.

---

## 5. `train.py`의 현재 역할

`train.py`는 전체 학습 파이프라인을 실행하기 위한 메인 스크립트다.

현재 주요 구성:

- `plot_g3_learning_curve()`: 최종 학습 곡선 저장
- `plot_g5_pred_vs_actual()`: 테스트셋 실제값과 예측값 산점도 저장
- `train_best_model_final()`: 주어진 `best_params`로 최종 모델 학습, 평가, 저장
- `main()`: 데이터 로드, 최종 학습, 그래프 생성 흐름 제어

### 현재 `main()`의 중요한 상태

현재 `train.py`의 1단계/2단계 탐색 호출은 주석 처리되어 있다.

```python
# stage1_results = run_stage1_random_search(train_data, val_data, device)
# stage2_results, best_params, best_model_weights = run_stage2_grid_search(
#     stage1_results, train_data, val_data, device
# )
```

또한 현재 `main()`에는 예전 best 값이 하드코딩되어 있다.

```python
best_params = {
    "learning_rate": 0.005,
    "num_layers": 5,
    "hidden_units": 128,
    "dropout_rate": 0.0,
    "batch_size": 128,
}
```

따라서 `python train.py`를 그대로 실행하면 최신 `stage2_grid_search.csv`의 best 조합을 자동으로 읽지 않는다. 최신 best 조합으로 최종 학습하려면 `train_best_model_final(best_params, ...)`에 최신 값을 직접 전달하거나, `train.py`의 `main()` 흐름을 최신 결과를 읽도록 바꿔야 한다.

### 현재 최신 실험 기준 best 조합

최근 실행된 `stage2_grid_search.csv` 기준 best 조합은 다음이다.

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

이 조합으로 최종 학습한 테스트 성능은 `models/scaler_config.json`에 저장되어 있다.

```text
test_mae  = 355,924.29
test_mape = 39.74%
test_r2   = 0.7901
```

---

## 6. 예측 실행 구조

`predict.py`는 아래 두 파일을 사용한다.

```text
models/best_model.pth
models/scaler_config.json
```

`scaler_config.json`에는 다음 정보가 들어간다.

- 학습 데이터 기준 feature mean
- 학습 데이터 기준 feature std
- `best_hyperparameters`
- 최종 테스트 성능 `test_performance`

`predict.py`는 CLI 입력값으로 7일 관객수와 보조 지표를 받고, 내부에서 다음 파생 피처를 계산한다.

```python
week1_total = sum(day_audiences)
day_over_day_ratio = round(day7 / day1, 4)
weekend_ratio = weekend_audience / week1_total
```

실행 예:

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

Windows 콘솔에서 이모지 출력 인코딩 문제가 나면 다음 환경변수를 지정해서 실행한다.

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

---

## 7. 시각화 산출물

`visualization/plot_results.py`의 `generate_all_plots()`가 `stage1`과 `stage2` 결과 CSV를 읽어 그래프를 생성한다.

저장 위치:

```text
visualization/figures/stage1/
visualization/figures/stage2/
```

생성 그래프:

| 그래프 | 내용 |
|---|---|
| G1 | 하이퍼파라미터별 MAE 박스플롯 |
| G2 | 하이퍼파라미터 2개 조합별 평균 MAE 히트맵 |
| G4 | 상위 10개 조합 MAE 비교 |
| G6 | RandomForestRegressor 기반 하이퍼파라미터 중요도 |
| G7 | 병렬 좌표 그래프 |

최종 학습 그래프는 `train_best_model_final()`에서 생성한다.

```text
visualization/figures/G3_learning_curve.png
visualization/figures/G5_pred_vs_actual.png
```

현재 환경에서는 matplotlib GUI backend 문제를 피하기 위해 `MPLBACKEND=Agg`를 지정하는 것이 안전하다.

```powershell
$env:MPLBACKEND = "Agg"
```

---

## 8. 산출물과 덮어쓰기 정책

현재 프로젝트는 실행할 때마다 최신 결과로 덮어쓰는 구조다.

덮어써지는 주요 파일:

```text
experiments/results/stage1_random_search.csv
experiments/results/stage1_best.csv
experiments/results/stage2_grid_candidates.csv
experiments/results/stage2_grid_search.csv
models/best_model.pth
models/scaler_config.json
visualization/figures/G3_learning_curve.png
visualization/figures/G5_pred_vs_actual.png
visualization/figures/stage1/*.png
visualization/figures/stage2/*.png
```

실행별 아카이브는 현재 구현되어 있지 않다.

---

## 9. 프로젝트 디렉터리 구조

```text
box-office-prediction/
  config.py
  train.py
  predict.py
  requirements.txt
  project_plan.md

  data/
    collect_data.py
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

---

## 10. 의존성

현재 `requirements.txt` 기준 주요 의존성:

```text
requests
pandas
numpy
torch
matplotlib
seaborn
```

그래프 G6는 `scikit-learn`의 `RandomForestRegressor`를 사용하므로 실행 환경에는 `scikit-learn`도 필요하다.

---

## 11. 현재 주의할 점

1. `train.py main()`은 최신 stage2 결과를 자동으로 읽지 않고, 예전 best 파라미터가 하드코딩되어 있다.
2. 최신 best 조합은 `hidden_units=512`지만, `train.py main()`의 하드코딩 값은 `hidden_units=128`이다.
3. `predict.py`는 `models/best_model.pth`와 `models/scaler_config.json`을 기준으로 동작한다.
4. 최종 테스트 성능은 `models/scaler_config.json`의 `test_performance`에 저장된다.
5. 탐색 결과의 검증 성능은 `experiments/results/stage2_grid_search.csv`에 저장된다.
6. Windows 콘솔에서는 `PYTHONIOENCODING=utf-8`, 그래프 생성 시에는 `MPLBACKEND=Agg`가 필요할 수 있다.

---

## 12. 전체 실행 흐름 요약

현재 코드 기준 권장 실행 흐름:

1. 데이터 수집 및 전처리
2. `run_stage1_random_search()`로 200회 Random Search
3. `run_stage2_grid_search()`로 상위 3개 주변 Grid Search
4. `stage2_grid_search.csv`에서 best 조합 확인
5. `train_best_model_final(best_params, ...)`로 최종 모델 저장
6. `generate_all_plots()`로 stage1/stage2 그래프 생성
7. `predict.py`로 신규 영화 예측

`train.py` 하나로 완전 자동화하려면, `main()`에서 주석 처리된 1단계/2단계 탐색을 활성화하고 하드코딩된 `best_params` 대신 2단계 결과의 `best_params`를 사용하도록 정리해야 한다.
