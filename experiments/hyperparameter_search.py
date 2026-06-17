# -*- coding: utf-8 -*-
"""
하이퍼파라미터 자동 탐색 모듈
=============================
1단계(무작위 탐색)와 2단계(정밀 격자 탐색)를 분리하여 수행하고,
각 단계별 실험 로그를 기록합니다. 최적의 모델 가중치와 파라미터를 저장합니다.
"""

import os
import sys
import time
import json
import random
import logging
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.model import BoxOfficeMLP
from utils.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import experiments.experiment_configs as exp_config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 재현성을 위한 시드 설정 함수
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# PyTorch 데이터셋 정의
# ============================================================
class BoxOfficeDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# 데이터 로드 및 전처리 (수동 스케일러 구현)
# ============================================================
class SimpleStandardScaler:
    """scikit-learn 없이 z-score 정규화를 수행하는 간단한 스케일러"""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # 분모가 0이 되는 것을 방지
        self.std[self.std == 0] = 1e-8

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


def load_and_split_data(csv_path: str, seed: int = 42) -> Tuple:
    """
    데이터를 로드하고 Train/Val/Test (70% : 15% : 15%) 비율로 분할합니다.
    피처 스케일링 및 타겟 로그 변환도 함께 적용합니다.
    """
    if not os.path.exists(csv_path):
        logger.error(f"최종 피처 데이터 파일이 존재하지 않습니다: {csv_path}")
        logger.error("먼저 utils/feature_engineering.py를 실행해 주세요.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    
    # 예측에 사용할 16개 피처명 정의
    feature_cols = [
        "day1_audience", "day2_audience", "day3_audience", "day4_audience",
        "day5_audience", "day6_audience", "day7_audience",
        "open_year", "open_month",
        "week1_total", "avg_screen_count", "avg_show_count",
        "avg_rank", "rank_trend", "day_over_day_ratio", "weekend_ratio"
    ]
    
    X = df[feature_cols].values
    y = df["target_audience"].values

    # 인덱스 무작위 셔플 (재현성을 위해 시드 고정)
    set_seed(seed)
    indices = np.arange(len(df))
    np.random.shuffle(indices)

    # 7 : 1.5 : 1.5 비율 분할 계산
    n_total = len(df)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # 피처 스케일링 (Train 기준으로 Fit)
    scaler = SimpleStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 타겟 로그 변환: log1p 적용 (관객수가 비대칭적으로 크므로 학습 안정화용)
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    y_test_log = np.log1p(y_test)

    logger.info(f"데이터셋 분할 완료:")
    logger.info(f"  Train: {len(X_train)}건 | Val: {len(X_val)}건 | Test: {len(X_test)}건")

    return (
        X_train_scaled, y_train_log, y_train,
        X_val_scaled, y_val_log, y_val,
        X_test_scaled, y_test_log, y_test,
        scaler
    )


# ============================================================
# 단일 모델 학습 및 평가 함수
# ============================================================
def train_and_evaluate(
    params: Dict,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    device: torch.device
) -> Tuple[float, float, float, Dict]:
    """
    지정된 하이퍼파라미터로 모델을 학습하고 검증 셋에서 평가합니다 (Early Stopping 적용).
    """
    X_train, y_train_log = train_data
    X_val, y_val_log, y_val_raw = val_data

    # 데이터 로더 구성
    train_dataset = BoxOfficeDataset(X_train, y_train_log)
    val_dataset = BoxOfficeDataset(X_val, y_val_log)

    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False)

    # 모델 선언
    model = BoxOfficeMLP(
        input_dim=16,
        num_layers=params["num_layers"],
        hidden_units=params["hidden_units"],
        dropout_rate=params["dropout_rate"]
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    # 조기 종료(Early Stopping) 변수 초기화
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, exp_config.MAX_EPOCHS + 1):
        # 학습 모드
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(batch_x)
        train_loss /= len(train_dataset)
        history["train_loss"].append(train_loss)

        # 검증 모드
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item() * len(batch_x)
        val_loss /= len(val_dataset)
        history["val_loss"].append(val_loss)

        # 조기 종료 체크
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= exp_config.EARLY_STOPPING_PATIENCE:
            break

    # 최적 가중치 복원
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # 최종 검증 평가 (로그 스케일 해제 후 실제 관객수로 계산)
    model.eval()
    val_preds_log = []
    with torch.no_grad():
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            val_preds_log.extend(pred.cpu().numpy().flatten())

    val_preds_log = np.array(val_preds_log)
    # 로그의 역변환 적용: expm1
    val_preds = np.expm1(val_preds_log)
    # 음수 예측 방지
    val_preds = np.clip(val_preds, 0, None)

    # 지표 계산
    mae = mean_absolute_error(y_val_raw, val_preds)
    mape = mean_absolute_percentage_error(y_val_raw, val_preds)
    r2 = r2_score(y_val_raw, val_preds)

    return mae, mape, r2, best_model_state


# ============================================================
# 1단계: Random Search (무작위 탐색)
# ============================================================
def run_stage1_random_search(
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    device: torch.device
) -> pd.DataFrame:
    """
    1단계: 무작위 조합으로 하이퍼파라미터를 탐색합니다.
    """
    logger.info("=" * 60)
    logger.info("🚀 [1단계] Random Search 시작")
    logger.info("=" * 60)

    results = []
    sampled_combinations = set()

    # 가능한 총 조합 수 계산
    total_possible = (
        len(exp_config.LEARNING_RATES)
        * len(exp_config.HIDDEN_LAYERS)
        * len(exp_config.HIDDEN_UNITS)
        * len(exp_config.DROPOUT_RATES)
        * len(exp_config.BATCH_SIZES)
    )
    num_trials = min(exp_config.STAGE1_NUM_TRIALS, total_possible)

    set_seed(42)  # 재현 가능한 무작위성

    for trial in range(1, num_trials + 1):
        # 고유한 조합이 샘플링될 때까지 반복
        while True:
            lr = random.choice(exp_config.LEARNING_RATES)
            layers = random.choice(exp_config.HIDDEN_LAYERS)
            units = random.choice(exp_config.HIDDEN_UNITS)
            dropout = random.choice(exp_config.DROPOUT_RATES)
            batch = random.choice(exp_config.BATCH_SIZES)
            
            combo = (lr, layers, units, dropout, batch)
            if combo not in sampled_combinations:
                sampled_combinations.add(combo)
                break

        params = {
            "learning_rate": lr,
            "num_layers": layers,
            "hidden_units": units,
            "dropout_rate": dropout,
            "batch_size": batch
        }

        start_t = time.time()
        mae, mape, r2, _ = train_and_evaluate(params, train_data, val_data, device)
        elapsed = time.time() - start_t

        logger.info(
            f"Trial {trial:3d}/{num_trials} | "
            f"LR={lr:<6.4f}, Layers={layers}, Units={units:3d}, Dropout={dropout:<3.1f}, Batch={batch:3d} | "
            f"MAE={mae:11,.1f}명, MAPE={mape:5.2f}%, R2={r2:6.4f} | 소요시간={elapsed:.1f}초"
        )

        results.append({
            "trial": trial,
            "learning_rate": lr,
            "num_layers": layers,
            "hidden_units": units,
            "dropout_rate": dropout,
            "batch_size": batch,
            "val_mae": mae,
            "val_mape": mape,
            "val_r2": r2,
            "time_seconds": elapsed
        })

    # 결과 DataFrame 저장
    os.makedirs(exp_config.RESULTS_DIR, exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv(exp_config.STAGE1_LOG_PATH, index=False, encoding="utf-8")
    logger.info(f"💾 1단계 결과 저장 완료: {exp_config.STAGE1_LOG_PATH}")
    stage1_best = df_results.nsmallest(3, "val_mae").reset_index(drop=True)
    stage1_best.insert(0, "rank", range(1, len(stage1_best) + 1))
    stage1_best.to_csv(exp_config.STAGE1_BEST_PATH, index=False, encoding="utf-8")
    logger.info(f"1단계 상위 3개 결과 저장 완료: {exp_config.STAGE1_BEST_PATH}")

    return df_results


# ============================================================
# 2단계: Grid Search (정밀 격자 탐색)
# ============================================================
def run_stage2_grid_search(
    stage1_results: pd.DataFrame,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    device: torch.device
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    1단계 성적이 좋은 상위 조합의 주변 영역을 정밀하게 탐색합니다.
    """
    logger.info("=" * 60)
    logger.info("🎯 [2단계] 정밀 Grid Search 시작")
    logger.info("=" * 60)

    # 1단계 최우수 3개 모델 선택
    top_3 = stage1_results.nsmallest(3, "val_mae")
    logger.info("1단계 상위 3개 조합:")
    for rank, (_, row) in enumerate(top_3.iterrows(), 1):
        logger.info(
            f"  순위 {rank}: MAE={row['val_mae']:,.1f} | "
            f"LR={row['learning_rate']}, Layers={int(row['num_layers'])}, "
            f"Units={int(row['hidden_units'])}, Dropout={row['dropout_rate']}, Batch={int(row['batch_size'])}"
        )

    # 상위 모델들의 값들을 기준으로 미세조정 격자(Grid) 생성
    # 1단계 결과를 분석해 가장 성적이 좋은 핵심 값들의 인접값들로 바운더리를 좁힙니다.
    best_lrs = top_3["learning_rate"].unique()
    best_layers = top_3["num_layers"].unique().astype(int)
    best_units = top_3["hidden_units"].unique().astype(int)
    best_dropouts = top_3["dropout_rate"].unique()
    best_batches = top_3["batch_size"].unique().astype(int)

    # 미세 범위 확장 (후보군 개수가 좁을 때 인접 요소를 조금 추가)
    def expand_lr(lrs):
        expanded = set(lrs)
        for lr in lrs:
            expanded.add(lr * 0.5)
            expanded.add(lr * 2.0)
        # 1단계 탐색범위인 [0.0001, 0.01] 사이로 제한
        return sorted([x for x in expanded if 0.00009 <= x <= 0.011])

    def expand_discrete(vals, allowed_pool):
        # 기존 풀 내에서 베스트 값과 그 앞뒤 원소를 추가
        idx_set = set()
        for v in vals:
            if v in allowed_pool:
                idx = allowed_pool.index(v)
                idx_set.add(idx)
                if idx > 0:
                    idx_set.add(idx - 1)
                if idx < len(allowed_pool) - 1:
                    idx_set.add(idx + 1)
        return sorted([allowed_pool[i] for i in idx_set])

    grid_lrs = expand_lr(best_lrs)
    grid_layers = expand_discrete(best_layers, exp_config.HIDDEN_LAYERS)
    grid_units = expand_discrete(best_units, exp_config.HIDDEN_UNITS)
    grid_dropouts = expand_discrete(best_dropouts, exp_config.DROPOUT_RATES)
    grid_batches = expand_discrete(best_batches, exp_config.BATCH_SIZES)

    logger.info("2단계 정밀 격자 탐색 범위 정의:")
    logger.info(f"  Learning Rate: {grid_lrs}")
    logger.info(f"  Hidden Layers: {grid_layers}")
    logger.info(f"  Hidden Units : {grid_units}")
    logger.info(f"  Dropout Rate : {grid_dropouts}")
    logger.info(f"  Batch Size   : {grid_batches}")

    # 그리드 조합 생성
    def to_json_values(values):
        return json.dumps([
            value.item() if isinstance(value, np.generic) else value
            for value in values
        ])

    total_grid_combinations = (
        len(grid_lrs)
        * len(grid_layers)
        * len(grid_units)
        * len(grid_dropouts)
        * len(grid_batches)
    )
    grid_candidates = pd.DataFrame([
        {"parameter": "learning_rate", "values": to_json_values(grid_lrs), "num_values": len(grid_lrs)},
        {"parameter": "num_layers", "values": to_json_values(grid_layers), "num_values": len(grid_layers)},
        {"parameter": "hidden_units", "values": to_json_values(grid_units), "num_values": len(grid_units)},
        {"parameter": "dropout_rate", "values": to_json_values(grid_dropouts), "num_values": len(grid_dropouts)},
        {"parameter": "batch_size", "values": to_json_values(grid_batches), "num_values": len(grid_batches)},
        {"parameter": "TOTAL_COMBINATIONS", "values": total_grid_combinations, "num_values": total_grid_combinations},
    ])
    os.makedirs(exp_config.RESULTS_DIR, exist_ok=True)
    grid_candidates.to_csv(exp_config.STAGE2_CANDIDATES_PATH, index=False, encoding="utf-8")
    logger.info(f"2단계 Grid Search 후보 저장 완료: {exp_config.STAGE2_CANDIDATES_PATH}")

    all_grid_combos = []
    for lr in grid_lrs:
        for layers in grid_layers:
            for units in grid_units:
                for dropout in grid_dropouts:
                    for batch in grid_batches:
                        all_grid_combos.append({
                            "learning_rate": lr,
                            "num_layers": layers,
                            "hidden_units": units,
                            "dropout_rate": dropout,
                            "batch_size": batch
                        })

    # 생성된 정밀 grid의 모든 조합을 실행합니다.
    num_grid_trials = len(all_grid_combos)
    grid_combos_to_run = all_grid_combos
    logger.info(f"2단계 Grid Search: 모든 후보조합 {num_grid_trials}개를 실행합니다.")

    logger.info(f"2단계 총 실험 조합 수: {num_grid_trials}개 (후보조합 {len(all_grid_combos)}개 중)")

    grid_results = []
    best_mae = float("inf")
    best_params = None
    best_model_weights = None

    for trial, params in enumerate(grid_combos_to_run, 1):
        start_t = time.time()
        mae, mape, r2, weights = train_and_evaluate(params, train_data, val_data, device)
        elapsed = time.time() - start_t

        logger.info(
            f"Grid Trial {trial:3d}/{num_grid_trials} | "
            f"LR={params['learning_rate']:<6.4f}, Layers={params['num_layers']}, "
            f"Units={params['hidden_units']:3d}, Dropout={params['dropout_rate']:<3.1f}, Batch={params['batch_size']:3d} | "
            f"MAE={mae:11,.1f}명, MAPE={mape:5.2f}%, R2={r2:6.4f} | 소요시간={elapsed:.1f}초"
        )

        grid_results.append({
            "trial": trial,
            "learning_rate": params["learning_rate"],
            "num_layers": params["num_layers"],
            "hidden_units": params["hidden_units"],
            "dropout_rate": params["dropout_rate"],
            "batch_size": params["batch_size"],
            "val_mae": mae,
            "val_mape": mape,
            "val_r2": r2,
            "time_seconds": elapsed
        })

        # 최고의 성능 모델 저장
        if mae < best_mae:
            best_mae = mae
            best_params = params
            best_model_weights = weights

    # 결과 저장
    df_grid_results = pd.DataFrame(grid_results)
    df_grid_results.to_csv(exp_config.STAGE2_LOG_PATH, index=False, encoding="utf-8")
    logger.info(f"💾 2단계 결과 저장 완료: {exp_config.STAGE2_LOG_PATH}")

    return df_grid_results, best_params, best_model_weights
