# -*- coding: utf-8 -*-
"""
프로젝트 메인 학습 및 자동 최적화 스크립트
=========================================
데이터 분할, 1단계(Random) 및 2단계(Grid) 탐색 실행, 최적 모델의 최종 평가,
스케일러 및 최적 가중치 저장, 시각화 스크립트 호출을 포함하는 통합 파이프라인입니다.

사용법:
    python train.py
"""

import os
import sys
import json
import time
import logging
from typing import Tuple, Dict, List

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "__cache__", "matplotlib"),
)

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import experiments.experiment_configs as exp_config
from experiments.hyperparameter_search import (
    load_and_split_data,
    run_stage1_random_search,
    run_stage2_grid_search,
    train_and_evaluate,
    BoxOfficeDataset
)
from models.model import BoxOfficeMLP
from utils.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from visualization.plot_results import generate_all_plots

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# G3: 최적 모델의 학습 곡선 (Learning Curve) 그리기
# ============================================================
def plot_g3_learning_curve(history: dict, output_dir: str):
    """최종 학습 중 기록된 Train 및 Val Loss 곡선을 그립니다."""
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss (MSE)", color="blue", linewidth=1.5)
    plt.plot(history["val_loss"], label="Val Loss (MSE)", color="red", linewidth=1.5)
    plt.title("최적 모델 학습 곡선 (Learning Curve)", fontsize=13, pad=15)
    plt.xlabel("에포크 (Epochs)", fontsize=11)
    plt.ylabel("손실 (Loss, MSE)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, "G3_learning_curve.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.info(f"📊 G3: 최적 모델 학습 곡선 저장 완료: {fig_path}")


# ============================================================
# G5: 실제 관객수 vs 예측 관객수 산점도 (Scatter Plot)
# ============================================================
def plot_g5_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, output_dir: str):
    """최종 테스트 세트의 실제 값과 모델의 예측 값을 비교하는 산점도를 그립니다."""
    plt.figure(figsize=(7, 7))
    
    # 단위: 만 명
    y_true_man = y_true / 10000.0
    y_pred_man = y_pred / 10000.0
    
    # 산점도 그리기
    sns.scatterplot(x=y_true_man, y=y_pred_man, alpha=0.6, color="purple")
    
    # 대각선 (완벽한 예측 선) 그리기
    max_val = max(y_true_man.max(), y_pred_man.max())
    plt.plot([0, max_val], [0, max_val], color="red", linestyle="--", linewidth=1.5, label="이상적 예측 (y=x)")
    
    plt.title("실제 누적 관객수 vs 예측 누적 관객수 산점도", fontsize=13, pad=15)
    plt.xlabel("실제 관객수 (단위: 만 명)", fontsize=11)
    plt.ylabel("예측 관객수 (단위: 만 명)", fontsize=11)
    plt.legend()
    plt.xlim(0, max_val * 1.05)
    plt.ylim(0, max_val * 1.05)
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, "G5_pred_vs_actual.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.info(f"📊 G5: 예측 vs 실제 산점도 저장 완료: {fig_path}")


# ============================================================
# 최적 파라미터로 최종 학습 및 스케일러 메타데이터 저장
# ============================================================
def train_best_model_final(
    best_params: dict,
    train_val_data: Tuple,  # Train + Val 데이터로 마지막 학습 수행 또는 Train으로 학습하고 가중치 결정
    test_data: Tuple,
    scaler,
    device: torch.device
):
    """
    최적의 하이퍼파라미터 조합으로 모델을 최종 학습하고,
    예측에 필요한 모델 가중치 파일 및 스케일러 설정 정보(JSON)를 저장합니다.
    """
    logger.info("=" * 60)
    logger.info("🏆 최적 하이퍼파라미터 기반 최종 모델 학습 및 평가")
    logger.info("=" * 60)
    logger.info(f"최적 파라미터: {best_params}")

    X_train, y_train_log, X_val, y_val_log, X_test, y_test_log, y_test_raw = test_data

    # 데이터 로더 준비
    train_dataset = BoxOfficeDataset(X_train, y_train_log)
    val_dataset = BoxOfficeDataset(X_val, y_val_log)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=False)

    # 모델 생성
    model = BoxOfficeMLP(
        input_dim=16,
        num_layers=best_params["num_layers"],
        hidden_units=best_params["hidden_units"],
        dropout_rate=best_params["dropout_rate"]
    ).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["learning_rate"])

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_weights = None
    
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, exp_config.MAX_EPOCHS + 1):
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= exp_config.EARLY_STOPPING_PATIENCE:
            break

    # 최적 가중치 로드
    model.load_state_dict({k: v.to(device) for k, v in best_model_weights.items()})

    # 테스트 데이터 세트 평가
    model.eval()
    test_preds_log = []
    test_loader = torch.utils.data.DataLoader(
        BoxOfficeDataset(X_test, y_test_log),
        batch_size=best_params["batch_size"],
        shuffle=False
    )
    
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            test_preds_log.extend(pred.cpu().numpy().flatten())

    test_preds = np.expm1(np.array(test_preds_log))
    test_preds = np.clip(test_preds, 0, None)

    # 지표 산출
    mae = mean_absolute_error(y_test_raw, test_preds)
    mape = mean_absolute_percentage_error(y_test_raw, test_preds)
    r2 = r2_score(y_test_raw, test_preds)

    logger.info("=" * 60)
    logger.info("🎯 최종 모델 테스트 세트 평가 결과:")
    logger.info(f"  MAE  : {mae:,.1f} 명 (평균 절대 오차)")
    logger.info(f"  MAPE : {mape:.2f} % (평균 절대 백분율 오차)")
    logger.info(f"  R2   : {r2:.4f} (결정계수)")
    logger.info("=" * 60)

    # 가중치 및 스케일러 정보 저장 경로 구성
    models_dir = os.path.join(config.PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_weight_path = os.path.join(models_dir, "best_model.pth")
    torch.save(best_model_weights, model_weight_path)
    logger.info(f"💾 최적 모델 가중치 저장 완료: {model_weight_path}")

    # 스케일러 정보 및 최적 하이퍼파라미터 JSON 메타데이터 저장
    scaler_metadata = {
        "mean": scaler.mean.tolist(),
        "std": scaler.std.tolist(),
        "best_hyperparameters": best_params,
        "test_performance": {
            "mae": mae,
            "mape": mape,
            "r2": r2
        }
    }
    
    metadata_path = os.path.join(models_dir, "scaler_config.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(scaler_metadata, f, indent=4, ensure_ascii=False)
    logger.info(f"💾 스케일러 정보 및 메타데이터 저장 완료: {metadata_path}")

    # G3, G5 그래프 출력 저장
    figures_dir = os.path.join(config.PROJECT_ROOT, "visualization", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    plot_g3_learning_curve(history, figures_dir)
    plot_g5_pred_vs_actual(y_test_raw, test_preds, figures_dir)


# ============================================================
# 메인 파이프라인 제어
# ============================================================
def main():
    start_time = time.time()
    
    # 0. 환경 장비(Device) 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"사용 장비 설정: {device}")

    # 1. 데이터 로드 및 분할
    (
        X_train, y_train_log, y_train,
        X_val, y_val_log, y_val,
        X_test, y_test_log, y_test,
        scaler
    ) = load_and_split_data(config.FINAL_CSV_PATH)

    train_data = (X_train, y_train_log)
    val_data = (X_val, y_val_log, y_val)
    test_data = (X_train, y_train_log, X_val, y_val_log, X_test, y_test_log, y_test)

    # 2. 1단계: Random Search 무작위 탐색 실행 (이미 완료됨)
    # stage1_results = run_stage1_random_search(train_data, val_data, device)

    # 3. 2단계: Grid Search 정밀 탐색 실행 (이미 완료됨)
    # stage2_results, best_params, best_model_weights = run_stage2_grid_search(
    #     stage1_results, train_data, val_data, device
    # )

    best_params = {'learning_rate': 0.005, 'num_layers': 5, 'hidden_units': 128, 'dropout_rate': 0.0, 'batch_size': 128}

    # 4. 최적 모델 최종 학습 및 평가 저장
    train_best_model_final(best_params, train_data, test_data, scaler, device)

    # 5. 시각화 그래프 자동 생성 (G1, G2, G4, G6, G7)
    generate_all_plots()

    elapsed_total = time.time() - start_time
    logger.info("=" * 60)
    logger.info("🎉 전체 최적화 파이프라인이 성공적으로 완료되었습니다!")
    logger.info(f"⌛ 총 소요 시간: {int(elapsed_total // 60)}분 {int(elapsed_total % 60)}초")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
