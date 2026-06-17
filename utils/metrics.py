# -*- coding: utf-8 -*-
"""
평가 지표 모듈
==============
모델 성능 평가를 위한 다양한 회귀 지표(MAE, MAPE, R2 등)를 계산합니다.
"""

import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MAE(Mean Absolute Error)를 계산합니다.

    Args:
        y_true: 실제값 배열
        y_pred: 예측값 배열

    Returns:
        MAE 값
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MAPE(Mean Absolute Percentage Error)를 계산합니다.
    실제가 0인 값은 제외하고 계산하거나 아주 작은 분모 처리를 수행합니다.

    Args:
        y_true: 실제값 배열
        y_pred: 예측값 배열

    Returns:
        MAPE 값 (백분율 단위, 예: 15.5%)
    """
    # 0 분모 방지
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R2 Score(결정계수)를 계산합니다.

    Args:
        y_true: 실제값 배열
        y_pred: 예측값 배열

    Returns:
        R2 Score 값
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""):
    """
    평가 지표를 화면에 출력합니다.

    Args:
        y_true: 실제값 배열
        y_pred: 예측값 배열
        prefix: 출력 시 머리말 문자열
    """
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    prefix_str = f"[{prefix}] " if prefix else ""
    print(f"{prefix_str}MAE : {mae:,.2f} 명")
    print(f"{prefix_str}MAPE: {mape:.2f} %")
    print(f"{prefix_str}R2  : {r2:.4f}")
