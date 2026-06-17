# -*- coding: utf-8 -*-
"""
결과 시각화 모듈
================
자동 실험 결과 CSV 데이터를 분석하여 7가지 종류의 분석 그래프를 생성합니다.
1단계(Random Search)와 2단계(Grid Search) 결과를 각각 시각화하여 저장합니다.
"""

import os
import sys
import logging

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "__cache__", "matplotlib"),
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import experiments.experiment_configs as exp_config

# 한글 깨짐 방지 및 스타일 설정
plt.rcParams["font.family"] = "Malgun Gothic"  # Windows 한글 폰트
plt.rcParams["axes.unicode_minus"] = False     # 마이너스 깨짐 방지
sns.set_theme(style="whitegrid", font="Malgun Gothic")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# G1: 하이퍼파라미터별 박스플롯 (5개 HP × 1장씩 = 5장)
# ============================================================
def plot_g1_boxplots(df: pd.DataFrame, output_dir: str):
    """각 하이퍼파라미터 값에 따른 MAE 분포를 박스플롯으로 그립니다."""
    hp_cols = ["learning_rate", "num_layers", "hidden_units", "dropout_rate", "batch_size"]
    hp_labels = ["학습률 (Learning Rate)", "은닉층 수 (Hidden Layers)", "은닉층 뉴런 수 (Hidden Units)", "드롭아웃 비율 (Dropout Rate)", "배치 크기 (Batch Size)"]

    for col, label in zip(hp_cols, hp_labels):
        plt.figure(figsize=(8, 5))
        # 범주형으로 변환하여 정렬된 상태로 표시
        df_sorted = df.copy()
        df_sorted[col] = df_sorted[col].astype(str)
        
        # 정렬 기준 설정 (원래 숫자형이었으므로 크기 순 정렬)
        unique_vals = sorted(df[col].unique())
        order = [str(x) for x in unique_vals]

        sns.boxplot(x=col, y="val_mae", data=df_sorted, order=order, palette="Blues_d")
        plt.title(f"하이퍼파라미터별 성능 영향: {label}", fontsize=13, pad=15)
        plt.xlabel(label, fontsize=11)
        plt.ylabel("검증 MAE (명, 낮을수록 우수)", fontsize=11)
        plt.tight_layout()
        
        fig_path = os.path.join(output_dir, f"G1_boxplot_{col}.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
    
    logger.info(f"  G1: 박스플롯 5장 생성 완료")


# ============================================================
# G2: 2D 히트맵 (2개 HP 조합별 평균 MAE, 10장)
# ============================================================
def plot_g2_heatmaps(df: pd.DataFrame, output_dir: str):
    """2개 하이퍼파라미터 간의 상호작용을 파악하기 위해 히트맵을 생성합니다."""
    hp_cols = ["learning_rate", "num_layers", "hidden_units", "dropout_rate", "batch_size"]
    hp_labels = ["LR", "Layers", "Units", "Dropout", "Batch"]
    
    count = 0
    # 5개 중 2개를 고르는 조합 (C(5,2) = 10가지)
    for i in range(len(hp_cols)):
        for j in range(i + 1, len(hp_cols)):
            col_x = hp_cols[i]
            col_y = hp_cols[j]
            label_x = hp_labels[i]
            label_y = hp_labels[j]
            
            # 피벗 테이블 생성 (평균 MAE 계산)
            pivot_df = df.groupby([col_y, col_x])["val_mae"].mean().unstack()
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(pivot_df, annot=True, fmt=",.0f", cmap="YlGnBu_r", cbar_kws={'label': '평균 검증 MAE'})
            plt.title(f"성능 상호작용 히트맵: {label_y} vs {label_x}", fontsize=13, pad=15)
            plt.xlabel(label_x, fontsize=11)
            plt.ylabel(label_y, fontsize=11)
            plt.tight_layout()
            
            fig_path = os.path.join(output_dir, f"G2_heatmap_{col_y}_vs_{col_x}.png")
            plt.savefig(fig_path, dpi=150)
            plt.close()
            count += 1
            
    logger.info(f"  G2: 히트맵 {count}장 생성 완료")


# ============================================================
# G4: Top-10 모델 바 차트 (1장)
# ============================================================
def plot_g4_top10_models(df: pd.DataFrame, output_dir: str):
    """상위 10개 최우수 하이퍼파라미터 조합의 MAE 성능을 바 차트로 비교합니다."""
    top_10 = df.nsmallest(10, "val_mae").copy()
    
    # 레이블명 축소 생성
    labels = []
    for idx, row in top_10.reset_index().iterrows():
        label = (
            f"Top {idx+1}\n"
            f"(LR:{row['learning_rate']}, L:{int(row['num_layers'])}, "
            f"U:{int(row['hidden_units'])}, D:{row['dropout_rate']}, B:{int(row['batch_size'])})"
        )
        labels.append(label)
        
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(10), top_10["val_mae"] / 10000.0, color="teal", width=0.6)
    
    # 바 위에 수치 표시 (만 명 단위)
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.0, 
            yval + 0.5, 
            f"{yval:.1f}만명", 
            ha='center', 
            va='bottom', 
            fontsize=9
        )
        
    plt.xticks(range(10), labels, rotation=45, ha="right", fontsize=9)
    plt.title("상위 10개 하이퍼파라미터 조합 성능 비교", fontsize=14, pad=20)
    plt.ylabel("검증 MAE (단위: 만 명, 낮을수록 우수)", fontsize=11)
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, "G4_top10_comparison.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.info(f"  G4: Top-10 비교 그래프 생성 완료")


# ============================================================
# G6: 하이퍼파라미터 중요도 분석 바 차트 (1장)
# ============================================================
def plot_g6_param_importance(df: pd.DataFrame, output_dir: str):
    """
    Random Forest를 간이로 학습시켜 각 하이퍼파라미터가 
    검증 MAE 성능 분산에 미치는 기여도(중요도)를 분석하고 그립니다.
    """
    from sklearn.ensemble import RandomForestRegressor
    
    hp_cols = ["learning_rate", "num_layers", "hidden_units", "dropout_rate", "batch_size"]
    hp_labels = ["학습률 (LR)", "레이어 수 (Layers)", "뉴런 수 (Units)", "드롭아웃 (Dropout)", "배치 크기 (Batch)"]
    
    X = df[hp_cols].copy()
    y = df["val_mae"].copy()
    
    # RF 학습을 통해 피처 중요도 추출
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(8, 5))
    plt.barh(range(len(hp_cols)), importances[indices], color="coral", align="center", height=0.5)
    plt.yticks(range(len(hp_cols)), [hp_labels[i] for i in indices])
    plt.xlabel("기여도 (중요도 비율)", fontsize=11)
    plt.title("성능(MAE) 결정에 미치는 하이퍼파라미터 중요도 분석", fontsize=13, pad=15)
    plt.gca().invert_yaxis()  # 높은 게 위에 오도록
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, "G6_parameter_importance.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.info(f"  G6: 파라미터 중요도 그래프 생성 완료")


# ============================================================
# G7: 병렬 좌표 그래프 (Parallel Coordinates, 1장)
# ============================================================
def plot_g7_parallel_coordinates(df: pd.DataFrame, output_dir: str):
    """
    하이퍼파라미터 조합의 전체 흐름과 성능 수렴을 나타내는 
    병렬 좌표 그래프를 Matplotlib으로 직접 구현합니다.
    """
    hp_cols = ["learning_rate", "num_layers", "hidden_units", "dropout_rate", "batch_size"]
    hp_labels = ["LR", "Layers", "Units", "Dropout", "Batch"]
    
    # 플로팅용 데이터셋 스케일링 (0~1 범위로 정규화)
    df_norm = df.copy()
    for col in hp_cols:
        min_v = df_norm[col].min()
        max_v = df_norm[col].max()
        if max_v != min_v:
            df_norm[col] = (df_norm[col] - min_v) / (max_v - min_v)
        else:
            df_norm[col] = 0.5
            
    # MAE 역순으로 컬러 맵핑하기 위한 정규화 (좋은 모델=붉은색/노란색 계열, 나쁜 모델=푸른색 계열)
    mae = df["val_mae"].values
    min_mae, max_mae = mae.min(), mae.max()
    if max_mae != min_mae:
        norm_mae = 1.0 - (mae - min_mae) / (max_mae - min_mae)  # 낮을수록 1에 가깝게
    else:
        norm_mae = np.ones_like(mae) * 0.5

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 컬러 맵 설정 (우수 모델 강조용)
    cmap = plt.get_cmap("turbo")
    
    # 각 실험 시도(Trial)를 선으로 그리기
    for idx, row in df_norm.iterrows():
        y_pts = [row[col] for col in hp_cols]
        # 성능에 따라 선 투명도 및 두께 조절 (우수한 모델을 더 굵고 뚜렷하게 그리기)
        color = cmap(norm_mae[idx])
        alpha = 0.15 + 0.65 * norm_mae[idx]
        linewidth = 0.5 + 2.0 * norm_mae[idx]
        
        ax.plot(range(len(hp_cols)), y_pts, color=color, alpha=alpha, linewidth=linewidth)
        
    # 축 설정
    ax.set_xticks(range(len(hp_cols)))
    ax.set_xticklabels(hp_labels, fontsize=12)
    
    # 각 축의 상단/하단에 실제 물리적인 하이퍼파라미터 최댓값/최솟값 텍스트 표시
    for col_idx, col in enumerate(hp_cols):
        min_val = df[col].min()
        max_val = df[col].max()
        
        # 텍스트 형식 지정
        fmt = ".4f" if col == "learning_rate" else ".1f" if col == "dropout_rate" else "d"
        ax.text(col_idx, 1.02, f"Max\n{max_val:{fmt}}", ha="center", va="bottom", color="darkred", fontsize=9)
        ax.text(col_idx, -0.02, f"Min\n{min_val:{fmt}}", ha="center", va="top", color="darkblue", fontsize=9)

    # 우측 컬러 바 추가
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_mae/10000.0, vmax=max_mae/10000.0))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, pad=0.05)
    cbar.set_label("검증 MAE (단위: 만 명, 낮을수록 우수)", fontsize=11)
    
    plt.title("전체 하이퍼파라미터 탐색 조합 및 성능 흐름 (병렬 좌표계)", fontsize=14, pad=25)
    plt.ylabel("정규화된 축 스케일 (0.0~1.0)", fontsize=11)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, "G7_parallel_coordinates.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.info(f"  G7: 병렬 좌표계 그래프 생성 완료")


# ============================================================
# 메인 통합 시각화 실행 함수
# ============================================================
def generate_all_plots():
    """1단계와 2단계 실험 CSV 결과 로그를 불러와서 각각 시각화 폴더에 저장합니다."""
    logger.info("=" * 60)
    logger.info("📊 실험 결과 시각화 분석 시작...")
    logger.info("=" * 60)

    fig_root_dir = os.path.join(config.PROJECT_ROOT, "visualization", "figures")
    os.makedirs(fig_root_dir, exist_ok=True)

    # 1. 1단계(Random Search) 시각화
    if os.path.exists(exp_config.STAGE1_LOG_PATH):
        logger.info("[1단계: Random Search 결과 시각화 실행]")
        stage1_dir = os.path.join(fig_root_dir, "stage1")
        os.makedirs(stage1_dir, exist_ok=True)
        
        df1 = pd.read_csv(exp_config.STAGE1_LOG_PATH)
        plot_g1_boxplots(df1, stage1_dir)
        plot_g2_heatmaps(df1, stage1_dir)
        plot_g4_top10_models(df1, stage1_dir)
        plot_g6_param_importance(df1, stage1_dir)
        plot_g7_parallel_coordinates(df1, stage1_dir)
    else:
        logger.warning(f"1단계 결과 파일이 없습니다: {exp_config.STAGE1_LOG_PATH}")

    # 2. 2단계(Grid Search) 시각화
    if os.path.exists(exp_config.STAGE2_LOG_PATH):
        logger.info("[2단계: Grid Search 결과 시각화 실행]")
        stage2_dir = os.path.join(fig_root_dir, "stage2")
        os.makedirs(stage2_dir, exist_ok=True)
        
        df2 = pd.read_csv(exp_config.STAGE2_LOG_PATH)
        plot_g1_boxplots(df2, stage2_dir)
        plot_g2_heatmaps(df2, stage2_dir)
        plot_g4_top10_models(df2, stage2_dir)
        plot_g6_param_importance(df2, stage2_dir)
        plot_g7_parallel_coordinates(df2, stage2_dir)
    else:
        logger.warning(f"2단계 결과 파일이 없습니다: {exp_config.STAGE2_LOG_PATH}")

    logger.info("=" * 60)
    logger.info("📊 모든 결과 시각화 그래프 저장 완료!")
    logger.info(f"💾 저장 경로: {fig_root_dir}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    generate_all_plots()
