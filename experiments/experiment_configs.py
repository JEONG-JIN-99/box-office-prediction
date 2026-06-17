# -*- coding: utf-8 -*-
"""
하이퍼파라미터 탐색 설정 파일
============================
자동 실험을 위해 5가지 핵심 하이퍼파라미터의 탐색 범위를 구성합니다.
"""

import os

# ============================================================
# 실험 결과 저장 경로 설정
# ============================================================
EXPERIMENT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_ROOT, "results")

# 각 스테이지별 로그 저장 경로
STAGE1_LOG_PATH = os.path.join(RESULTS_DIR, "stage1_random_search.csv")
STAGE1_BEST_PATH = os.path.join(RESULTS_DIR, "stage1_best.csv")
STAGE2_CANDIDATES_PATH = os.path.join(RESULTS_DIR, "stage2_grid_candidates.csv")
STAGE2_LOG_PATH = os.path.join(RESULTS_DIR, "stage2_grid_search.csv")

# ============================================================
# 하이퍼파라미터 탐색 범위 (Search Space)
# ============================================================
# HP1: 학습률 (Learning Rate)
LEARNING_RATES = [0.0001, 0.0005, 0.001, 0.005, 0.01]

# HP2: 은닉 레이어 수 (Hidden Layers)
HIDDEN_LAYERS = [2, 3, 4, 5]

# HP3: 은닉 뉴런 수 (Hidden Units)
HIDDEN_UNITS = [64, 128, 256, 512]

# HP4: 드롭아웃 비율 (Dropout Rates)
DROPOUT_RATES = [0.0, 0.1, 0.2, 0.3, 0.5]

# HP5: 배치 크기 (Batch Sizes)
BATCH_SIZES = [16, 32, 64, 128]

# ============================================================
# 실험 실행 설정
# ============================================================
# 1단계 무작위 탐색 시도 횟수
STAGE1_NUM_TRIALS = 200

# 2단계 정밀 탐색 시도 횟수 (유망 영역 조합의 격자 탐색)
# 1단계 탐색에서 성능이 가장 좋았던 상위 3개 조합의 주변 영역을 Grid로 구성합니다.
# 2단계 격자 탐색 시도 횟수 (1단계 상위 성적 파라미터 조합 기반)

# 에포크 수 (각 모델의 최대 학습 에포크)
MAX_EPOCHS = 100

# Early Stopping 인내심 (검증 손실이 이 횟수만큼 나아지지 않으면 조기 종료)
EARLY_STOPPING_PATIENCE = 10
