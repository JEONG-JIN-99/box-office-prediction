# -*- coding: utf-8 -*-
"""
한국 박스오피스 예측 프로젝트 - 설정 파일
==========================================
API 키, 파일 경로, 수집 기간 등 프로젝트 전반의 설정을 관리합니다.
"""

import os

# ============================================================
# 프로젝트 루트 경로 (이 파일이 위치한 디렉토리 기준)
# ============================================================
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CONFIG_DIR)

# ============================================================
# KOFIC(영화진흥위원회) Open API 설정
# ============================================================
# API 키: https://www.kobis.or.kr/kobisopenapi/ 에서 발급받은 키를 입력하세요
API_KEY = "23b44594b6bac57a23888276eed1f277"

# 일별 박스오피스 API 엔드포인트
API_BASE_URL = (
    "http://www.kobis.or.kr/kobisopenapi/webservice/rest/"
    "boxoffice/searchDailyBoxOfficeList.json"
)

# API 호출 간 대기 시간 (초) - 과도한 요청 방지
API_RATE_LIMIT = 0.5

# API 호출 실패 시 최대 재시도 횟수
API_MAX_RETRIES = 3

# 재시도 시 대기 시간 (초)
API_RETRY_DELAY = 2.0

# API 요청 타임아웃 (초)
API_TIMEOUT = 30

# 일일 API 호출 제한 횟수 (초과 시 자동 중단, 다음 날 이어서 수집)
API_DAILY_LIMIT = 3000

# ============================================================
# 데이터 수집 기간 설정
# ============================================================
COLLECT_START_DATE = "2010-01-01"
COLLECT_END_DATE = "2025-12-31"

# ============================================================
# 파일 경로 설정
# ============================================================
# 원본 데이터 디렉토리
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# 전처리된 데이터 디렉토리
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# 원본 일별 박스오피스 CSV 파일 경로
RAW_CSV_PATH = os.path.join(RAW_DATA_DIR, "daily_boxoffice.csv")

# 전처리 중간 결과 (영화별 7일 데이터) CSV 파일 경로
INTERMEDIATE_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, "movie_features.csv")

# 최종 피처 엔지니어링 결과 CSV 파일 경로
FINAL_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, "movie_features_final.csv")

# ============================================================
# 수집 대상 필드 목록 (KOFIC API 응답에서 추출할 필드)
# ============================================================
EXTRACT_FIELDS = [
    "movieNm",    # 영화명
    "openDt",     # 개봉일
    "audiCnt",    # 해당일 관객수
    "audiAcc",    # 누적 관객수
    "scrnCnt",    # 스크린 수
    "showCnt",    # 상영 횟수
    "rank",       # 박스오피스 순위
    "rankInten",  # 순위 변동 (전일 대비)
    "movieCd",    # 영화 코드 (고유 식별자)
]

# ============================================================
# 전처리 설정
# ============================================================
# 개봉 후 수집할 일수 (첫 7일간의 데이터 사용)
FEATURE_DAYS = 7
