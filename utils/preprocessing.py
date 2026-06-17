# -*- coding: utf-8 -*-
"""
데이터 전처리 모듈
==================
수집된 원본 일별 박스오피스 데이터를 영화 단위로 정리하고,
개봉 후 첫 7일간의 데이터를 기반으로 기본 피처를 생성합니다.

사용법:
    python utils/preprocessing.py

입력:
    data/raw/daily_boxoffice.csv  (수집된 원본 데이터)

출력:
    data/processed/movie_features.csv  (영화별 7일 데이터 + 기본 피처)
"""

import os
import sys
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import data_config as config

# ============================================================
# 로깅 설정
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """
    원본 일별 박스오피스 CSV 파일을 로드합니다.

    Args:
        csv_path: 원본 CSV 파일 경로

    Returns:
        로드된 DataFrame
    """
    if not os.path.exists(csv_path):
        logger.error(f"원본 데이터 파일이 존재하지 않습니다: {csv_path}")
        logger.error("먼저 data/collect_data.py를 실행하여 데이터를 수집해 주세요.")
        sys.exit(1)

    logger.info(f"원본 데이터 로드 중: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8", dtype=str)
    logger.info(f"  로드 완료: {len(df):,}행, {len(df.columns)}열")

    return df


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    원본 데이터의 타입 변환 및 기본 정제를 수행합니다.

    Args:
        df: 원본 DataFrame

    Returns:
        정제된 DataFrame
    """
    logger.info("데이터 타입 변환 및 정제 중...")

    # 숫자형 필드 변환
    numeric_fields = ["audiCnt", "audiAcc", "scrnCnt", "showCnt", "rank", "rankInten"]
    for field in numeric_fields:
        df[field] = pd.to_numeric(df[field], errors="coerce")

    # 결측치 확인 및 로그
    missing_counts = df[numeric_fields].isnull().sum()
    if missing_counts.any():
        logger.warning("숫자 변환 시 결측치 발생:")
        for field, count in missing_counts.items():
            if count > 0:
                logger.warning(f"  {field}: {count}건")

    # 결측치가 있는 행 제거 (movieCd나 openDt가 비어있으면 의미 없음)
    before_len = len(df)
    df = df.dropna(subset=["movieCd", "openDt", "audiCnt"])
    after_len = len(df)
    if before_len != after_len:
        logger.info(f"  필수 필드 결측 행 제거: {before_len - after_len}건")

    # openDt의 형식 정규화 (공백이나 하이픈 제거 후 일관된 형식으로)
    df["openDt"] = df["openDt"].str.strip().str.replace("-", "", regex=False)

    # date 필드도 정규화
    df["date"] = df["date"].str.strip().str.replace("-", "", regex=False)

    logger.info(f"  정제 완료: {len(df):,}행")
    return df


def compute_day_offset(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 레코드에 대해 개봉일로부터 몇 번째 날인지 계산합니다.
    day_offset = 1이면 개봉 첫째 날, 2이면 둘째 날, ...

    Args:
        df: 정제된 DataFrame

    Returns:
        day_offset 열이 추가된 DataFrame
    """
    logger.info("개봉일 기준 날짜 오프셋 계산 중...")

    # 날짜 파싱
    df["date_dt"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["open_dt"] = pd.to_datetime(df["openDt"], format="%Y%m%d", errors="coerce")

    # 날짜 파싱 실패 행 제거
    before_len = len(df)
    df = df.dropna(subset=["date_dt", "open_dt"])
    after_len = len(df)
    if before_len != after_len:
        logger.warning(f"  날짜 파싱 실패 행 제거: {before_len - after_len}건")

    # 개봉일로부터의 오프셋 (일 수) 계산: 개봉일 당일 = 1
    df["day_offset"] = (df["date_dt"] - df["open_dt"]).dt.days + 1

    # 개봉일 이전 데이터(day_offset <= 0) 또는 비정상적으로 먼 데이터 제거
    before_len = len(df)
    df = df[df["day_offset"] >= 1].copy()
    after_len = len(df)
    if before_len != after_len:
        logger.info(f"  개봉일 이전 데이터 제거: {before_len - after_len}건")

    logger.info(f"  오프셋 계산 완료: {len(df):,}행")
    return df


def extract_first_n_days(df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """
    각 영화의 개봉 후 첫 N일간의 데이터만 추출합니다.

    Args:
        df: day_offset이 포함된 DataFrame
        n_days: 추출할 일수 (기본 7일)

    Returns:
        첫 N일 데이터만 포함된 DataFrame
    """
    logger.info(f"개봉 후 첫 {n_days}일 데이터 추출 중...")

    # 첫 N일 데이터만 필터링
    df_filtered = df[df["day_offset"] <= n_days].copy()

    logger.info(f"  필터링 후: {len(df_filtered):,}행")
    return df_filtered


def build_movie_features(df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """
    영화별로 첫 N일 데이터를 피벗하여 피처를 생성합니다.

    생성되는 피처:
        - movieCd: 영화 코드
        - movieNm: 영화명
        - openDt: 개봉일 (YYYYMMDD)
        - day1_audience ~ dayN_audience: 일별 관객수
        - day1_scrnCnt ~ dayN_scrnCnt: 일별 스크린 수
        - day1_showCnt ~ dayN_showCnt: 일별 상영 횟수
        - day1_rank ~ dayN_rank: 일별 순위
        - day1_rankInten ~ dayN_rankInten: 일별 순위 변동
        - open_year: 개봉 연도
        - open_month: 개봉 월
        - target_audience: 최종 누적 관객수 (타겟)

    Args:
        df: 첫 N일로 필터링된 DataFrame
        n_days: 사용할 일수

    Returns:
        영화별 피처 DataFrame
    """
    logger.info("영화별 피처 생성 중...")

    # 각 영화의 전체 데이터에서 최종 누적 관객수 산출을 위해 원본 필요
    # (여기서는 이미 필터링된 데이터를 사용하므로, 원본에서 max audiAcc를 별도 계산)
    # → 이 함수 호출 전에 전체 데이터에서 타겟을 계산해야 함

    movies = df.groupby("movieCd")
    results = []

    for movie_cd, group in movies:
        # 해당 영화가 N일치 데이터를 모두 갖고 있는지 확인
        available_days = sorted(group["day_offset"].unique())
        required_days = set(range(1, n_days + 1))

        if not required_days.issubset(set(available_days)):
            # N일치 데이터가 불완전한 영화는 제외
            continue

        # 기본 정보 추출 (첫 번째 행에서)
        first_row = group.iloc[0]
        movie_nm = first_row["movieNm"]
        open_dt = first_row["openDt"]

        # 피처 딕셔너리 초기화
        feature = {
            "movieCd": movie_cd,
            "movieNm": movie_nm,
            "openDt": open_dt,
        }

        # 일별 데이터를 피벗 (day_offset 기준으로 정렬)
        for day in range(1, n_days + 1):
            day_data = group[group["day_offset"] == day]
            if len(day_data) == 0:
                continue

            # 같은 날짜에 중복 데이터가 있을 경우 첫 번째 사용
            row = day_data.iloc[0]

            feature[f"day{day}_audience"] = int(row["audiCnt"])
            feature[f"day{day}_scrnCnt"] = int(row["scrnCnt"]) if pd.notna(row["scrnCnt"]) else 0
            feature[f"day{day}_showCnt"] = int(row["showCnt"]) if pd.notna(row["showCnt"]) else 0
            feature[f"day{day}_rank"] = int(row["rank"]) if pd.notna(row["rank"]) else 0
            feature[f"day{day}_rankInten"] = int(row["rankInten"]) if pd.notna(row["rankInten"]) else 0

        # 개봉 연도, 월 추출
        try:
            open_date = datetime.strptime(open_dt, "%Y%m%d")
            feature["open_year"] = open_date.year
            feature["open_month"] = open_date.month
        except (ValueError, TypeError):
            logger.warning(f"  영화 {movie_cd}({movie_nm}): 개봉일 파싱 실패 ({open_dt})")
            continue

        results.append(feature)

    result_df = pd.DataFrame(results)
    logger.info(f"  피처 생성 완료: {len(result_df):,}개 영화")
    return result_df


def compute_target(raw_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    각 영화의 최종 누적 관객수(타겟 변수)를 계산합니다.
    전체 원본 데이터에서 각 영화의 audiAcc 최대값을 사용합니다.

    Args:
        raw_df: 전체 원본 DataFrame (필터링 전)
        feature_df: 영화별 피처 DataFrame

    Returns:
        target_audience 열이 추가된 DataFrame
    """
    logger.info("타겟 변수 (최종 누적 관객수) 계산 중...")

    # 영화별 최대 누적 관객수 계산
    target = raw_df.groupby("movieCd")["audiAcc"].max().reset_index()
    target.columns = ["movieCd", "target_audience"]

    # 피처 DataFrame과 병합
    result_df = feature_df.merge(target, on="movieCd", how="left")

    # 타겟이 없는 영화 제거
    before_len = len(result_df)
    result_df = result_df.dropna(subset=["target_audience"])
    result_df["target_audience"] = result_df["target_audience"].astype(int)
    after_len = len(result_df)

    if before_len != after_len:
        logger.warning(f"  타겟 결측 영화 제거: {before_len - after_len}건")

    logger.info(f"  타겟 계산 완료: {len(result_df):,}개 영화")
    return result_df


def preprocess():
    """
    전처리 메인 함수: 원본 데이터를 읽어 영화별 7일 피처를 생성합니다.
    """
    logger.info("=" * 60)
    logger.info("데이터 전처리를 시작합니다...")
    logger.info("=" * 60)

    # 1. 원본 데이터 로드
    raw_df = load_raw_data(config.RAW_CSV_PATH)

    # 2. 데이터 정제 (타입 변환, 결측치 처리)
    cleaned_df = clean_raw_data(raw_df)

    # 3. 개봉일 기준 날짜 오프셋 계산
    offset_df = compute_day_offset(cleaned_df)

    # 4. 첫 7일 데이터 추출
    first_week_df = extract_first_n_days(offset_df, config.FEATURE_DAYS)

    # 5. 영화별 피처 생성 (일별 관객수, 스크린수, 상영횟수, 순위, 순위변동)
    feature_df = build_movie_features(first_week_df, config.FEATURE_DAYS)

    if feature_df.empty:
        logger.error("피처 생성 결과가 비어있습니다. 원본 데이터를 확인해 주세요.")
        sys.exit(1)

    # 6. 타겟 변수 계산 (전체 기간의 최대 누적 관객수)
    feature_df = compute_target(offset_df, feature_df)

    # 7. 결과 저장
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    feature_df.to_csv(config.INTERMEDIATE_CSV_PATH, index=False, encoding="utf-8")
    logger.info(f"전처리 결과 저장: {config.INTERMEDIATE_CSV_PATH}")

    # 8. 결과 요약
    logger.info("=" * 60)
    logger.info("전처리 완료 요약:")
    logger.info(f"  총 영화 수: {len(feature_df):,}개")
    logger.info(f"  피처 수: {len(feature_df.columns)}개")
    logger.info(f"  피처 목록: {list(feature_df.columns)}")
    logger.info(f"  타겟(관객수) 통계:")
    logger.info(f"    최소: {feature_df['target_audience'].min():,}명")
    logger.info(f"    최대: {feature_df['target_audience'].max():,}명")
    logger.info(f"    평균: {feature_df['target_audience'].mean():,.0f}명")
    logger.info(f"    중앙값: {feature_df['target_audience'].median():,.0f}명")
    logger.info("=" * 60)

    return feature_df


if __name__ == "__main__":
    preprocess()
