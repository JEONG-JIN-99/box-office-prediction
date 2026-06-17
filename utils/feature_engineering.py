# -*- coding: utf-8 -*-
"""
피처 엔지니어링 모듈
=====================
전처리된 영화별 7일 데이터에서 파생 피처를 생성합니다.

사용법:
    python utils/feature_engineering.py

입력:
    data/processed/movie_features.csv  (전처리 결과)

출력:
    data/processed/movie_features_final.csv  (최종 피처)

최종 16개 피처:
    1-7.  day1_audience ~ day7_audience  (일별 관객수)
    8.    open_year                      (개봉 연도)
    9.    open_month                     (개봉 월)
    10.   week1_total                    (첫째 주 총 관객수)
    11.   avg_screen_count               (7일 평균 스크린 수)
    12.   avg_show_count                 (7일 평균 상영 횟수)
    13.   avg_rank                       (7일 평균 순위)
    14.   rank_trend                     (7일 평균 순위 변동)
    15.   day_over_day_ratio             (day7 / day1 비율)
    16.   weekend_ratio                  (주말 관객 비율)

타겟:
    target_audience  (최종 누적 관객수)
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


def load_intermediate_data(csv_path: str) -> pd.DataFrame:
    """
    전처리된 중간 결과 CSV를 로드합니다.

    Args:
        csv_path: 중간 결과 CSV 파일 경로

    Returns:
        로드된 DataFrame
    """
    if not os.path.exists(csv_path):
        logger.error(f"전처리 결과 파일이 존재하지 않습니다: {csv_path}")
        logger.error("먼저 utils/preprocessing.py를 실행해 주세요.")
        sys.exit(1)

    logger.info(f"전처리 데이터 로드 중: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8")
    logger.info(f"  로드 완료: {len(df):,}행, {len(df.columns)}열")
    return df


def add_week1_total(df: pd.DataFrame) -> pd.DataFrame:
    """
    첫째 주 총 관객수를 계산합니다.
    week1_total = day1_audience + day2_audience + ... + day7_audience

    Args:
        df: 피처 DataFrame

    Returns:
        week1_total 열이 추가된 DataFrame
    """
    logger.info("피처 생성: week1_total (첫째 주 총 관객수)")

    audience_cols = [f"day{i}_audience" for i in range(1, config.FEATURE_DAYS + 1)]
    df["week1_total"] = df[audience_cols].sum(axis=1)

    logger.info(f"  평균 첫째 주 관객수: {df['week1_total'].mean():,.0f}명")
    return df


def add_avg_screen_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    7일 평균 스크린 수를 계산합니다.

    Args:
        df: 피처 DataFrame

    Returns:
        avg_screen_count 열이 추가된 DataFrame
    """
    logger.info("피처 생성: avg_screen_count (7일 평균 스크린 수)")

    scrn_cols = [f"day{i}_scrnCnt" for i in range(1, config.FEATURE_DAYS + 1)]
    df["avg_screen_count"] = df[scrn_cols].mean(axis=1).round(2)

    return df


def add_avg_show_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    7일 평균 상영 횟수를 계산합니다.

    Args:
        df: 피처 DataFrame

    Returns:
        avg_show_count 열이 추가된 DataFrame
    """
    logger.info("피처 생성: avg_show_count (7일 평균 상영 횟수)")

    show_cols = [f"day{i}_showCnt" for i in range(1, config.FEATURE_DAYS + 1)]
    df["avg_show_count"] = df[show_cols].mean(axis=1).round(2)

    return df


def add_avg_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    7일 평균 박스오피스 순위를 계산합니다.

    Args:
        df: 피처 DataFrame

    Returns:
        avg_rank 열이 추가된 DataFrame
    """
    logger.info("피처 생성: avg_rank (7일 평균 순위)")

    rank_cols = [f"day{i}_rank" for i in range(1, config.FEATURE_DAYS + 1)]
    df["avg_rank"] = df[rank_cols].mean(axis=1).round(2)

    return df


def add_rank_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    7일 평균 순위 변동을 계산합니다.
    양수: 순위 하락 추세, 음수: 순위 상승 추세

    Args:
        df: 피처 DataFrame

    Returns:
        rank_trend 열이 추가된 DataFrame
    """
    logger.info("피처 생성: rank_trend (7일 평균 순위 변동)")

    rank_inten_cols = [f"day{i}_rankInten" for i in range(1, config.FEATURE_DAYS + 1)]
    df["rank_trend"] = df[rank_inten_cols].mean(axis=1).round(2)

    return df


def add_day_over_day_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    첫날 대비 7일차 관객수 비율을 계산합니다.
    day_over_day_ratio = day7_audience / day1_audience

    비율이 1보다 크면 관객수 증가 추세, 1보다 작으면 감소 추세를 나타냅니다.
    day1_audience가 0인 경우 0으로 처리합니다.

    Args:
        df: 피처 DataFrame

    Returns:
        day_over_day_ratio 열이 추가된 DataFrame
    """
    logger.info("피처 생성: day_over_day_ratio (day7 / day1 비율)")

    # 0으로 나누는 것을 방지
    df["day_over_day_ratio"] = np.where(
        df["day1_audience"] > 0,
        (df["day7_audience"] / df["day1_audience"]).round(4),
        0.0,
    )

    return df


def add_weekend_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    주말(토요일, 일요일) 관객 비율을 계산합니다.

    개봉일(openDt)을 기준으로 개봉 후 7일 각각이 주말인지 판별하여,
    주말 관객수의 합 / 전체 7일 관객수의 합을 계산합니다.

    예: 개봉일이 수요일이면
        day1(수), day2(목), day3(금), day4(토★), day5(일★), day6(월), day7(화)
        weekend_ratio = (day4 + day5) / (day1 + ... + day7)

    Args:
        df: 피처 DataFrame (openDt, day1~day7_audience 포함)

    Returns:
        weekend_ratio 열이 추가된 DataFrame
    """
    logger.info("피처 생성: weekend_ratio (주말 관객 비율)")

    weekend_ratios = []

    for _, row in df.iterrows():
        try:
            # 개봉일 파싱
            open_dt_str = str(row["openDt"]).strip()
            # openDt가 정수로 읽혔을 수 있으므로 처리
            if "." in open_dt_str:
                open_dt_str = open_dt_str.split(".")[0]
            open_date = datetime.strptime(open_dt_str, "%Y%m%d")
        except (ValueError, TypeError):
            # 파싱 실패 시 0으로 처리
            weekend_ratios.append(0.0)
            continue

        # 7일간 주말 관객수 합산
        weekend_audience = 0
        total_audience = 0

        for day in range(1, config.FEATURE_DAYS + 1):
            day_audience = row.get(f"day{day}_audience", 0)
            if pd.isna(day_audience):
                day_audience = 0
            day_audience = int(day_audience)

            total_audience += day_audience

            # 해당 일자의 요일 확인 (개봉일 + (day-1)일)
            current_date = open_date + timedelta(days=day - 1)
            weekday = current_date.weekday()  # 0=월, 1=화, ..., 5=토, 6=일

            if weekday in (5, 6):  # 토요일 또는 일요일
                weekend_audience += day_audience

        # 비율 계산 (전체 관객이 0이면 0)
        if total_audience > 0:
            weekend_ratios.append(round(weekend_audience / total_audience, 4))
        else:
            weekend_ratios.append(0.0)

    df["weekend_ratio"] = weekend_ratios

    logger.info(f"  평균 주말 관객 비율: {df['weekend_ratio'].mean():.4f}")
    return df


def select_final_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    최종 16개 피처 + 타겟 + 메타정보를 선택하여 정리합니다.

    Args:
        df: 모든 피처가 포함된 DataFrame

    Returns:
        최종 피처만 포함된 DataFrame
    """
    logger.info("최종 피처 선택 및 정리 중...")

    # 메타 정보 (모델 학습에는 사용하지 않지만 참조용)
    meta_cols = ["movieCd", "movieNm", "openDt"]

    # 16개 피처
    feature_cols = [
        # 일별 관객수 (7개)
        "day1_audience", "day2_audience", "day3_audience", "day4_audience",
        "day5_audience", "day6_audience", "day7_audience",
        # 시간 정보 (2개)
        "open_year", "open_month",
        # 파생 피처 (7개)
        "week1_total", "avg_screen_count", "avg_show_count",
        "avg_rank", "rank_trend", "day_over_day_ratio", "weekend_ratio",
    ]

    # 타겟
    target_col = ["target_audience"]

    # 최종 컬럼 순서
    final_cols = meta_cols + feature_cols + target_col

    # 존재하지 않는 컬럼 확인
    missing_cols = [col for col in final_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"누락된 컬럼: {missing_cols}")
        sys.exit(1)

    result_df = df[final_cols].copy()
    logger.info(f"  최종 피처 수: {len(feature_cols)}개")
    logger.info(f"  최종 컬럼: {final_cols}")

    return result_df


def engineer_features():
    """
    피처 엔지니어링 메인 함수: 전처리 결과에서 파생 피처를 생성합니다.
    """
    logger.info("=" * 60)
    logger.info("피처 엔지니어링을 시작합니다...")
    logger.info("=" * 60)

    # 1. 전처리 결과 로드
    df = load_intermediate_data(config.INTERMEDIATE_CSV_PATH)

    # 2. 파생 피처 생성
    df = add_week1_total(df)
    df = add_avg_screen_count(df)
    df = add_avg_show_count(df)
    df = add_avg_rank(df)
    df = add_rank_trend(df)
    df = add_day_over_day_ratio(df)
    df = add_weekend_ratio(df)

    # 3. 최종 피처 선택
    final_df = select_final_features(df)

    # 4. 결과 저장
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    final_df.to_csv(config.FINAL_CSV_PATH, index=False, encoding="utf-8")
    logger.info(f"최종 피처 저장: {config.FINAL_CSV_PATH}")

    # 5. 결과 요약
    logger.info("=" * 60)
    logger.info("피처 엔지니어링 완료 요약:")
    logger.info(f"  총 영화 수: {len(final_df):,}개")
    logger.info(f"  피처 수: 16개")
    logger.info(f"")
    logger.info("  [피처 통계]")

    # 주요 피처 통계 출력
    stat_cols = [
        "week1_total", "avg_screen_count", "avg_show_count",
        "avg_rank", "rank_trend", "day_over_day_ratio", "weekend_ratio",
        "target_audience",
    ]
    for col in stat_cols:
        logger.info(
            f"    {col:25s} | "
            f"평균: {final_df[col].mean():>12,.2f} | "
            f"중앙값: {final_df[col].median():>12,.2f} | "
            f"최소: {final_df[col].min():>12,.2f} | "
            f"최대: {final_df[col].max():>12,.2f}"
        )

    logger.info("=" * 60)

    return final_df


if __name__ == "__main__":
    engineer_features()
