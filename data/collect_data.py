# -*- coding: utf-8 -*-
"""
한국 박스오피스 데이터 수집 스크립트
====================================
KOFIC(영화진흥위원회) Open API를 사용하여 일별 박스오피스 데이터를 수집합니다.

사용법:
    python data/collect_data.py

기능:
    - 2010-01-01 ~ 2025-12-31 기간의 일별 박스오피스 데이터 수집
    - 이미 수집된 날짜는 건너뛰는 이어받기(resume) 기능
    - API 호출 실패 시 자동 재시도
    - 진행 상황 실시간 표시
"""

import os
import sys
import time
import csv
import logging
from datetime import datetime, timedelta

import requests

# 프로젝트 루트를 sys.path에 추가하여 config 모듈 임포트 가능하게 함
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

# CSV 헤더: API 조회 날짜 + 추출 대상 9개 필드
CSV_HEADER = ["date"] + config.EXTRACT_FIELDS


def load_collected_dates(csv_path: str) -> set:
    """
    이미 수집된 날짜 목록을 CSV 파일에서 읽어 반환합니다.
    이어받기(resume) 기능에 사용됩니다.

    Args:
        csv_path: 원본 데이터 CSV 파일 경로

    Returns:
        수집 완료된 날짜 문자열(YYYYMMDD)의 집합
    """
    collected = set()
    if not os.path.exists(csv_path):
        return collected

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                date_val = row.get("date", "").strip()
                if date_val:
                    collected.add(date_val)
    except Exception as e:
        logger.warning(f"기존 CSV 파일 읽기 실패: {e}")

    return collected


def generate_date_range(start_date: str, end_date: str) -> list:
    """
    시작일부터 종료일까지의 날짜 목록을 생성합니다.

    Args:
        start_date: 시작 날짜 (YYYY-MM-DD 형식)
        end_date: 종료 날짜 (YYYY-MM-DD 형식)

    Returns:
        YYYYMMDD 형식 날짜 문자열 리스트
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # 종료일이 오늘 이후이면 어제까지만 수집 (당일 데이터는 미확정)
    yesterday = datetime.now() - timedelta(days=1)
    if end > yesterday:
        end = yesterday
        logger.info(f"종료일을 어제({end.strftime('%Y-%m-%d')})로 조정합니다 (당일 데이터 미확정).")

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)

    return dates


def fetch_daily_boxoffice(target_date: str) -> list:
    """
    특정 날짜의 일별 박스오피스 데이터를 API에서 가져옵니다.

    Args:
        target_date: 조회 대상 날짜 (YYYYMMDD 형식)

    Returns:
        영화별 데이터 딕셔너리 리스트. 실패 시 빈 리스트 반환.
    """
    params = {
        "key": config.API_KEY,
        "targetDt": target_date,
    }

    for attempt in range(1, config.API_MAX_RETRIES + 1):
        try:
            response = requests.get(
                config.API_BASE_URL,
                params=params,
                timeout=config.API_TIMEOUT,
            )
            response.raise_for_status()

            data = response.json()

            # API 응답 구조 확인
            box_office_result = data.get("boxOfficeResult")
            if box_office_result is None:
                logger.error(f"[{target_date}] 응답에 'boxOfficeResult'가 없습니다: {data}")
                return []

            daily_list = box_office_result.get("dailyBoxOfficeList", [])

            # 필요한 필드만 추출
            results = []
            for movie in daily_list:
                row = {"date": target_date}
                for field in config.EXTRACT_FIELDS:
                    row[field] = movie.get(field, "")
                results.append(row)

            return results

        except requests.exceptions.Timeout:
            logger.warning(
                f"[{target_date}] 요청 타임아웃 (시도 {attempt}/{config.API_MAX_RETRIES})"
            )
        except requests.exceptions.ConnectionError:
            logger.warning(
                f"[{target_date}] 연결 오류 (시도 {attempt}/{config.API_MAX_RETRIES})"
            )
        except requests.exceptions.HTTPError as e:
            logger.warning(
                f"[{target_date}] HTTP 오류: {e} (시도 {attempt}/{config.API_MAX_RETRIES})"
            )
        except (ValueError, KeyError) as e:
            logger.error(f"[{target_date}] 응답 파싱 실패: {e}")
            return []  # 파싱 오류는 재시도 불필요
        except Exception as e:
            logger.error(f"[{target_date}] 예기치 않은 오류: {e}")
            return []

        # 재시도 전 대기
        if attempt < config.API_MAX_RETRIES:
            wait_time = config.API_RETRY_DELAY * attempt  # 점진적 대기
            logger.info(f"  {wait_time}초 후 재시도...")
            time.sleep(wait_time)

    logger.error(f"[{target_date}] 최대 재시도 횟수 초과. 해당 날짜 건너뜁니다.")
    return []


def save_rows_to_csv(csv_path: str, rows: list, write_header: bool):
    """
    수집된 데이터 행을 CSV 파일에 추가 저장합니다.

    Args:
        csv_path: 저장할 CSV 파일 경로
        rows: 저장할 데이터 행 리스트
        write_header: 헤더 행 작성 여부
    """
    mode = "w" if write_header else "a"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def collect_data():
    """
    메인 수집 함수: 설정된 기간의 일별 박스오피스 데이터를 수집합니다.
    이미 수집된 날짜는 자동으로 건너뜁니다.
    """
    # API 키 확인
    if config.API_KEY == "YOUR_API_KEY_HERE" or not config.API_KEY:
        logger.error("=" * 60)
        logger.error("API 키가 설정되지 않았습니다!")
        logger.error("data/data_config.py 파일에서 API_KEY 값을 설정해 주세요.")
        logger.error("API 키 발급: https://www.kobis.or.kr/kobisopenapi/")
        logger.error("=" * 60)
        sys.exit(1)

    # 저장 디렉토리 생성
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)

    # 이미 수집된 날짜 확인 (이어받기)
    collected_dates = load_collected_dates(config.RAW_CSV_PATH)
    if collected_dates:
        logger.info(f"기존 수집 데이터 발견: {len(collected_dates)}일치 수집 완료")

    # 수집 대상 날짜 범위 생성
    all_dates = generate_date_range(config.COLLECT_START_DATE, config.COLLECT_END_DATE)
    total_dates = len(all_dates)
    logger.info(f"전체 수집 대상 기간: {config.COLLECT_START_DATE} ~ {config.COLLECT_END_DATE}")
    logger.info(f"전체 날짜 수: {total_dates}일")

    # 아직 수집하지 않은 날짜만 필터링
    remaining_dates = [d for d in all_dates if d not in collected_dates]
    logger.info(f"수집 필요 날짜: {len(remaining_dates)}일")

    if not remaining_dates:
        logger.info("모든 날짜의 데이터가 이미 수집되었습니다. 종료합니다.")
        return

    # CSV 파일에 헤더가 필요한지 확인
    need_header = not os.path.exists(config.RAW_CSV_PATH) or os.path.getsize(config.RAW_CSV_PATH) == 0

    # 수집 시작
    logger.info("=" * 60)
    logger.info("데이터 수집을 시작합니다...")
    logger.info("=" * 60)

    collected_count = 0
    error_count = 0
    total_movies = 0
    api_call_count = 0  # 일일 API 호출 횟수 추적
    hit_daily_limit = False  # 일일 제한 도달 여부
    start_time = time.time()

    for idx, target_date in enumerate(remaining_dates, 1):
        # 진행 상황 표시
        formatted_date = f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}"
        progress_pct = (idx / len(remaining_dates)) * 100

        # 경과 시간 및 예상 남은 시간 계산
        elapsed = time.time() - start_time
        if idx > 1:
            avg_time_per_date = elapsed / (idx - 1)
            remaining_time = avg_time_per_date * (len(remaining_dates) - idx)
            eta_str = _format_seconds(remaining_time)
        else:
            eta_str = "계산 중..."

        logger.info(
            f"[{idx}/{len(remaining_dates)}] {formatted_date} 수집 중... "
            f"({progress_pct:.1f}%) | 예상 남은 시간: {eta_str}"
        )

        # API 호출
        rows = fetch_daily_boxoffice(target_date)

        if rows:
            # CSV에 저장
            save_rows_to_csv(config.RAW_CSV_PATH, rows, write_header=need_header)
            need_header = False  # 최초 1회만 헤더 작성

            collected_count += 1
            total_movies += len(rows)
        else:
            error_count += 1
            logger.warning(f"  → {formatted_date}: 데이터 없음 또는 수집 실패")

        # 일일 API 호출 제한 확인
        api_call_count += 1
        if api_call_count >= config.API_DAILY_LIMIT:
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"⚠️  일일 API 호출 제한({config.API_DAILY_LIMIT}회)에 도달했습니다.")
            logger.info(f"   내일 다시 실행하면 나머지 데이터를 이어서 수집합니다.")
            logger.info(f"   남은 수집 일수: {len(remaining_dates) - idx}일")
            logger.info("=" * 60)
            hit_daily_limit = True
            break

        # API 호출 간 대기 (마지막 요청 후에는 불필요)
        if idx < len(remaining_dates):
            time.sleep(config.API_RATE_LIMIT)

    # 수집 완료 요약
    elapsed_total = time.time() - start_time
    logger.info("=" * 60)
    if hit_daily_limit:
        logger.info("데이터 수집 일시 중단 (일일 제한 도달)")
    else:
        logger.info("데이터 수집 완료!")
    logger.info(f"  금일 API 호출 횟수: {api_call_count}회")
    logger.info(f"  성공: {collected_count}일")
    logger.info(f"  실패: {error_count}일")
    logger.info(f"  수집된 총 영화 레코드: {total_movies}건")
    logger.info(f"  소요 시간: {_format_seconds(elapsed_total)}")
    logger.info(f"  저장 위치: {config.RAW_CSV_PATH}")
    if hit_daily_limit:
        logger.info(f"")
        logger.info(f"  👉 내일 동일한 명령으로 다시 실행해 주세요.")
        logger.info(f"     python data/collect_data.py")
    logger.info("=" * 60)


def _format_seconds(seconds: float) -> str:
    """초를 시:분:초 형식의 문자열로 변환합니다."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}시간 {minutes}분 {secs}초"
    elif minutes > 0:
        return f"{minutes}분 {secs}초"
    else:
        return f"{secs}초"


if __name__ == "__main__":
    collect_data()
