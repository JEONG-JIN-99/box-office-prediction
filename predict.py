# -*- coding: utf-8 -*-
"""
최종 관객수 예측 실행 스크립트
==============================
학습 완료된 최적 모델과 스케일러 메타데이터를 로드하여,
사용자가 입력한 개봉 첫 주(7일간) 실적을 기반으로 최종 누적 관객수를 예측합니다.

사용법:
    python predict.py --day1 150000 --day2 120000 --day3 95000 \
                      --day4 80000 --day5 110000 --day6 180000 --day7 200000 \
                      --open_date 2026-06-07
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
import numpy as np
import torch

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import data_config as config
from models.model import BoxOfficeMLP

def parse_args():
    parser = argparse.ArgumentParser(description="영화 최종 누적 관객수 예측 프로그램")
    
    # 7일간 일별 관객수 (필수)
    parser.add_argument("--day1", type=int, required=True, help="개봉 1일차 관객수")
    parser.add_argument("--day2", type=int, required=True, help="개봉 2일차 관객수")
    parser.add_argument("--day3", type=int, required=True, help="개봉 3일차 관객수")
    parser.add_argument("--day4", type=int, required=True, help="개봉 4일차 관객수")
    parser.add_argument("--day5", type=int, required=True, help="개봉 5일차 관객수")
    parser.add_argument("--day6", type=int, required=True, help="개봉 6일차 관객수")
    parser.add_argument("--day7", type=int, required=True, help="개봉 7일차 관객수")

    # 선택적 정보 입력
    parser.add_argument("--open_date", type=str, default=None, help="개봉일 (YYYY-MM-DD 또는 YYYYMMDD, 기본값: 오늘)")
    parser.add_argument("--avg_screen", type=float, default=600.0, help="7일 평균 스크린 수 (기본값: 600)")
    parser.add_argument("--avg_show", type=float, default=2500.0, help="7일 평균 상영 횟수 (기본값: 2500)")
    parser.add_argument("--avg_rank", type=float, default=3.0, help="7일 평균 박스오피스 순위 (기본값: 3.0)")
    parser.add_argument("--rank_trend", type=float, default=0.0, help="7일 평균 순위 변동 (기본값: 0.0)")

    return parser.parse_args()


def main():
    args = parse_args()
    
    models_dir = os.path.join(config.PROJECT_ROOT, "models")
    metadata_path = os.path.join(models_dir, "scaler_config.json")
    model_weight_path = os.path.join(models_dir, "best_model.pth")
    
    # 모델 및 메타데이터 존재 여부 확인
    if not os.path.exists(metadata_path) or not os.path.exists(model_weight_path):
        print("=" * 60)
        print("❌ 예측 불가: 학습된 최적 모델 또는 설정 파일이 존재하지 않습니다!")
        print("   먼저 python train.py를 실행해 학습을 진행해 주세요.")
        print("=" * 60)
        sys.exit(1)

    # 1. 설정 로드
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    scaler_mean = np.array(metadata["mean"])
    scaler_std = np.array(metadata["std"])
    best_params = metadata["best_hyperparameters"]

    # 2. 개봉일 파싱 및 시기 피처 추출
    if args.open_date:
        clean_date_str = args.open_date.strip().replace("-", "")
    else:
        clean_date_str = datetime.now().strftime("%Y%m%d")
        
    try:
        open_date = datetime.strptime(clean_date_str, "%Y%m%d")
    except ValueError:
        print(f"❌ 개봉일 날짜 형식이 잘못되었습니다: {args.open_date}. YYYY-MM-DD 형식으로 입력하세요.")
        sys.exit(1)

    open_year = open_date.year
    open_month = open_date.month

    # 3. 파생 피처 계산
    day_audiences = [args.day1, args.day2, args.day3, args.day4, args.day5, args.day6, args.day7]
    week1_total = sum(day_audiences)
    
    # day7 / day1 비율
    day_over_day_ratio = round(args.day7 / args.day1, 4) if args.day1 > 0 else 0.0

    # 주말 관객 비율 (개봉일을 요일 기준으로 삼아 주말에 해당하는 날짜의 관객수 합산)
    weekend_audience = 0
    for day_idx, audience in enumerate(day_audiences):
        current_date = open_date + timedelta(days=day_idx)
        # 5=토요일, 6=일요일
        if current_date.weekday() in (5, 6):
            weekend_audience += audience
            
    weekend_ratio = round(weekend_audience / week1_total, 4) if week1_total > 0 else 0.0

    # 4. 피처 벡터 순서 조립 (총 16개)
    # feature_cols 순서:
    # 1-7: day1_audience ~ day7_audience
    # 8: open_year, 9: open_month, 10: week1_total, 
    # 11: avg_screen_count, 12: avg_show_count, 13: avg_rank, 14: rank_trend, 
    # 15: day_over_day_ratio, 16: weekend_ratio
    feature_vector = np.array([
        args.day1, args.day2, args.day3, args.day4, args.day5, args.day6, args.day7,
        open_year, open_month, week1_total,
        args.avg_screen, args.avg_show, args.avg_rank, args.rank_trend,
        day_over_day_ratio, weekend_ratio
    ], dtype=np.float32)

    # 5. 피처 스케일링
    feature_vector_scaled = (feature_vector - scaler_mean) / scaler_std
    feature_tensor = torch.tensor(feature_vector_scaled, dtype=torch.float32).unsqueeze(0)

    # 6. 모델 초기화 및 가중치 불러오기
    model = BoxOfficeMLP(
        input_dim=16,
        num_layers=best_params["num_layers"],
        hidden_units=best_params["hidden_units"],
        dropout_rate=best_params["dropout_rate"]
    )
    
    # CPU로 가중치 로드
    model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))
    model.eval()

    # 7. 예측 수행
    with torch.no_grad():
        pred_log = model(feature_tensor).item()
        
    # 로그 스케일 역변환 (expm1)
    predicted_audience = np.expm1(pred_log)
    predicted_audience = max(0.0, predicted_audience)

    # 8. 예측 오차 범위를 기반으로 신뢰 구간 설정 (테스트 세트 MAPE 활용)
    test_mape = metadata["test_performance"]["mape"] / 100.0  # 퍼센트를 소수로 변환
    error_margin = predicted_audience * test_mape
    lower_bound = max(week1_total, predicted_audience - error_margin)
    upper_bound = predicted_audience + error_margin

    # 결과 출력
    print("\n" + "=" * 60)
    print("🎬  영화 최종 누적 관객수 예측 결과")
    print("=" * 60)
    print(f"개봉일           : {open_date.strftime('%Y년 %m월 %d일')} ({['월','화','수','목','금','토','일'][open_date.weekday()]}요일)")
    print(f"첫 주 누적 관객수: {week1_total:,.0f} 명")
    print(f"7일 평균 스크린수: {args.avg_screen:,.0f} 스크린")
    print(f"주말 관객 비율   : {weekend_ratio * 100:.2f} %")
    print("-" * 60)
    print(f"💡 예측 최종 관객수: {predicted_audience:,.0f} 명 (약 {predicted_audience/10000.0:.1f} 만 명)")
    print(f"🔒 예측 범위(신뢰) : {lower_bound:,.0f} ~ {upper_bound:,.0f} 명")
    print(f"   (테스트 세트 MAPE 오차 기준: ±{test_mape*100:.1f}%)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
