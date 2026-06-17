# -*- coding: utf-8 -*-
"""
딥러닝 모델 모듈
================
영화 관객수 예측을 위한 PyTorch 기반 Multi-Layer Perceptron (MLP) 모델을 정의합니다.
하이퍼파라미터(레이어 수, 뉴런 수, 드롭아웃 비율)에 따라 동적으로 아키텍처가 구성됩니다.
"""

import torch
import torch.nn as nn


class BoxOfficeMLP(nn.Module):
    """
    관객수 예측을 위한 동적 MLP 모델
    """

    def __init__(self, input_dim: int = 16, num_layers: int = 3, hidden_units: int = 256, dropout_rate: float = 0.2):
        """
        모델 초기화

        Args:
            input_dim: 입력 피처 차원 수 (기본값: 16)
            num_layers: 은닉층 개수 (2~5개)
            hidden_units: 은닉층 뉴런 수 (64~512개)
            dropout_rate: 드롭아웃 비율 (0.0~0.5)
        """
        super(BoxOfficeMLP, self).__init__()
        
        layers = []
        in_dim = input_dim

        for i in range(num_layers):
            # 1. 선형 결합 층 (Linear Layer)
            layers.append(nn.Linear(in_dim, hidden_units))
            
            # 2. 배치 정규화 (Batch Normalization) - 학습 안정성 향상
            layers.append(nn.BatchNorm1d(hidden_units))
            
            # 3. 활성화 함수 (ReLU)
            layers.append(nn.ReLU())
            
            # 4. 드롭아웃 (Dropout) - 과적합 방지
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
                
            in_dim = hidden_units

        # 출력층: 최종 1개의 관객수 값을 출력 (회귀 분석이므로 활성화 함수 없음)
        layers.append(nn.Linear(hidden_units, 1))

        # 순차적 레이어 구성
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파 (Forward Pass)

        Args:
            x: 입력 텐서 (shape: [batch_size, input_dim])

        Returns:
            예측 텐서 (shape: [batch_size, 1])
        """
        return self.network(x)


if __name__ == "__main__":
    # 모델 정의가 잘 돌아가는지 간단히 테스트
    test_model = BoxOfficeMLP(input_dim=16, num_layers=3, hidden_units=128, dropout_rate=0.2)
    print("모델 구조 확인:")
    print(test_model)
    
    # 더미 데이터로 작동 확인
    dummy_input = torch.randn(5, 16)
    dummy_output = test_model(dummy_input)
    print(f"입력 크기: {dummy_input.shape} -> 출력 크기: {dummy_output.shape}")
