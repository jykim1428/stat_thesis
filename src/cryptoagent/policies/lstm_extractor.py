"""
9월 2주차: 커스텀 LSTM 정책망 (SB3 BaseFeaturesExtractor)

transformer_extractor.py와 동일한 입출력 계약을 따른다 - 세 모델(MLP/LSTM/
Transformer)을 공정 비교하려면 관측 처리 방식이 같아야 하므로, PortfolioOptimizationEnv
observation (batch, feature, asset, time) = (batch, 3, 8, lookback)을 그대로
입력으로 받고, 자산별 독립 시퀀스로 나눠 시간 축을 인코딩한 뒤 자산 축을
concat해서 최종 feature vector로 압축하는 흐름을 그대로 재사용한다.

트랜스포머는 self-attention으로 시간 축을 인코딩하지만, LSTM은 순환 구조로
시간 축을 인코딩한다는 점만 다르다. 마지막 시점의 hidden state를 그 자산의
시계열 요약으로 쓴다 (트랜스포머 쪽의 "시간 축 평균 풀링" 대신).

변환 순서: (batch, feature, asset, time)
        -> permute(batch, asset, time, feature)   [transformer_extractor.py와 동일]
        -> reshape(batch*asset, time, feature)     [자산별 독립 시퀀스로]
        -> LSTM(시간 축을 순환으로 인코딩)
        -> 마지막 시점 hidden state -> (batch*asset, hidden_size)
        -> reshape(batch, asset*hidden_size) -> 최종 feature vector

작게 시작: 기본 hidden_size=32, num_layers=2 - transformer_extractor.py의
d_model=32, n_layers=2와 파라미터 규모를 맞춰서 세 모델(피처·lookback·시드
동일 조건) 공정 비교가 가능하게 함.
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LSTMFeaturesExtractor(BaseFeaturesExtractor):
    """PortfolioOptimizationEnv observation (feature, asset, time) -> feature vector.

    train_ppo_lstm.py에서 이렇게 연결:

        policy_kwargs = dict(
            features_extractor_class=LSTMFeaturesExtractor,
            features_extractor_kwargs=dict(hidden_size=32, num_layers=2),
        )
        model = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, ...)

    lookback window는 env 생성 시 time_window로 바꾸면 되고, 이 클래스는
    observation_space.shape에서 자동으로 읽어오므로 코드 수정이 필요 없다
    (transformer_extractor.py와 동일).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        # transformer_extractor.py의 d_model % n_heads 검증과 같은 자리 -
        # nn.LSTM은 num_layers==1일 때 dropout>0을 조용히 무시하고 경고만
        # 띄우므로, 여기서 생성 시점에 명시적으로 막는다.
        if num_layers < 1:
            raise ValueError(f"num_layers({num_layers})는 1 이상이어야 합니다.")
        if num_layers == 1 and dropout > 0:
            raise ValueError(
                f"num_layers=1일 땐 dropout={dropout}이 적용되지 않습니다 "
                "(PyTorch nn.LSTM 제약). num_layers를 2 이상으로 올리거나 dropout=0으로 두세요."
            )

        n_features, n_assets, lookback = observation_space.shape
        features_dim = n_assets * hidden_size
        super().__init__(observation_space, features_dim=features_dim)

        self.n_features = n_features
        self.n_assets = n_assets
        self.lookback = lookback
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch, feature, asset, time)
        batch_size = observations.shape[0]

        # (batch, feature, asset, time) -> (batch, asset, time, feature)
        x = observations.permute(0, 2, 3, 1)
        # (batch, asset, time, feature) -> (batch*asset, time, feature)
        x = x.reshape(batch_size * self.n_assets, self.lookback, self.n_features)

        _, (h_n, _) = self.lstm(x)      # h_n: (num_layers, batch*asset, hidden_size)
        x = h_n[-1]                     # 마지막 레이어의 마지막 시점 hidden state -> (batch*asset, hidden_size)

        x = x.reshape(batch_size, self.n_assets * self.hidden_size)  # (batch, features_dim)
        return x
