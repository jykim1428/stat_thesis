"""
9월 1주차: 커스텀 트랜스포머 정책망 (SB3 BaseFeaturesExtractor)

PortfolioOptimizationEnv의 observation은 (batch, feature, asset, time) =
(batch, 3, 8, lookback) 형태로 들어온다 (observation_shape.md에서 확정된 값,
lookback=50이 기본이지만 env 생성 시 time_window로 바뀌면 여기도 따라감).

트랜스포머 인코더는 "시간 축"을 시퀀스로 본다 - 각 자산의 시계열 패턴을
독립적으로 인코딩한 뒤, 자산 축은 concat해서 최종 feature vector로 압축한다.
자산 개수(8)가 바뀌면 features_dim도 따라 바뀐다.

변환 순서: (batch, feature, asset, time)
        -> permute(batch, asset, time, feature)   [observation_shape.md 문서 기준]
        -> reshape(batch*asset, time, feature)     [자산별 독립 시퀀스로]
        -> Linear(feature -> d_model) + positional encoding
        -> TransformerEncoder(시간 축에 self-attention)
        -> 시간 축 평균 풀링 -> (batch*asset, d_model)
        -> reshape(batch, asset*d_model) -> 최종 feature vector

작게 시작: 기본 2 layer, 4 head (체크리스트 요구사항 - 이 데이터 규모에서
파라미터 많으면 오버피팅 직행).
"""

from __future__ import annotations

import math

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PositionalEncoding(nn.Module):
    """표준 sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1), :]


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """PortfolioOptimizationEnv observation (feature, asset, time) -> feature vector.

    train_ppo_transformer.py에서 이렇게 연결:

        policy_kwargs = dict(
            features_extractor_class=TransformerFeaturesExtractor,
            features_extractor_kwargs=dict(d_model=32, n_heads=4, n_layers=2),
        )
        model = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, ...)

    lookback window는 env 생성 시 time_window로 바꾸면 되고, 이 클래스는
    observation_space.shape에서 자동으로 읽어오므로 코드 수정이 필요 없다.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
    ):
        if d_model % n_heads != 0:
            raise ValueError(f"d_model({d_model})은 n_heads({n_heads})로 나누어 떨어져야 합니다.")

        n_features, n_assets, lookback = observation_space.shape
        features_dim = n_assets * d_model
        super().__init__(observation_space, features_dim=features_dim)

        self.n_features = n_features
        self.n_assets = n_assets
        self.lookback = lookback
        self.d_model = d_model

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max(lookback, 500))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch, feature, asset, time)
        batch_size = observations.shape[0]

        # (batch, feature, asset, time) -> (batch, asset, time, feature)
        x = observations.permute(0, 2, 3, 1)
        # (batch, asset, time, feature) -> (batch*asset, time, feature)
        x = x.reshape(batch_size * self.n_assets, self.lookback, self.n_features)

        x = self.input_proj(x)          # (batch*asset, time, d_model)
        x = self.pos_encoding(x)
        x = self.encoder(x)             # (batch*asset, time, d_model)

        x = x.mean(dim=1)               # 시간 축 평균 풀링 -> (batch*asset, d_model)
        x = x.reshape(batch_size, self.n_assets * self.d_model)  # (batch, features_dim)
        return x
