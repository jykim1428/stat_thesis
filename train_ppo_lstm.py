"""
9월 2주차: 커스텀 LSTM 정책망 (CryptoAgent)
리포 루트에서 실행: python experiments/train_ppo_lstm.py

train_ppo_mlp.py(8월 2주차 MLP 베이스라인)와 train_ppo_transformer.py(9월 1주차)는
건드리지 않고 별도 스크립트로 둔다. policy_kwargs로 features_extractor만
LSTMFeaturesExtractor로 바꾼 것 외에는 둘과 구조 동일 (같은 env, 같은 공용
스펙 출력) - 세 모델을 나란히 비교 가능하게.

세 모델 공정 비교 조건 (피처·lookback·시드)
--------------------------------------------
TIME_WINDOW=50, FEATURES=["close","high","low"], SEED=42 -
train_ppo_mlp.py / train_ppo_transformer.py와 동일하게 고정.
hidden_size=32, num_layers=2는 transformer_extractor.py의
d_model=32, n_layers=2와 파라미터 규모를 맞춘 값 (lstm_extractor.py 참고).

작게 시작: hidden_size=32, num_layers=2. 체크리스트 요구사항(이 데이터
규모에서 파라미터 많으면 오버피팅 직행) 반영 - 트랜스포머와 동일 기준.

W&B/tensorboard 연동
---------------------
train_ppo_mlp.py / train_ppo_transformer.py 패턴을 그대로 이식:
wandb.init(sync_tensorboard=True)로 러닝을 열고, train()에
tensorboard_log/callback을 넘겨 WandbCallback으로 학습 로그를 같은
프로젝트(cryptoagent-ppo)에 기록한다. config에 hidden_size/num_layers를
추가로 남겨 MLP/Transformer run과 W&B 대시보드에서 구분 가능하게 함.

공용 모듈
--------
make_env/backtest/sanity_check는 cryptoagent.training.common에서 가져옴
(train_ppo_mlp.py, train_ppo_transformer.py와 동일 - 복붙 없음). 이 정책망
(LSTM) 고유 로직(policy_kwargs, HIDDEN_SIZE/NUM_LAYERS)만 이 파일에 있다.
"""

from __future__ import annotations

import os
from functools import partial

import shimmy
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from cryptoagent.policies.lstm_extractor import LSTMFeaturesExtractor
from cryptoagent.training.common import backtest, make_env as _make_env, sanity_check

# ── 설정 ─────────────────────────────────────────────
# MLP/Transformer와 공정 비교를 위해 아래 세 값은 동일하게 고정
TIME_WINDOW = 50
FEATURES = ["close", "high", "low"]
SEED = 42

INITIAL_AMOUNT = 100_000

HIDDEN_SIZE = 32  # transformer_extractor.py의 D_MODEL과 파라미터 규모 맞춤
NUM_LAYERS = 2    # transformer_extractor.py의 N_LAYERS와 동일
DROPOUT = 0.1

TOTAL_TIMESTEPS = 50_000  # 스모크 테스트용 기본값. 본 실험 규모는 게이트 통과 후 조정

RESULTS_DIR = "results/ppo_lstm"
MODEL_PATH = os.path.join(RESULTS_DIR, "ppo_lstm.zip")
BACKTEST_PATH = os.path.join(RESULTS_DIR, "backtest_test.csv")
# ─────────────────────────────────────────────────────


# make_env/backtest/sanity_check는 cryptoagent.training.common 공용 모듈에서 가져옴
# (train_ppo_mlp.py, train_ppo_transformer.py와 동일 패턴). 이 스크립트 설정값만
# 미리 바인딩해서 기존 make_env(split) 호출부는 그대로 쓸 수 있게 함.
make_env = partial(
    _make_env,
    features=FEATURES,
    initial_amount=INITIAL_AMOUNT,
    time_window=TIME_WINDOW,
)


def train(
    train_env,
    tensorboard_log: str | None = None,
    callback=None,
) -> PPO:
    """train split으로 PPO(LSTM policy) 학습.

    train_ppo_transformer.py의 train()과 동일 구조 - MlpPolicy 대신
    policy_kwargs로 features_extractor_class만 교체.
    """
    gym_env = shimmy.GymV21CompatibilityV0(env=train_env)
    vec_env = DummyVecEnv([lambda: gym_env])

    policy_kwargs = dict(
        features_extractor_class=LSTMFeaturesExtractor,
        features_extractor_kwargs=dict(
            hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT
        ),
    )

    model = PPO(
        "MlpPolicy",  # feature extractor만 LSTM, 이후 정책/가치 헤드는 SB3 기본 MLP
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=SEED,
        tensorboard_log=tensorboard_log,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    return model


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    run = wandb.init(
        entity="choieuna0711-student",
        project="cryptoagent-ppo",
        name="ppo_lstm_sanity_baseline",
        config={
            "total_timesteps": TOTAL_TIMESTEPS,
            "seed": SEED,
            "time_window": TIME_WINDOW,
            "features": FEATURES,
            "initial_amount": INITIAL_AMOUNT,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
        },
        sync_tensorboard=True,
    )

    print(f"=== [1/3] train split으로 PPO(LSTM, hidden_size={HIDDEN_SIZE}, "
          f"num_layers={NUM_LAYERS}, lookback={TIME_WINDOW}) 학습 ===")
    train_env = make_env("train")
    model = train(
        train_env,
        tensorboard_log=f"runs/{run.id}",
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"{RESULTS_DIR}/wandb_models/{run.id}",
        ),
    )
    model.save(MODEL_PATH)
    print(f"모델 저장: {MODEL_PATH}")

    print("\n=== [2/3] test split 백테스트 ===")
    test_env = make_env("test")
    backtest_df = backtest(model, test_env)
    backtest_df.to_csv(BACKTEST_PATH)
    print(f"백테스트 결과 저장: {BACKTEST_PATH}  shape={backtest_df.shape}")

    print("\n=== [3/3] Sanity Check ===")
    sanity_check(backtest_df)
    run.finish()


if __name__ == "__main__":
    main()
