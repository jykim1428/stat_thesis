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
hidden_size=32, num_layers=2는 transformer_extractor.py의 d_model=32,
n_layers=2와 representation width/depth를 맞춘 값 (실제 extractor
파라미터 수는 구조 차이로 Transformer 17,216개, LSTM 13,184개로 다름).

PPO 하이퍼파라미터는 세 모델 모두 SB3 기본값을 그대로 사용함
(learning_rate 등 별도 지정 없음).

작게 시작: hidden_size=32, num_layers=2. 체크리스트 요구사항(이 데이터
규모에서 파라미터 많으면 오버피팅 직행) 반영 - 트랜스포머와 동일 기준.

오버피팅 모니터링
------------------
EvalCallback으로 학습 도중 val split 성과를 주기적으로 측정해 W&B에
기록함 (src/cryptoagent/training/overfitting_monitor.py). MLP/
Transformer와 동일하게 eval_freq=10240, val env는 train과 동일 경로
(GymV21CompatibilityV0 -> Monitor)로 구성.

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

--eval-split (코덱스 리뷰 반영)
--------------------------------
기본값을 val로 바꾸고, test는 --eval-split test로 명시했을 때만 사용
(train_ppo_mlp.py와 동일 원칙 - test-set contamination 방지).
"""

from __future__ import annotations

import argparse
import os
from functools import partial

import shimmy
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from cryptoagent.policies.lstm_extractor import LSTMFeaturesExtractor
from cryptoagent.training.common import backtest, make_env as _make_env, sanity_check
from cryptoagent.training.overfitting_monitor import make_eval_callback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-split", choices=("val", "test"), default="val",
        help="백테스트에 쓸 split. test는 최종 평가 1회에만 명시적으로 사용할 것.",
    )
    return parser.parse_args()

# ── 설정 ─────────────────────────────────────────────
# MLP/Transformer와 공정 비교를 위해 아래 세 값은 동일하게 고정
TIME_WINDOW = 50
FEATURES = ["close", "high", "low"]
SEED = 42

INITIAL_AMOUNT = 100_000

HIDDEN_SIZE = 32  # transformer_extractor.py의 D_MODEL과 representation width 맞춤
NUM_LAYERS = 2    # transformer_extractor.py의 N_LAYERS와 동일
DROPOUT = 0.1

TOTAL_TIMESTEPS = 50_000  # 스모크 테스트용 기본값. 본 실험 규모는 게이트 통과 후 조정

EVAL_FREQ = 10_240  # MLP/Transformer와 동일 (5 * rollout(2048))

RESULTS_DIR = "results/ppo_lstm"
MODEL_PATH = os.path.join(RESULTS_DIR, "ppo_lstm.zip")
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


def make_val_env():
    """오버피팅 모니터링용 val env factory.

    train과 동일한 API 변환 경로(GymV21CompatibilityV0 -> Monitor)를
    거치도록 함 - MLP/Transformer와 동일 패턴.
    """
    env = make_env("val")
    gym_env = shimmy.GymV21CompatibilityV0(env=env)
    return Monitor(gym_env)


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
    vec_env = DummyVecEnv([lambda: Monitor(gym_env)])

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
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    backtest_path = os.path.join(RESULTS_DIR, f"backtest_{args.eval_split}.csv")

    if args.eval_split == "test":
        print(
            "[경고] --eval-split test로 실행합니다. test는 최종 평가 1회에만 "
            "사용해야 합니다 (반복 실행 시 test-set contamination 위험)."
        )

    run = wandb.init(
        entity="choieuna0711-student",
        project="cryptoagent-ppo",
        name=f"ppo_lstm_{args.eval_split}",
        config={
            "total_timesteps": TOTAL_TIMESTEPS,
            "seed": SEED,
            "time_window": TIME_WINDOW,
            "features": FEATURES,
            "initial_amount": INITIAL_AMOUNT,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "eval_freq": EVAL_FREQ,
            "eval_split": args.eval_split,
        },
        sync_tensorboard=True,
    )

    print(f"=== [1/3] train split으로 PPO(LSTM, hidden_size={HIDDEN_SIZE}, "
          f"num_layers={NUM_LAYERS}, lookback={TIME_WINDOW}) 학습 ===")
    train_env = make_env("train")

    eval_callback = make_eval_callback(make_val_env, RESULTS_DIR, eval_freq=EVAL_FREQ)

    model = train(
        train_env,
        tensorboard_log=f"runs/{run.id}",
        callback=CallbackList([
            WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"{RESULTS_DIR}/wandb_models/{run.id}",
            ),
            eval_callback,
        ]),
    )
    model.save(MODEL_PATH)
    print(f"모델 저장: {MODEL_PATH}")

    print(f"\n=== [2/3] {args.eval_split} split 백테스트 ===")
    eval_env = make_env(args.eval_split)
    backtest_df = backtest(model, eval_env)
    backtest_df.to_csv(backtest_path)
    print(f"백테스트 결과 저장: {backtest_path}  shape={backtest_df.shape}")

    print("\n=== [3/3] Sanity Check ===")
    sanity_check(backtest_df)
    run.finish()


if __name__ == "__main__":
    main()