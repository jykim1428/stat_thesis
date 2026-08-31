"""
9월 1주차: 커스텀 트랜스포머 정책망 (CryptoAgent)
리포 루트에서 실행: python experiments/train_ppo_transformer.py

train_ppo_mlp.py(8월 2주차 MLP 베이스라인, 8월 게이트 비교표에 쓰이는 결과)는
건드리지 않고 별도 스크립트로 둔다. policy_kwargs로 features_extractor만
TransformerFeaturesExtractor로 바꾼 것 외에는 train_ppo_mlp.py와 구조 동일
(같은 env, 같은 공용 스펙 출력) - MLP와 나중에 나란히 비교 가능하게.

착수 조건 (팀 합의)
--------------------
코드 작성/스모크 테스트는 8월 게이트 전에 미리 해도 되지만, 본 실험
(TOTAL_TIMESTEPS 규모 학습 + 결과를 비교표에 반영)은 8월 게이트
(PPO(MLP) vs 전통모델 비교표 완성) 통과 후 진행한다.

lookback window(TIME_WINDOW)와 트랜스포머 크기(D_MODEL, N_HEADS, N_LAYERS)는
전부 하이퍼파라미터로 노출되어 있다 - env_portfolio_optimization.py의
observation_space가 TIME_WINDOW에 맞춰 자동으로 (3, 8, TIME_WINDOW)가 되므로
extractor 쪽 코드 수정 없이 50~168h 등 다른 lookback으로 바로 실험 가능
(50, 168 두 값으로 직접 검증함).

작게 시작: 기본 2 layer / 4 head, d_model=32 (파라미터 약 1.7만개).
체크리스트 요구사항(이 데이터 규모에서 파라미터 많으면 오버피팅 직행) 반영.

오버피팅 모니터링 (9월 3주차, 은아)
--------------------------------------
EvalCallback으로 학습 도중 val split 성과를 주기적으로 측정해 W&B에
기록함 (src/cryptoagent/training/overfitting_monitor.py).

eval_freq=10240(=5*rollout 크기)으로 설정 - 2048마다 평가하면 val
에피소드 길이(6,838 step) 때문에 평가 비용이 학습 비용의 3배 이상이
되어 비효율적임을 스모크 테스트로 확인, 10,240으로 조정함
(50,000 스텝 기준 약 5회 평가).

train episode(수만 스텝)가 PPO rollout(2048)보다 길어
rollout/ep_rew_mean과 eval/mean_reward를 직접 비교하기는 어려움.
대신 학습 안정성(approx_kl, explained_variance 등)과 일반화
(eval/mean_reward, val Sharpe/MDD/turnover)를 각각 관찰하는 것으로
목표를 재정의함. 평가 지점이 적을 때 단기 등락만으로 오버피팅을
단정하지 않고, 충분한 학습 구간에 걸친 추세로 판단할 것.

val 환경도 train과 동일한 API 변환 경로(GymV21CompatibilityV0 ->
Monitor)를 거치도록 함 - SB3의 "Monitor wrapper 없음" 경고 제거 및
경로 일관성 확보.
참고: make_env / backtest / sanity_check는 train_ppo_mlp.py와 공용으로
쓰기 위해 cryptoagent.training.common으로 추출됨 (8월 2주차 이월 작업).

W&B/tensorboard 연동 (9월 1주차 은아 구현)
--------------------------------------------
train_ppo_mlp.py 패턴을 그대로 이식: wandb.init(sync_tensorboard=True)로
러닝을 열고, train()에 tensorboard_log/callback을 넘겨 WandbCallback으로
학습 로그를 같은 프로젝트(cryptoagent-ppo)에 기록한다.
"""

from __future__ import annotations

import os
from functools import partial

import shimmy
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from cryptoagent.envs.env_portfolio_optimization import PortfolioOptimizationEnv
from cryptoagent.training.common import backtest, make_env as _make_env, sanity_check
from cryptoagent.policies.transformer_extractor import TransformerFeaturesExtractor
from cryptoagent.training.overfitting_monitor import make_eval_callback

# ── 설정 ─────────────────────────────────────────────
TIME_WINDOW = 50  # lookback. 50~168h 범위에서 하이퍼파라미터로 조정 가능 (검증됨)
FEATURES = ["close", "high", "low"]
INITIAL_AMOUNT = 100_000

D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2

TOTAL_TIMESTEPS = 50_000  # 스모크 테스트용 기본값. 본 실험 규모는 게이트 통과 후 조정
SEED = 42

EVAL_FREQ = 10_240  # 5 * rollout(2048). val 에피소드가 길어(6,838 step)
                     # 2048 주기는 평가 비용이 과도함 (코덱스 리뷰 반영)

RESULTS_DIR = "results/ppo_transformer"
MODEL_PATH = os.path.join(RESULTS_DIR, "ppo_transformer.zip")
BACKTEST_PATH = os.path.join(RESULTS_DIR, "backtest_test.csv")
# ─────────────────────────────────────────────────────


# make_env/backtest/sanity_check는 cryptoagent.training.common 공용 모듈에서 가져옴
# (train_ppo_mlp.py, train_ppo_lstm.py와 동일 패턴). 이 스크립트 설정값만
# 미리 바인딩해서 make_env(split) 형태로 세 스크립트가 동일한 인터페이스를 갖게 함
# - 오케스트레이션에서 모델 종류와 무관하게 make_env("train")로 호출 가능.
make_env = partial(
    _make_env,
    features=FEATURES,
    initial_amount=INITIAL_AMOUNT,
    time_window=TIME_WINDOW,
)


def make_val_env():
    """오버피팅 모니터링용 val env factory.

    train과 동일한 API 변환 경로(GymV21CompatibilityV0 -> Monitor)를
    거치도록 함 - SB3의 "Monitor wrapper 없음" 경고 제거 및 train/val
    경로 일관성 확보 (코덱스 리뷰 반영).
    """
    env = make_env("val")
    gym_env = shimmy.GymV21CompatibilityV0(env=env)
    return Monitor(gym_env)


def train(
    train_env: PortfolioOptimizationEnv,
    tensorboard_log: str | None = None,
    callback=None,
) -> PPO:
    """train split으로 PPO(Transformer policy) 학습.

    train_ppo_mlp.py의 train()과 동일 구조 - MlpPolicy 대신
    policy_kwargs로 features_extractor_class만 교체.

    (9월 1주차 은아 구현) tensorboard_log, callback 파라미터를 추가해
    train_ppo_mlp.py와 동일한 방식으로 WandbCallback 연결.

    표준화 관련 하이퍼파라미터(learning_rate 등)는 표준화 미적용
    최종 결정에 따라 넣지 않음 - SB3 PPO 기본값 사용 (MLP/LSTM과 통일).
    """
    gym_env = shimmy.GymV21CompatibilityV0(env=train_env)
    vec_env = DummyVecEnv([lambda: Monitor(gym_env)])

    policy_kwargs = dict(
        features_extractor_class=TransformerFeaturesExtractor,
        features_extractor_kwargs=dict(
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS
        ),
    )

    model = PPO(
        "MlpPolicy",
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
        name=f"ppo_transformer_d{D_MODEL}_h{N_HEADS}_l{N_LAYERS}_lb{TIME_WINDOW}",
        config={
            "total_timesteps": TOTAL_TIMESTEPS,
            "seed": SEED,
            "time_window": TIME_WINDOW,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "features": FEATURES,
            "initial_amount": INITIAL_AMOUNT,
            "eval_freq": EVAL_FREQ,
        },
        sync_tensorboard=True,
    )

    print(f"=== [1/3] train split으로 PPO(Transformer, d_model={D_MODEL}, "
          f"heads={N_HEADS}, layers={N_LAYERS}, lookback={TIME_WINDOW}) 학습 ===")
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