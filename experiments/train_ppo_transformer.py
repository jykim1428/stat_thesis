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
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import shimmy
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from cryptoagent.envs.adapter import load_env_ready_df, patch_seed_method
from cryptoagent.envs.env_portfolio_optimization import PortfolioOptimizationEnv
from cryptoagent.policies.transformer_extractor import TransformerFeaturesExtractor

# ── 설정 ─────────────────────────────────────────────
TIME_WINDOW = 50  # lookback. 50~168h 범위에서 하이퍼파라미터로 조정 가능 (검증됨)
FEATURES = ["close", "high", "low"]
INITIAL_AMOUNT = 100_000

D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2

TOTAL_TIMESTEPS = 50_000  # 스모크 테스트용 기본값. 본 실험 규모는 게이트 통과 후 조정
SEED = 42

RESULTS_DIR = "results/ppo_transformer"
MODEL_PATH = os.path.join(RESULTS_DIR, "ppo_transformer.zip")
BACKTEST_PATH = os.path.join(RESULTS_DIR, "backtest_test.csv")
# ─────────────────────────────────────────────────────


def make_env(split: str) -> PortfolioOptimizationEnv:
    df = load_env_ready_df(split=split)
    env = PortfolioOptimizationEnv(
        df=df,
        initial_amount=INITIAL_AMOUNT,
        time_column="date",
        tic_column="tic",
        features=FEATURES,
        time_window=TIME_WINDOW,
    )
    patch_seed_method(env)
    return env


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
    """
    gym_env = shimmy.GymV21CompatibilityV0(env=train_env)
    vec_env = DummyVecEnv([lambda: gym_env])

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


def backtest(model: PPO, eval_env: PortfolioOptimizationEnv) -> pd.DataFrame:
    """train_ppo_mlp.py의 backtest()와 동일 - 공용 스펙 DataFrame 생성."""
    gym_env = shimmy.GymV21CompatibilityV0(env=eval_env)

    obs, _ = gym_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = gym_env.step(action)
        done = terminated or truncated

    result = pd.DataFrame(
        {
            "date": eval_env._date_memory,
            "returns": eval_env._portfolio_return_memory,
            "portfolio_values": eval_env._asset_memory["final"],
            "weights": [w.tolist() for w in eval_env._final_weights],
            "target_weights": [w.tolist() for w in eval_env._actions_memory],
        }
    )
    result["date"] = pd.to_datetime(result["date"])
    result = result.set_index("date")
    return result


def sanity_check(backtest_df: pd.DataFrame) -> None:
    weight_sums = backtest_df["weights"].apply(sum)
    max_dev = (weight_sums - 1.0).abs().max()
    assert max_dev < 1e-3, f"비중 합이 1에서 {max_dev}만큼 벗어남"

    assert not backtest_df["returns"].isna().any(), "returns에 NaN 존재"
    assert not backtest_df["portfolio_values"].isna().any(), "portfolio_values에 NaN 존재"
    assert np.isfinite(backtest_df["portfolio_values"]).all(), "portfolio_values에 inf 존재"

    print(f"[sanity_check] OK - 비중 합 최대 편차: {max_dev:.2e}")
    print(f"[sanity_check] OK - 최종 포트폴리오 가치: {backtest_df['portfolio_values'].iloc[-1]:,.2f}")


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
        },
        sync_tensorboard=True,
    )

    print(f"=== [1/3] train split으로 PPO(Transformer, d_model={D_MODEL}, "
          f"heads={N_HEADS}, layers={N_LAYERS}, lookback={TIME_WINDOW}) 학습 ===")
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