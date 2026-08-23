"""LSTM L-raw vs L-std validation ablation.

Run from the CryptoAgent repository root (not from this output directory):
    python experiments/train_ppo_lstm_ablation.py --condition raw --seed 42
    python experiments/train_ppo_lstm_ablation.py --condition std --seed 42

Use the same seed set for both conditions. The default evaluation split is
``val`` deliberately; do not use ``test`` while choosing a condition.

train_ppo_transformer_ablation.py와 동일한 구조/설정(learning_rate,
n_epochs, target_kl, seed 세트)을 그대로 따름 - LSTM 고유 부분은
features_extractor_class와 hidden_size/num_layers뿐.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import shimmy
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback

from cryptoagent.envs.adapter import load_env_ready_df, patch_seed_method
from cryptoagent.envs.env_portfolio_optimization import PortfolioOptimizationEnv
from cryptoagent.envs.normalize_wrapper import TrainStandardizeWrapper, compute_train_stats
from cryptoagent.policies.lstm_extractor import LSTMFeaturesExtractor


TIME_WINDOW = 50
FEATURES = ["close", "high", "low"]
INITIAL_AMOUNT = 100_000
HIDDEN_SIZE, NUM_LAYERS, DROPOUT = 32, 2, 0.1
TOTAL_TIMESTEPS = 50_000
LEARNING_RATE, N_EPOCHS, TARGET_KL = 1e-4, 5, 0.02
RESULTS_ROOT = Path("results/ppo_lstm_ablation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("raw", "std"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    # test is available only for a pre-registered final run, never for tuning.
    parser.add_argument("--eval-split", choices=("val", "test"), default="val")
    return parser.parse_args()


def make_env(split: str) -> PortfolioOptimizationEnv:
    env = PortfolioOptimizationEnv(
        df=load_env_ready_df(split=split),
        initial_amount=INITIAL_AMOUNT,
        time_column="date",
        tic_column="tic",
        features=FEATURES,
        time_window=TIME_WINDOW,
    )
    patch_seed_method(env)
    return env


def maybe_standardize(env, stats: dict | None):
    return TrainStandardizeWrapper(env, stats=stats) if stats is not None else env


def train(train_env, seed: int, tensorboard_log: str, callback) -> PPO:
    gym_env = shimmy.GymV21CompatibilityV0(env=train_env)
    vec_env = DummyVecEnv([lambda: gym_env])
    return PPO(
        "MlpPolicy",
        vec_env,
        seed=seed,
        verbose=1,
        tensorboard_log=tensorboard_log,
        policy_kwargs={
            "features_extractor_class": LSTMFeaturesExtractor,
            "features_extractor_kwargs": {
                "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS, "dropout": DROPOUT,
            },
        },
        # Identical in L-raw and L-std: this is essential for the ablation.
        learning_rate=LEARNING_RATE,
        n_epochs=N_EPOCHS,
        target_kl=TARGET_KL,
    ).learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)


def backtest(model: PPO, eval_env) -> pd.DataFrame:
    gym_env = shimmy.GymV21CompatibilityV0(env=eval_env)
    obs, _ = gym_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = gym_env.step(action)
        done = terminated or truncated

    base_env = eval_env.unwrapped
    result = pd.DataFrame({
        "date": base_env._date_memory,
        "returns": base_env._portfolio_return_memory,
        "portfolio_values": base_env._asset_memory["final"],
        "weights": [weights.tolist() for weights in base_env._final_weights],
    })
    return result.assign(date=pd.to_datetime(result["date"])).set_index("date")


def summarize(backtest_df: pd.DataFrame) -> dict[str, float]:
    returns = backtest_df["returns"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    values = backtest_df["portfolio_values"].astype(float)
    if returns.std(ddof=0) == 0:
        sharpe = float("nan")
    else:
        # 1-hour crypto data: 24 * 365 observations per year.
        sharpe = float(np.sqrt(24 * 365) * returns.mean() / returns.std(ddof=0))
    drawdown = values / values.cummax() - 1.0
    weights = np.vstack(backtest_df["weights"].to_numpy())
    turnover = float(np.abs(np.diff(weights, axis=0)).sum(axis=1).mean() / 2) if len(weights) > 1 else 0.0
    return {
        "final_value": float(values.iloc[-1]),
        "cumulative_return": float(values.iloc[-1] / values.iloc[0] - 1.0),
        "sharpe_hourly_annualized": sharpe,
        "max_drawdown": float(drawdown.min()),
        "mean_turnover": turnover,
    }


def save_stats(stats: dict, path: Path) -> None:
    np.savez(path, mean=stats["mean"], std=stats["std"],
             tic_order=np.asarray(stats["tic_order"]), features=np.asarray(stats["features"]))


def sanity_check(backtest_df: pd.DataFrame) -> None:
    weight_sums = backtest_df["weights"].apply(sum)
    max_dev = (weight_sums - 1.0).abs().max()
    assert max_dev < 1e-3, f"비중 합이 1에서 {max_dev}만큼 벗어남"
    assert not backtest_df["returns"].isna().any(), "returns에 NaN 존재"
    assert not backtest_df["portfolio_values"].isna().any(), "portfolio_values에 NaN 존재"
    assert np.isfinite(backtest_df["portfolio_values"]).all(), "portfolio_values에 inf 존재"
    print(f"[sanity_check] OK - 비중 합 최대 편차: {max_dev:.2e}")


def main() -> None:
    args = parse_args()
    use_std = args.condition == "std"

    results_dir = RESULTS_ROOT / args.condition / f"seed_{args.seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        entity="choieuna0711-student",
        project="cryptoagent-ppo",
        name=f"lstm_{args.condition}_seed{args.seed}_{args.eval_split}",
        config={
            "total_timesteps": TOTAL_TIMESTEPS,
            "seed": args.seed,
            "time_window": TIME_WINDOW,
            "features": FEATURES,
            "initial_amount": INITIAL_AMOUNT,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "standardization": use_std,
            "learning_rate": LEARNING_RATE,
            "n_epochs": N_EPOCHS,
            "target_kl": TARGET_KL,
            "eval_split": args.eval_split,
        },
        sync_tensorboard=True,
    )

    try:
        raw_train_env = make_env("train")
        train_stats = compute_train_stats(raw_train_env) if use_std else None
        train_env = maybe_standardize(raw_train_env, train_stats)

        model = train(
            train_env,
            seed=args.seed,
            tensorboard_log=f"runs/{run.id}",
            callback=WandbCallback(
                gradient_save_freq=100,
                model_save_path=str(results_dir / "wandb_models" / run.id),
            ),
        )
        model.save(str(results_dir / "model.zip"))

        if train_stats is not None:
            save_stats(train_stats, results_dir / "train_stats.npz")

        eval_env = maybe_standardize(make_env(args.eval_split), train_stats)
        backtest_df = backtest(model, eval_env)
        backtest_df.to_csv(results_dir / f"{args.eval_split}_backtest.csv")

        sanity_check(backtest_df)
        metrics = summarize(backtest_df)
        with open(results_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(json.dumps(metrics, indent=2))
        print(f"saved: {results_dir}")
        wandb.log(metrics)
    finally:
        run.finish()


if __name__ == "__main__":
    main()