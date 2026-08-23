"""Transformer T-raw vs T-std validation ablation.

Run from the CryptoAgent repository root (not from this output directory):
    python experiments/train_ppo_transformer_ablation.py --condition raw --seed 42
    python experiments/train_ppo_transformer_ablation.py --condition std --seed 42

Use the same seed set for both conditions.  The default evaluation split is
``val`` deliberately; do not use ``test`` while choosing a condition.
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
from cryptoagent.policies.transformer_extractor import TransformerFeaturesExtractor


TIME_WINDOW = 50
FEATURES = ["close", "high", "low"]
INITIAL_AMOUNT = 100_000
D_MODEL, N_HEADS, N_LAYERS = 32, 4, 2
TOTAL_TIMESTEPS = 50_000
LEARNING_RATE, N_EPOCHS, TARGET_KL = 1e-4, 5, 0.02
RESULTS_ROOT = Path("results/ppo_transformer_ablation")


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
            "features_extractor_class": TransformerFeaturesExtractor,
            "features_extractor_kwargs": {
                "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS,
            },
        },
        # Identical in T-raw and T-std: this is essential for the ablation.
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


def main() -> None:
    args = parse_args()
    run_dir = RESULTS_ROOT / args.condition / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"transformer_{args.condition}_seed{args.seed}_{args.eval_split}"
    config = {
        "condition": args.condition, "seed": args.seed, "eval_split": args.eval_split,
        "time_window": TIME_WINDOW, "features": FEATURES, "d_model": D_MODEL,
        "n_heads": N_HEADS, "n_layers": N_LAYERS, "total_timesteps": TOTAL_TIMESTEPS,
        "learning_rate": LEARNING_RATE, "n_epochs": N_EPOCHS, "target_kl": TARGET_KL,
        "standardization": args.condition == "std",
        "standardization_method": "train_only_per_asset_feature" if args.condition == "std" else None,
        "standardization_clip": [-5.0, 5.0] if args.condition == "std" else None,
    }
    run = wandb.init(entity="choieuna0711-student", project="cryptoagent-ppo",
                     name=run_name, config=config, sync_tensorboard=True)
    try:
        raw_train_env = make_env("train")
        stats = compute_train_stats(raw_train_env) if args.condition == "std" else None
        train_env = maybe_standardize(raw_train_env, stats)
        model = train(
            train_env, args.seed, tensorboard_log=f"runs/{run.id}",
            callback=WandbCallback(gradient_save_freq=100, model_save_path=str(run_dir / "wandb_models")),
        )

        model_path = run_dir / "model.zip"
        model.save(str(model_path))
        stats_path = run_dir / "train_stats.npz"
        if stats is not None:
            save_stats(stats, stats_path)

        raw_eval_env = make_env(args.eval_split)
        eval_env = maybe_standardize(raw_eval_env, stats)
        backtest_df = backtest(model, eval_env)
        if backtest_df["returns"].isna().any() or not np.isfinite(backtest_df["portfolio_values"]).all():
            raise ValueError("백테스트 결과에 NaN/inf가 있습니다.")
        backtest_path = run_dir / f"backtest_{args.eval_split}.csv"
        backtest_df.to_csv(backtest_path)
        metrics = summarize(backtest_df)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        wandb.log(metrics)

        artifact = wandb.Artifact(name=f"{run_name}-{run.id}", type="transformer-experiment",
                                  metadata=config | metrics)
        artifact.add_file(str(model_path))
        artifact.add_file(str(backtest_path))
        artifact.add_file(str(run_dir / "metrics.json"))
        if stats is not None:
            artifact.add_file(str(stats_path))
        run.log_artifact(artifact)
        print(json.dumps(metrics, indent=2))
        print(f"saved: {run_dir}")
    finally:
        run.finish()


if __name__ == "__main__":
    main()
