"""MLP ablation: raw vs zscore vs zscore_clip (val 기준)

Run from the repository root:
    python experiments/train_ppo_mlp_ablation.py --condition raw --seed 42
    python experiments/train_ppo_mlp_ablation.py --condition zscore --seed 42
    python experiments/train_ppo_mlp_ablation.py --condition zscore_clip --seed 42

목적: train-only standardization(z-score, 자산별 표준화)을 최종
실험(본실험)에 쓸지 말지 결정하기 위한 ablation. val 3 seed
(42/202/777) 결과만으로 판단하고 test는 이 스크립트에서 다루지 않음.

train_ppo_transformer_ablation.py와 완전히 동일한 구조 - ARCHITECTURE
및 FEATURES_EXTRACTOR_CLASS만 다름 (MLP는 커스텀 extractor 없이 SB3
기본 FlattenExtractor 사용).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import shimmy
import stable_baselines3
import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback

from cryptoagent.envs.adapter import load_env_ready_df, patch_seed_method
from cryptoagent.envs.env_portfolio_optimization import PortfolioOptimizationEnv
from cryptoagent.envs.normalize_wrapper import TrainStandardizeWrapper, compute_train_stats

# ── 아키텍처 고유 설정 ──────────────────────────────
ARCHITECTURE = "mlp"
FEATURES_EXTRACTOR_CLASS = None  # MLP는 커스텀 extractor 없음 (SB3 기본 FlattenExtractor)
FEATURES_EXTRACTOR_KWARGS = {}
# ──────────────────────────────────────────────────────────────

TIME_WINDOW = 50
FEATURES = ["close", "high", "low"]
INITIAL_AMOUNT = 100_000
TOTAL_TIMESTEPS = 50_000
RESULTS_ROOT = Path(f"results/ppo_{ARCHITECTURE}_ablation")

CONDITIONS = {
    "raw": {"use_zscore": False, "clip": None},
    "zscore": {"use_zscore": True, "clip": None},
    "zscore_clip": {"use_zscore": True, "clip": (-5.0, 5.0)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=list(CONDITIONS.keys()), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
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


def maybe_standardize(env, stats, clip):
    return TrainStandardizeWrapper(env, stats=stats, clip=clip) if stats is not None else env


def train(train_env, seed: int, tensorboard_log: str, callback) -> PPO:
    gym_env = shimmy.GymV21CompatibilityV0(env=train_env)
    vec_env = DummyVecEnv([lambda: Monitor(gym_env)])

    policy_kwargs = {}
    if FEATURES_EXTRACTOR_CLASS is not None:
        policy_kwargs = {
            "features_extractor_class": FEATURES_EXTRACTOR_CLASS,
            "features_extractor_kwargs": FEATURES_EXTRACTOR_KWARGS,
        }

    return PPO(
        "MlpPolicy",
        vec_env,
        seed=seed,
        verbose=1,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
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
        "weights": [w.tolist() for w in base_env._final_weights],
    })
    return result.assign(date=pd.to_datetime(result["date"])).set_index("date")


def summarize(backtest_df: pd.DataFrame) -> dict[str, float]:
    returns = backtest_df["returns"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    values = backtest_df["portfolio_values"].astype(float)
    sharpe = (float(np.sqrt(24 * 365) * returns.mean() / returns.std(ddof=0))
              if returns.std(ddof=0) != 0 else float("nan"))
    drawdown = values / values.cummax() - 1.0
    weights = np.vstack(backtest_df["weights"].to_numpy())
    turnover = (float(np.abs(np.diff(weights, axis=0)).sum(axis=1).mean() / 2)
                if len(weights) > 1 else 0.0)
    return {
        "final_value": float(values.iloc[-1]),
        "cumulative_return": float(values.iloc[-1] / values.iloc[0] - 1.0),
        "sharpe_hourly_annualized": sharpe,
        "max_drawdown": float(drawdown.min()),
        "mean_turnover": turnover,
    }


def sanity_check(backtest_df: pd.DataFrame, split: str) -> None:
    if len(backtest_df) == 0:
        raise ValueError(f"[{split}] backtest 결과가 비어있음")
    if not np.isfinite(backtest_df["returns"]).all():
        raise ValueError(f"[{split}] returns에 NaN/inf 존재")
    if backtest_df["portfolio_values"].isna().any():
        raise ValueError(f"[{split}] portfolio_values에 NaN 존재")
    if not np.isfinite(backtest_df["portfolio_values"]).all():
        raise ValueError(f"[{split}] portfolio_values에 inf 존재")
    weight_sums = backtest_df["weights"].apply(sum)
    max_dev = (weight_sums - 1.0).abs().max()
    if max_dev >= 1e-3:
        raise ValueError(f"[{split}] 비중 합이 1에서 {max_dev}만큼 벗어남")
    print(f"[sanity_check:{split}] OK - 비중 합 최대 편차: {max_dev:.2e}")


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def save_artifacts(results_dir: Path, train_stats, cond: dict, args) -> None:
    if train_stats is not None:
        np.savez(
            results_dir / "normalization_stats.npz",
            mean=train_stats["mean"],
            std=train_stats["std"],
            tic_order=np.array(train_stats["tic_order"]),
            features=np.array(train_stats["features"]),
        )

    experiment_config = {
        "architecture": ARCHITECTURE,
        "condition": args.condition,
        "seed": args.seed,
        "clip": cond["clip"],
        "features": FEATURES,
        "features_extractor_kwargs": FEATURES_EXTRACTOR_KWARGS,
        "time_window": TIME_WINDOW,
        "initial_amount": INITIAL_AMOUNT,
        "total_timesteps": TOTAL_TIMESTEPS,
    }
    with open(results_dir / "experiment_config.json", "w") as f:
        json.dump(experiment_config, f, indent=2)

    manifest = {
        "git_commit": get_git_commit(),
        "sb3_version": stable_baselines3.__version__,
        "torch_version": torch.__version__,
    }
    with open(results_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def has_complete_prior_run(results_dir: Path, needs_stats: bool) -> bool:
    required = ["model.zip", "experiment_config.json", "manifest.json", "val_metrics.json"]
    if needs_stats:
        required.append("normalization_stats.npz")
    return all((results_dir / name).exists() for name in required)


def main() -> None:
    args = parse_args()
    cond = CONDITIONS[args.condition]

    results_dir = RESULTS_ROOT / args.condition / f"seed_{args.seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    if has_complete_prior_run(results_dir, needs_stats=cond["use_zscore"]) and not args.overwrite:
        raise FileExistsError(
            f"{results_dir}에 이미 완료된 실행 결과가 있습니다. "
            f"덮어쓰려면 --overwrite를 붙이세요."
        )

    run = wandb.init(
        entity="choieuna0711-student",
        project="cryptoagent-ppo",
        name=f"{ARCHITECTURE}_{args.condition}_seed{args.seed}",
        group=f"{ARCHITECTURE}_{args.condition}",
        config={
            "architecture": ARCHITECTURE,
            "total_timesteps": TOTAL_TIMESTEPS,
            "seed": args.seed,
            "condition": args.condition,
            "clip": cond["clip"],
            "features_extractor_kwargs": FEATURES_EXTRACTOR_KWARGS,
            "sb3_version": stable_baselines3.__version__,
            "torch_version": torch.__version__,
            "git_commit": get_git_commit(),
        },
        sync_tensorboard=True,
    )

    try:
        raw_train_env = make_env("train")
        train_stats = compute_train_stats(raw_train_env) if cond["use_zscore"] else None
        train_env = maybe_standardize(raw_train_env, train_stats, cond["clip"])

        model = train(
            train_env,
            seed=args.seed,
            tensorboard_log=f"runs/{run.id}",
            callback=WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"{results_dir}/wandb_models/{run.id}",
            ),
        )
        model.save(str(results_dir / "model.zip"))
        save_artifacts(results_dir, train_stats, cond, args)

        eval_env = maybe_standardize(make_env("val"), train_stats, cond["clip"])
        backtest_df = backtest(model, eval_env)
        backtest_df.to_csv(results_dir / "val_backtest.csv")
        sanity_check(backtest_df, "val")

        metrics = summarize(backtest_df)
        with open(results_dir / "val_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("[val]", json.dumps(metrics, indent=2))

        wandb.run.summary.update({f"val/{k}": v for k, v in metrics.items()})
    finally:
        run.finish()


if __name__ == "__main__":
    main()