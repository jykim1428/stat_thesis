"""Transformer ablation: raw vs zscore vs zscore_clip (val 기준)

Run from the repository root:
    python experiments/train_ppo_transformer_ablation.py --condition raw --seed 42
    python experiments/train_ppo_transformer_ablation.py --condition zscore --seed 42
    python experiments/train_ppo_transformer_ablation.py --condition zscore_clip --seed 42

목적: train-only standardization(z-score, 자산별 표준화)을 최종
실험(본실험)에 쓸지 말지 결정하기 위한 ablation. 정규화 방법론 자체를
최적화하는 게 목적이 아니라 "켤지 끌지" 이진 결정이 목적이므로,
val 3 seed(42/202/777) 결과만으로 판단하고 test는 이 스크립트에서
다루지 않음 (표준화 결정이 끝난 뒤 본실험에서만 test 평가).

세 조건 정의
------------
raw:         정규화 없음 (env의 by_previous_time만)
zscore:      train stats로 z-score만 적용, clip 없음
zscore_clip: z-score + clip(-5, 5) - 이전 실험에서 쓰던 "std" 조건과 동일

리뷰 반영 사항 
----------------------------------------------------------
1. raw/std 2조건 -> raw/zscore/zscore_clip 3조건 (z-score 효과와
   clip 효과를 분리해서 봄)
2. 세 모델(mlp/transformer/lstm) PPO 하이퍼파라미터를 SB3 기본값으로
   통일 (learning_rate 등 명시하지 않음)
3. 학습은 condition/seed당 1번만 하고 그 model로 val을 평가 (이전엔
   val/test 실행이 서로 다른 학습이 되는 문제가 있었음)
4. train()에도 Monitor 적용 (val과 동일 경로) - rollout/ep_rew_mean
   기록 확인됨
5. normalize_with_stats() 공용 함수로 wrapper/verify_on_split 계산
   통일 (clip 누락으로 실제 정책 입력과 다른 값을 보고하던 문제 해결)
6. 결과 재현에 필요한 metadata(stats, config, git commit, 버전)를
   결과 디렉터리에 저장
7. 완료된 실행 재실행 시 조용히 덮어쓰지 않도록 방어 (--overwrite
   필요)
8. sanity_check을 assert 대신 명시적 예외로 변경 (python -O에서도
   유지됨)

범위에서 제외한 것 (팀 논의 결과)
------------------------------------
- 학습/평가 스크립트 물리적 분리, artifact hash 검증: 이 ablation의
  목적(표준화 on/off 결정)에 비해 과하다고 판단해 채택하지 않음.
  본실험(표준화 결정 이후 최종 모델)에서는 별도로 test를 신중하게
  평가할 것.
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
from cryptoagent.envs.evaluate import compute_turnover_from_weights
from cryptoagent.policies.transformer_extractor import TransformerFeaturesExtractor

# ── 아키텍처 고유 설정 (MLP/LSTM 복사 시 이 블록만 바꾸면 됨) ──────
ARCHITECTURE = "transformer"
FEATURES_EXTRACTOR_CLASS = TransformerFeaturesExtractor
FEATURES_EXTRACTOR_KWARGS = {"d_model": 32, "n_heads": 4, "n_layers": 2}
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
        # SB3 기본값 그대로 사용 (learning_rate 등 명시 안 함) - 세 모델 통일
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
        "target_weights": [w.tolist() for w in base_env._actions_memory],
    })
    return result.assign(date=pd.to_datetime(result["date"])).set_index("date")


def summarize(backtest_df: pd.DataFrame) -> dict[str, float]:
    """turnover는 evaluate.py의 compute_turnover_from_weights를 재사용한다.

    이전 버전은 |weights[t]-weights[t-1]|로 turnover를 근사했는데, weights는
    가격 변동을 반영한 사후 비중이라 이 방식은 가격 변동분까지 거래량으로
    잘못 포함해 실제보다 과대계상한다(실측 약 1.2~1.45배). target_weights(그
    스텝의 목표 비중)와 직전 weights의 차이로 계산해야 진짜 거래량이 나온다.
    """
    returns = backtest_df["returns"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    values = backtest_df["portfolio_values"].astype(float)
    sharpe = (float(np.sqrt(24 * 365) * returns.mean() / returns.std(ddof=0))
              if returns.std(ddof=0) != 0 else float("nan"))
    drawdown = values / values.cummax() - 1.0
    turnover_series = compute_turnover_from_weights(
        backtest_df["weights"], backtest_df["target_weights"]
    )
    turnover = float(turnover_series.mean()) if len(turnover_series) > 1 else 0.0
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

    # weights/target_weights 벡터 안에 NaN이 섞이면 sum()이 NaN이 되고,
    # pandas.Series.max()는 기본적으로 skipna=True라 그 행이 통계에서
    # 조용히 빠져 max_dev가 정상값으로 나온다 - 반드시 벡터 원소 단위로
    # 먼저 finite 여부를 확인해야 이 케이스를 놓치지 않는다.
    for col in ("weights", "target_weights"):
        if not backtest_df[col].apply(lambda w: np.isfinite(w).all()).all():
            raise ValueError(f"[{split}] {col}에 NaN 또는 inf 존재")

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
    required = ["model.zip", "experiment_config.json", "manifest.json", "val_metrics.json", "val_backtest.csv"]
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