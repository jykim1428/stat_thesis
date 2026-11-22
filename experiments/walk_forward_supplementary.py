"""동결 이후 보강 분석 (2026-11, 디펜스 대비 D2 결정).

동결된 모형(model.zip)과 백테스트(backtest_oos.csv)를 입력으로만 사용한다.
재학습하지 않고, 동결 디렉터리(results/walk_forward/, results/frozen_summaries/)에
쓰지 않는다. 결과는 results/walk_forward_supplementary/에 저장한다.

B-1 정적 학습비중 기준선
    각 시드의 OOS 시간평균 목표 비중을 고정하고 매 시간 그 비중으로
    리밸런싱하는 비-RL 전략을 같은 환경·같은 평가 모듈로 채점한다.
    PPO 성과가 이 기준선과 같다면, 성과는 상태 의존적 운용이 아니라
    고정 배분에서 나온 것이다.

B-2 입력 섭동 검사
    (a) 저장 모형이 저장된 백테스트의 목표 비중을 재현하는지(체크포인트 확인)
    (b) OOS 관측이 실제로 시간에 따라 변하는지
    (c) 극단적으로 다른 합성 관측에서 출력이 얼마나 바뀌는지
    (d) 같은 시드로 새로 초기화한 학습 전 정책의 출력과 얼마나 다른지
"""

from __future__ import annotations

import ast
import json
import os
import subprocess

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

import cryptoagent.policies.transformer_extractor  # noqa: F401  (model.zip 역직렬화에 필요)
from cryptoagent.envs.adapter import load_env_ready_df_by_date
from cryptoagent.envs.env_portfolio_optimization import PortfolioOptimizationEnv
from cryptoagent.envs.evaluate import evaluate
from cryptoagent.training.common import trim_warmup_rows

TIME_WINDOW = 50
FEATURES = ["close", "high", "low"]
INITIAL_AMOUNT = 100_000
COST_RATE = 0.001
SEEDS = [42, 43, 44, 45, 46]
CANDIDATES = {"mlp": "candidate03", "transformer": "candidate04"}
FOLDS = {
    "fold1_final": ("bull_2024", "2023-10-16", "2025-01-02"),
    "fold2_final": ("choppy_2025", "2025-01-02", "2026-01-01"),
}
OUT_DIR = "results/walk_forward_supplementary"


def make_env(start: str, end_exclusive: str) -> PortfolioOptimizationEnv:
    return PortfolioOptimizationEnv(
        df=load_env_ready_df_by_date(start, end_exclusive, warmup_hours=TIME_WINDOW),
        initial_amount=INITIAL_AMOUNT,
        time_column="date",
        tic_column="tic",
        features=FEATURES,
        time_window=TIME_WINDOW,
        cwd=OUT_DIR,
    )


def run_dir(policy: str, fold: str, seed: int) -> str:
    return f"results/walk_forward/{policy}/{fold}/{CANDIDATES[policy]}/seed{seed}"


def load_backtest(policy: str, fold: str, seed: int) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(run_dir(policy, fold, seed), "backtest_oos.csv"), index_col=0)
    df.index = pd.to_datetime(df.index)
    for col in ("weights", "target_weights"):
        df[col] = df[col].map(ast.literal_eval)  # CSV에는 문자열로 저장돼 있음
    return df


def stacked(col: pd.Series) -> np.ndarray:
    return np.vstack(col.map(lambda v: ast.literal_eval(v) if isinstance(v, str) else v).to_numpy())


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def l1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b).sum(axis=-1)


# --------------------------------------------------------------------------
# B-1 정적 학습비중 기준선
# --------------------------------------------------------------------------
def backtest_constant(env: PortfolioOptimizationEnv, weights: np.ndarray) -> pd.DataFrame:
    env.reset()
    done = False
    while not done:
        _, _, terminated, _ = env.step(weights)
        done = terminated
    out = pd.DataFrame(
        {
            "date": env._date_memory,
            "returns": env._portfolio_return_memory,
            "portfolio_values": env._asset_memory["final"],
            "weights": [w.tolist() for w in env._final_weights],
            "target_weights": [w.tolist() for w in env._actions_memory],
        }
    )
    out["date"] = pd.to_datetime(out["date"])
    return out.set_index("date")


def net_returns(df: pd.DataFrame) -> pd.Series:
    w = stacked(df["weights"])
    t = stacked(df["target_weights"])
    turnover = np.r_[0.0, np.abs(t[1:] - w[:-1]).sum(axis=1) / 2]
    return pd.Series(df["returns"].to_numpy() - turnover * COST_RATE, index=df.index)


def run_static_baseline() -> list[dict]:
    rows = []
    for fold, (label, start, end_ex) in FOLDS.items():
        for policy in CANDIDATES:
            for seed in SEEDS:
                ppo_df = load_backtest(policy, fold, seed)
                w_static = stacked(ppo_df["target_weights"]).mean(axis=0)
                w_static = w_static / w_static.sum()

                base_df = trim_warmup_rows(backtest_constant(make_env(start, end_ex), w_static), oos_start=start)
                m_ppo = evaluate(ppo_df, cost_rate=COST_RATE)
                m_base = evaluate(base_df, cost_rate=COST_RATE)

                r_ppo, r_base = net_returns(ppo_df), net_returns(base_df)
                common = r_ppo.index.intersection(r_base.index)
                diff = r_ppo.loc[common] - r_base.loc[common]
                rows.append(
                    {
                        "policy": policy,
                        "fold_label": label,
                        "seed": seed,
                        "n_common": int(len(common)),
                        "ppo_sharpe": m_ppo["sharpe"],
                        "static_sharpe": m_base["sharpe"],
                        "sharpe_diff": m_ppo["sharpe"] - m_base["sharpe"],
                        "ppo_cagr": m_ppo["cagr"],
                        "static_cagr": m_base["cagr"],
                        "ppo_avg_turnover": m_ppo["avg_turnover"],
                        "static_avg_turnover": m_base["avg_turnover"],
                        "return_corr": float(np.corrcoef(r_ppo.loc[common], r_base.loc[common])[0, 1]),
                        "max_abs_return_diff": float(diff.abs().max()),
                        "tracking_error_ann": float(diff.std(ddof=1) * np.sqrt(8760)),
                    }
                )
                print(f"[B-1] {policy:11s} {label:11s} seed{seed}  "
                      f"PPO {m_ppo['sharpe']:.6f}  static {m_base['sharpe']:.6f}  "
                      f"max|Δr| {rows[-1]['max_abs_return_diff']:.2e}")
    return rows


# --------------------------------------------------------------------------
# B-2 입력 섭동 검사
# --------------------------------------------------------------------------
def collect_observations(start: str, end_ex: str) -> tuple[np.ndarray, list]:
    """OOS 에피소드의 관측을 순서대로 모은다.

    return_last_action=False이므로 관측은 행동과 무관하다. 어떤 행동으로
    진행해도 같은 관측 열이 나온다.
    """
    env = make_env(start, end_ex)
    obs = env.reset()  # new_gym_api=False: 관측만 반환
    if isinstance(obs, tuple):
        obs = obs[0]
    observations, dates = [obs], [env._date_memory[-1]]
    hold = np.r_[1.0, np.zeros(8)]
    done = False
    while not done:
        obs, _, terminated, _ = env.step(hold)
        done = terminated
        if not done:
            observations.append(obs)
            dates.append(env._date_memory[-1])
    return np.asarray(observations, dtype=np.float32), pd.to_datetime(dates)


def policy_weights(model: PPO, obs: np.ndarray, chunk: int = 2048) -> np.ndarray:
    outs = []
    for i in range(0, len(obs), chunk):
        a, _ = model.predict(obs[i:i + chunk], deterministic=True)
        outs.append(a)
    return softmax(np.concatenate(outs).astype(np.float64))


def synthetic_observations(real: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    shape = real.shape[1:]
    n = 256
    base = real[rng.integers(0, len(real), n)]
    return {
        "flat_no_change": np.ones((n, *shape), dtype=np.float32),
        "rally_+5pct_every_hour": np.full((n, *shape), 1.05, dtype=np.float32),
        "crash_-5pct_every_hour": np.full((n, *shape), 0.95, dtype=np.float32),
        "iid_lognormal_sigma_5pct": np.exp(rng.normal(0, 0.05, (n, *shape))).astype(np.float32),
        "real_deviation_x10": np.clip(1 + 10 * (base - 1), 0.05, None).astype(np.float32),
        "real_assets_reversed": base[:, :, ::-1, :].copy(),
    }


def fresh_policy(model: PPO, env: PortfolioOptimizationEnv, seed: int) -> PPO:
    return PPO("MlpPolicy", env, policy_kwargs=model.policy_kwargs, seed=seed, device="cpu")


def run_perturbation() -> list[dict]:
    rng = np.random.default_rng(20261122)
    rows = []
    for fold, (label, start, end_ex) in FOLDS.items():
        real_obs, obs_dates = collect_observations(start, end_ex)
        # (b) 관측이 시간에 따라 변하는가
        obs_time_std = float(real_obs.std(axis=0).mean())
        obs_step_change = float(np.abs(np.diff(real_obs, axis=0)).mean())
        synth = synthetic_observations(real_obs, rng)
        env_for_init = make_env(start, end_ex)

        for policy in CANDIDATES:
            for seed in SEEDS:
                model = PPO.load(os.path.join(run_dir(policy, fold, seed), "model.zip"), device="cpu")
                w_real = policy_weights(model, real_obs)
                w_ref = w_real.mean(axis=0)

                # (a) 체크포인트가 저장된 백테스트를 재현하는가 (날짜 기준 정렬)
                stored = load_backtest(policy, fold, seed)
                stored_tw = pd.DataFrame(stacked(stored["target_weights"]), index=stored.index)
                pred_tw = pd.DataFrame(w_real, index=obs_dates)
                common = stored_tw.index.intersection(pred_tw.index)
                reproduce_max = float(np.abs(stored_tw.loc[common].to_numpy() - pred_tw.loc[common].to_numpy()).max())

                # (c) 합성 관측 반응
                synth_resp = {}
                for name, x in synth.items():
                    w = policy_weights(model, x)
                    synth_resp[name] = {
                        "max_l1_vs_real_mean": float(l1(w, w_ref).max()),
                        "cash_weight_mean": float(w[:, 0].mean()),
                    }

                # (d) 학습 전 정책과의 거리
                init = fresh_policy(model, env_for_init, seed)
                w_init = policy_weights(init, real_obs)

                row = {
                    "policy": policy,
                    "fold_label": label,
                    "seed": seed,
                    "n_obs": int(len(real_obs)),
                    "n_reproduce_compared": int(len(common)),
                    "obs_time_std_mean": obs_time_std,
                    "obs_mean_abs_step_change": obs_step_change,
                    "checkpoint_reproduce_max_abs_diff": reproduce_max,
                    "real_max_l1_vs_mean": float(l1(w_real, w_ref).max()),
                    "real_cash_mean": float(w_ref[0]),
                    "synthetic": synth_resp,
                    "init_cash_mean": float(w_init[:, 0].mean()),
                    "init_max_abs_dev_from_1_9": float(np.abs(w_init.mean(axis=0) - 1 / 9).max()),
                    "trained_vs_init_l1": float(l1(w_ref, w_init.mean(axis=0))),
                }
                rows.append(row)
                worst = max(v["max_l1_vs_real_mean"] for v in synth_resp.values())
                print(f"[B-2] {policy:11s} {label:11s} seed{seed}  재현 {reproduce_max:.1e}  "
                      f"실제관측 L1 {row['real_max_l1_vs_mean']:.1e}  합성 최대 L1 {worst:.1e}  "
                      f"학습전 현금 {row['init_cash_mean']:.4f}  학습후 {row['real_cash_mean']:.4f}")
    return rows


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    meta = {"script": "experiments/walk_forward_supplementary.py", "git_head": git_head(),
            "inputs": "results/walk_forward/*/fold*_final/*/seed*/{model.zip,backtest_oos.csv} (읽기 전용)",
            "cost_rate": COST_RATE}

    static = run_static_baseline()
    with open(os.path.join(OUT_DIR, "static_baseline.json"), "w") as f:
        json.dump({"meta": meta, "rows": static}, f, indent=2, ensure_ascii=False)

    perturb = run_perturbation()
    with open(os.path.join(OUT_DIR, "perturbation.json"), "w") as f:
        json.dump({"meta": meta, "rows": perturb}, f, indent=2, ensure_ascii=False)

    print(f"\n저장: {OUT_DIR}/static_baseline.json, {OUT_DIR}/perturbation.json")


if __name__ == "__main__":
    main()
