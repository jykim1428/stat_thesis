"""
9월 4주차: Walk-forward OOS 구간에서 전통 벤치마크 재실행 (CryptoAgent)
리포 루트에서 실행: python experiments/walk_forward_benchmarks.py

배경
----
experiments/benchmark_{buy_and_hold_equal_weight,markowitz,risk_parity}.py는
전부 split="test"(2023-10-18~2025-12-31, 19,320행) 기준으로 실행되었다.
반면 walk_forward_orchestrator.py의 PPO OOS 평가는 bull_2024
(2023-10-16~2025-01-02)와 choppy_2025(2025-01-02~2026-01-01)로 날짜 경계가
다르다 - 국면 경계(docs/walk_forward_design.md)와 고정 split 라벨의 경계가
다르기 때문. 그대로 비교하면 기간이 안 맞아 공정하지 않으므로, 동일한 벤치마크
로직을 make_env_by_date()로 bull_2024/choppy_2025 구간에 재실행한다.

기존 benchmark_*.py 스크립트는 건드리지 않는다 (test split 기준 결과는 그
결과대로 유지). 이 스크립트는 별도 산출물(results/walk_forward_benchmarks/)을
만든다.

워밍업 처리
-----------
Buy&Hold/Equal-Weight는 매 스텝 고정 action을 반복하므로 관측(과거 시점)을
전혀 쓰지 않지만, walk_forward_orchestrator.py의 PPO OOS와 동일하게
enable_warmup=True + trim_warmup_rows()를 적용해 비교 조건을 통일한다
(docs/walk_forward_design.md "모든 정책망에 동일한 조건 적용" 원칙을
벤치마크에도 동일하게 확장 - 어차피 고정 action에는 결과가 바뀌지 않는다).

Markowitz/Risk Parity는 all_returns.loc[:current_date].tail(720h)로 이미
DB 전체 기간에서 과거 수익률을 직접 참조하므로(env가 반환하는 관측이 아님)
워밍업 여부와 무관하게 정확하다 - 단, PPO/Buy&Hold와 조건을 통일하기 위해
env 자체는 동일하게 enable_warmup=True로 생성한다.

거래비용은 evaluate.py 단계에서 PPO와 동일하게 0.1%로 통일 적용한다 (팀
합의 사항, benchmark_buy_and_hold_equal_weight.py와 동일 원칙).
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models
from scipy.optimize import minimize

from cryptoagent.envs.adapter import load_env_ready_df_by_date
from cryptoagent.envs.env_portfolio_optimization import PortfolioOptimizationEnv
from cryptoagent.training.common import trim_warmup_rows

TIME_WINDOW = 50
FEATURES = ["close", "high", "low"]
INITIAL_AMOUNT = 100_000
CASH_IDX = 0
BTC_IDX = 4
N_ASSETS = 8
TIC_ORDER = ["ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
MV_RP_WINDOW_HOURS = 720
MAX_WEIGHT_PER_ASSET = 0.3

RESULTS_DIR = "results/walk_forward_benchmarks"

# walk_forward_orchestrator.py와 동일 (docs/walk_forward_design.md)
FOLDS = {
    "fold1_oos": ("2023-10-16", "2025-01-02"),
    "fold2_oos": ("2025-01-02", "2026-01-01"),
}


def make_env(start: str, end_exclusive: str) -> PortfolioOptimizationEnv:
    return PortfolioOptimizationEnv(
        df=load_env_ready_df_by_date(start, end_exclusive, warmup_hours=TIME_WINDOW),
        initial_amount=INITIAL_AMOUNT,
        time_column="date",
        tic_column="tic",
        features=FEATURES,
        time_window=TIME_WINDOW,
    )


def load_wide_close_prices() -> pd.DataFrame:
    """Markowitz/Risk Parity가 rolling window로 참조할 전체 기간 종가.

    DB 전체를 읽어야 fold 시작 시점 직전 720시간(30일)도 과거로 참조 가능
    (benchmark_markowitz.py/benchmark_risk_parity.py와 동일 원칙).
    """
    df = load_env_ready_df_by_date("2021-01-01", "2026-01-01")
    wide = df.pivot(index="date", columns="tic", values="close")
    return wide[TIC_ORDER].sort_index()


def backtest_fixed_action(env: PortfolioOptimizationEnv, action: np.ndarray) -> pd.DataFrame:
    env.reset()
    done = False
    while not done:
        _, _, terminated, _ = env.step(action)
        done = terminated

    result = pd.DataFrame(
        {
            "date": env._date_memory,
            "returns": env._portfolio_return_memory,
            "portfolio_values": env._asset_memory["final"],
            "weights": [w.tolist() for w in env._final_weights],
            "target_weights": [w.tolist() for w in env._actions_memory],
        }
    )
    result["date"] = pd.to_datetime(result["date"])
    return result.set_index("date")


def buy_and_hold_btc_action() -> np.ndarray:
    action = np.zeros(1 + N_ASSETS, dtype=np.float32)
    action[BTC_IDX] = 1.0
    return action


def equal_weight_action() -> np.ndarray:
    action = np.zeros(1 + N_ASSETS, dtype=np.float32)
    action[1:1 + N_ASSETS] = 1.0 / N_ASSETS
    return action


def min_variance_action(returns_window: pd.DataFrame, fallback_counter: dict) -> np.ndarray:
    action = np.zeros(1 + N_ASSETS, dtype=np.float32)
    try:
        S = risk_models.CovarianceShrinkage(
            returns_window, returns_data=True, frequency=24 * 365
        ).ledoit_wolf()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ef = EfficientFrontier(None, S, weight_bounds=(0, MAX_WEIGHT_PER_ASSET))
            ef.min_volatility()

        cleaned = ef.clean_weights()
        weights = np.array([cleaned[t] for t in TIC_ORDER], dtype=np.float64)

        if (not np.all(np.isfinite(weights))
                or weights.sum() <= 0
                or np.any(weights < -1e-4)
                or np.any(weights > 1.0)):
            raise ValueError(f"비정상 weights: {weights}")

        weights = np.clip(weights, 0, MAX_WEIGHT_PER_ASSET)
        for _ in range(10):
            if weights.sum() <= 0:
                break
            weights = weights / weights.sum()
            weights = np.clip(weights, 0, MAX_WEIGHT_PER_ASSET)
        weights = weights / weights.sum()

    except Exception as e:
        fallback_counter["count"] += 1
        if fallback_counter["count"] <= 5:
            print(f"[Markowitz 폴백 #{fallback_counter['count']}] {e}")
        weights = np.full(N_ASSETS, 1.0 / N_ASSETS, dtype=np.float64)

    action[1:1 + N_ASSETS] = weights.astype(np.float32)
    return action


def _risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    port_var = w @ cov @ w
    port_vol = np.sqrt(max(port_var, 1e-12))
    marginal = cov @ w / port_vol
    return w * marginal


def _erc_objective(w: np.ndarray, cov: np.ndarray) -> float:
    rc = _risk_contributions(w, cov)
    target = rc.mean()
    return np.sum((rc - target) ** 2)


def risk_parity_action(returns_window: pd.DataFrame, fallback_counter: dict) -> np.ndarray:
    action = np.zeros(1 + N_ASSETS, dtype=np.float32)
    try:
        S = risk_models.CovarianceShrinkage(
            returns_window, returns_data=True, frequency=24 * 365
        ).ledoit_wolf()
        cov = S.values

        n = N_ASSETS
        w0 = np.full(n, 1.0 / n)
        bounds = [(1e-6, MAX_WEIGHT_PER_ASSET) for _ in range(n)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                _erc_objective, w0, args=(cov,),
                method="SLSQP", bounds=bounds, constraints=constraints,
                options={"maxiter": 200, "ftol": 1e-10},
            )

        if not result.success:
            raise RuntimeError(f"ERC 최적화 실패: {result.message}")

        weights = np.array(result.x, dtype=np.float64)

        if (not np.all(np.isfinite(weights))
                or weights.sum() <= 0
                or np.any(weights < -1e-4)
                or np.any(weights > 1.0)):
            raise ValueError(f"비정상 weights: {weights}")

        weights = np.clip(weights, 0, MAX_WEIGHT_PER_ASSET)
        for _ in range(10):
            if weights.sum() <= 0:
                break
            weights = weights / weights.sum()
            weights = np.clip(weights, 0, MAX_WEIGHT_PER_ASSET)
        weights = weights / weights.sum()

    except Exception as e:
        fallback_counter["count"] += 1
        if fallback_counter["count"] <= 5:
            print(f"[Risk Parity 폴백 #{fallback_counter['count']}] {e}")
        weights = np.full(N_ASSETS, 1.0 / N_ASSETS, dtype=np.float64)

    action[1:1 + N_ASSETS] = weights.astype(np.float32)
    return action


def backtest_rolling_window(
    env: PortfolioOptimizationEnv,
    wide_prices: pd.DataFrame,
    window_hours: int,
    action_fn,
) -> pd.DataFrame:
    all_returns = wide_prices.pct_change().dropna()
    fallback_counter = {"count": 0}

    env.reset()
    done = False
    step_count = 0

    while not done:
        current_date = env._date_memory[-1]
        window = all_returns.loc[:current_date].tail(window_hours)

        if len(window) < window_hours // 2:
            action = np.zeros(1 + N_ASSETS, dtype=np.float32)
            action[1:1 + N_ASSETS] = 1.0 / N_ASSETS
        else:
            action = action_fn(window, fallback_counter)

        _, _, terminated, _ = env.step(action)
        done = terminated
        step_count += 1
        if step_count % 5000 == 0:
            print(f"  ...{step_count} 스텝 진행 중")

    result = pd.DataFrame(
        {
            "date": env._date_memory,
            "returns": env._portfolio_return_memory,
            "portfolio_values": env._asset_memory["final"],
            "weights": [w.tolist() for w in env._final_weights],
            "target_weights": [w.tolist() for w in env._actions_memory],
        }
    )
    result["date"] = pd.to_datetime(result["date"])
    print(f"  총 폴백 횟수: {fallback_counter['count']} / {len(result)}")
    return result.set_index("date")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    wide_prices = load_wide_close_prices()

    for fold_name, (start, end_exclusive) in FOLDS.items():
        fold_dir = os.path.join(RESULTS_DIR, fold_name)
        os.makedirs(fold_dir, exist_ok=True)
        print(f"\n{'=' * 20} {fold_name} ({start} ~ {end_exclusive}) {'=' * 20}")

        print("\n=== Buy & Hold BTC ===")
        env = make_env(start, end_exclusive)
        bnh_df = backtest_fixed_action(env, buy_and_hold_btc_action())
        bnh_df = trim_warmup_rows(bnh_df, oos_start=start)
        bnh_df.to_csv(f"{fold_dir}/buy_and_hold_btc.csv")
        print(f"최종 가치: {bnh_df['portfolio_values'].iloc[-1]:,.2f}")

        print("\n=== Equal-Weight ===")
        env = make_env(start, end_exclusive)
        ew_df = backtest_fixed_action(env, equal_weight_action())
        ew_df = trim_warmup_rows(ew_df, oos_start=start)
        ew_df.to_csv(f"{fold_dir}/equal_weight.csv")
        print(f"최종 가치: {ew_df['portfolio_values'].iloc[-1]:,.2f}")

        print("\n=== Minimum Variance Portfolio (Markowitz) ===")
        env = make_env(start, end_exclusive)
        mv_df = backtest_rolling_window(env, wide_prices, MV_RP_WINDOW_HOURS, min_variance_action)
        mv_df = trim_warmup_rows(mv_df, oos_start=start)
        mv_df.to_csv(f"{fold_dir}/markowitz.csv")
        print(f"최종 가치: {mv_df['portfolio_values'].iloc[-1]:,.2f}")

        print("\n=== Risk Parity (ERC) ===")
        env = make_env(start, end_exclusive)
        rp_df = backtest_rolling_window(env, wide_prices, MV_RP_WINDOW_HOURS, risk_parity_action)
        rp_df = trim_warmup_rows(rp_df, oos_start=start)
        rp_df.to_csv(f"{fold_dir}/risk_parity.csv")
        print(f"최종 가치: {rp_df['portfolio_values'].iloc[-1]:,.2f}")


if __name__ == "__main__":
    main()
