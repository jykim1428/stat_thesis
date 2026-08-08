"""
8월 4주차: 전통 벤치마크 - Minimum Variance Portfolio
리포 루트에서 실행: python experiments/benchmark_markowitz.py

방법론
------
- Rolling window: 720시간(30일). 매 시점 t 기준 과거 720시간의 수익률만 사용해
  공분산을 추정 (미래 데이터 참조 없음, 데이터 누수 방지 원칙 준수)
- 공분산 추정: 표본공분산 대신 Ledoit-Wolf shrinkage 사용 (추정 오차 완화)
- 개별 자산 비중 상한: 30% (weight_bounds=(0, 0.3))
- 최적화: 기대수익률(mu) 추정 없이 분산만 최소화하는 min_volatility()
- 폴백: 최적화 실패 시 equal-weight

명칭에 대해
-----------
Minimum Variance Portfolio는 위험회피계수(gamma)가 무한대로 수렴할 때
도출되는 Efficient Frontier 최좌단 포트폴리오로, Markowitz Mean-Variance
(MV) 프레임워크에 포함되는 특수 케이스임. 기대수익률(mu) 추정 없이
공분산(Sigma)에만 의존해 최적화함.

공용 인터페이스: train_ppo_mlp.py의 backtest()와 동일 스펙
(date 인덱스 + returns/portfolio_values/weights).
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models

from cryptoagent.envs.adapter import load_env_ready_df
from cryptoagent.envs.env_portfolio_optimization import PortfolioOptimizationEnv

TIME_WINDOW = 50
FEATURES = ["close", "high", "low"]
INITIAL_AMOUNT = 100_000
N_ASSETS = 8
TIC_ORDER = ["ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
MV_WINDOW_HOURS = 720
MAX_WEIGHT_PER_ASSET = 0.3

RESULTS_DIR = "results/benchmarks"


def make_env(split: str) -> PortfolioOptimizationEnv:
    df = load_env_ready_df(split=split)
    return PortfolioOptimizationEnv(
        df=df,
        initial_amount=INITIAL_AMOUNT,
        time_column="date",
        tic_column="tic",
        features=FEATURES,
        time_window=TIME_WINDOW,
    )


def load_wide_close_prices() -> pd.DataFrame:
    df = load_env_ready_df(split=None)
    wide = df.pivot(index="date", columns="tic", values="close")
    return wide[TIC_ORDER].sort_index()


def min_variance_action(returns_window: pd.DataFrame) -> np.ndarray:
    """returns_window: (관측치 x 8자산) 단순수익률 DataFrame, 이미 과거 데이터만 포함.

    기대수익률(mu) 추정 없이, 공분산(Ledoit-Wolf shrinkage)만으로 분산을
    최소화하는 비중을 계산 (Minimum Variance Portfolio).
    """
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
        min_variance_action.fallback_count += 1
        if min_variance_action.fallback_count <= 5:
            print(f"[폴백 #{min_variance_action.fallback_count}] {e}")
        weights = np.full(N_ASSETS, 1.0 / N_ASSETS, dtype=np.float64)

    action[1:1 + N_ASSETS] = weights.astype(np.float32)
    return action


min_variance_action.fallback_count = 0


def backtest_min_variance(env: PortfolioOptimizationEnv, wide_prices: pd.DataFrame) -> pd.DataFrame:
    all_returns = wide_prices.pct_change().dropna()

    env.reset()
    done = False
    step_count = 0

    while not done:
        current_date = env._date_memory[-1]
        window = all_returns.loc[:current_date].tail(MV_WINDOW_HOURS)

        if len(window) < MV_WINDOW_HOURS // 2:
            action = np.zeros(1 + N_ASSETS, dtype=np.float32)
            action[1:1 + N_ASSETS] = 1.0 / N_ASSETS
        else:
            action = min_variance_action(window)

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
        }
    )
    result["date"] = pd.to_datetime(result["date"])
    return result.set_index("date")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== Minimum Variance Portfolio (720h window, Ledoit-Wolf) ===")
    wide_prices = load_wide_close_prices()
    env = make_env("test")
    mv_df = backtest_min_variance(env, wide_prices)
    mv_df.to_csv(f"{RESULTS_DIR}/markowitz.csv")
    print(f"최종 가치: {mv_df['portfolio_values'].iloc[-1]:,.2f}")
    print(f"총 폴백 횟수: {min_variance_action.fallback_count} / {len(mv_df)}")


if __name__ == "__main__":
    main()