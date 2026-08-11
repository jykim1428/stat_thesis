"""
8월 4주차: 전통 벤치마크 - Risk Parity (Equal Risk Contribution)
리포 루트에서 실행: python experiments/benchmark_risk_parity.py

방법론
------
- 공분산 행렬(Ledoit-Wolf shrinkage 적용)을 반영해, 각 자산이 포트폴리오
  전체 리스크에 기여하는 정도(risk contribution)가 균등해지는 비중을
  scipy.optimize(SLSQP)로 탐색 (Equal Risk Contribution, ERC)
- Rolling window: 720시간(30일). Markowitz 벤치마크와 동일 window/shrinkage/
  weight_bounds(0, 0.3)를 사용해 두 벤치마크 간 방법론 일관성을 맞춤
- 실패 시 equal-weight로 폴백 (실제 실행 결과 폴백 0회 - Markowitz보다
  안정적. 기대수익률 추정이 없어 노이즈에 덜 민감하기 때문으로 판단)

참고 - 단순 버전과의 차이
------------------------
자산별 변동성(표준편차)만 반영하는 "Inverse Volatility"는 계산이 훨씬
간단하지만 자산 간 상관관계를 무시함. 본 구현은 공분산 전체를 반영하는
엄밀한 정의의 Risk Parity(ERC)임.

공용 인터페이스: train_ppo_mlp.py의 backtest()와 동일 스펙
(date 인덱스 + returns/portfolio_values/weights).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from pypfopt import risk_models
from scipy.optimize import minimize

from cryptoagent.envs.adapter import load_env_ready_df
from cryptoagent.envs.env_portfolio_optimization import PortfolioOptimizationEnv

TIME_WINDOW = 50
FEATURES = ["close", "high", "low"]
INITIAL_AMOUNT = 100_000
N_ASSETS = 8
TIC_ORDER = ["ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
RP_WINDOW_HOURS = 720  # Markowitz와 동일 (방법론 일관성)
MAX_WEIGHT_PER_ASSET = 0.3  # Markowitz와 동일 근거 (Jagannathan & Ma 2003 등)

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


def _risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    port_var = w @ cov @ w
    port_vol = np.sqrt(max(port_var, 1e-12))
    marginal = cov @ w / port_vol
    return w * marginal  # 자산별 risk contribution


def _erc_objective(w: np.ndarray, cov: np.ndarray) -> float:
    rc = _risk_contributions(w, cov)
    target = rc.mean()
    return np.sum((rc - target) ** 2)


def risk_parity_action(returns_window: pd.DataFrame) -> np.ndarray:
    """returns_window: (관측치 x 8자산) 단순수익률 DataFrame, 이미 과거 데이터만 포함.

    ERC(Equal Risk Contribution) 최적화: 각 자산의 risk contribution이 균등해지는
    비중을 scipy.optimize.minimize(SLSQP)로 탐색. 실패 시 equal-weight로 폴백.
    """
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
        risk_parity_action.fallback_count += 1
        if risk_parity_action.fallback_count <= 5:
            print(f"[폴백 #{risk_parity_action.fallback_count}] {e}")
        weights = np.full(N_ASSETS, 1.0 / N_ASSETS, dtype=np.float64)

    action[1:1 + N_ASSETS] = weights.astype(np.float32)
    return action


risk_parity_action.fallback_count = 0


def backtest_risk_parity(env: PortfolioOptimizationEnv, wide_prices: pd.DataFrame) -> pd.DataFrame:
    all_returns = wide_prices.pct_change().dropna()

    env.reset()
    done = False
    step_count = 0

    while not done:
        current_date = env._date_memory[-1]
        window = all_returns.loc[:current_date].tail(RP_WINDOW_HOURS)

        if len(window) < RP_WINDOW_HOURS // 2:
            action = np.zeros(1 + N_ASSETS, dtype=np.float32)
            action[1:1 + N_ASSETS] = 1.0 / N_ASSETS
        else:
            action = risk_parity_action(window)

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

    print("=== Risk Parity (Equal Risk Contribution, 720h window) ===")
    wide_prices = load_wide_close_prices()
    env = make_env("test")
    rp_df = backtest_risk_parity(env, wide_prices)
    rp_df.to_csv(f"{RESULTS_DIR}/risk_parity.csv")
    print(f"최종 가치: {rp_df['portfolio_values'].iloc[-1]:,.2f}")
    print(f"총 폴백 횟수: {risk_parity_action.fallback_count} / {len(rp_df)}")


if __name__ == "__main__":
    main()