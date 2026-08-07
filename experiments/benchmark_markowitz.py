"""
8월 4주차: 전통 벤치마크 - Markowitz Mean-Variance (Constrained)
리포 루트에서 실행: python experiments/benchmark_markowitz.py

방법론
------
- Rolling window: 720시간(30일). 매 시점 t 기준 과거 720시간의 수익률만 사용해
  mu(기대수익률)/공분산을 추정 (미래 데이터 참조 없음, 데이터 누수 방지 원칙 준수)
- 공분산 추정: 표본공분산 대신 Ledoit-Wolf shrinkage 사용 (추정 오차 완화)
- 개별 자산 비중 상한: 30% (weight_bounds=(0, 0.3))
- 폴백 순서: max_sharpe 실패 -> min_volatility -> equal-weight

왜 이런 구조인가 (중요 - 실험적으로 검증된 사실)
------------------------------------------------
Unconstrained Markowitz(제약 없음)를 그대로 적용하면 극단적 자산 쏠림이
발생함을 실험으로 확인함:
  - window=50h(관측치 부족): 90%+ 몰빵 44.6%, 최종수익 17배(비현실적)
  - window=720h로 확장해도 제약이 없으면 오히려 쏠림 악화(몰빵 71.3%)
    -> 원인은 window 길이가 아니라 weight_bounds 부재. Mean-Variance
       최적화는 "가장 효율적인 자산 하나"에 수렴하는 경향이 있음
       (학술적으로 "Estimation Error Maximizer" 문제로 알려짐)
현재 설정(720h + shrinkage + weight_bounds 0.3)을 적용한 뒤에는
전 벤치마크와 동일한 스케일(약 3.7배)로 정상화됨. 개별 자산 상한 30%는
동일비중(1/8=12.5%)의 2.4배 수준이며 Jagannathan & Ma(2003) 및
암호화폐 포트폴리오 관련 최근 문헌에서 흔히 쓰이는 값.

공용 인터페이스: train_ppo_mlp.py의 backtest()와 동일 스펙
(date 인덱스 + returns/portfolio_values/weights).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models

from cryptoagent.envs.adapter import load_env_ready_df
from cryptoagent.envs.env_portfolio_optimization import PortfolioOptimizationEnv

TIME_WINDOW = 50
FEATURES = ["close", "high", "low"]
INITIAL_AMOUNT = 100_000
N_ASSETS = 8
TIC_ORDER = ["ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
MV_WINDOW_HOURS = 720
MAX_WEIGHT_PER_ASSET = 0.3   # ← 이 줄 추가! (Jagannathan & Ma 2003 근거, 동일비중의 2.4배)

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
    """split 구분 없이 전체 기간 가격을 불러와 (date x tic) wide 형태로 변환.
    test 구간의 rolling window가 test 시작 이전(train/val) 데이터도
    필요로 하기 때문에 전체를 로드한다 - 단, 각 시점 t에서는 t 이전
    데이터만 잘라서 쓰므로 미래 데이터 누수는 없다.
    """
    df = load_env_ready_df(split=None)  # 전체 기간
    wide = df.pivot(index="date", columns="tic", values="close")
    return wide[TIC_ORDER].sort_index()


def markowitz_action(returns_window: pd.DataFrame) -> np.ndarray:
    action = np.zeros(1 + N_ASSETS, dtype=np.float32)
    try:
        mu = expected_returns.mean_historical_return(returns_window, returns_data=True, frequency=24 * 365)
        S = risk_models.CovarianceShrinkage(returns_window, returns_data=True, frequency=24 * 365).ledoit_wolf()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                ef = EfficientFrontier(mu, S, weight_bounds=(0, MAX_WEIGHT_PER_ASSET))
                ef.max_sharpe(risk_free_rate=0.0)
            except Exception as e1:
                try:
                    ef = EfficientFrontier(mu, S, weight_bounds=(0, MAX_WEIGHT_PER_ASSET))
                    ef.min_volatility()
                except Exception as e2:
                    raise RuntimeError(f"max_sharpe 실패({e1}) / min_vol도 실패({e2})")

        cleaned = ef.clean_weights()
        weights = np.array([cleaned[t] for t in TIC_ORDER], dtype=np.float64)

        if (not np.all(np.isfinite(weights)) 
                or weights.sum() <= 0 
                or np.any(weights < -0.01) # -1e-4 -> -0.01로 완화 (미세 반올림 오차 허용)
                or np.any(weights > 1.0)):  # 상한 근처 오차도 허용
            raise ValueError(f"비정상 weights: {weights}")
        weights = np.clip(weights, 0, MAX_WEIGHT_PER_ASSET)
        for _ in range(10): # 반복하며 clip 후 재정규화가 다시 상한을 넘지 않도록 수렴시킴
            if weights.sum() <= 0:
                break
            weights = weights / weights.sum()
            weights = np.clip(weights, 0, MAX_WEIGHT_PER_ASSET)
        weights = weights / weights.sum()  # 마지막에 한 번 더 정규화해 합=1 보장

    except Exception as e:
        markowitz_action.fallback_count += 1
        if markowitz_action.fallback_count <= 5:  # 처음 5번만 출력해서 로그 폭주 방지
            print(f"[폴백 #{markowitz_action.fallback_count}] {e}")
        weights = np.full(N_ASSETS, 1.0 / N_ASSETS, dtype=np.float64)

    action[1:1 + N_ASSETS] = weights.astype(np.float32)
    return action

markowitz_action.fallback_count = 0


def backtest_markowitz(env: PortfolioOptimizationEnv, wide_prices: pd.DataFrame) -> pd.DataFrame:
    all_returns = wide_prices.pct_change().dropna()

    obs = env.reset()
    done = False
    step_count = 0
    fallback_count = 0

    while not done:
        current_date = env._date_memory[-1]
        window = all_returns.loc[:current_date].tail(MV_WINDOW_HOURS)

        if len(window) < MV_WINDOW_HOURS // 2:  # window 데이터가 너무 적으면 equal-weight
            action = np.zeros(1 + N_ASSETS, dtype=np.float32)
            action[1:1 + N_ASSETS] = 1.0 / N_ASSETS
        else:
            action = markowitz_action(window)

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

    print("=== Markowitz Mean-Variance (720h rolling window, max Sharpe) ===")
    wide_prices = load_wide_close_prices()
    env = make_env("test")
    mv_df = backtest_markowitz(env, wide_prices)
    mv_df.to_csv(f"{RESULTS_DIR}/markowitz.csv")
    print(f"최종 가치: {mv_df['portfolio_values'].iloc[-1]:,.2f}")
    print(f"총 폴백 횟수: {markowitz_action.fallback_count} / 19320")   # ← 추가


if __name__ == "__main__":
    main()