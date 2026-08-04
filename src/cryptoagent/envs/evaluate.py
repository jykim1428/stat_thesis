"""
공용 평가 모듈 (3주차)
====================
공용 인터페이스 스펙(착수 전 5분 문서)에 맞춘 evaluate() 함수.

PPO 백테스트가 끝나면 나오는 아래 형태의 DataFrame을 그대로 입력으로 받는다:

    # index: date (datetime)
    returns            # 스텝별 포트폴리오 수익률
    portfolio_values   # 스텝별 포트폴리오 가치
    weights            # 스텝별 비중 벡터 (자산 8개 + 현금), 각 행이 length-9 array

env.step() 종료 시 반환되는 metrics_df(date/returns/rewards/portfolio_values)와
env._final_weights를 그대로 합치면 이 형태가 된다 (env_portfolio_optimization.py L228-239 참고).

- 아직 준영의 실제 백테스트 결과가 없으므로, 더미 데이터로 함수부터 완성.
- 나중에 실제 metrics_df로 교체(실연결)만 하면 됨.

의존성: pandas, numpy, empyrical-reloaded (import는 `empyrical`로 동일)
    pip install empyrical-reloaded --break-system-packages
"""

import numpy as np
import pandas as pd
import empyrical as ep

PERIODS_PER_YEAR = 24 * 365  # 시간(hourly) 스텝 기준 연율화 상수. 일단위 리밸런싱이면 252로 변경.
N_ASSETS_PLUS_CASH = 9  # 자산 8개 + 현금


def compute_turnover_from_weights(weights_col: pd.Series) -> pd.Series:
    """
    metrics_df["weights"] 컬럼(각 원소가 length-9 벡터)으로부터
    시점별 turnover = sum(|w_t - w_{t-1}|) / 2 를 계산.
    """
    W = np.vstack(weights_col.to_numpy())  # (T, 9)
    delta = np.abs(np.diff(W, axis=0)).sum(axis=1) / 2.0
    turnover = pd.Series(0.0, index=weights_col.index)
    turnover.iloc[1:] = delta
    return turnover


def evaluate(
    metrics_df: pd.DataFrame,
    cost_rate: float = 0.001,
    periods_per_year: int = PERIODS_PER_YEAR,
    risk_free: float = 0.0,
    returns_include_cost: bool = False,
) -> dict:
    """
    공용 인터페이스 DataFrame 하나로 성과 지표 전체를 계산.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        index=date(datetime), columns=["returns", "portfolio_values", "weights"]
        weights 컬럼의 각 원소는 length-9 벡터(자산 8개 + 현금)
    cost_rate : float
        편도 거래비용률 (기본 0.1% = 0.001)
    periods_per_year : int
        연율화 상수. hourly=24*365(기본), daily=252, weekly=52
    risk_free : float
        무위험 수익률 (기간당). 기본 0
    returns_include_cost : bool
        metrics_df["returns"]가 이미 거래비용을 반영한 값이면 True로 설정
        (예: env 내부에 transaction fee가 이미 구현되어 있는 경우).
        기본은 False -> evaluate()가 turnover 기반으로 비용을 직접 차감.

    Returns
    -------
    dict with keys:
        sharpe, sortino, calmar, mdd, vol, cagr, avg_turnover, total_cost, n_periods
    """
    required_cols = {"returns", "portfolio_values", "weights"}
    missing = required_cols - set(metrics_df.columns)
    if missing:
        raise ValueError(f"metrics_df에 다음 컬럼이 없습니다: {missing}")

    turnover = compute_turnover_from_weights(metrics_df["weights"])
    costs = turnover * cost_rate

    if returns_include_cost:
        net_returns = metrics_df["returns"]
    else:
        net_returns = metrics_df["returns"] - costs

    net_returns = net_returns.dropna()

    metrics = {
        "sharpe": ep.sharpe_ratio(net_returns, risk_free=risk_free, period="daily",
                                   annualization=periods_per_year),
        "sortino": ep.sortino_ratio(net_returns, required_return=risk_free,
                                     period="daily", annualization=periods_per_year),
        "calmar": ep.calmar_ratio(net_returns, period="daily", annualization=periods_per_year),
        "mdd": ep.max_drawdown(net_returns),
        "vol": ep.annual_volatility(net_returns, period="daily", annualization=periods_per_year),
        "cagr": ep.annual_return(net_returns, period="daily", annualization=periods_per_year),
        "avg_turnover": turnover.mean(),
        "total_cost": costs.sum(),
        "n_periods": len(net_returns),
    }
    return metrics


def print_report(metrics: dict, name: str = "Portfolio") -> None:
    print(f"\n===== {name} 평가 리포트 =====")
    print(f"  Sharpe        : {metrics['sharpe']:.4f}")
    print(f"  Sortino       : {metrics['sortino']:.4f}")
    print(f"  Calmar        : {metrics['calmar']:.4f}")
    print(f"  MDD           : {metrics['mdd']:.4%}")
    print(f"  Volatility    : {metrics['vol']:.4%}")
    print(f"  CAGR          : {metrics['cagr']:.4%}")
    print(f"  Avg Turnover  : {metrics['avg_turnover']:.4%}")
    print(f"  Total Cost    : {metrics['total_cost']:.4%}")
    print(f"  N periods     : {metrics['n_periods']}")


# ------------------------------------------------------------------
# 더미 데이터로 함수 완성도 테스트 (나중에 실제 metrics_df로 교체)
# ------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    n_periods = 24 * 90  # 90일치 hourly 더미 데이터
    dates = pd.date_range("2026-01-01", periods=n_periods, freq="h")

    # 더미 비중 벡터: 자산 8개 + 현금 1개, 랜덤워크 후 정규화
    raw_w = np.abs(np.random.normal(loc=1.0, scale=0.3, size=(n_periods, N_ASSETS_PLUS_CASH)))
    W = raw_w / raw_w.sum(axis=1, keepdims=True)

    # 더미 포트폴리오 수익률 (거래비용 반영 전 gross)
    gross_returns = np.random.normal(loc=0.00002, scale=0.015, size=n_periods)

    # 더미 portfolio_values: 누적곱으로 생성 (실제로는 env가 산출)
    portfolio_values = 10000 * np.cumprod(1 + gross_returns)

    dummy_metrics_df = pd.DataFrame(
        {
            "returns": gross_returns,
            "portfolio_values": portfolio_values,
            "weights": list(W),  # 각 행이 length-9 벡터
        },
        index=dates,
    )
    dummy_metrics_df.index.name = "date"

    result = evaluate(dummy_metrics_df, cost_rate=0.001, periods_per_year=PERIODS_PER_YEAR)
    print_report(result, name="Dummy (metrics_df 공용 인터페이스)")
    print("\n[OK] 공용 인터페이스 형식으로 evaluate() 정상 동작 확인. 실연결 준비됨.")
    print("     (실제 데이터로 교체 시: metrics_df = env가 만든 date/returns/portfolio_values + weights 합친 df)")
