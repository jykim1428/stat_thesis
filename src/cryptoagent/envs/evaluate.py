"""
공용 평가 모듈 (3주차)
====================

PPO 백테스트가 끝나면 나오는 아래 형태의 DataFrame을 그대로 입력으로 받는다:

    # index: date (datetime)
    returns            # 스텝별 포트폴리오 수익률
    portfolio_values   # 스텝별 포트폴리오 가치
    weights            # 스텝별 비중 벡터 (자산 8개 + 현금), 각 행이 length-9 array

env.step() 종료 시 반환되는 metrics_df(date/returns/rewards/portfolio_values)와
env._final_weights를 그대로 합치면 이 형태가 된다 (env_portfolio_optimization.py L228-239 참고).

- 더미 데이터로 함수부터 완성.
- 나중에 실제 metrics_df로 교체(실연결)만 하면 됨.

의존성: pandas, numpy, empyrical-reloaded (import는 `empyrical`로 동일, requirements.txt에 포함됨)
"""

import numpy as np
import pandas as pd
import empyrical as ep

PERIODS_PER_YEAR = 24 * 365  # 시간(hourly) 스텝 기준 연율화 상수. 일단위 리밸런싱이면 252로 변경.
N_ASSETS_PLUS_CASH = 9  # 자산 8개 + 현금


def _validate_weights_array(W: np.ndarray, name: str) -> None:
    """weights/target_weights 배열 하나에 대한 최소 검증 (shape/finite/합/범위)."""
    if W.ndim != 2:
        raise ValueError(f"{name}는 2차원(시점 x 자산)이어야 하는데 shape={W.shape}")
    if W.shape[0] < 1:
        raise ValueError(f"{name}에 최소 1행 이상 필요함 (현재 {W.shape[0]}행)")
    if not np.isfinite(W).all():
        raise ValueError(f"{name}에 NaN/inf가 포함되어 있음")
    if (W < -1e-6).any() or (W > 1 + 1e-6).any():
        raise ValueError(f"{name}의 원소가 [0, 1] 범위를 벗어남 (min={W.min()}, max={W.max()})")
    row_sums = W.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        bad = np.sum(~np.isclose(row_sums, 1.0, atol=1e-3))
        raise ValueError(f"{name}의 {bad}개 행에서 비중 합이 1이 아님")


def compute_turnover_from_weights(
    weights_col: pd.Series,
    target_weights_col: pd.Series | None = None,
    allow_legacy_turnover: bool = False,
) -> pd.Series:
    """
    시점별 turnover(실제 거래량)를 계산.

    turnover[t] = sum(|target_weights[t] - weights[t-1]|) / 2
    (t 시점에 에이전트가 지시한 목표 비중과, 그 직전 스텝이 끝난 뒤의
    실제 비중 사이의 차이 - 이게 진짜 리밸런싱 거래량이다.)

    target_weights_col이 없을 때 allow_legacy_turnover=True로 명시하면
    구버전 근사(turnover[t] = sum(|weights[t] - weights[t-1]|) / 2)를 쓸 수
    있지만, 이 값은 가격 변동으로 인한 비중 변화까지 거래량으로 포함해
    과대계상된다 (배수는 전략마다 다르며 실측상 약 1.2~1.45배). 과거 결과를
    참고할 때만 명시적으로 켤 것 - 기본값은 False라 target_weights가 없으면
    ValueError가 난다.
    """
    W = np.vstack(weights_col.to_numpy())  # (T, N)
    _validate_weights_array(W, "weights")

    if target_weights_col is None:
        if not allow_legacy_turnover:
            raise ValueError(
                "target_weights가 없습니다. turnover는 반드시 target_weights[t]와 "
                "weights[t-1]의 차이로 계산해야 정확합니다 (weights끼리의 차이는 가격 "
                "변동분까지 거래량으로 잘못 포함해 과대계상됩니다). backtest()가 "
                "target_weights를 저장하도록 갱신 후 재실행하세요. 과거 결과를 부득이 "
                "그대로 쓰려면 allow_legacy_turnover=True를 명시적으로 넘기세요."
            )
        delta = np.abs(np.diff(W, axis=0)).sum(axis=1) / 2.0
    else:
        if len(target_weights_col) != len(weights_col) or not target_weights_col.index.equals(weights_col.index):
            raise ValueError("target_weights와 weights의 길이/인덱스가 일치하지 않습니다")
        target = np.vstack(target_weights_col.to_numpy())  # (T, N)
        _validate_weights_array(target, "target_weights")
        if target.shape != W.shape:
            raise ValueError(f"target_weights shape({target.shape})와 weights shape({W.shape})가 다름")
        # turnover[t] = |target[t] - W[t-1]|, t=1..T-1 (t=0은 초기 진입이라 0)
        delta = np.abs(target[1:] - W[:-1]).sum(axis=1) / 2.0

    turnover = pd.Series(0.0, index=weights_col.index)
    turnover.iloc[1:] = delta
    return turnover


def evaluate(
    metrics_df: pd.DataFrame,
    cost_rate: float = 0.001,
    periods_per_year: int = PERIODS_PER_YEAR,
    risk_free: float = 0.0,
    returns_include_cost: bool = False,
    allow_legacy_turnover: bool = False,
) -> dict:
    """
    공용 인터페이스 DataFrame 하나로 성과 지표 전체를 계산.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        index=date(datetime), columns=["returns", "portfolio_values", "weights", "target_weights"]
        weights/target_weights 컬럼의 각 원소는 length-9 벡터(자산 8개 + 현금).
        weights는 가격 변동을 반영한 사후(end-of-period) 비중, target_weights는
        해당 스텝에서 지시한 리밸런싱 목표 비중이다 (backtest() 참고). target_weights가
        없으면 allow_legacy_turnover=True를 명시하지 않는 한 ValueError가 난다.
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
    allow_legacy_turnover : bool
        target_weights 컬럼이 없을 때, weights끼리의 차이로 turnover를 근사하는
        구버전 방식을 명시적으로 허용할지 여부. 이 근사치는 가격 변동으로 인한
        비중 변화까지 거래량으로 잘못 포함해 실제보다 과대계상된다(배수는 전략마다
        다름, 실측 약 1.2~1.45배). 과거 결과를 부득이 그대로 참고할 때만 True로 설정.

    Returns
    -------
    dict with keys:
        sharpe, sortino, calmar, mdd, vol, cagr, avg_turnover, total_cost, n_periods
    """
    required_cols = {"returns", "portfolio_values", "weights"}
    missing = required_cols - set(metrics_df.columns)
    if missing:
        raise ValueError(f"metrics_df에 다음 컬럼이 없습니다: {missing}")

    target_weights_col = metrics_df["target_weights"] if "target_weights" in metrics_df.columns else None

    turnover = compute_turnover_from_weights(
        metrics_df["weights"], target_weights_col, allow_legacy_turnover=allow_legacy_turnover
    )
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
