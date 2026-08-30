"""PPO 학습 스크립트 공용 모듈 (make_env / backtest / sanity_check).
...(기존 docstring 유지)...

Train-only Standardization 지원
--------------------------------
make_env()에 stats 파라미터를 추가해, TrainStandardizeWrapper를
공용 모듈 레벨에서 선택적으로 적용할 수 있게 함 (stats=None이면 원본
그대로, 기존 호출부는 코드 변경 없이 동작).

backtest()는 eval_env가 원본이든 wrapper로 감싼 버전이든 모두 처리
하도록 .unwrapped로 원본 env의 private 메모리에 접근함 (gym.Wrapper는
밑줄 속성을 자동 위임하지 않음 - MLP 적용 때 발견된 버그, 여기서도
동일하게 발생할 수 있어 선제 반영).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shimmy

from cryptoagent.envs.adapter import load_env_ready_df, patch_seed_method
from cryptoagent.envs.env_portfolio_optimization import PortfolioOptimizationEnv
from cryptoagent.envs.normalize_wrapper import TrainStandardizeWrapper


def make_env(
    split: str,
    *,
    features: list[str],
    initial_amount: float,
    time_window: int,
    stats: dict | None = None,
    clip: tuple[float, float] | None = (-5.0, 5.0),
) -> PortfolioOptimizationEnv:
    """split("train"/"val"/"test")에 맞는 PortfolioOptimizationEnv를 생성.

    stats가 주어지면 TrainStandardizeWrapper로 감싸서 반환 (train-only
    standardization 적용). stats=None(기본값)이면 기존과 동일하게 원본
    env를 그대로 반환해 하위 호환됨.
    """
    df = load_env_ready_df(split=split)
    env = PortfolioOptimizationEnv(
        df=df,
        initial_amount=initial_amount,
        time_column="date",
        tic_column="tic",
        features=features,
        time_window=time_window,
    )
    patch_seed_method(env)

    if stats is not None:
        env = TrainStandardizeWrapper(env, stats=stats, clip=clip)

    return env


def backtest(model, eval_env) -> pd.DataFrame:
    """학습된 모델을 eval_env에서 deterministic하게 굴려 공용 스펙 DataFrame을 만든다.

    eval_env는 원본 PortfolioOptimizationEnv이거나 TrainStandardizeWrapper로
    감싼 버전일 수 있음. gym.Wrapper는 밑줄(_) 속성을 자동 위임하지 않으므로
    .unwrapped로 원본 env를 꺼내서 private 메모리에 접근해야 함.

    공용 스펙 (팀 합의):
        index: date (datetime)
        columns: returns / portfolio_values / weights / target_weights

    weights는 가격 변동을 반영한 사후(post-trade) 비중, target_weights는
    해당 스텝에서 에이전트가 실제로 지시한 리밸런싱 목표 비중이다
    (env._actions_memory, env_portfolio_optimization.py 참고). 거래량(turnover)은
    반드시 target_weights[t]와 weights[t-1](직전 스텝 사후 비중)의 차이로
    계산해야 한다 - weights[t]-weights[t-1]로 계산하면 가격 변동으로 인한
    비중 변화까지 거래량으로 잘못 잡아 turnover가 과대계상된다
    (evaluate.py의 compute_turnover_from_weights 참고).
    """
    gym_env = shimmy.GymV21CompatibilityV0(env=eval_env)
    obs, _ = gym_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = gym_env.step(action)
        done = terminated or truncated

    base_env = eval_env.unwrapped

    result = pd.DataFrame(
        {
            "date": base_env._date_memory,
            "returns": base_env._portfolio_return_memory,
            "portfolio_values": base_env._asset_memory["final"],
            "weights": [w.tolist() for w in base_env._final_weights],
            "target_weights": [w.tolist() for w in base_env._actions_memory],
        }
    )
    result["date"] = pd.to_datetime(result["date"])
    result = result.set_index("date")
    return result


def sanity_check(backtest_df: pd.DataFrame) -> None:
    """비중 합=1, NaN/inf 없는지 최소 확인 (학습 스크립트 자체 방어용)."""
    # weights/target_weights 벡터 안에 NaN이 섞이면 sum()이 NaN이 되고,
    # pandas.Series.max()는 기본적으로 skipna=True라 그 행이 통계에서
    # 조용히 빠져 max_dev가 정상값으로 나온다 - 반드시 벡터 원소 단위로
    # 먼저 finite 여부를 확인해야 이 케이스를 놓치지 않는다.
    weight_cols = ["weights"] + (["target_weights"] if "target_weights" in backtest_df.columns else [])
    for col in weight_cols:
        assert backtest_df[col].apply(lambda w: np.isfinite(w).all()).all(), f"{col}에 NaN 또는 inf 존재"

    weight_sums = backtest_df["weights"].apply(sum)
    max_dev = (weight_sums - 1.0).abs().max()
    assert max_dev < 1e-3, f"비중 합이 1에서 {max_dev}만큼 벗어남"

    assert np.isfinite(backtest_df["returns"]).all(), "returns에 NaN 또는 inf 존재"
    assert not backtest_df["portfolio_values"].isna().any(), "portfolio_values에 NaN 존재"
    assert np.isfinite(backtest_df["portfolio_values"]).all(), "portfolio_values에 inf 존재"

    print(f"[sanity_check] OK - 비중 합 최대 편차: {max_dev:.2e}")
    print(f"[sanity_check] OK - 최종 포트폴리오 가치: {backtest_df['portfolio_values'].iloc[-1]:,.2f}")