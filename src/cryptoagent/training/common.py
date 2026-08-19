"""PPO 학습 스크립트 공용 모듈 (make_env / backtest / sanity_check).

train_ppo_mlp.py, train_ppo_transformer.py 에 거의 동일하게 복붙되어 있던
세 함수를 여기로 추출한 것. LSTM 스크립트 추가 전 선행 작업 - 이제 3개
스크립트로 늘어나도 로직이 한 곳(여기)에만 있어서 복붙 안 생김.

각 스크립트(train_ppo_*.py)는 자기 설정값(TIME_WINDOW, FEATURES,
INITIAL_AMOUNT, SEED 등)을 여기 함수에 인자로 넘기기만 하면 됨 - 정책망
종류(MLP/Transformer/LSTM)에 따라 달라지는 부분(features_extractor,
policy_kwargs)은 각 스크립트에 그대로 남겨둔다. 이 모듈은 정책망과
무관한 env 생성/백테스트/sanity check만 담당.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shimmy

from cryptoagent.envs.adapter import load_env_ready_df, patch_seed_method
from cryptoagent.envs.env_portfolio_optimization import PortfolioOptimizationEnv


def make_env(
    split: str,
    *,
    features: list[str],
    initial_amount: float,
    time_window: int,
) -> PortfolioOptimizationEnv:
    """split("train"/"val"/"test")에 맞는 PortfolioOptimizationEnv를 생성.

    각 학습 스크립트의 FEATURES/INITIAL_AMOUNT/TIME_WINDOW 설정값을
    키워드 인자로 그대로 넘겨서 쓴다.
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
    patch_seed_method(env)  # SB3 PPO(seed=...) 재현성을 위해 shimmy가 요구하는 seed() 보강
    return env


def backtest(model, eval_env: PortfolioOptimizationEnv) -> pd.DataFrame:
    """학습된 모델을 eval_env에서 deterministic하게 굴려 공용 스펙 DataFrame을 만든다.

    공용 스펙 (팀 합의):
        index: date (datetime)
        columns: returns / portfolio_values / weights
    """
    gym_env = shimmy.GymV21CompatibilityV0(env=eval_env)

    obs, _ = gym_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = gym_env.step(action)
        done = terminated or truncated

    # env._date_memory 등은 reset() 시 초기값 1개로 시작해 매 step마다 append되므로
    # 전부 동일 길이. reset() 시점의 초기값(t=0, 액션 이전)까지 포함된 전체 시계열이다.
    result = pd.DataFrame(
        {
            "date": eval_env._date_memory,
            "returns": eval_env._portfolio_return_memory,
            "portfolio_values": eval_env._asset_memory["final"],
            "weights": [w.tolist() for w in eval_env._final_weights],
        }
    )
    result["date"] = pd.to_datetime(result["date"])
    result = result.set_index("date")
    return result


def sanity_check(backtest_df: pd.DataFrame) -> None:
    """비중 합=1, NaN/inf 없는지 최소 확인 (학습 스크립트 자체 방어용)."""
    weight_sums = backtest_df["weights"].apply(sum)
    max_dev = (weight_sums - 1.0).abs().max()
    assert max_dev < 1e-3, f"비중 합이 1에서 {max_dev}만큼 벗어남"

    assert not backtest_df["returns"].isna().any(), "returns에 NaN 존재"
    assert not backtest_df["portfolio_values"].isna().any(), "portfolio_values에 NaN 존재"
    assert np.isfinite(backtest_df["portfolio_values"]).all(), "portfolio_values에 inf 존재"

    print(f"[sanity_check] OK - 비중 합 최대 편차: {max_dev:.2e}")
    print(f"[sanity_check] OK - 최종 포트폴리오 가치: {backtest_df['portfolio_values'].iloc[-1]:,.2f}")
