"""
8월 4주차: 전통 벤치마크 - Buy & Hold BTC, Equal-Weight (은아 담당)

공용 인터페이스: train_ppo_mlp.py의 backtest()와 동일 스펙
(date 인덱스 + returns/portfolio_values/weights) - verify_backtest_spec.py로
그대로 검증 가능.

action 벡터 구조 (실제 실행 결과로 검증, 코드 추측 아님):
    index 0        : 현금
    index 1~8      : ADA, AVAX, BNB, BTC, DOGE, ETH, SOL, XRP (알파벳순, env._tic_list 기준)
    검증 방법: index 0에 1.0을 넣고 여러 스텝 굴렸을 때 portfolio_value가
    BTC 가격 변동과 무관하게 고정되는 것으로 확인 (현금은 가격 변동이 없으므로).

action -> weights 변환 규칙 (env_portfolio_optimization.py L289-292):
    합이 정확히 1이고 전부 0 이상이면 -> action을 그대로 weights로 사용
    그 외의 경우 -> softmax_normalization()을 거쳐 근사값으로 변환
    본 스크립트의 action은 항상 합=1, 전부 0 이상으로 설계해 softmax 근사 오차 없이
    정확한 목표 비중(Buy&Hold=BTC 100%, Equal-Weight=8자산 각 1/8)을 만든다.

test split 백테스트 기간은 train_ppo_mlp.py와 완전히 동일함
(2023-10-18 ~ 2025-12-31, 19,320행) - 동일 조건 비교 검증 완료.

거래비용/수수료는 이 단계에서 반영하지 않음. evaluate() 단계에서
PPO 결과와 동일하게 0.1% 일괄 적용 예정 (팀 합의 사항, 8월 3주차 유지 담당).
"""
import os
import numpy as np
import pandas as pd

from cryptoagent.envs.adapter import load_env_ready_df
from cryptoagent.envs.env_portfolio_optimization import PortfolioOptimizationEnv

TIME_WINDOW = 50
FEATURES = ["close", "high", "low"]
INITIAL_AMOUNT = 100_000
CASH_IDX = 0
BTC_IDX = 4
N_ASSETS = 8

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


def backtest_fixed_action(env: PortfolioOptimizationEnv, action: np.ndarray) -> pd.DataFrame:
    """model.predict() 대신 고정 action을 매 스텝 반복 - train_ppo_mlp.py의 backtest() 구조 재사용.

    주의: train_ppo_mlp.py의 backtest()는 shimmy.GymV21CompatibilityV0로 감싼 env를 써서
    5-tuple(obs, reward, terminated, truncated, info)을 반환하지만, 여기서는 원본
    PortfolioOptimizationEnv를 shimmy 없이 직접 쓰므로 4-tuple(obs, reward, done, info)을
    반환한다. 두 함수가 겉보기엔 비슷해 보여도 반환값 개수가 다르므로 혼동 주의.
    """
    env.reset()
    done = False
    while not done:
        _, _, terminated, _ = env.step(action)   # 4개만 반환
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
    """BTC 100% 고정 비중.

    매 스텝 [현금0%, ..., BTC 100%, ...]를 그대로 다시 넣는 방식으로 구현.
    '진짜' Buy&Hold(최초 1회 매수 후 매매 없음, 비중은 가격변동에 따라 자연 변화)와
    다른 방식이지만, 자산이 BTC 1개뿐이라 이미 100% BTC인 상태에서 다시 100% BTC를
    지시해도 실제 거래(turnover)가 발생하지 않아 결과적으로 동일하다.
    (자산이 2개 이상 섞인 벤치마크라면 이 방식과 진짜 Buy&Hold는 달라짐 - 주의)
    """
    action = np.zeros(1 + N_ASSETS, dtype=np.float32)
    action[BTC_IDX] = 1.0
    return action


def equal_weight_action() -> np.ndarray:
    """8자산 균등 비중(각 1/8), 현금 0%. 매 스텝 동일 비중을 유지하도록 재조정."""
    action = np.zeros(1 + N_ASSETS, dtype=np.float32)
    action[1:1 + N_ASSETS] = 1.0 / N_ASSETS
    return action


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== Buy & Hold BTC ===")
    env = make_env("test")
    bnh_df = backtest_fixed_action(env, buy_and_hold_btc_action())
    bnh_df.to_csv(f"{RESULTS_DIR}/buy_and_hold_btc.csv")
    print(f"최종 가치: {bnh_df['portfolio_values'].iloc[-1]:,.2f}")

    print("\n=== Equal-Weight ===")
    env = make_env("test")
    ew_df = backtest_fixed_action(env, equal_weight_action())
    ew_df.to_csv(f"{RESULTS_DIR}/equal_weight.csv")
    print(f"최종 가치: {ew_df['portfolio_values'].iloc[-1]:,.2f}")


if __name__ == "__main__":
    main()