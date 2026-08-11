"""
  1) PortfolioOptimizationEnv에 실제 데이터 연결 (adapter.py 경유)
  2) observation shape 확인 (USAGE.md에 문서화된 값과 일치하는지 재검증)
"""

from __future__ import annotations

from cryptoagent.envs.adapter import load_env_ready_df # 실제 데이터 연결
from cryptoagent.envs.env_portfolio_optimization import PortfolioOptimizationEnv

# USAGE.md에 문서화된 확정 설정 
FEATURES = ["close", "high", "low"]
TIME_WINDOW = 50
N_TICS_EXPECTED = 8  # freeze_metadata.json 기준 역산된 확정값


def main():
    print("=" * 60)
    print("[Step 1] 데이터 연결 (adapter.load_env_ready_df)")
    print("=" * 60)
    df = load_env_ready_df(split="train") # 실제 데이터 로딩
    n_tics = df["tic"].nunique() # crypto asset 8개 로딩
    print(f"df shape: {df.shape}")
    print(f"종목 수(n): {n_tics}")
    print(f"종목 목록: {sorted(df['tic'].unique())}")

    if n_tics != N_TICS_EXPECTED:
        print(
            f"⚠️  경고: 기대했던 종목 수({N_TICS_EXPECTED})와 다릅니다 (실제 {n_tics}). "
            f"USAGE.md의 shape 문서를 재확인/갱신하세요."
        )

    print()
    print("=" * 60)
    print(f"[Step 2] PortfolioOptimizationEnv 생성 (features={FEATURES}, time_window={TIME_WINDOW})")
    print("=" * 60)
    env = PortfolioOptimizationEnv( #FinRL PortfolioOptimizationEnv 연결
        df=df,
        initial_amount=100_000,
        time_column="date",
        tic_column="tic",
        features=FEATURES,
        time_window=TIME_WINDOW,
    )
    print(f"action_space: {env.action_space.shape}") # action_space 확인
    print(f"episode_length: {env.episode_length}")

    print()
    print("=" * 60)
    print("[Step 3] Observation shape 확인")
    print("=" * 60)
    obs = env.reset() # observation shape 확인 (초기 observation 생성)
    expected_shape = (len(FEATURES), n_tics, TIME_WINDOW)

    print(f"실제 obs.shape : {obs.shape}")
    print(f"기대 shape     : {expected_shape}  (= (f, n, time_window))")
    print(f"USAGE.md 문서값 : (3, 8, 50)")

    if obs.shape == expected_shape:
        print("✅ 공식과 일치")
    else:
        raise AssertionError(f"obs.shape {obs.shape} != expected {expected_shape}")

    if obs.shape == (3, N_TICS_EXPECTED, 50):
        print("✅ USAGE.md 문서값과도 일치 — 문서 신뢰 가능, 갱신 불필요")
    else:
        print("⚠️  USAGE.md 문서값과 다름 — 데이터가 바뀐 것으로 보임, USAGE.md 갱신 필요")

    print()
    print("=" * 60)
    print("✅ 데이터 연결 확인 완료")
    print("✅ observation shape 확인 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
