"""PPO 학습 스크립트 공용 모듈 (make_env / backtest / sanity_check).

train_ppo_mlp.py, train_ppo_transformer.py, train_ppo_lstm.py에 복붙되어
있던 세 함수를 여기로 추출한 것. 정책망 종류(MLP/Transformer/LSTM)에
따라 달라지는 부분(features_extractor, policy_kwargs)은 각 스크립트에
그대로 남겨두고, env 생성/백테스트/sanity check만 이 모듈이 담당한다.

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

from cryptoagent.envs.adapter import load_env_ready_df, load_env_ready_df_by_date, patch_seed_method
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
    cwd: str = "./",
) -> PortfolioOptimizationEnv:
    """split("train"/"val"/"test")에 맞는 PortfolioOptimizationEnv를 생성.

    stats가 주어지면 TrainStandardizeWrapper로 감싸서 반환 (train-only
    standardization 적용). stats=None(기본값)이면 기존과 동일하게 원본
    env를 그대로 반환해 하위 호환됨.

    cwd: PortfolioOptimizationEnv가 에피소드 종료마다 자동 저장하는 그림
    (results/rl/*.png)의 저장 위치 기준 디렉토리. 기본값("./")은 항상 같은
    경로를 가리키므로, 여러 env 인스턴스를 병렬로 실행하면(예: Optuna
    다중 trial) 서로 다른 프로세스가 동일 파일을 동시에 덮어써 파일
    충돌이 날 수 있다. 병렬 실행 시에는 trial마다 고유한 cwd를 지정할 것.
    """
    df = load_env_ready_df(split=split)
    env = PortfolioOptimizationEnv(
        df=df,
        initial_amount=initial_amount,
        time_column="date",
        tic_column="tic",
        features=features,
        time_window=time_window,
        cwd=cwd,
    )
    patch_seed_method(env)

    if stats is not None:
        env = TrainStandardizeWrapper(env, stats=stats, clip=clip)

    return env


def make_env_by_date(
    start: str,
    end: str,
    *,
    features: list[str],
    initial_amount: float,
    time_window: int,
    warmup_hours: int = 0,
    stats: dict | None = None,
    clip: tuple[float, float] | None = (-5.0, 5.0),
    cwd: str = "./",
) -> PortfolioOptimizationEnv:
    """[start, end] 날짜 범위로 PortfolioOptimizationEnv를 생성 (walk-forward fold용).

    make_env(split=...)와 달리 고정 split 라벨이 아니라 임의의 날짜 범위를
    받는다. warmup_hours > 0이면 start보다 그만큼 앞선 시점부터 데이터를
    포함해서 로드한다 (docs/walk_forward_design.md의 워밍업 처리 참고 -
    PortfolioOptimizationEnv는 첫 관측을 만드는 데 time_window개 과거 시점이
    필요하므로, OOS 구간 시작부터만 로드하면 그 구간 초반이 관측 워밍업으로
    소모되어 거래/수익률 기록에서 누락된다).

    warmup_hours를 쓸 경우 반환된 env로 backtest()를 돌린 결과에는 워밍업
    구간의 행이 섞여 있으므로, trim_warmup_rows()로 실제 OOS 시작 시점
    이전 행을 잘라내야 한다.

    cwd: make_env()와 동일 - 병렬 실행(Optuna 다중 trial 등) 시 trial마다
    고유한 값을 지정해 results/rl/*.png 파일 충돌을 피할 것.
    """
    df = load_env_ready_df_by_date(start=start, end=end, warmup_hours=warmup_hours)
    env = PortfolioOptimizationEnv(
        df=df,
        initial_amount=initial_amount,
        time_column="date",
        tic_column="tic",
        features=features,
        cwd=cwd,
        time_window=time_window,
    )
    patch_seed_method(env)

    if stats is not None:
        env = TrainStandardizeWrapper(env, stats=stats, clip=clip)

    return env


def trim_warmup_rows(backtest_df: pd.DataFrame, oos_start: str) -> pd.DataFrame:
    """backtest() 결과에서 oos_start 이전(워밍업 구간) 행을 제거.

    make_env_by_date(..., warmup_hours=N)으로 만든 env를 backtest()에 넘기면
    결과 DataFrame의 인덱스가 (oos_start - N시간)부터 시작한다. 실제 성과지표
    계산과 CSV 저장에는 oos_start 이후 행만 남겨야 워밍업 구간의 수익률/거래가
    섞여 들어가지 않는다 (docs/walk_forward_design.md 참고).

    reset() 시점의 초기 행(t=0, returns=0)은 oos_start 이전 시각을 가지므로
    이 필터링으로 자동으로 함께 제거된다.
    """
    oos_start_ts = pd.Timestamp(oos_start)
    trimmed = backtest_df[backtest_df.index >= oos_start_ts].copy()
    if trimmed.empty:
        raise ValueError(
            f"oos_start={oos_start_ts} 이후 행이 없음 - warmup_hours 설정이나 "
            f"backtest_df 인덱스 범위를 확인하세요"
        )
    return trimmed


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

    # 환경은 reset 시점의 초기값 1개와 각 step 결과를 각각의 메모리에 추가한다.
    # 메모리가 같은 길이인지 확인해 date/returns/value/weights가
    # 같은 시점을 가리킨다는 공용 결과 스펙을 보장한다.
    assert (
        len(base_env._date_memory)
        == len(base_env._portfolio_return_memory)
        == len(base_env._asset_memory["final"])
        == len(base_env._final_weights)
        == len(base_env._actions_memory)
    ), "환경 메모리 길이가 일치하지 않음"

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