"""
9월 2주차: Train-only Standardization Wrapper (은아 담당)

env의 기존 정규화(by_previous_time, 1차 - 시점 간 비율만 계산, 데이터
누수 없음 검증됨)는 그대로 유지하고, 그 위에 train 구간 통계로 fit한
표준화(2차)를 추가하는 wrapper.

배경
----
관측값이 1.0 근처 좁은 범위에 몰려있고 자산별 변동성 차이가 있어,
신경망 학습 안정성을 위해 자산별 표준화를 시도.

방식 - env.step() 순회를 버리고 train_env._df 직접 계산으로 전환
------------------------------------------------------------
[문제 발견] compute_train_stats()가 처음에 env.step()을 액션으로
순회하며 obs를 모으는 방식이었음. 이 방식은 lookback window(예: 50h)가
한 스텝씩 겹치며 진행되므로, train 데이터의 각 시점이 최대
time_window번까지 중복으로 통계 계산에 들어가는 구조적 오류가 있었음.

[해결] train_env._df(env가 생성 시점에 이미 by_previous_time까지
전처리를 마친 데이터)를 직접 읽어, 각 시점을 정확히 1번씩만 계산하도록
전환. action/seed 개념 자체가 없어 항상 100% 동일한 값이 재현됨.

[ddof 통일] pandas.std() 기본값(ddof=1)이 기존 numpy 기반 계산
(ddof=0)과 다름을 발견, std(ddof=0) 명시로 통일.

[dtype 통일] mean/std를 float32로 저장, _normalize() 반환값도
float32로 캐스팅 - SB3/PyTorch 입력(float32)과 일관성 확보.

train stats의 자산/feature 순서가 적용 대상 env와
다르면, shape은 동일해서 에러 없이 조용히 잘못된 통계가 적용될 위험이
있음 (예: BTC 통계를 ETH에 적용). compute_train_stats()가 stats에
tic_order/features를 같이 저장하고, TrainStandardizeWrapper가
생성 시점에 이를 대상 env와 대조해 다르면 ValueError로 즉시 실패하도록
수정 (fail-fast).

gym.Wrapper를 상속한 이유 (TrainStandardizeWrapper)
----------------------------------------------------
일반 클래스로 감싸면 shimmy/SB3가 진짜 gym 환경으로 인식 못 할 위험이
있어 gym.Wrapper 상속으로 변경.

주의 - 아직 팀 논의/검증 전 단계
--------------------------------
이 wrapper는 작성만 해두고 아직 기본 학습 스크립트(train_ppo_mlp.py 등)
에는 연결하지 않음. 적용 범위(MLP 포함 여부)와 시점(지금 vs 3주차)을
팀과 논의 후 결정 예정.

사용법 (적용하기로 결정된 경우)
--------------------------------
    train_env = make_env("train")
    stats = compute_train_stats(train_env)   # train으로만 fit (seed 불필요)
    train_env = TrainStandardizeWrapper(train_env, stats=stats)

    test_env = make_env("test")
    test_env = TrainStandardizeWrapper(test_env, stats=stats)  # 동일 stats 재사용 (transform만)

검증 재현 방법 (팀원 누구나 동일 결과 확인 가능)
--------------------------------------------
    python -m cryptoagent.envs.normalize_wrapper
"""
from __future__ import annotations

import gym
import numpy as np
from gym import spaces


class TrainStandardizeWrapper(gym.Wrapper):
    """PortfolioOptimizationEnv를 감싸서 obs를 표준화해서 내보낸다.

    gym.Wrapper를 상속해 action_space/observation_space 등을 원본에서
    자동으로 물려받되, observation_space는 표준화 후 값 범위에 맞게
    __init__에서 명시적으로 재정의한다.

    stats는 반드시 train split으로만 계산해서 val/test에 그대로 적용
    (fit은 train에서만, transform은 어디든 동일 - 데이터 누수 방지).
    stats=None이면 표준화 없이 원본 obs를 그대로 통과시킨다 (비활성 모드).

    stats에 tic_order/features가 포함되어 있으면, 적용 대상 env의
    자산/feature 순서와 정확히 일치하는지 생성 시점에 검증한다
    (코덱스 리뷰 반영 - 순서가 다르면 조용히 잘못된 통계가 적용될
    위험이 있어 fail-fast로 막음).
    """

    def __init__(self, env, stats: dict | None = None):
        super().__init__(env)

        if stats is not None:
            if "tic_order" in stats:
                env_tic_order = list(env._tic_list)
                if list(stats["tic_order"]) != env_tic_order:
                    raise ValueError(
                        f"자산 순서 불일치: stats={stats['tic_order']}, "
                        f"env={env_tic_order}"
                    )
            if "features" in stats:
                if list(stats["features"]) != list(env._features):
                    raise ValueError(
                        f"feature 순서 불일치: stats={stats['features']}, "
                        f"env={env._features}"
                    )

        # SB3/PyTorch 입력 일관성을 위해 float32로 통일
        if stats is not None:
            self.stats = {
                "mean": stats["mean"].astype(np.float32),
                "std": stats["std"].astype(np.float32),
            }
        else:
            self.stats = None

        if stats is not None:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=env.observation_space.shape,
                dtype=np.float32,
            )

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        # gym(구식): obs만 반환 / gymnasium(신식): (obs, info) 2-tuple 반환
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            return self._normalize(obs), info
        return self._normalize(result)

    def step(self, action):
        result = self.env.step(action)
        obs = result[0]
        return (self._normalize(obs), *result[1:])
    
    def _normalize(self, obs):
        if self.stats is None:
            return obs
        normalized = ((np.asarray(obs, dtype=np.float32) - self.stats["mean"])
                   / self.stats["std"])
    # 표준화 후 값이 최대 ±36까지 튀는 극단치가 존재함 (DOGEUSDT 등 급등락이 심한 자산 특성). 
    # 신경망 학습 안정성을 위해 일반적인 clipping 관행(예: ±5)을 적용
        normalized = np.clip(normalized, -5.0, 5.0)
        return normalized.astype(np.float32)


def compute_train_stats(train_env) -> dict:
    """... (기존 docstring 유지, 아래 검증 추가 설명만 덧붙임)

    (코덱스 리뷰 반영) mean/std의 NaN/inf, train std가 tolerance
    이하인 축(상수 feature 의심)을 여기서 직접 검증. wrapper 생성이
    아니라 stats 계산 시점에 막아야, "wrapper는 생성됐지만 내부
    통계가 이미 잘못된" 상황을 방지할 수 있음.
    """
    df = train_env._df
    tic_col = train_env._tic_column
    features = train_env._features
    tic_order = list(train_env._tic_list)

    mean_per_tic = df.groupby(tic_col)[features].mean().loc[tic_order]
    std_per_tic = df.groupby(tic_col)[features].std(ddof=0).loc[tic_order]

    mean = mean_per_tic.to_numpy().T[:, :, np.newaxis].astype(np.float32)
    raw_std = std_per_tic.to_numpy().T[:, :, np.newaxis].astype(np.float32)

    # NaN/inf, 상수(zero-std) feature를 여기서 fail-fast
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(raw_std)):
        raise ValueError("train stats에 NaN/inf가 포함되어 있음 - 원본 데이터 확인 필요")

    tolerance = 1e-6
    if np.any(raw_std <= tolerance):
        bad_count = int(np.sum(raw_std <= tolerance))
        raise ValueError(
            f"train 구간에서 표준편차가 {tolerance} 이하인 axis가 {bad_count}개 "
            f"발견됨 (상수 feature 의심) - 표준화 대상에서 제외하거나 원본 데이터 확인 필요"
        )

    expected_shape = (len(features), len(tic_order), 1)
    if mean.shape != expected_shape or raw_std.shape != expected_shape:
        raise ValueError(f"stats shape 불일치: mean={mean.shape}, std={raw_std.shape}, 기대값={expected_shape}")

    std = raw_std + 1e-8  # 검증 통과 후에만 epsilon 추가 (0 나눗셈 방지용)

    return {"mean": mean, "std": std, "tic_order": tic_order, "features": features}


def verify_on_split(train_env, target_env) -> dict:
    """train stats를 target_env(test/val)의 데이터에 적용했을 때의
    검증 통계를 계산.

    per_asset_feature_std는 (asset, feature) 형태로 계산 - 자산 안에서
    close/high/low가 뭉쳐진 단일 값이 되지 않도록 분리.

    max_min_ratio: 표준편차가 0인 feature를 조용히 제외하지 않음.
    0이 하나라도 있고 양수 std가 존재하면 inf를 반환하고 경고를 출력,
    전부 0이면 nan을 반환함.

    df 기반이라 seed 불필요, 항상 동일한 결과. 표준편차는 ddof=0으로
    통일.
    """
    stats = compute_train_stats(train_env)

    target_df = target_env._df
    tic_col = target_env._tic_column
    features = target_env._features
    tic_order = list(target_env._tic_list)

    per_asset_feature_std = np.zeros((len(tic_order), len(features)))
    all_normalized = []
    for i, tic in enumerate(tic_order):
        tic_data = target_df[target_df[tic_col] == tic][features].to_numpy()
        normalized = (tic_data - stats["mean"][:, i, 0]) / stats["std"][:, i, 0]
        per_asset_feature_std[i, :] = normalized.std(axis=0, ddof=0)
        all_normalized.append(normalized)

    all_normalized = np.concatenate(all_normalized)
    flat_std = per_asset_feature_std.flatten()

    has_zero = np.any(flat_std <= 1e-12)
    has_positive = np.any(flat_std > 1e-12)

    if has_zero and has_positive:
        print(f"[WARNING] 표준편차가 0에 가까운 feature가 "
              f"{int((flat_std <= 1e-12).sum())}개 발견됨 (상수값 의심) "
              f"- max_min_ratio를 inf로 처리함")
        max_min_ratio = float("inf")
    elif not has_positive:
        max_min_ratio = float("nan")
    else:
        max_min_ratio = float(flat_std.max() / flat_std.min())

    return {
        "overall_mean": float(all_normalized.mean()),
        "overall_std": float(all_normalized.std(ddof=0)),
        "per_asset_feature_std": per_asset_feature_std,
        "tic_order": tic_order,
        "features": features,
        "max_min_ratio": max_min_ratio,
    }


if __name__ == "__main__":
    from cryptoagent.envs.adapter import load_env_ready_df
    from cryptoagent.envs.env_portfolio_optimization import PortfolioOptimizationEnv

    def _make_env(split: str) -> PortfolioOptimizationEnv:
        df = load_env_ready_df(split=split)
        return PortfolioOptimizationEnv(
            df=df,
            initial_amount=100_000,
            time_column="date",
            tic_column="tic",
            features=["close", "high", "low"],
            time_window=50,
        )

    print("=== Train-only Standardization 검증 (df 기반, seed 불필요) ===\n")
    train_env = _make_env("train")
    test_env = _make_env("test")

    print("[검증] train_env._df 정규화 확인:")
    print(train_env._df[["close", "high", "low"]].describe().loc[["mean", "std"]])
    print("  -> 값이 1.0 근처면 by_previous_time 정규화가 적용된 것\n")

    # 순서 검증 동작 확인 (정상 케이스)
    stats = compute_train_stats(train_env)
    wrapped_test = TrainStandardizeWrapper(test_env, stats=stats)
    print("[검증] TrainStandardizeWrapper 순서 검증 통과 (정상 케이스)\n")

    result = verify_on_split(train_env, test_env)

    print(f"전체 평균          : {result['overall_mean']:.4f}")
    print(f"전체 표준편차        : {result['overall_std']:.4f}")
    print(f"\n자산별 x feature별 표준편차 (asset x feature):")
    import pandas as pd
    df_std = pd.DataFrame(
        result["per_asset_feature_std"],
        index=result["tic_order"],
        columns=result["features"],
    )
    print(df_std)
    print(f"\n최대/최소 비율(전체): {result['max_min_ratio']:.4f}")
    print("\n[OK] df 기반 계산이라 항상 동일한 값이 재현됨 (seed 무관).")
    
    print("\n[검증] train 자기 자신에 stats 적용 시 std≈1인지 확인 (계산 정합성 증명):")
    train_self_result = verify_on_split(train_env, train_env)
    print(f"  train 전체 표준편차: {train_self_result['overall_std']:.4f} (1.0에 가까워야 정상)")
    print(f"  train max/min 비율: {train_self_result['max_min_ratio']:.4f} (1.0에 가까워야 정상)")