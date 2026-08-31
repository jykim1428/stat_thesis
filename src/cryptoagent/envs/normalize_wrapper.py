"""
9월 2주차: Train-only Standardization Wrapper (은아 담당)

env의 기존 정규화(by_previous_time, 1차 - 시점 간 비율만 계산, 데이터
누수 없음 검증됨)는 그대로 유지하고, 그 위에 train 구간 통계로 fit한
표준화(2차)를 추가하는 wrapper.

배경
----
관측값이 1.0 근처 좁은 범위에 몰려있고 자산별 변동성 차이가 있어,
신경망 학습 안정성을 위해 자산별 표준화를 시도.

핵심 기능
---------
- compute_train_stats(): train split 데이터에서 자산별/feature별
  평균·표준편차를 계산 (train으로만 fit, 데이터 누수 없음)
- TrainStandardizeWrapper: 위 통계로 env의 관측값을 표준화해서 내보냄.
  clip 파라미터로 이상치 클리핑 여부 선택 가능 (기본 (-5.0, 5.0))
- verify_on_split(): train stats를 다른 split(val/test)에 적용했을 때
  실제로 어떤 분포가 나오는지 검증하는 함수
- normalize_with_stats(): 위 두 곳이 공유하는 계산 공식 - wrapper와
  verify 함수가 항상 같은 로직을 쓰도록 통일함

안전장치
--------
- train stats의 자산/feature 순서가 적용 대상 env와 다르면 ValueError로
  즉시 실패 (조용히 잘못된 통계가 적용되는 것 방지)
- gym.Wrapper를 상속해 SB3/shimmy와의 호환성 확보

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


def normalize_with_stats(
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    clip: tuple[float, float] | None,
) -> np.ndarray:
    """(x - mean) / std 후 선택적으로 clip을 적용하는 공용 순수 함수.

    TrainStandardizeWrapper._normalize()와 verify_on_split()이 각각
    독립적으로 이 계산을 중복 구현하면서, 한쪽만 수정되고 다른 쪽이
    누락되는 문제(clip 반영 누락)가 실제로 발생했음. 계산식을 이
    함수 하나로 통일해 두 곳이 항상 같은 로직을 쓰도록 정리함.

    values, mean, std는 NumPy broadcasting이 되는 어떤 shape이든
    받을 수 있음 (예: wrapper에서는 (feature, asset, time) 전체 obs,
    verify_on_split에서는 (time, feature) 형태의 자산 하나만 등).
    """
    result = (values - mean) / std
    return np.clip(result, clip[0], clip[1]) if clip is not None else result


class TrainStandardizeWrapper(gym.Wrapper):
    """... (기존 docstring 유지) ...

    PortfolioOptimizationEnv(return_last_action=True)는 observation을
    {"state": (feature, asset, time), "last_action": (...)} 형태의 Dict로
    반환한다 (return_last_action=False 기본값이면 Box). 이 wrapper는 두
    경우 모두 지원한다 - Dict일 때는 "state" 원소만 표준화하고
    "last_action"은 그대로 통과시키며, observation_space도 Dict 구조를
    유지한 채 "state"의 space만 Box(-inf, inf)로 갱신한다.
    """

    def __init__(self, env, stats: dict | None = None, clip: tuple[float, float] | None = (-5.0, 5.0)):
        super().__init__(env)

        self._is_dict_obs = isinstance(env.observation_space, spaces.Dict)

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

        if stats is not None:
            self.stats = {
                "mean": stats["mean"].astype(np.float32),
                "std": stats["std"].astype(np.float32),
            }
        else:
            self.stats = None

        self.clip = clip

        if stats is not None:
            if self._is_dict_obs:
                state_space = env.observation_space["state"]
                self.observation_space = spaces.Dict({
                    "state": spaces.Box(
                        low=-np.inf, high=np.inf,
                        shape=state_space.shape,
                        dtype=np.float32,
                    ),
                    "last_action": env.observation_space["last_action"],
                })
            else:
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=env.observation_space.shape,
                    dtype=np.float32,
                )

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
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
        if self._is_dict_obs:
            state = np.asarray(obs["state"], dtype=np.float32)
            normalized_state = normalize_with_stats(state, self.stats["mean"], self.stats["std"], self.clip)
            return {"state": normalized_state.astype(np.float32), "last_action": obs["last_action"]}
        obs = np.asarray(obs, dtype=np.float32)
        normalized = normalize_with_stats(obs, self.stats["mean"], self.stats["std"], self.clip)
        return normalized.astype(np.float32)


def compute_train_stats(train_env) -> dict:
    """train split의 자산×feature별 mean/std를 계산.

    (코덱스 리뷰 반영) mean/std의 NaN/inf, train std가 tolerance 이하인
    축(상수 feature 의심)을 여기서 직접 검증. wrapper 생성이 아니라 stats
    계산 시점에 막아야 "wrapper는 생성됐지만 내부 통계가 이미 잘못된"
    상황을 방지할 수 있음.
    """
    df = train_env._df
    tic_col = train_env._tic_column
    features = train_env._features
    tic_order = list(train_env._tic_list)

    mean_per_tic = df.groupby(tic_col)[features].mean().loc[tic_order]
    std_per_tic = df.groupby(tic_col)[features].std(ddof=0).loc[tic_order]

    mean = mean_per_tic.to_numpy().T[:, :, np.newaxis].astype(np.float32)
    raw_std = std_per_tic.to_numpy().T[:, :, np.newaxis].astype(np.float32)

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


def verify_on_split(train_env, target_env, clip: tuple[float, float] | None = (-5.0, 5.0)) -> dict:
    """train stats를 target_env(test/val)의 데이터에 적용했을 때의
    검증 통계를 계산.

    normalize_with_stats()를 wrapper와 동일하게 재사용 (계산 로직을 공용 함수로 추출해 중복/누락 위험을 원천 제거).
    float32로 캐스팅해 실제 학습 입력과 동일한 dtype 경로로 검증.

    ticker 순서는 stats["tic_order"]와 target_env의 실제 tic_order가
    일치하는지 확인 후 사용 (순서가 다르면 조용히 잘못된 stats가 적용될 위험 방지).

    clip=None이면 clip 없는 순수 z-score 통계, clip=(-5.0, 5.0)이
    기본값으로 실제 wrapper와 동일한 값을 재현함.
    """
    stats = compute_train_stats(train_env)

    target_df = target_env._df
    tic_col = target_env._tic_column
    features = target_env._features
    tic_order = list(target_env._tic_list)

    if stats["tic_order"] != tic_order:
        raise ValueError(
            f"train과 target의 자산 순서가 다릅니다: "
            f"train={stats['tic_order']}, target={tic_order}"
        )
    if stats["features"] != features:
        raise ValueError(
            f"train과 target의 feature 순서가 다릅니다: "
            f"train={stats['features']}, target={features}"
        )

    per_asset_feature_std = np.zeros((len(tic_order), len(features)))
    all_normalized = []
    for i, tic in enumerate(tic_order):
        tic_data = target_df[target_df[tic_col] == tic][features].to_numpy(dtype=np.float32)
        if not np.all(np.isfinite(tic_data)):
            raise ValueError(f"target_env({tic})의 데이터에 NaN/inf가 포함되어 있음 - 원본 데이터 확인 필요")
        mean_i = stats["mean"][:, i, 0]
        std_i = stats["std"][:, i, 0]
        normalized = normalize_with_stats(tic_data, mean_i, std_i, clip)
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
        "clip_applied": clip is not None,
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
    
    print("=== clip 미적용 (순수 z-score) ===")
    result_no_clip = verify_on_split(train_env, test_env, clip=None)
    print(f"전체 표준편차: {result_no_clip['overall_std']:.4f}")
    print(f"최대/최소 비율: {result_no_clip['max_min_ratio']:.4f}")

    print("\n=== clip 적용 (실제 정책 입력과 동일) ===")
    result_clipped = verify_on_split(train_env, test_env, clip=(-5.0, 5.0))
    print(f"전체 표준편차: {result_clipped['overall_std']:.4f}")
    print(f"최대/최소 비율: {result_clipped['max_min_ratio']:.4f}")

    print("\n[검증] train 자기 자신에 stats 적용 시 std≈1인지 확인 (계산 정합성 증명):")
    train_self_result = verify_on_split(train_env, train_env, clip=None)
    print(f"  train 전체 표준편차: {train_self_result['overall_std']:.4f} (1.0에 가까워야 정상)")
    print(f"  train max/min 비율: {train_self_result['max_min_ratio']:.4f} (1.0에 가까워야 정상)")
    