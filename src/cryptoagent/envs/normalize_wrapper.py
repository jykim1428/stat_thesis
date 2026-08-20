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
time_window번까지 중복으로 통계 계산에 들어가는 구조적 오류가 있었음
(seed 고정 여부와 무관하게 항상 발생하는 문제 - 처음에는 seed
미고정이 원인이라고 잘못 판단했었으나, 실제로는 seed와 무관하게
env.step() 방식 자체가 중복 카운트 문제를 안고 있었음).

[해결] train_env._df(env가 생성 시점에 이미 by_previous_time까지
전처리를 마친 데이터, _preprocess_data()/_normalize_dataframe()에서
확인)를 직접 읽어, 각 시점을 정확히 1번씩만 계산하도록 전환. 이
방식은 action/seed 개념 자체가 없어 항상 100% 동일한 값이 재현됨.

[검증 - private attribute 의존성] train_env._df 사용이 안전한지
__main__ 블록에서 실행 시점마다 실제 mean/std 값을 출력해 확인
(mean≈1.0001, std≈0.013~0.014로 by_previous_time 정규화가 적용된
값임을 재확인).

[ddof 통일] pandas.std() 기본값(ddof=1)이 기존 numpy 기반 계산
(ddof=0)과 다름을 발견, std(ddof=0) 명시로 통일.

[팀 리뷰 반영 - 2차] 아래 3가지 추가 수정:
1. observation_space를 표준화 후 값 범위에 맞게 명시적으로 갱신
   (SB3 환경 검사/정책 입력 검증 이슈 방지)
2. reset()이 gym(구식, obs만 반환)과 gymnasium(신식, (obs,info) 튜플
   반환)을 모두 안전하게 처리하도록 수정 (지금까지는 PortfolioOptimizationEnv가
   구식 방식만 써서 우연히 문제가 안 드러났었음)
3. verify_on_split()의 per_asset_std를 (asset, feature)로 나눠서
   계산 - 기존에는 close/high/low가 뭉쳐진 값이었음. max_min_ratio도
   0으로 나누기 방어 추가
   -> 재계산 결과 최대/최소 비율이 1.59배가 아니라 2.02배로 확인됨.
      뭉쳐진 값은 feature 간 극단치가 상쇄되어 실제보다 낮게
      나타났던 것.

검증 결과 (train stats를 test split에 적용)
-------------------------------------------
    python src/cryptoagent/envs/normalize_wrapper.py 실행 시 seed
    없이 항상 동일한 값 출력됨. 자산별 x feature별 표준편차 표를
    출력하도록 개선(아래 __main__ 참고).

gym.Wrapper를 상속한 이유 (TrainStandardizeWrapper)
----------------------------------------------------
일반 클래스로 감싸면 shimmy/SB3가 진짜 gym 환경으로 인식 못 할 위험이
있어 (action_space 등 속성 누락 가능) gym.Wrapper 상속으로 변경.

feature/자산/lookback 축 정합성 확인 필요 (팀 리뷰 지적)
-------------------------------------------------------
observation의 실제 축 순서(feature, asset, time)가 features/_tic_list
순서와 정확히 일치하는지는 env_portfolio_optimization.py의
_get_state_and_info_from_time_index()에서 `for tic in self._tic_list`
순서대로 state를 쌓는 것으로 확인됨 - compute_train_stats의 tic_order
= list(train_env._tic_list)로 동일 순서를 사용하도록 맞춰져 있음.

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
    python src/cryptoagent/envs/normalize_wrapper.py
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
    """

    def __init__(self, env, stats: dict | None = None):
        super().__init__(env)
        # SB3/PyTorch 입력 일관성을 위해 float32로 통일 (팀 리뷰 지적)
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
        # len==2까지 확인해 더 엄격하게 판별 (팀 리뷰 지적)
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
        # float32로 캐스팅해 observation_space.dtype과 일치시킴 (팀 리뷰 지적)
        return ((np.asarray(obs, dtype=np.float32) - self.stats["mean"])
                 / self.stats["std"]).astype(np.float32)


def compute_train_stats(train_env) -> dict:
    """... (docstring 동일, float32 저장 부분만 추가 설명) ...

    mean/std는 float32로 저장 - SB3/PyTorch 입력 dtype(float32)과
    통일하기 위함 (팀 리뷰 지적).
    """
    df = train_env._df
    tic_col = train_env._tic_column
    features = train_env._features
    tic_order = list(train_env._tic_list)

    mean_per_tic = df.groupby(tic_col)[features].mean().loc[tic_order]
    std_per_tic = df.groupby(tic_col)[features].std(ddof=0).loc[tic_order]

    mean = mean_per_tic.to_numpy().T[:, :, np.newaxis].astype(np.float32)
    std = (std_per_tic.to_numpy().T[:, :, np.newaxis] + 1e-8).astype(np.float32)

    return {"mean": mean, "std": std}


def verify_on_split(train_env, target_env) -> dict:
    """... (docstring 동일) ...

    max_min_ratio: 표준편차가 0인 feature를 조용히 제외하지 않음
    (팀 리뷰 지적 - 0인 항목을 숨기면 "상수 feature 존재"라는 실제
    문제를 못 보게 됨). 0이 하나라도 있고 양수 std가 존재하면
    inf를 반환하고 경고를 출력, 전부 0이면 nan을 반환함.
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
        print(f"[WARNING] 표준편차가 0에 가까운 feature가 {int((flat_std <= 1e-12).sum())}개 "
              f"발견됨 (상수값 의심) - max_min_ratio를 inf로 처리함")
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