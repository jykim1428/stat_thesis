"""
9월 2주차: Train-only Standardization Wrapper (은아 담당)

env의 기존 정규화(by_previous_time, 1차 - 시점 간 비율만 계산, 데이터
누수 없음 검증됨)는 그대로 유지하고, 그 위에 train 구간 통계로 fit한
표준화(2차)를 추가하는 wrapper.


배경
----
관측값이 1.0 근처 좁은 범위에 몰려있고 자산별 변동성 차이가 있어,
신경망 학습 안정성을 위해 자산별 표준화를 시도.

검증 결과 (train stats를 test split에 적용, 실사용 시나리오 기준)
------------------------------------------------------------
    TEST split 전체 평균: -0.0068
    TEST split 전체 표준편차: 0.6842 (1.0 아님)
    자산별 표준편차: 0.552 ~ 0.810 (약 1.47배 차이)

train으로 fit한 통계를 test에 적용하면 완벽히 0/1로 맞춰지지는 않음.
train/test 시기의 실제 시장 변동성 차이(자연스러운 분포 시프트)로
추정됨. 표준화 전(자산별 최대 4.3배 차이, 단일 스냅샷 기준) 대비
격차는 줄었으나(1.47배) 완전히 해소되지는 않는 것으로 확인.

gym.Wrapper를 상속한 이유
-------------------------
일반 클래스로 감싸면 shimmy/SB3가 진짜 gym 환경으로 인식 못 할 위험이
있어 (action_space 등 속성 누락 가능) gym.Wrapper 상속으로 변경, 표준
방식을 따름.

주의 - 아직 팀 논의/검증 전 단계
--------------------------------
이 wrapper는 작성만 해두고 아직 기본 학습 스크립트(train_ppo_mlp.py 등)
에는 연결하지 않음. 적용 범위(MLP 포함 여부)와 시점(지금 vs 3주차)을
팀과 논의 후 결정 예정.

검증 완료 사항
--------------
- shimmy.GymV21CompatibilityV0로 감싸 reset()/step() 정상 동작 확인
- obs shape (3, 8, 50) 정상 유지 (broadcasting 오류 없음)
- train stats를 test split에 적용한 결과는 위 "검증 결과" 참고

사용법 (적용하기로 결정된 경우)
--------------------------------
    train_env = make_env("train")
    stats = compute_train_stats(train_env)   # train으로만 fit
    train_env = TrainStandardizeWrapper(train_env, stats=stats)

    test_env = make_env("test")
    test_env = TrainStandardizeWrapper(test_env, stats=stats)  # 동일 stats 재사용 (transform만)
"""
from __future__ import annotations

import gym
import numpy as np


class TrainStandardizeWrapper(gym.Wrapper):
    """PortfolioOptimizationEnv를 감싸서 obs를 표준화해서 내보낸다.

    gym.Wrapper를 상속해 action_space/observation_space 등을 원본에서
    자동으로 물려받고, SB3/shimmy와의 호환성을 보장한다.

    stats는 반드시 train split으로만 계산해서 val/test에 그대로 적용
    (fit은 train에서만, transform은 어디든 동일 - 데이터 누수 방지).
    stats=None이면 표준화 없이 원본 obs를 그대로 통과시킨다 (비활성 모드).
    """

    def __init__(self, env, stats: dict | None = None):
        super().__init__(env)
        self.stats = stats

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._normalize(obs)

    def step(self, action):
        result = self.env.step(action)
        obs = result[0]
        return (self._normalize(obs), *result[1:])

    def _normalize(self, obs):
        if self.stats is None:
            return obs
        obs = np.array(obs)
        return (obs - self.stats["mean"]) / self.stats["std"]


def compute_train_stats(train_env) -> dict:
    """train split 전체를 순회하며 자산별(feature별) 평균/표준편차를 계산.

    반드시 train_env(split="train")에만 사용할 것 - val/test env로
    호출하면 데이터 누수가 됨.

    mean/std는 obs와 동일하게 (feature, asset, 1) 3차원으로 반환해
    (feature, asset, time) 형태의 obs와 정확히 broadcasting되도록 함.
    """
    obs = train_env.reset()
    all_obs = [np.array(obs)]

    done = False
    while not done:
        action = train_env.action_space.sample()
        step_result = train_env.step(action)
        obs, done = step_result[0], step_result[2]
        all_obs.append(np.array(obs))

    stacked = np.stack(all_obs)  # (steps, feature, asset, time)

    # obs는 (feature, asset, time) 3차원이므로 mean/std도 3차원으로 맞춤
    mean = stacked.mean(axis=(0, 3), keepdims=True)[0]  # (feature, asset, 1)
    std = stacked.std(axis=(0, 3), keepdims=True)[0] + 1e-8

    return {"mean": mean, "std": std}