"""
9월 3주차: 오버피팅 모니터링 (train↑ val↓ 확인)

SB3의 EvalCallback을 이용해 학습 도중 주기적으로 val split 성과를
측정하고 W&B에 자동 기록한다. rollout/ep_rew_mean(train)과
eval/mean_reward(val)를 W&B 대시보드에서 나란히 비교하면, train은
계속 좋아지는데 val이 어느 시점부터 정체/악화되는지 확인할 수 있다.

주의 - shimmy를 여기서 직접 감싸지 않음 (코덱스 리뷰 반영)
-------------------------------------------------------------
DummyVecEnv([make_val_env])만 쓰고 shimmy.GymV21CompatibilityV0로 감싸지
않는다. SB3 2.x의 DummyVecEnv는 구형 Gym 환경도 내부적으로 처리 가능하며,
여기서 수동으로 한 번 더 감싸면 이중 변환/API 불일치가 생길 수 있다.
(train() 쪽에서 이미 shimmy로 감싼 것과는 별개 - 거기는 gym_env를
DummyVecEnv에 넣기 전에 필요해서 쓴 것이고, 여기 eval 쪽은 make_val_env
자체가 이미 원본 gym env를 반환하므로 DummyVecEnv가 알아서 처리함)

env를 미리 만든 객체가 아니라 "만드는 함수(factory)"로 받는 이유
-------------------------------------------------------------------
이미 생성된 env 객체를 lambda로 감싸 반환하면 reset 상태 관리가
꼬이기 쉬움. make_val_env()를 호출할 때마다 새로 env를 만들도록 강제.

Frozen train stats 원칙 (중요 - 데이터 누수 방지)
----------------------------------------------------
make_val_env가 표준화를 쓴다면, 반드시 train에서 미리 계산해둔
train_stats를 "그대로" 넘겨서 감싸야 한다. val에서 새로 fit하거나,
reset()/step() 도중 통계를 갱신하는 방식은 전부 데이터 누수다.

    피해야 할 것:
        TrainStandardizeWrapper(make_env("val"))  # val로 새로 fit하면 누수

    올바른 것:
        train_stats = compute_train_stats(raw_train_env)  # train에서 1번만
        def make_val_env():
            return TrainStandardizeWrapper(make_env("val"), stats=train_stats)

best_model 체크포인트의 의미 (중요 - 최종 평가 기준과 연결)
----------------------------------------------------------------
best_model_save_path에 저장되는 모델은 "eval/mean_reward가 최고였던
시점"의 체크포인트다. 프로젝트의 최종 비교 지표는 Sharpe인데, reward
최고 지점이 Sharpe 최고 지점과 항상 일치하지 않을 수 있다. 지금은
1차로 reward 기준 best를 보되, 최종 모델 선택 시에는 이 체크포인트의
Sharpe/MDD/turnover도 별도로 확인해서 기준 불일치가 없는지 점검할 것.

사용법
------
    def make_val_env():
        env = make_env("val")
        if train_stats is not None:
            env = TrainStandardizeWrapper(env, stats=train_stats)
        return env

    eval_callback = make_eval_callback(make_val_env, results_dir)
    model.learn(total_timesteps=..., callback=CallbackList([wandb_callback, eval_callback]))
"""
from __future__ import annotations

import os
from typing import Callable

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv


def make_eval_callback(
    make_val_env: Callable[[], object],
    results_dir: str,
    eval_freq: int = 2048,
    n_eval_episodes: int = 1,
    n_envs: int = 1,
) -> EvalCallback:
    """make_val_env()로 val 환경을 생성해 학습 도중 주기적 평가를 수행.

    make_val_env: 호출할 때마다 새 val env를 만들어 반환하는 함수(factory).
        표준화를 쓴다면 이 함수 안에서 frozen train_stats로 감싸야 함
        (모듈 docstring 참고).

    eval_freq: 몇 (콜백 관점의) step마다 평가할지. n_envs=1이면 PPO의
        rollout 크기(기본 2048)와 맞춰야 매 iteration 직후 평가되어
        해석이 쉬움. 병렬 환경(n_envs>1)을 쓴다면
        eval_freq = max(2048 // n_envs, 1)로 조정할 것 - 이 함수는
        n_envs를 받아 자동으로 보정한다.

    n_eval_episodes: val split이 하나의 연속된 국면(예: 2023 상반기)
        이라 사실상 에피소드가 1개뿐이므로 기본값 1. 이 경우
        eval/std_reward가 0으로 나오는 것은 정상.
    """
    val_vec_env = DummyVecEnv([make_val_env])

    adjusted_eval_freq = max(eval_freq // n_envs, 1)

    return EvalCallback(
        val_vec_env,
        eval_freq=adjusted_eval_freq,
        n_eval_episodes=n_eval_episodes,
        log_path=os.path.join(results_dir, "eval_logs"),
        best_model_save_path=os.path.join(results_dir, "best_model"),
        deterministic=True,
        verbose=1,
    )