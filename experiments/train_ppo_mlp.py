"""
8월 2주차: Vanilla PPO(MLP) 학습 + 백테스트 (CryptoAgent)
리포 루트에서 실행: python experiments/train_ppo_mlp.py

파이프라인
----------
1. train split으로 PPO(MlpPolicy) 학습
2. test split으로 학습된 정책을 그리디(deterministic)하게 굴려 백테스트
3. 팀 공용 스펙 DataFrame(date 인덱스 + returns/portfolio_values/weights) 저장

공용 인터페이스 (8월 2~3주차 팀 합의 스펙)
------------------------------------------
index: date (datetime)
columns:
    returns           # 스텝별 포트폴리오 수익률
    portfolio_values  # 스텝별 포트폴리오 가치
    weights           # 스텝별 비중 벡터 (자산 8개 + 현금, 리스트)

env._date_memory / env._portfolio_return_memory / env._asset_memory["final"] /
env._final_weights는 reset() 시 초기값 1개로 시작해 매 step()마다 append되므로
전부 길이가 같다 (PortfolioOptimizationEnv 내부 구현, env_portfolio_optimization.py 참고).

주의: CSV로 저장하면 weights 컬럼은 "[1.0, 0.0, ...]" 형태의 문자열이 된다.
다시 읽을 때는 ast.literal_eval(row["weights"])로 리스트로 복원할 것
(pandas.read_csv만으로는 str로 남는다).

산출물
------
- results/ppo_mlp/backtest_test.csv : 공용 스펙 백테스트 결과 (test split, .gitignore 처리됨)
- results/ppo_mlp/ppo_mlp.zip       : 학습된 SB3 모델 (.gitignore 처리됨)
- runs/{run.id}/                    : SB3 tensorboard 로그 (.gitignore 처리됨)
- W&B run (project=cryptoagent-ppo) : 학습 로그(loss/entropy/approx_kl 등),
  train()의 tensorboard_log/callback 파라미터로 연결됨.
  로컬 wandb/ 폴더도 생성되나 .gitignore 처리됨 - 결과는 W&B 웹사이트에서 확인

참고: make_env / backtest / sanity_check는 train_ppo_transformer.py와 공용으로
쓰기 위해 cryptoagent.training.common으로 추출됨 (8월 2주차 이월 작업).
"""

from __future__ import annotations

import os
from functools import partial

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import shimmy

from cryptoagent.training.common import backtest, make_env as _make_env, sanity_check

# ── 설정 ─────────────────────────────────────────────
TIME_WINDOW = 50
FEATURES = ["close", "high", "low"]
INITIAL_AMOUNT = 100_000

TOTAL_TIMESTEPS = 50_000  # sanity-check용 기본값. 본 실험 규모는 8월 4주차 이후 조정
SEED = 42

RESULTS_DIR = "results/ppo_mlp"
MODEL_PATH = os.path.join(RESULTS_DIR, "ppo_mlp.zip")
BACKTEST_PATH = os.path.join(RESULTS_DIR, "backtest_test.csv")
# ─────────────────────────────────────────────────────


# make_env/backtest/sanity_check는 cryptoagent.training.common 공용 모듈에서 가져옴
# (train_ppo_transformer.py, train_ppo_lstm.py와 동일 패턴). 이 스크립트 설정값만
# 미리 바인딩해서 make_env(split) 형태로 세 스크립트가 동일한 인터페이스를 갖게 함
# - 오케스트레이션에서 모델 종류와 무관하게 make_env("train")로 호출 가능.
make_env = partial(
    _make_env,
    features=FEATURES,
    initial_amount=INITIAL_AMOUNT,
    time_window=TIME_WINDOW,
)


def train(
    train_env,
    tensorboard_log: str | None = None,
    callback=None,
) -> PPO:
    """train split으로 PPO(MlpPolicy) 학습.

    B(W&B) 담당자는 여기 vec_env를 만든 뒤 PPO(...) 생성자에
    tensorboard_log= 를 지정하고 model.learn(..., callback=WandbCallback())로
    콜백만 얹으면 된다 - 이 함수 시그니처/구조는 유지.

    (8월 2주차 은아 구현) tensorboard_log, callback 파라미터를 추가해
    위 안내대로 연결. 원본 로직(vec_env 생성, PPO 하이퍼파라미터)은 미변경.
    """
    gym_env = shimmy.GymV21CompatibilityV0(env=train_env)
    vec_env = DummyVecEnv([lambda: gym_env])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=SEED,
        tensorboard_log=tensorboard_log,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    return model


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    run = wandb.init(
        entity="choieuna0711-student",
        project="cryptoagent-ppo",
        name="ppo_mlp_sanity_baseline",
        config={
            "total_timesteps": TOTAL_TIMESTEPS,
            "seed": SEED,
            "time_window": TIME_WINDOW,
            "features": FEATURES,
            "initial_amount": INITIAL_AMOUNT,
        },
        sync_tensorboard=True,
    )

    print("=== [1/3] train split으로 PPO(MLP) 학습 ===")
    train_env = make_env("train")
    model = train(
        train_env,
        tensorboard_log=f"runs/{run.id}",
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"{RESULTS_DIR}/wandb_models/{run.id}",
        ),
    )
    model.save(MODEL_PATH)
    print(f"모델 저장: {MODEL_PATH}")

    print("\n=== [2/3] test split 백테스트 ===")
    test_env = make_env("test")
    backtest_df = backtest(model, test_env)
    backtest_df.to_csv(BACKTEST_PATH)
    print(f"백테스트 결과 저장: {BACKTEST_PATH}  shape={backtest_df.shape}")

    print("\n=== [3/3] Sanity Check ===")
    sanity_check(backtest_df)
    run.finish()


if __name__ == "__main__":
    main()
