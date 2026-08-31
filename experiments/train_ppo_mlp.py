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

주의: CSV로 저장하면 weights 컬럼은 "[1.0, 0.0, ...]" 형태의 문자열이 된다.
다시 읽을 때는 ast.literal_eval(row["weights"])로 리스트로 복원할 것
(pandas.read_csv만으로는 str로 남는다).

산출물
------
- results/ppo_mlp/backtest_test.csv : 공용 스펙 백테스트 결과 (test split, .gitignore 처리됨)
- results/ppo_mlp/ppo_mlp.zip       : 학습된 SB3 모델 (.gitignore 처리됨)
- W&B run (project=cryptoagent-ppo) : 학습 로그(loss/entropy/approx_kl 등),
  train()의 tensorboard_log/callback 파라미터로 연결됨.
  로컬 wandb/ 폴더도 생성되나 .gitignore 처리됨 - 결과는 W&B 웹사이트에서 확인

오버피팅 모니터링
------------------
EvalCallback으로 학습 도중 val split 성과를 주기적으로 측정해 W&B에
기록함 (src/cryptoagent/training/overfitting_monitor.py).

eval_freq=10240(=5*rollout 크기)으로 설정 - 2048마다 평가하면 val
에피소드 길이가 길어 평가 비용이 학습 비용 대비 과도해짐을 트랜스포머
스크립트에서 스모크 테스트로 확인, 10,240으로 조정함 (50,000 스텝
기준 약 5회 평가).

train episode가 PPO rollout(2048)보다 길어 rollout/ep_rew_mean과
eval/mean_reward를 직접 비교하기는 어려움. 대신 학습 안정성
(approx_kl, explained_variance 등)과 일반화(eval/mean_reward, val
Sharpe/MDD/turnover)를 각각 관찰하는 것으로 목표를 재정의함. 평가
지점이 적을 때 단기 등락만으로 오버피팅을 단정하지 않고, 충분한
학습 구간에 걸친 추세로 판단할 것.

val 환경도 train과 동일한 API 변환 경로(GymV21CompatibilityV0 ->
Monitor)를 거치도록 함 - SB3의 "Monitor wrapper 없음" 경고 제거 및
경로 일관성 확보.

train_ppo_transformer.py에서 먼저 검증됨: EvalCallback 추가 전/후로
동일 seed 실행 시 최종 결과값이 완전히 동일함을 확인 - 모니터링
추가가 실제 학습 결과에 영향을 주지 않고 관찰만 한다는 것을 확인함.
참고: make_env / backtest / sanity_check는 train_ppo_transformer.py와 공용으로
쓰기 위해 cryptoagent.training.common으로 추출됨 (8월 2주차 이월 작업).

--eval-split (코덱스 리뷰 반영, "기존 학습 엔트리포인트가 자동으로 test를
본다"는 지적)
--------------------------------------------------------------------------
과거에는 항상 test split으로 백테스트했는데, 이러면 하이퍼파라미터 조정이나
버그 수정 후 재학습·재실행할 때마다 test 성능을 반복 관찰하게 되어
"test는 최종 1회 평가에만 쓴다"는 팀 원칙(test-set contamination 방지)이
깨진다. 기본값을 val로 바꾸고, test는 --eval-split test로 명시했을 때만
사용하도록 해서 실수로 test를 보는 것을 막는다.
"""

from __future__ import annotations

import argparse
import os
from functools import partial

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import shimmy

from cryptoagent.training.overfitting_monitor import make_eval_callback
from cryptoagent.training.common import backtest, make_env as _make_env, sanity_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-split", choices=("val", "test"), default="val",
        help="백테스트에 쓸 split. test는 최종 평가 1회에만 명시적으로 사용할 것.",
    )
    return parser.parse_args()

# ── 설정 ─────────────────────────────────────────────
TIME_WINDOW = 50
FEATURES = ["close", "high", "low"]
INITIAL_AMOUNT = 100_000

TOTAL_TIMESTEPS = 50_000  # sanity-check용 기본값. 본 실험 규모는 8월 4주차 이후 조정
SEED = 42

EVAL_FREQ = 10_240  # 5 * rollout(2048). val 에피소드가 길어 2048 주기는 비효율적
                     # (트랜스포머 스모크 테스트로 확인)

RESULTS_DIR = "results/ppo_mlp"
MODEL_PATH = os.path.join(RESULTS_DIR, "ppo_mlp.zip")
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


def make_val_env():
    """오버피팅 모니터링용 val env factory.

    train과 동일한 API 변환 경로(GymV21CompatibilityV0 -> Monitor)를
    거치도록 함 - SB3의 "Monitor wrapper 없음" 경고 제거 및 train/val
    경로 일관성 확보.
    """
    env = make_env("val")
    gym_env = shimmy.GymV21CompatibilityV0(env=env)
    return Monitor(gym_env)


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
    위 안내대로 연결.
    """
    gym_env = shimmy.GymV21CompatibilityV0(env=train_env)
    vec_env = DummyVecEnv([lambda: Monitor(gym_env)])

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
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    backtest_path = os.path.join(RESULTS_DIR, f"backtest_{args.eval_split}.csv")

    if args.eval_split == "test":
        print(
            "[경고] --eval-split test로 실행합니다. test는 최종 평가 1회에만 "
            "사용해야 합니다 (반복 실행 시 test-set contamination 위험)."
        )

    run = wandb.init(
        entity="choieuna0711-student",
        project="cryptoagent-ppo",
        name=f"ppo_mlp_{args.eval_split}",
        config={
            "total_timesteps": TOTAL_TIMESTEPS,
            "seed": SEED,
            "time_window": TIME_WINDOW,
            "features": FEATURES,
            "initial_amount": INITIAL_AMOUNT,
            "eval_freq": EVAL_FREQ,
            "eval_split": args.eval_split,
        },
        sync_tensorboard=True,
    )

    print("=== [1/3] train split으로 PPO(MLP) 학습 ===")
    train_env = make_env("train")

    eval_callback = make_eval_callback(make_val_env, RESULTS_DIR, eval_freq=EVAL_FREQ)

    model = train(
        train_env,
        tensorboard_log=f"runs/{run.id}",
        callback=CallbackList([
            WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"{RESULTS_DIR}/wandb_models/{run.id}",
            ),
            eval_callback,
        ]),
    )
    model.save(MODEL_PATH)
    print(f"모델 저장: {MODEL_PATH}")

    print(f"\n=== [2/3] {args.eval_split} split 백테스트 ===")
    eval_env = make_env(args.eval_split)
    backtest_df = backtest(model, eval_env)
    backtest_df.to_csv(backtest_path)
    print(f"백테스트 결과 저장: {backtest_path}  shape={backtest_df.shape}")

    print("\n=== [3/3] Sanity Check ===")
    sanity_check(backtest_df)
    run.finish()


if __name__ == "__main__":
    main()