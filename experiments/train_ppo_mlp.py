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
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import shimmy
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from cryptoagent.envs.adapter import load_env_ready_df, patch_seed_method
from cryptoagent.envs.env_portfolio_optimization import PortfolioOptimizationEnv
from cryptoagent.training.overfitting_monitor import make_eval_callback

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
BACKTEST_PATH = os.path.join(RESULTS_DIR, "backtest_test.csv")
# ─────────────────────────────────────────────────────


def make_env(split: str) -> PortfolioOptimizationEnv:
    df = load_env_ready_df(split=split)
    env = PortfolioOptimizationEnv(
        df=df,
        initial_amount=INITIAL_AMOUNT,
        time_column="date",
        tic_column="tic",
        features=FEATURES,
        time_window=TIME_WINDOW,
    )
    patch_seed_method(env)
    return env


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
    train_env: PortfolioOptimizationEnv,
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


def backtest(model: PPO, eval_env: PortfolioOptimizationEnv) -> pd.DataFrame:
    gym_env = shimmy.GymV21CompatibilityV0(env=eval_env)

    obs, _ = gym_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = gym_env.step(action)
        done = terminated or truncated

    result = pd.DataFrame(
        {
            "date": eval_env._date_memory,
            "returns": eval_env._portfolio_return_memory,
            "portfolio_values": eval_env._asset_memory["final"],
            "weights": [w.tolist() for w in eval_env._final_weights],
        }
    )
    result["date"] = pd.to_datetime(result["date"])
    result = result.set_index("date")
    return result


def sanity_check(backtest_df: pd.DataFrame) -> None:
    # weights 벡터 안에 NaN이 섞이면 sum()이 NaN이 되고, pandas.Series.max()의
    # 기본 skipna=True 때문에 그 행이 통계에서 조용히 빠져 max_dev가 정상값으로
    # 나온다 - 벡터 원소 단위로 먼저 finite 여부를 확인해야 이 케이스를 놓치지 않는다.
    assert backtest_df["weights"].apply(lambda w: np.isfinite(w).all()).all(), "weights에 NaN 또는 inf 존재"

    weight_sums = backtest_df["weights"].apply(sum)
    max_dev = (weight_sums - 1.0).abs().max()
    assert max_dev < 1e-3, f"비중 합이 1에서 {max_dev}만큼 벗어남"

    assert np.isfinite(backtest_df["returns"]).all(), "returns에 NaN 또는 inf 존재"
    assert not backtest_df["portfolio_values"].isna().any(), "portfolio_values에 NaN 존재"
    assert np.isfinite(backtest_df["portfolio_values"]).all(), "portfolio_values에 inf 존재"

    print(f"[sanity_check] OK - 비중 합 최대 편차: {max_dev:.2e}")
    print(f"[sanity_check] OK - 최종 포트폴리오 가치: {backtest_df['portfolio_values'].iloc[-1]:,.2f}")


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
            "eval_freq": EVAL_FREQ,
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