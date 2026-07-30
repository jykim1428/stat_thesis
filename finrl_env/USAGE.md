# 이 프로젝트에서 PortfolioOptimizationEnv 쓰는 법

`env_portfolio_optimization.py`는 [AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL)에서
벤더링한 원본 그대로다 (출처는 파일 상단 주석 참고). 아래는 우리 데이터에 맞춘 어댑터 사용법.

## 데이터 연결

`adapter.py`가 Week4에서 동결한 `data/crypto_market.db`의 `feature_table`을
env가 기대하는 컬럼명(`date`, `tic`, `close`/`high`/`low`)으로 변환한다.

```python
import sys
sys.path.insert(0, "finrl_env")
from adapter import load_env_ready_df
from env_portfolio_optimization import PortfolioOptimizationEnv

df = load_env_ready_df(split="train")  # "train" / "val" / "test" / None(전체)

env = PortfolioOptimizationEnv(
    df=df,
    initial_amount=100000,
    time_column="date",
    tic_column="tic",
    features=["close", "high", "low"],
    time_window=50,  # lookback window (하이퍼파라미터, 9월 트랜스포머와 공유)
)
```

`data/crypto_market.db`는 `.gitignore` 처리되어 있으므로, 없으면 먼저
`python Week4_data_freeze.py`로 로컬 재생성.

## 확인된 shape (8종목, time_window=50 기준)

- `observation_space`: `Box(3, 8, 50)` = (feature 3개 [close, high, low], 자산 8종목, lookback 50시간)
- `action_space`: `Box(9,)` = 자산 8종목 비중 + 현금 비중 1개, 합이 1이어야 함(안 맞으면 내부에서 softmax 정규화)
- `env.reset()` → `np.ndarray`, shape `(3, 8, 50)`
- `env.step(action)` → `(obs, reward, done, info)` **4-tuple** (구식 gym API, `truncated` 없음)
  - `reward`는 `ln(V_t / V_{t-1})` (포트폴리오 가치 로그수익률)

9월 트랜스포머/LSTM feature extractor는 `(batch, 3, 8, 50)` → 원하는 시퀀스 형태로
reshape하는 것이 전제. `features` 리스트를 늘리면 (RSI, MACD 등 추가) 첫 번째 차원이 커짐.

## SB3 PPO에 물리기 (8월 2주차용)

`PortfolioOptimizationEnv`는 구식 `gym.Env`를 상속하는데, SB3 2.9.0은 `gymnasium.Env`만
받는다 (`check_env()`가 `AssertionError: must inherit from gymnasium.Env`로 즉시 막음).
`shimmy.GymV21CompatibilityV0`로 감싸면 해결된다:

```python
import shimmy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

gym_env = shimmy.GymV21CompatibilityV0(env=raw_env)
vec_env = DummyVecEnv([lambda: gym_env])

model = PPO("MlpPolicy", vec_env, n_steps=64, batch_size=32)
model.learn(total_timesteps=128)
```

2026-07-30 Python 3.13.7 / SB3 2.9.0 / shimmy 2.0.1 기준으로 `learn()`과 `predict()`까지
확인 완료. `check_env()`는 `env.seed()`가 없다는 별개 경고를 내지만 학습/추론은 막지 않음.

## 알려진 이슈

- `gym==0.26.2`(구식)가 NumPy 2.0에 대해 "unmaintained" 경고를 매 실행마다 띄움.
  동작 자체엔 영향 없음 (2026-07-30 기준 확인).
- `env_portfolio_optimization.py`는 최상단에서 `quantstats`를 강제 import함 (`requirements.txt`에 포함됨).
- `check_env(raw_env)` (shimmy 래핑 전)는 gymnasium 상속 요구 때문에 무조건 실패함 — 정상, shimmy로 감싼 뒤 학습하면 됨.
