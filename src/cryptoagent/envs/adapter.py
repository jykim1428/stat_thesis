"""features.parquet(Week4 동결 산출물) -> PortfolioOptimizationEnv 입력 포맷 변환.

PortfolioOptimizationEnv가 기대하는 컬럼명(date, tic, close/high/low)과
우리 데이터 컬럼명(Open_time, Symbol, Close/High/Low)이 달라 여기서만 매핑한다.
원본 features.parquet/컬럼명은 바꾸지 않는다 (7월 데이터 동결 원칙).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

# src/cryptoagent/envs/adapter.py 기준 리포 루트로 4단계 위 (envs -> cryptoagent -> src -> root)
REPO_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = REPO_ROOT / "data" / "raw" / "crypto_market_features.db"
TABLE_NAME = "feature_table"

COLUMN_RENAME = {
    "Open_time": "date",
    "Symbol": "tic",
    "Close": "close",
    "High": "high",
    "Low": "low",
    "Open": "open",
    "Volume": "volume",
}


def load_env_ready_df(split: str | None = None) -> pd.DataFrame:
    """Week4에서 동결된 feature_table을 읽어 env 입력 포맷으로 변환.

    split: "train" / "val" / "test" 중 하나. None이면 전체 반환.
    """
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)

    if split is not None:
        df = df[df["split"] == split].copy()

    df = df.rename(columns=COLUMN_RENAME)
    df["date"] = pd.to_datetime(df["date"])
    # PortfolioOptimizationEnv는 order_df=True일 때 자체적으로 date, tic 기준 정렬함
    return df.reset_index(drop=True)


def patch_seed_method(env) -> None:
    """PortfolioOptimizationEnv에 gym.Env.seed()를 붙여준다.

    gym==0.26.2부터 Env 베이스 클래스가 seed()를 제거해서, shimmy.GymV21CompatibilityV0가
    reset(seed=...)를 호출할 때 AttributeError가 난다. PortfolioOptimizationEnv엔 구식
    관례의 _seed()가 남아있으므로 그걸 호출하는 seed()만 얹어준다. SB3 PPO(..., seed=...)로
    재현성을 확보하려면 shimmy로 감싸기 전에 이 함수를 호출해야 한다.
    """
    if not hasattr(env, "seed"):
        env.seed = env._seed
