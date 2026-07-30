"""features.parquet(Week4 동결 산출물) -> PortfolioOptimizationEnv 입력 포맷 변환.

PortfolioOptimizationEnv가 기대하는 컬럼명(date, tic, close/high/low)과
우리 데이터 컬럼명(Open_time, Symbol, Close/High/Low)이 달라 여기서만 매핑한다.
원본 features.parquet/컬럼명은 바꾸지 않는다 (7월 데이터 동결 원칙).
"""

from __future__ import annotations

import sqlite3

import pandas as pd

DB_PATH = "data/crypto_market.db"
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
