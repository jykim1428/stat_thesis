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


def load_env_ready_df_by_date(
    start: str,
    end: str,
    warmup_hours: int = 0,
) -> pd.DataFrame:
    """날짜 범위(start~end, 양끝 포함)로 feature_table을 읽어 env 입력 포맷으로 변환.

    walk-forward 평가에서 fold 경계가 split 라벨과 다르게 잡힐 때(예: OOS
    구간 직전 N시간을 observation 워밍업으로 포함해야 하는 경우) 사용한다.

    warmup_hours > 0이면 start보다 warmup_hours시간 앞선 시점부터 데이터를
    포함해서 반환한다 (docs/walk_forward_design.md의 "TIME_WINDOW=50 워밍업
    처리" 참고 - PortfolioOptimizationEnv가 첫 관측을 만드는 데 time_window개
    과거 시점이 필요하므로, OOS 구간 시작부터만 로드하면 그 구간의 처음
    time_window개 시점이 관측 워밍업으로 소모되어 거래/수익률 기록에서
    누락된다). 워밍업으로 포함된 행 자체는 미래 정보가 아니라 과거 데이터이므로
    누수가 아니다 - 실제 거래/reward 기록을 OOS 시작 시점부터로 제한하는 건
    이 함수를 호출하는 쪽(백테스트 결과에서 워밍업 구간을 잘라내는 로직)의
    책임이다.

    split 컬럼은 이 경로에서 쓰지 않으므로 반환 DataFrame에 남겨두되 값의
    의미를 보장하지 않는다 (여러 split에 걸친 날짜 범위를 요청할 수 있음).
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if warmup_hours > 0:
        start_ts = start_ts - pd.Timedelta(hours=warmup_hours)

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)

    df = df.rename(columns=COLUMN_RENAME)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()

    if df.empty:
        raise ValueError(f"날짜 범위 [{start_ts}, {end_ts}]에 해당하는 데이터가 없음")

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
