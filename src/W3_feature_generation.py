"""
============================================================
Week 3 : Technical Indicator Generation
============================================================

[목적]
- 암호화폐 포트폴리오 최적화를 위한 기술적 지표(Technical Indicators)를 생성한다.
- OHLCV 데이터를 기반으로 추세(Trend), 모멘텀(Momentum),
  변동성(Volatility), 거래량(Volume) 관련 Feature를 계산한다.
- 생성된 Feature는 이후 강화학습 모델의 입력(State)으로 활용된다.

[입력 데이터]
- data/crypto_market.db
- 분석 기간 : 2021-01-01 ~ 2025-12-31

[산출물]
- results/feature_dataset.csv

[생성 지표]
- SMA20
- EMA20
- RSI14
- ATR14
- MACD
- MACD_SIGNAL
- MACD_HIST
- OBV

[비고]
- 기술지표만 생성한다.
- 결측치 제거 및 Feature Scaling은 이후 단계에서 수행한다.
"""

import os
import sqlite3

import pandas as pd

from ta.trend import (
    SMAIndicator,
    EMAIndicator,
    MACD
)

from ta.momentum import RSIIndicator

from ta.volatility import AverageTrueRange

from ta.volume import OnBalanceVolumeIndicator

# =====================================================
# 설정
# =====================================================

DB_PATH = "data/crypto_market.db"

SAVE_DIR = "results"

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

# =====================================================
# DB Load
# =====================================================

print("=" * 60)
print("Load Database")
print("=" * 60)

conn = sqlite3.connect(DB_PATH)

df = pd.read_sql(
    """
    SELECT
        Open_time,
        Symbol,
        Open,
        High,
        Low,
        Close,
        Volume
    FROM ohlcv_data
    """,
    conn
)

conn.close()

# =====================================================
# 전처리
# =====================================================

df["Open_time"] = pd.to_datetime(df["Open_time"])

numeric_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

for col in numeric_columns:

    df[col] = df[col].astype(float)

df = df[
    (df["Open_time"] >= "2021-01-01") &
    (df["Open_time"] <= "2025-12-31 23:59:59")
]

print()

print(f"Rows    : {len(df):,}")
print(f"Symbols : {df['Symbol'].nunique()}")

# =====================================================
# Technical Indicator Generation
# =====================================================

feature_list = []

symbols = sorted(
    df["Symbol"].unique()
)

for symbol in symbols:

    print("=" * 60)
    print(symbol)
    print("=" * 60)

    temp = (
        df[
            df["Symbol"] == symbol
        ]
        .sort_values("Open_time")
        .copy()
    )

    # -------------------------------------------------
    # SMA20
    # -------------------------------------------------

    temp["SMA20"] = (
        SMAIndicator(
            close=temp["Close"],
            window=20
        )
        .sma_indicator()
    )

    # -------------------------------------------------
    # EMA20
    # -------------------------------------------------

    temp["EMA20"] = (
        EMAIndicator(
            close=temp["Close"],
            window=20
        )
        .ema_indicator()
    )

    # -------------------------------------------------
    # RSI14
    # -------------------------------------------------

    temp["RSI14"] = (
        RSIIndicator(
            close=temp["Close"],
            window=14
        )
        .rsi()
    )

    # -------------------------------------------------
    # ATR14
    # -------------------------------------------------

    temp["ATR14"] = (
        AverageTrueRange(
            high=temp["High"],
            low=temp["Low"],
            close=temp["Close"],
            window=14
        )
        .average_true_range()
    )
    
    # -------------------------------------------------
    # MACD
    # -------------------------------------------------

    macd = MACD(
        close=temp["Close"],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )

    temp["MACD"] = (
        macd
        .macd()
    )

    temp["MACD_SIGNAL"] = (
        macd
        .macd_signal()
    )

    temp["MACD_HIST"] = (
        macd
        .macd_diff()
    )

    # -------------------------------------------------
    # OBV
    # -------------------------------------------------

    temp["OBV"] = (
        OnBalanceVolumeIndicator(
            close=temp["Close"],
            volume=temp["Volume"]
        )
        .on_balance_volume()
    )

    # -------------------------------------------------
    # 저장
    # -------------------------------------------------

    feature_list.append(temp)

# =====================================================
# Merge
# =====================================================

feature_df = (
    pd.concat(
        feature_list,
        ignore_index=True
    )
    .sort_values(
        ["Open_time", "Symbol"]
    )
    .reset_index(drop=True)
)

# =====================================================
# Feature 확인
# =====================================================

print()

print("=" * 60)
print("Generated Features")
print("=" * 60)

feature_columns = [

    "SMA20",

    "EMA20",

    "RSI14",

    "ATR14",

    "MACD",

    "MACD_SIGNAL",

    "MACD_HIST",

    "OBV"

]

for feature in feature_columns:

    print(feature)

print()

print("=" * 60)
print("Preview")
print("=" * 60)

print(

    feature_df[
        [
            "Open_time",
            "Symbol"
        ] +
        feature_columns
    ].head()

)

print()

print(f"Rows    : {len(feature_df):,}")

print(f"Columns : {len(feature_df.columns)}")

# =====================================================
# CSV 저장
# =====================================================

save_path = os.path.join(
    SAVE_DIR,
    "feature_dataset.csv"
)

feature_df.to_csv(
    save_path,
    index=False,
    encoding="utf-8-sig"
)

# =====================================================
# 저장 결과 확인
# =====================================================

print()

print("=" * 60)
print("Feature Dataset Information")
print("=" * 60)

print(f"Start Date : {feature_df['Open_time'].min()}")
print(f"End Date   : {feature_df['Open_time'].max()}")

print()

print(f"Total Rows    : {len(feature_df):,}")
print(f"Total Columns : {len(feature_df.columns)}")

print()

print("Feature Columns")

feature_columns = [
    "SMA20",
    "EMA20",
    "RSI14",
    "ATR14",
    "MACD",
    "MACD_SIGNAL",
    "MACD_HIST",
    "OBV"
]

for feature in feature_columns:
    print(f" - {feature}")

print()

print("=" * 60)
print("Feature Generation Finished")
print("=" * 60)

print()

print("Saved File")
print(save_path)