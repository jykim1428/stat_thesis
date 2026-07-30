"""
============================================================
Week 2 : 수익률 분포 및 꼬리위험(Tail Risk) 분석
============================================================

[목적]
- 암호화폐 자산별 시간별 수익률 분포 특성을 분석한다.
- 왜도(Skewness)와 첨도(Kurtosis)를 계산하여
  암호화폐 수익률의 비대칭성과 Heavy-tail 특성을 확인한다.

[입력 데이터]
- data/crypto_market.db

- 분석 기간:
  2021-01-01 ~ 2025-12-31

- 대상 자산:
  BTCUSDT
  ETHUSDT
  BNBUSDT
  SOLUSDT
  XRPUSDT
  ADAUSDT
  DOGEUSDT
  AVAXUSDT


[산출물]
- results/tail_statistics.csv


[활용]
- 암호화폐 시장의 비정규성과 극단적 변동성을 확인한다.
- 강화학습 기반 포트폴리오 최적화 모델 적용 필요성의 근거 자료로 활용한다.
"""

import os
import sqlite3

import pandas as pd


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
        Close
    FROM ohlcv_data
    """,
    conn
)


conn.close()


# =====================================================
# Preprocessing
# =====================================================

df["Open_time"] = pd.to_datetime(
    df["Open_time"]
)


df["Close"] = df["Close"].astype(float)


df = df[
    (df["Open_time"] >= "2021-01-01") &
    (df["Open_time"] <= "2025-12-31 23:59:59")
]


# =====================================================
# Return & Distribution Statistics
# =====================================================

results = []


for symbol in sorted(df["Symbol"].unique()):

    temp = (
        df[df["Symbol"] == symbol]
        .sort_values("Open_time")
        .copy()
    )


    # 시간별 수익률 계산

    temp["Return"] = (
        temp["Close"]
        .pct_change()
    )


    temp = temp.dropna(
        subset=["Return"]
    )


    results.append({

        "Symbol": symbol,

        "Mean_Return":
            temp["Return"].mean(),

        "Std_Return":
            temp["Return"].std(),

        "Skewness":
            temp["Return"].skew(),

        "Kurtosis":
            temp["Return"].kurt(),

        "Min_Return":
            temp["Return"].min(),

        "Max_Return":
            temp["Return"].max()

    })


# =====================================================
# 결과 저장
# =====================================================

tail_statistics = pd.DataFrame(results)


save_path = os.path.join(
    SAVE_DIR,
    "tail_statistics.csv"
)


tail_statistics.to_csv(
    save_path,
    index=False,
    encoding="utf-8-sig"
)


# =====================================================
# 출력
# =====================================================

print("=" * 60)
print("Tail Risk Statistics")
print("=" * 60)

print(tail_statistics)


print("=" * 60)
print("Saved :", save_path)