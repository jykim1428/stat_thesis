"""
============================================================
Week 2 : 시장 국면(Regime) 분석
============================================================

[목적]
- BTC를 시장 대표 자산(Benchmark)으로 사용하여 전체 시장의 흐름을 파악한다.
- 시간별 수익률, 누적수익률(Cumulative Return), 24시간 이동변동성(Rolling Volatility)을 계산한다.
- 누적수익률과 변동성 그래프를 통해 상승장(Bull), 하락장(Bear), 횡보장(Side) 후보 구간을 확인한다.

[입력 데이터]
- data/crypto_market.db
- 분석 기간 : 2021-01-01 ~ 2025-12-31

[산출물]
- results/btc_regime_analysis.png

[비고]
- BTC는 암호화폐 시장을 대표하는 자산으로 판단하여 시장 국면 분석의 기준으로 사용하였다.
- 이후 시장 국면 날짜를 확정하고 Train / Validation / Test 데이터셋을 구성하는 기준 자료로 활용한다.
"""

import os
import sqlite3

import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# 설정
# =====================================================

DB_PATH = "data/crypto_market.db"
SAVE_DIR = "results"

os.makedirs(SAVE_DIR, exist_ok=True)

# =====================================================
# 데이터 불러오기
# =====================================================

print("=" * 60)
print("Load Database")
print("=" * 60)

conn = sqlite3.connect(DB_PATH)

df = pd.read_sql(
    "SELECT * FROM ohlcv_data",
    conn
)

conn.close()

# =====================================================
# 전처리
# =====================================================

df["Open_time"] = pd.to_datetime(df["Open_time"])
df["Close"] = df["Close"].astype(float)

# =====================================================
# 분석 기간 고정
# (2021-01-01 ~ 2025-12-31)
# =====================================================

df = df[
    (df["Open_time"] >= "2021-01-01") &
    (df["Open_time"] <= "2025-12-31 23:59:59")
]

print(f"Filtered Rows : {len(df):,}")

# =====================================================
# BTC 데이터만 사용
# (시장 전체를 대표하는 Benchmark)
# =====================================================

btc = (
    df[df["Symbol"] == "BTCUSDT"]
    .copy()
    .sort_values("Open_time")
)

print(f"BTC Rows : {len(btc):,}")

# =====================================================
# 시간별 수익률
# =====================================================

btc["Return"] = btc["Close"].pct_change()

# =====================================================
# 누적수익률
# =====================================================

btc["Cumulative_Return"] = (
    1 + btc["Return"]
).cumprod()

# =====================================================
# 24시간 Rolling Volatility
# =====================================================

btc["Rolling_Volatility"] = (
    btc["Return"]
    .rolling(window=24)
    .std()
)

# =====================================================
# 그래프
# =====================================================

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(18, 10),
    sharex=True
)

# -----------------------------------------------------
# 1. 누적수익률
# -----------------------------------------------------

ax1.plot(
    btc["Open_time"],
    btc["Cumulative_Return"],
    linewidth=2
)

ax1.set_title("BTC Cumulative Return")
ax1.set_ylabel("Cumulative Return")

ax1.grid(True)

# -----------------------------------------------------
# 2. Rolling Volatility
# -----------------------------------------------------

ax2.plot(
    btc["Open_time"],
    btc["Rolling_Volatility"],
    linewidth=1.5
)

ax2.set_title("24-Hour Rolling Volatility")
ax2.set_xlabel("Date")
ax2.set_ylabel("Volatility")

ax2.grid(True)

plt.tight_layout()

# =====================================================
# 저장
# =====================================================

save_path = os.path.join(
    SAVE_DIR,
    "btc_regime_analysis.png"
)

plt.savefig(
    save_path,
    dpi=300
)

plt.show()

print("\nFigure Saved")

print(save_path)

print("=" * 60)
print("Analysis Finished")
print("=" * 60)

print()

print("===== BTC 기간 확인 =====")
print("Start :", btc["Open_time"].min())
print("End   :", btc["Open_time"].max())

print()

print("BTC 행 개수 :", len(btc))