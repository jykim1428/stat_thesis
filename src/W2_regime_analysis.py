"""
============================================================
Week 2 : 시장 국면(Regime) 분석
============================================================

[목적]
- BTC를 시장 대표 자산(Benchmark)으로 사용하여 전체 시장 흐름을 파악한다.
- 누적수익률(Cumulative Return)과 24시간 이동변동성(Rolling Volatility)을 계산한다.
- 상승장(Bull), 하락장(Bear), 횡보장(Side) 후보 구간을 확인한다.
- 후보 국면 경계선을 시각화하여 최종 국면 날짜 확정의 기준 자료로 활용한다.

[입력 데이터]
- data/crypto_market.db
- 분석 기간 : 2021-01-01 ~ 2025-12-31

[산출물]
- results/regime_analysis.png

[활용]
- 이후 regime_definition.py에서 최종 국면 날짜 확정
- Train / Validation / Test 데이터 분할 기준 설정
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
# 후보 Regime Boundary
# (분석 후 수정 가능)
# =====================================================

REGIME_BOUNDARIES = [

    ("2021-11-10", "Peak"),

    ("2022-12-31", "Bear End"),

    ("2023-07-01", "Bull Start")

]


# =====================================================
# DB Load
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
# Preprocessing
# =====================================================

df["Open_time"] = pd.to_datetime(df["Open_time"])
df["Close"] = df["Close"].astype(float)


df = df[
    (df["Open_time"] >= "2021-01-01") &
    (df["Open_time"] <= "2025-12-31 23:59:59")
]


# =====================================================
# BTC Benchmark
# =====================================================

btc = (
    df[df["Symbol"] == "BTCUSDT"]
    .sort_values("Open_time")
    .copy()
)


print(f"BTC Rows : {len(btc):,}")


# =====================================================
# Return Calculation
# =====================================================

btc["Return"] = btc["Close"].pct_change()


# =====================================================
# Cumulative Return
# =====================================================

btc["Cumulative_Return"] = (
    1 + btc["Return"]
).cumprod()


# =====================================================
# Rolling Volatility
# 24-hour window
# =====================================================

btc["Rolling_Volatility"] = (
    btc["Return"]
    .rolling(window=24)
    .std()
)


# =====================================================
# Visualization
# =====================================================

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(18, 10),
    sharex=True
)


# -----------------------------------------------------
# Cumulative Return
# -----------------------------------------------------

ax1.plot(
    btc["Open_time"],
    btc["Cumulative_Return"],
    linewidth=2
)

ax1.set_title(
    "BTC Cumulative Return"
)

ax1.set_ylabel(
    "Cumulative Return"
)


# -----------------------------------------------------
# Rolling Volatility
# -----------------------------------------------------

ax2.plot(
    btc["Open_time"],
    btc["Rolling_Volatility"],
    linewidth=1.5
)

ax2.set_title(
    "BTC 24-Hour Rolling Volatility"
)

ax2.set_ylabel(
    "Volatility"
)

ax2.set_xlabel(
    "Date"
)


# =====================================================
# Regime Boundary 표시
# =====================================================

for date, label in REGIME_BOUNDARIES:

    date = pd.to_datetime(date)

    ax1.axvline(
        date,
        linestyle="--",
        linewidth=2
    )

    ax2.axvline(
        date,
        linestyle="--",
        linewidth=2
    )

    ax1.text(
        date,
        ax1.get_ylim()[1],
        label,
        rotation=90,
        fontsize=10,
        verticalalignment="top"
    )


# =====================================================
# Layout
# =====================================================

ax1.grid(True)
ax2.grid(True)

plt.tight_layout()


# =====================================================
# Save
# =====================================================

save_path = os.path.join(
    SAVE_DIR,
    "regime_analysis.png"
)


plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =====================================================
# Result Check
# =====================================================

print("=" * 60)
print("Regime Analysis Finished")
print("=" * 60)

print("Start :", btc["Open_time"].min())
print("End   :", btc["Open_time"].max())

print("BTC Rows :", len(btc))

print("Saved :", save_path)