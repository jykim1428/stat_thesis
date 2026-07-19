import os
import sqlite3

import pandas as pd

# =====================================================
# 설정
# =====================================================

DB_PATH = "data/crypto_market.db"
SAVE_DIR = "results"

os.makedirs(SAVE_DIR, exist_ok=True)

# =====================================================
# DB Load
# =====================================================

conn = sqlite3.connect(DB_PATH)

df = pd.read_sql(
    "SELECT * FROM ohlcv_data",
    conn
)

conn.close()

df["Open_time"] = pd.to_datetime(df["Open_time"])
df["Close"] = df["Close"].astype(float)

df = df[
    (df["Open_time"] >= "2021-01-01") &
    (df["Open_time"] <= "2025-12-31 23:59:59")
]

# =====================================================
# Return 계산
# =====================================================

result = []

for symbol in sorted(df["Symbol"].unique()):

    temp = (
        df[df["Symbol"] == symbol]
        .sort_values("Open_time")
        .copy()
    )

    temp["Return"] = temp["Close"].pct_change()

    result.append({

        "Symbol": symbol,

        "Skewness":
            temp["Return"].skew(),

        "Kurtosis":
            temp["Return"].kurt()

    })

tail = pd.DataFrame(result)

print("=" * 60)
print(tail)
print("=" * 60)

tail.to_csv(
    os.path.join(
        SAVE_DIR,
        "tail_statistics.csv"
    ),
    index=False
)

print("Saved : tail_statistics.csv")

### 데이터 확인 -> 왜도가 너무 크게 나옴 ###

print("="*60)
print(symbol)

print(temp["Return"].describe())

print("\nLargest Returns")
print(temp["Return"].nlargest(5))

print("\nSmallest Returns")
print(temp["Return"].nsmallest(5))

### 티커별로 조금 더 확인 ###
for symbol in sorted(df["Symbol"].unique()):
    
    temp = (
        df[df["Symbol"] == symbol]
        .sort_values("Open_time")
        .copy()
    )

    temp["Return"] = temp["Close"].pct_change()

    print("="*60)
    print(symbol)

    print("Max Return")
    print(temp["Return"].max())

    print("Min Return")
    print(temp["Return"].min())
    
### BTC 만 한번 확인 ###
btc = (
    df[df["Symbol"] == "BTCUSDT"]
    .sort_values("Open_time")
    .copy()
)

btc["Return"] = btc["Close"].pct_change()

print("Skew :", btc["Return"].skew())
print("Kurt :", btc["Return"].kurt())

print(btc["Return"].quantile([0.001,0.01,0.99,0.999]))