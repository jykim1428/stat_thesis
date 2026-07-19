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
    "SELECT Open_time, Symbol, Close FROM ohlcv_data",
    conn
)

conn.close()

# =====================================================
# 전처리
# =====================================================

df["Open_time"] = pd.to_datetime(df["Open_time"])

df = df[
    (df["Open_time"] >= "2021-01-01") &
    (df["Open_time"] <= "2025-12-31 23:59:59")
]

# =====================================================
# 국면 정의
# =====================================================

REGIMES = [
    ("2021-01-01", "2021-11-09", "Bull"),
    ("2021-11-10", "2022-12-31", "Bear"),
    ("2023-01-01", "2023-10-31", "Side"),
    ("2023-11-01", "2025-08-31", "Bull"),
    ("2025-09-01", "2025-12-31", "Bear")
]

# =====================================================
# 결과 생성
# =====================================================

result = []

for start, end, regime in REGIMES:

    temp = df[
        (df["Open_time"] >= start) &
        (df["Open_time"] <= end)
    ]

    result.append({
        "Regime": regime,
        "Start": start,
        "End": end,
        "Rows": len(temp),
        "Days": (
            pd.to_datetime(end) -
            pd.to_datetime(start)
        ).days + 1
    })

result = pd.DataFrame(result)

# =====================================================
# 저장
# =====================================================

save_path = os.path.join(
    SAVE_DIR,
    "regime_definition.csv"
)

result.to_csv(
    save_path,
    index=False,
    encoding="utf-8-sig"
)

print("="*60)
print(result)
print("="*60)

print("Saved :", save_path)