## 해당 파일에서 진행할 사항
#1. DB 불러오기 2. 커버리지 확인 3. 시간 갭 확인 4. 중복 확인 5. pivot 확인

import sqlite3
import pandas as pd

# =====================================================
# DB 연결
# =====================================================

DB_PATH = "data/crypto_market.db"

conn = sqlite3.connect(DB_PATH)

df = pd.read_sql(
    "SELECT * FROM ohlcv_data",
    conn
)

conn.close()

df["Open_time"] = pd.to_datetime(df["Open_time"])

print("=" * 60)
print("Data Loaded")
print("=" * 60)
print(df.head())
print()
print(f"Total Rows : {len(df):,}")
print()

# =====================================================
# 1. 자산별 커버리지 확인
# =====================================================

print("=" * 60)
print("1. Coverage Check")
print("=" * 60)

coverage = (
    df.groupby("Symbol")
      .agg(
          Start=("Open_time", "min"),
          End=("Open_time", "max"),
          Rows=("Open_time", "count")
      )
      .reset_index()
)

print(coverage)

# =====================================================
# 2. 시간 Gap 확인
# =====================================================

print()
print("=" * 60)
print("2. Time Gap Check")
print("=" * 60)

for symbol in sorted(df["Symbol"].unique()):

    temp = (
        df[df["Symbol"] == symbol]
        .sort_values("Open_time")
        .copy()
    )

    temp["Gap"] = temp["Open_time"].diff()

    gap = temp[
        temp["Gap"] > pd.Timedelta(hours=1)
    ]

    print(f"\n{symbol}")

    if len(gap) == 0:

        print("Gap 없음")

    else:

        print(f"Gap 개수 : {len(gap)}")

        print(
            gap[
                ["Open_time", "Gap"]
            ].head()
        )

# =====================================================
# 3. 중복 확인
# =====================================================

print()
print("=" * 60)
print("3. Duplicate Check")
print("=" * 60)

duplicate = df.duplicated(
    subset=["Open_time", "Symbol"]
).sum()

print(f"Duplicate Rows : {duplicate}")

# =====================================================
# 4. Pivot 생성
# =====================================================

print()
print("=" * 60)
print("4. Pivot Check")
print("=" * 60)

pivot = df.pivot(
    index="Open_time",
    columns="Symbol",
    values="Close"
)

print(f"Before DropNA : {pivot.shape}")

pivot_clean = pivot.dropna()

print(f"After DropNA  : {pivot_clean.shape}")

# =====================================================
# 5. 실제 학습기간
# =====================================================

print()
print("=" * 60)
print("5. Common Training Period")
print("=" * 60)

print(
    "Start :",
    pivot_clean.index.min()
)

print(
    "End   :",
    pivot_clean.index.max()
)

print(
    f"Total Hours : {len(pivot_clean):,}"
)

print()
print("=" * 60)
print("Validation Finished")
print("=" * 60)