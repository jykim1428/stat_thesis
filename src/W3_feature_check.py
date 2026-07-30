"""
============================================================
Week 3 : Feature Dataset Validation
============================================================

[목적]
- 생성된 Feature Dataset이 정상적으로 생성되었는지 확인한다.
- Feature별 결측치(NaN) 개수와 데이터 구조를 점검한다.
- 데이터 전처리는 수행하지 않으며, 생성 결과만 검증한다.

[입력 데이터]
- results/feature_dataset.csv

[산출물]
- Console Output

[비고]
- NaN 제거는 수행하지 않는다.
- Feature Scaling은 수행하지 않는다.
- Week 4에서 전처리를 진행하기 전 데이터 상태를 확인하기 위한 코드이다.
"""

import pandas as pd

# =====================================================
# 설정
# =====================================================

FILE_PATH = "results/feature_dataset.csv"

# =====================================================
# 데이터 불러오기
# =====================================================

print("=" * 60)
print("Load Feature Dataset")
print("=" * 60)

df = pd.read_csv(
    FILE_PATH,
    parse_dates=["Open_time"]
)

# =====================================================
# 기본 정보
# =====================================================

print()

print("=" * 60)
print("Dataset Information")
print("=" * 60)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")

print()

print(f"Start Date : {df['Open_time'].min()}")
print(f"End Date   : {df['Open_time'].max()}")

print()

print("Symbols")

print(df["Symbol"].unique())

# =====================================================
# Feature 목록
# =====================================================

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

print()

print("=" * 60)
print("Feature List")
print("=" * 60)

for feature in feature_columns:

    print(feature)

# =====================================================
# NaN 확인
# =====================================================

print()

print("=" * 60)
print("NaN Count")
print("=" * 60)

print(

    df[
        feature_columns
    ].isnull().sum()

)

# =====================================================
# 코인별 데이터 개수
# =====================================================

print()

print("=" * 60)
print("Rows by Symbol")
print("=" * 60)

print(

    df.groupby("Symbol")
      .size()

)

# =====================================================
# 기술지표 미리보기
# =====================================================

print()

print("=" * 60)
print("Preview")
print("=" * 60)

print(

    df[
        [

            "Open_time",

            "Symbol",

            "Close"

        ]

        +

        feature_columns

    ].head()

)

print()

print("=" * 60)
print("Validation Finished")
print("=" * 60)