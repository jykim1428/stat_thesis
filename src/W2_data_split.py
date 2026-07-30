"""
============================================================
Week 2 : Regime 기반 Dataset Split
============================================================

[목적]
- 각 시장 국면(Bull/Bear/Side) 내부에서
  Train / Validation / Test 데이터를 시간 순서대로 분할한다.

- 모든 데이터셋이 다양한 시장 국면을 포함하도록 구성한다.

- 시간 순서를 유지하여 미래 데이터가 과거 학습에 포함되는
  데이터 누수(Data Leakage)를 방지한다.

[산출물]
- results/train.csv
- results/validation.csv
- results/test.csv
- results/dataset_split_summary.csv
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
# Regime별 Split 기준
#
# End timestamp는 포함하지 않음
# 예)
# 2021-08-07 00:00 이전 데이터까지 포함
# =====================================================

SPLIT_CONFIG = {


    "Bull_2021": {

        "Train":
        ("2021-01-01 00:00:00",
         "2021-08-07 00:00:00"),

        "Validation":
        ("2021-08-07 00:00:00",
         "2021-09-23 00:00:00"),

        "Test":
        ("2021-09-23 00:00:00",
         "2021-11-10 00:00:00")
    },


    "Bear_2022": {

        "Train":
        ("2021-11-10 00:00:00",
         "2022-08-28 00:00:00"),

        "Validation":
        ("2022-08-28 00:00:00",
         "2022-10-29 00:00:00"),

        "Test":
        ("2022-10-29 00:00:00",
         "2023-01-01 00:00:00")
    },


    "Side_2023": {

        "Train":
        ("2023-01-01 00:00:00",
         "2023-08-01 00:00:00"),

        "Validation":
        ("2023-08-01 00:00:00",
         "2023-09-15 00:00:00"),

        "Test":
        ("2023-09-15 00:00:00",
         "2023-11-01 00:00:00")
    },


    "Bull_2023_2025": {

        "Train":
        ("2023-11-01 00:00:00",
         "2025-02-11 00:00:00"),

        "Validation":
        ("2025-02-11 00:00:00",
         "2025-05-22 00:00:00"),

        "Test":
        ("2025-05-22 00:00:00",
         "2025-09-01 00:00:00")
    },


    "Bear_2025": {

        "Train":
        ("2025-09-01 00:00:00",
         "2025-11-24 00:00:00"),

        "Validation":
        ("2025-11-24 00:00:00",
         "2025-12-12 00:00:00"),

        "Test":
        ("2025-12-12 00:00:00",
         "2026-01-01 00:00:00")
    }

}



# =====================================================
# DB Load
# =====================================================

conn = sqlite3.connect(DB_PATH)

df = pd.read_sql(
    """
    SELECT *
    FROM ohlcv_data
    """,
    conn
)

conn.close()



df["Open_time"] = pd.to_datetime(
    df["Open_time"]
)



# =====================================================
# Split
# =====================================================

train_list = []
validation_list = []
test_list = []

summary = []



for regime, split_dict in SPLIT_CONFIG.items():

    for split_name, (start, end) in split_dict.items():

        temp = df[
            (df["Open_time"] >= start)
            &
            (df["Open_time"] < end)
        ].copy()


        temp["Regime"] = regime
        temp["Split"] = split_name


        if split_name == "Train":

            train_list.append(temp)

        elif split_name == "Validation":

            validation_list.append(temp)

        else:

            test_list.append(temp)



        summary.append({

            "Regime": regime,

            "Split": split_name,

            "Defined_Start": start,

            "Defined_End": end,

            "Actual_Start":
                temp["Open_time"].min(),

            "Actual_End":
                temp["Open_time"].max(),

            "Rows":
                len(temp)

        })



# =====================================================
# Dataset 생성
# =====================================================

train = pd.concat(
    train_list,
    ignore_index=True
)

validation = pd.concat(
    validation_list,
    ignore_index=True
)

test = pd.concat(
    test_list,
    ignore_index=True
)



# =====================================================
# 중복 검사
# =====================================================

def check_duplicate(data):

    return data.duplicated(
        subset=[
            "Symbol",
            "Open_time"
        ]
    ).sum()



print("="*60)
print("Duplicate Check")
print("="*60)

print(
    "Train:",
    check_duplicate(train)
)

print(
    "Validation:",
    check_duplicate(validation)
)

print(
    "Test:",
    check_duplicate(test)
)



# =====================================================
# 저장
# =====================================================

train.to_csv(
    os.path.join(
        SAVE_DIR,
        "train.csv"
    ),
    index=False
)


validation.to_csv(
    os.path.join(
        SAVE_DIR,
        "validation.csv"
    ),
    index=False
)


test.to_csv(
    os.path.join(
        SAVE_DIR,
        "test.csv"
    ),
    index=False
)


pd.DataFrame(summary).to_csv(
    os.path.join(
        SAVE_DIR,
        "dataset_split_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)



print("="*60)
print("Dataset Split Finished")
print("="*60)

print(
    "Train:",
    train.shape
)

print(
    "Validation:",
    validation.shape
)

print(
    "Test:",
    test.shape
)