"""
Week 4: 최종 학습 데이터셋 구축 및 데이터 동결 (Final Dataset Build & Freeze)
CryptoAgent - 암호화폐 포트폴리오 최적화 PPO 프로젝트
리포 루트에서 실행: python src/cryptoagent/pipeline/Week4_data_freeze.py

파이프라인 순서
--------------
Week1(무결성 검증) -> Week2(국면 정의) -> Week3(피처 생성) -> Week4(본 파일)

이 파일이 하는 일
------------------
1. Week3 산출물(data/processed/features.parquet) 로드
2. Week2 산출물(data/processed/regime_definition.csv)을 기준으로 각 행에 regime/split 매핑
3. 데이터 검증 (행/열 개수, 결측치, 중복, regime/split 분포) + Summary Report 출력
4. SQLite(data/raw/crypto_market_features.db)에 feature_table로 저장 (재실행 시 REPLACE)
5. results/data_dictionary.csv 생성 (컬럼별 설명/타입/단위)
6. results/freeze_metadata.json 생성 (데이터셋 버전 동결 메타데이터)
7. 최종 완료 로그 출력

주의
----
- Week2, Week3 코드는 건드리지 않습니다. 이 파일은 그 결과물만 입력으로 사용합니다.
- 스케일링(train fit / val-test transform)은 이 단계에서 수행하지 않습니다.
- 거시지표(DXY, VIX)는 이번 구현 범위에서 제외합니다.
- SQLite는 if_exists="replace", CSV/JSON은 매 실행마다 덮어쓰기이므로
  재실행해도 동일한 입력에 대해 동일한 산출물이 나오는 idempotent 구조입니다.
  (단, freeze_metadata.json의 created_at은 "동결이 언제 확정됐는지" 기록이 목적이라
   실행 시각 그대로 갱신됩니다. 그 외 통계값은 입력이 같으면 항상 동일합니다.)
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd

# ------------------------------------------------------------------
# 경로 설정 (상수로 관리 - 필요 시 이 블록만 수정)
# ------------------------------------------------------------------
FEATURES_PARQUET_PATH = "data/processed/features.parquet"                  # Week3 산출물
REGIME_CSV_PATH = "data/processed/regime_definition.csv"               # Week2 산출물
DB_PATH = "data/raw/crypto_market_features.db"
FEATURE_TABLE_NAME = "feature_table"

RESULTS_DIR = "results"
DATA_DICTIONARY_PATH = os.path.join(RESULTS_DIR, "data_dictionary.csv")
FREEZE_METADATA_PATH = os.path.join(RESULTS_DIR, "freeze_metadata.json")

TIME_COL = "Open_time"
SYMBOL_COL = "Symbol"

DATASET_NAME = "crypto_ppo_feature_dataset"
DATASET_VERSION = "v1.1"


# ------------------------------------------------------------------
# 1. Feature Dataset 로드
# ------------------------------------------------------------------
def load_features(path: str) -> pd.DataFrame:
    """Week3에서 생성한 features.parquet을 로드하고 시간 컬럼을 datetime으로 변환."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"features.parquet을 찾을 수 없습니다: '{path}'. "
            f"Week3_feature_generation.py를 먼저 실행했는지 확인하세요."
        )

    df = pd.read_parquet(path)

    if TIME_COL not in df.columns:
        raise KeyError(f"'{TIME_COL}' 컬럼이 features.parquet에 없습니다. 컬럼명을 확인하세요.")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    return df


# ------------------------------------------------------------------
# 2. Regime 정보 로드 및 조인
# ------------------------------------------------------------------
def load_regime_definition(path: str) -> pd.DataFrame:
    """Week2에서 생성한 regime_definition.csv를 로드하고 날짜 컬럼을 datetime으로 변환."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"regime_definition.csv를 찾을 수 없습니다: '{path}'. "
            f"Week2_regime_definition.py를 먼저 실행했는지 확인하세요."
        )

    regimes = pd.read_csv(path)

    required_cols = {"regime", "split", "start", "end"}
    missing = required_cols - set(regimes.columns)
    if missing:
        raise KeyError(f"regime_definition.csv에 필요한 컬럼이 없습니다: {missing}")

    regimes["start"] = pd.to_datetime(regimes["start"])
    regimes["end"] = pd.to_datetime(regimes["end"]) + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
    return regimes


def assign_regime_and_split(df: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    """
    각 행의 Open_time이 어느 regime 구간에 속하는지 판단하여 regime/split 컬럼을 부여.

    regime 구간 개수는 소수(수 개~수십 개)이므로, 전체 행에 대해 반복문(iterrows)을
    돌리는 대신 '구간별로' 반복하며 boolean mask로 일괄 할당한다.
    -> 가독성과 성능을 동시에 확보하는 방식.
    """
    df = df.copy()
    df["regime"] = pd.NA
    df["split"] = pd.NA

    for _, row in regimes.iterrows():
        in_range = (df[TIME_COL] >= row["start"]) & (df[TIME_COL] <= row["end"])
        df.loc[in_range, "regime"] = row["regime"]
        df.loc[in_range, "split"] = row["split"]

    unmatched = df["regime"].isna().sum()
    if unmatched > 0:
        print(f"  [경고] 어떤 regime 구간에도 속하지 않는 행이 {unmatched}개 있습니다. "
              f"regime_definition.csv의 날짜 범위를 확인하세요.")

    return df


# ------------------------------------------------------------------
# 3. 데이터 검증
# ------------------------------------------------------------------
def validate_dataset(df: pd.DataFrame) -> dict:
    """데이터 검증 지표를 계산 + 출력하고, 이후 단계(메타데이터 생성)에서 재사용할 수 있도록 dict로 반환."""
    n_rows, n_cols = df.shape
    n_missing = int(df.isnull().sum().sum())
    n_duplicates = int(df.duplicated().sum())
    regime_counts = df["regime"].value_counts(dropna=False).to_dict()
    split_counts = df["split"].value_counts(dropna=False).to_dict()

    print("\n" + "=" * 50)
    print("데이터 검증 (Data Validation)")
    print("=" * 50)
    print(f"행 개수        : {n_rows:,}")
    print(f"컬럼 개수      : {n_cols}")
    print(f"컬럼 목록      : {list(df.columns)}")
    print(f"결측치 개수    : {n_missing:,}")
    print(f"중복 행 개수   : {n_duplicates:,}")
    print(f"regime별 개수  : {regime_counts}")
    print(f"split별 개수   : {split_counts}")

    print("\n--- Summary Report ---")
    print(f"기간: {df[TIME_COL].min()} ~ {df[TIME_COL].max()}")
    if SYMBOL_COL in df.columns:
        print(f"심볼 수: {df[SYMBOL_COL].nunique()}개 ({sorted(df[SYMBOL_COL].unique())})")
    missing_ratio = n_missing / (n_rows * n_cols) * 100 if n_rows and n_cols else 0
    print(f"결측치 비율: {missing_ratio:.4f}%")
    print("-" * 50)

    return {
        "total_rows": n_rows,
        "total_columns": n_cols,
        "missing_values": n_missing,
        "duplicate_rows": n_duplicates,
        "regime_distribution": {str(k): int(v) for k, v in regime_counts.items()},
        "split_distribution": {str(k): int(v) for k, v in split_counts.items()},
    }


# ------------------------------------------------------------------
# 4. SQLite 저장
# ------------------------------------------------------------------
def save_to_sqlite(df: pd.DataFrame, db_path: str, table_name: str) -> None:
    """feature_table로 SQLite에 저장. 기존 테이블이 있으면 REPLACE (idempotent)."""
    try:
        # DB 파일이 위치할 폴더가 없으면 생성 (data/ 폴더 등)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # datetime 컬럼은 SQLite에 문자열로 저장되므로 명시적으로 ISO 포맷 변환
        df_to_save = df.copy()
        df_to_save[TIME_COL] = df_to_save[TIME_COL].astype(str)

        with sqlite3.connect(db_path) as conn:
            df_to_save.to_sql(table_name, conn, if_exists="replace", index=False)

        print(f"\n[SQLite] '{table_name}' 테이블 저장 완료 -> {db_path}")

    except sqlite3.Error as e:
        raise RuntimeError(f"SQLite 저장 중 오류 발생: {e}") from e


# ------------------------------------------------------------------
# 5. Data Dictionary 자동 생성
# ------------------------------------------------------------------
# 대표적인 금융 Feature에 대한 설명/단위 매핑.
# Week3 pandas-ta 실제 컬럼명(RSI_14, MACD_12_26_9 등)과
# 프롬프트 예시 표기(RSI14, MACD 등)를 모두 커버하도록 별칭을 함께 등록.
FEATURE_DESCRIPTIONS = {
    # 원본 OHLCV
    "Open_time": ("Candle open timestamp", "datetime"),
    "Symbol": ("Trading pair symbol", "categorical"),
    "Open": ("Opening price", "USDT"),
    "High": ("Highest price in the interval", "USDT"),
    "Low": ("Lowest price in the interval", "USDT"),
    "Close": ("Closing price", "USDT"),
    "Volume": ("Traded volume in the interval", "base asset units"),

    # 이동평균
    "SMA_20": ("Simple Moving Average (20)", "USDT"),
    "SMA_50": ("Simple Moving Average (50)", "USDT"),
    "SMA20": ("Simple Moving Average (20)", "USDT"),
    "SMA50": ("Simple Moving Average (50)", "USDT"),
    "EMA_20": ("Exponential Moving Average (20)", "USDT"),
    "EMA_50": ("Exponential Moving Average (50)", "USDT"),
    "EMA20": ("Exponential Moving Average (20)", "USDT"),
    "EMA50": ("Exponential Moving Average (50)", "USDT"),

    # 모멘텀 / 변동성
    "RSI_14": ("Relative Strength Index (14)", "0-100 (unitless)"),
    "RSI14": ("Relative Strength Index (14)", "0-100 (unitless)"),
    "ATR_14": ("Average True Range (14)", "USDT"),
    "ATR14": ("Average True Range (14)", "USDT"),
    "OBV": ("On-Balance Volume", "base asset units (cumulative)"),

    # MACD (pandas-ta 실제 컬럼명 + 프롬프트 예시 표기)
    "MACD_12_26_9": ("MACD line (12, 26)", "USDT"),
    "MACDh_12_26_9": ("MACD histogram", "USDT"),
    "MACDs_12_26_9": ("MACD signal line (9)", "USDT"),
    "MACD": ("MACD Indicator", "USDT"),
    "MACD_SIGNAL": ("MACD signal line", "USDT"),
    "MACD_HIST": ("MACD histogram", "USDT"),

    # Week4에서 추가되는 컬럼
    "regime": ("Market regime label (Week2 definition)", "categorical"),
    "split": ("Train/val/test split label", "categorical"),
}

DEFAULT_DESCRIPTION = "Feature generated in Week3"
DEFAULT_UNIT = "N/A"


def generate_data_dictionary(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """컬럼별 (Column, Data Type, Description, Unit) 표를 만들어 CSV로 저장."""
    rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        description, unit = FEATURE_DESCRIPTIONS.get(col, (DEFAULT_DESCRIPTION, DEFAULT_UNIT))
        rows.append({
            "Column": col,
            "Data Type": dtype,
            "Description": description,
            "Unit": unit,
        })

    dictionary_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    dictionary_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[Data Dictionary] 저장 완료 -> {path}")

    return dictionary_df


# ------------------------------------------------------------------
# 6. Dataset Freeze Metadata 생성
# ------------------------------------------------------------------
def generate_freeze_metadata(validation_summary: dict, path: str) -> None:
    """데이터셋 버전 동결 시점의 메타데이터를 JSON으로 저장."""
    metadata = {
        "dataset_name": DATASET_NAME,
        "version": DATASET_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_rows": validation_summary["total_rows"],
        "total_columns": validation_summary["total_columns"],
        "missing_values": validation_summary["missing_values"],
        "duplicate_rows": validation_summary["duplicate_rows"],
        "regime_distribution": validation_summary["regime_distribution"],
        "split_distribution": validation_summary["split_distribution"],
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[Freeze Metadata] 저장 완료 -> {path}")


# ------------------------------------------------------------------
# 7. 메인 파이프라인
# ------------------------------------------------------------------
def main():
    try:
        print("=== [1/6] Feature Dataset 로드 ===")
        features_df = load_features(FEATURES_PARQUET_PATH)
        print(f"  로드 완료: {features_df.shape[0]:,}행 x {features_df.shape[1]}열")

        print("\n=== [2/6] Regime 정보 조인 ===")
        regimes_df = load_regime_definition(REGIME_CSV_PATH)
        final_df = assign_regime_and_split(features_df, regimes_df)

        print(f"  regime/split 컬럼 추가 완료")

        print("\n=== [3/6] 데이터 검증 ===")
        validation_summary = validate_dataset(final_df)

        print("\n=== [4/6] SQLite 저장 ===")
        save_to_sqlite(final_df, DB_PATH, FEATURE_TABLE_NAME)

        print("\n=== [5/6] Data Dictionary 생성 ===")
        generate_data_dictionary(final_df, DATA_DICTIONARY_PATH)

        print("\n=== [6/6] Freeze Metadata 생성 ===")
        generate_freeze_metadata(validation_summary, FREEZE_METADATA_PATH)

        print("\n======================================")
        print("Week4 Final Dataset Build Complete")
        print("SQLite Saved     : OK")
        print("Data Dictionary  : OK")
        print("Freeze Metadata  : OK")
        print(f"Rows             : {validation_summary['total_rows']}")
        print(f"Columns          : {validation_summary['total_columns']}")
        print("======================================")

    except FileNotFoundError as e:
        print(f"\n[오류] 필요한 입력 파일을 찾을 수 없습니다.\n{e}")
        raise
    except KeyError as e:
        print(f"\n[오류] 필요한 컬럼이 없습니다.\n{e}")
        raise
    except RuntimeError as e:
        print(f"\n[오류] 파이프라인 실행 중 문제가 발생했습니다.\n{e}")
        raise
    except Exception as e:
        print(f"\n[오류] 예상치 못한 오류가 발생했습니다.\n{e}")
        raise


if __name__ == "__main__":
    main()