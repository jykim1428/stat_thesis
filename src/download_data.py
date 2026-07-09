import os
import sqlite3

import pandas as pd

from dotenv import load_dotenv
from binance.client import Client

from datetime import datetime
from dateutil.relativedelta import relativedelta

# =====================================================
# API KEY
# =====================================================

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

client = Client(API_KEY, API_SECRET)

# =====================================================
# 설정
# =====================================================

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT"
]

INTERVAL = Client.KLINE_INTERVAL_1HOUR

DB_PATH = "data/crypto_market.db"

COLUMNS = [
    "Open_time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Close_time",
    "Quote_asset_volume",
    "Number_of_trades",
    "Taker_buy_base",
    "Taker_buy_quote",
    "Ignore"
]

# =====================================================
# 다운로드
# =====================================================

def download_symbol(symbol):

    print(f"\nDownloading {symbol}")

    start_date = (
        datetime.today() -
        relativedelta(years=5)
    ).strftime("%Y-%m-%d")

    end_date = datetime.today().strftime("%Y-%m-%d")

    klines = client.get_historical_klines(
        symbol=symbol,
        interval=INTERVAL,
        start_str=start_date,
        end_str=end_date
    )

    print(f"{len(klines)} rows downloaded")

    df = pd.DataFrame(
        klines,
        columns=COLUMNS
    )

    df["Open_time"] = pd.to_datetime(
        df["Open_time"],
        unit="ms"
    )

    df["Close_time"] = pd.to_datetime(
        df["Close_time"],
        unit="ms"
    )

    numeric_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Quote_asset_volume",
        "Taker_buy_base",
        "Taker_buy_quote"
    ]

    df[numeric_cols] = df[numeric_cols].astype(float)

    df["Number_of_trades"] = (
        df["Number_of_trades"]
        .astype(int)
    )

    df["Symbol"] = symbol

    return df


# =====================================================
# DB 초기화
# =====================================================

def initialize_database():

    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        "DROP TABLE IF EXISTS ohlcv_data"
    )

    conn.commit()

    conn.close()

    print("기존 테이블 삭제 완료")


# =====================================================
# 저장
# =====================================================

def save_database(df):

    conn = sqlite3.connect(DB_PATH)

    df.to_sql(
        "ohlcv_data",
        conn,
        if_exists="append",
        index=False
    )

    total = pd.read_sql(
        "SELECT COUNT(*) AS cnt FROM ohlcv_data",
        conn
    )

    conn.close()

    print(
        f"현재 DB Row : {total.iloc[0,0]}"
    )


# =====================================================
# 메인
# =====================================================

def main():

    print("="*60)
    print("Binance Download Start")
    print("="*60)

    initialize_database()

    total_rows = 0

    for symbol in SYMBOLS:

        df = download_symbol(symbol)

        save_database(df)

        total_rows += len(df)

        print(
            f"{symbol} 저장 완료\n"
        )

    print("="*60)
    print("Download Finished")
    print(f"Total Rows : {total_rows}")
    print("="*60)


if __name__ == "__main__":

    main()