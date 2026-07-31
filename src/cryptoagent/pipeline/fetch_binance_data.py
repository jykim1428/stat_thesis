import os
import sqlite3
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

if not API_KEY or not API_SECRET:
    raise RuntimeError(".env 파일에 BINANCE_API_KEY / BINANCE_API_SECRET을 설정하세요")

client = Client(API_KEY, API_SECRET)

SYMBOLS = ["BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"]  # 필요한 종목으로 수정
INTERVAL = Client.KLINE_INTERVAL_1HOUR
START_STR = "2021-01-01"  # 조회 시작일 (종목별 상장일 이전이면 자동으로 상장일부터 반환됨)
DB_PATH = "data/raw/binance_ohlcv.db"  # 리포 루트에서 실행 전제

COLUMNS = [
    "Open_time", "Open", "High", "Low", "Close", "Volume",
    "Close_time", "Quote_asset_volume", "Number_of_trades",
    "Taker_buy_base_vol", "Taker_buy_quote_vol", "Ignore",
]


def fetch_symbol(symbol: str) -> pd.DataFrame:
    print(f"[{symbol}] 수집 중...")
    klines = client.get_historical_klines(symbol, INTERVAL, START_STR)
    df = pd.DataFrame(klines, columns=COLUMNS)

    df["Open_time"] = pd.to_datetime(df["Open_time"], unit="ms")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)

    df["Symbol"] = symbol
    print(f"[{symbol}] {len(df)}행 수집 완료 ({df['Open_time'].min()} ~ {df['Open_time'].max()})")
    return df[["Open_time", "Open", "High", "Low", "Close", "Volume", "Symbol"]]


def main():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    for symbol in SYMBOLS:
        df = fetch_symbol(symbol)
        df.to_sql("ohlcv_data", conn, if_exists="append", index=False)
    conn.close()
    print(f"\n저장 완료: {DB_PATH}")


if __name__ == "__main__":
    main()
