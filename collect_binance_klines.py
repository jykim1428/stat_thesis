import time
import sqlite3
import requests
import pandas as pd
from datetime import datetime, timezone

# =====================================================
# 설정
# =====================================================

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT",
           "SOLUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT"]

INTERVAL = "1h"          # 1시간봉. 다른 봉이 필요하면 "1d", "15m" 등으로 변경
START_DATE = "2021-01-01"
END_DATE = "2025-12-31 23:59:59"

DB_PATH = "crypto_market_new.db"   # 저장할 sqlite 파일명
TABLE_NAME = "ohlcv_data"

BASE_URL = "https://api.binance.com/api/v3/klines"
LIMIT = 1000              # 바이낸스 klines 1회 요청 최대 개수
SLEEP_SEC = 0.3           # 요청 사이 딜레이 (rate limit 방지용)

COLUMNS = [
    "Open_time", "Open", "High", "Low", "Close", "Volume",
    "Close_time", "Quote_asset_volume", "Number_of_trades",
    "Taker_buy_base", "Taker_buy_quote", "Ignore",
]


def to_ms(dt_str: str) -> int:
    """'YYYY-MM-DD' 또는 'YYYY-MM-DD HH:MM:SS' 문자열 -> UTC 기준 밀리초 timestamp"""
    dt = pd.Timestamp(dt_str, tz="UTC")
    return int(dt.timestamp() * 1000)


def fetch_symbol(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """한 심볼에 대해 start_ms ~ end_ms 구간 전체를 페이지네이션하며 수집"""
    all_rows = []
    cursor = start_ms

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": LIMIT,
        }
        resp = requests.get(BASE_URL, params=params, timeout=10)

        if resp.status_code != 200:
            print(f"  [경고] {symbol} 요청 실패 (status={resp.status_code}): {resp.text[:200]}")
            time.sleep(2)
            continue

        data = resp.json()
        if not data:
            break  # 더 이상 받아올 데이터 없음

        all_rows.extend(data)

        last_open_time = data[-1][0]
        cursor = last_open_time + 1  # 다음 요청은 마지막 캔들 다음 시점부터

        # 진행상황 출력
        last_dt = pd.to_datetime(last_open_time, unit="ms", utc=True)
        print(f"  {symbol}: {len(all_rows)} rows 누적, 현재 {last_dt}")

        # 마지막 페이지(받은 개수가 limit보다 적음)면 종료
        if len(data) < LIMIT:
            break

        time.sleep(SLEEP_SEC)

    if not all_rows:
        return pd.DataFrame(columns=COLUMNS + ["Symbol"])

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    df["Symbol"] = symbol

    # 타입 변환
    df["Open_time"] = pd.to_datetime(df["Open_time"], unit="ms", utc=True).dt.tz_localize(None)
    df["Close_time"] = pd.to_datetime(df["Close_time"], unit="ms", utc=True).dt.tz_localize(None)
    for col in ["Open", "High", "Low", "Close", "Volume",
                "Quote_asset_volume", "Taker_buy_base", "Taker_buy_quote"]:
        df[col] = df[col].astype(float)
    df["Number_of_trades"] = df["Number_of_trades"].astype(int)

    return df


def main():
    start_ms = to_ms(START_DATE)
    end_ms = to_ms(END_DATE)

    conn = sqlite3.connect(DB_PATH)

    for i, symbol in enumerate(SYMBOLS, 1):
        print(f"\n[{i}/{len(SYMBOLS)}] {symbol} 수집 시작 "
              f"({START_DATE} ~ {END_DATE})")
        df = fetch_symbol(symbol, INTERVAL, start_ms, end_ms)

        if df.empty:
            print(f"  -> {symbol}: 데이터 없음, 스킵")
            continue

        # 실제 상장일이 START_DATE보다 늦으면 여기서 바로 확인 가능
        actual_start = df["Open_time"].min()
        if actual_start > pd.Timestamp(START_DATE):
            print(f"  ⚠️ {symbol} 실제 데이터 시작일: {actual_start} "
                  f"(요청한 {START_DATE}보다 늦음 -> 해당 시점 이전엔 상장 전이었을 가능성)")

        df.to_sql(TABLE_NAME, conn, if_exists="append", index=False)
        print(f"  -> {symbol}: {len(df)} rows 저장 완료")

    conn.close()
    print(f"\n전체 수집 완료. DB 저장 위치: {DB_PATH}")


if __name__ == "__main__":
    main()
