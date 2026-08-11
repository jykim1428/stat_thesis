import sqlite3
import pandas as pd

conn = sqlite3.connect(r"C:\Users\ilove\Downloads\crypto_market_new.db")

# 1) DB 안에 어떤 테이블들이 있는지 확인
print(pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn))

# 2) ohlcv_data 테이블의 컬럼명 확인
df_sample = pd.read_sql("SELECT * FROM ohlcv_data LIMIT 5", conn)
print(df_sample.columns.tolist())
print(df_sample)

conn.close()