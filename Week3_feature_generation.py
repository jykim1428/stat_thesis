"""
Week 3: 피처 생성 (CryptoAgent)

pandas-ta로 종목별 기술지표(RSI, MACD, ATR, OBV, SMA, EMA)를 계산.
스케일링은 하지 않음 (train-only fit 원칙 때문에 학습 단계로 미룸) — 원본 피처만 저장.

설정 4줄(DB_PATH, TS_COL, SYM_COL, PRICE_COLS)만 본인 환경에 맞게.
산출물: features.parquet (long format: Open_time, Symbol, ... raw OHLCV, ... 지표들)
"""
import sqlite3
import numpy as np
import pandas as pd
import pandas_ta as ta

# ── 설정 ─────────────────────────────────────────────
DB_PATH    = "crypto_market.db"
TS_COL     = "Open_time"
SYM_COL    = "Symbol"
PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]
END_DATE   = "2025-12-31"   # 2026년 데이터는 이번 분석에서 제외

# 지표 파라미터 (기본값)
RSI_LEN     = 14
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
ATR_LEN     = 14
SMA_WINDOWS = [20, 50]
EMA_WINDOWS = [20, 50]
# ─────────────────────────────────────────────────────


def load_ohlcv():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM ohlcv_data", con)
    con.close()

    ts = df[TS_COL]
    if pd.api.types.is_numeric_dtype(ts):
        df[TS_COL] = pd.to_datetime(ts, unit="ms" if ts.max() > 1e12 else "s")
    else:
        df[TS_COL] = pd.to_datetime(ts)

    for c in PRICE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values([SYM_COL, TS_COL]).reset_index(drop=True)
    df = df[df[TS_COL] <= END_DATE]
    return df


def add_indicators(g: pd.DataFrame) -> pd.DataFrame:
    """단일 심볼에 대해 기술지표 컬럼 추가"""
    g = g.sort_values(TS_COL).reset_index(drop=True)

    g["RSI_14"] = ta.rsi(g["Close"], length=RSI_LEN)

    macd = ta.macd(g["Close"], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIG)
    g = pd.concat([g, macd], axis=1)

    g["ATR_14"] = ta.atr(g["High"], g["Low"], g["Close"], length=ATR_LEN)
    g["OBV"]    = ta.obv(g["Close"], g["Volume"])

    for w in SMA_WINDOWS:
        g[f"SMA_{w}"] = ta.sma(g["Close"], length=w)
    for w in EMA_WINDOWS:
        g[f"EMA_{w}"] = ta.ema(g["Close"], length=w)

    return g


def main():
    df = load_ohlcv()

    feats = (df.groupby(SYM_COL, group_keys=False)[df.columns]
               .apply(add_indicators))

    # 결측치 처리: 지표는 warm-up 구간(각 지표의 lookback window)에서 NaN 발생.
    # 종목별로 가장 긴 warm-up(SMA/EMA 50봉) 이후부터만 남겨서 지표 계산이 끝난 구간만 사용.
    warmup = max(RSI_LEN, MACD_SLOW + MACD_SIG, ATR_LEN, *SMA_WINDOWS, *EMA_WINDOWS)
    feats = (feats.groupby(SYM_COL, group_keys=False)[feats.columns]
                  .apply(lambda g: g.iloc[warmup:]))

    n_before = len(feats)
    feats = feats.dropna()
    n_after = len(feats)
    print(f"warm-up 제거 후 행수: {n_before} -> dropna 후: {n_after} (제거 {n_before - n_after}행)")

    feature_cols = [c for c in feats.columns if c not in [TS_COL, SYM_COL] + PRICE_COLS]
    print(f"\n생성된 피처 목록 ({len(feature_cols)}개):")
    for c in feature_cols:
        print(f"  - {c}")

    feats = feats.sort_values([SYM_COL, TS_COL]).reset_index(drop=True)
    feats.to_parquet("features.parquet", index=False)
    print(f"\n저장 완료: features.parquet  shape={feats.shape}")

    with open("feature_list.md", "w", encoding="utf-8") as f:
        f.write("# Week 3 피처 목록\n\n")
        f.write(f"- 원본 OHLCV: {PRICE_COLS}\n")
        f.write(f"- 기술지표 ({len(feature_cols)}개): {feature_cols}\n\n")
        f.write(f"## 결측치 처리 규칙\n\n")
        f.write(f"- 지표 계산에 필요한 최대 warm-up 구간({warmup}봉)을 종목별로 앞에서 잘라냄\n")
        f.write(f"- 이후 남은 NaN은 dropna로 제거\n")
        f.write(f"- 최종 shape: {feats.shape}\n\n")
        f.write(f"## 스케일링\n\n")
        f.write("- 이번 단계에서는 하지 않음. train-only fit 원칙에 따라 학습 단계로 이관.\n")
        f.write("- 거시지표(DXY, VIX)는 이번 주 범위에서 제외.\n")


if __name__ == "__main__":
    main()
