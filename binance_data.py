import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

# matplotlib 한글 폰트 설정 (Windows 환경 기준, Mac이면 'AppleGothic'으로 변경)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

conn = sqlite3.connect(r"C:\Users\ilove\Downloads\crypto_market_new.db")  # 사용자 환경에 맞게 수정 필요
df = pd.read_sql("SELECT * FROM ohlcv_data", conn, parse_dates=["Open_time"])
conn.close()
output_dir = r"C:\Users\ilove\OneDrive\바탕 화면\outputs"  # 사용자 환경에 맞게 수정 필요
os.makedirs(output_dir, exist_ok=True)

df = df.sort_values(["Symbol", "Open_time"]).reset_index(drop=True)

# =====================================================
# 1. 자산별 커버리지 확인
# =====================================================

coverage = (
    df.groupby("Symbol")["Open_time"]
    .agg(start="min", end="max", rows="count")
    .reset_index()
)
coverage["span_days"] = (coverage["end"] - coverage["start"]).dt.days
coverage = coverage.sort_values("start", ascending=False)

print("=" * 70)
print("1. 자산별 커버리지")
print("=" * 70)
print(coverage.to_string(index=False))

common_start = coverage["start"].max()
common_end = coverage["end"].min()
print(f"\n공통 가능 구간: {common_start} ~ {common_end}")
print(f"공통 구간 길이: {(common_end - common_start).days} days")

# =====================================================
# 2. 시간 갭 점검 (1h 아닌 간격)
# =====================================================

print("\n" + "=" * 70)
print("2. 시간 갭 점검")
print("=" * 70)

gap_records = []
for symbol, g in df.groupby("Symbol"):
    g = g.sort_values("Open_time")
    diffs = g["Open_time"].diff().dropna()
    bad = diffs[diffs != pd.Timedelta(hours=1)]
    for idx in bad.index:
        gap_records.append({
            "Symbol": symbol,
            "gap_start": g.loc[idx - 1, "Open_time"] if (idx - 1) in g.index else None,
            "gap_end": g.loc[idx, "Open_time"],
            "gap_size": bad[idx]
        })

gap_df = pd.DataFrame(gap_records)

if gap_df.empty:
    print("갭 없음 — 모든 구간이 정확히 1시간 간격")
else:
    print(f"총 {len(gap_df)}개 갭 발견")
    print(gap_df.groupby("Symbol").size().to_string())
    print("\n갭 상세 (상위 20개):")
    print(gap_df.sort_values("gap_size", ascending=False).head(20).to_string(index=False))

# =====================================================
# 3. 중복 (timestamp, symbol) 확인
# =====================================================

print("\n" + "=" * 70)
print("3. 중복 확인")
print("=" * 70)

dup_mask = df.duplicated(subset=["Open_time", "Symbol"], keep=False)
dup_count = dup_mask.sum()

if dup_count == 0:
    print("중복 없음")
else:
    print(f"중복 {dup_count}건 발견")
    dup_df = df[dup_mask].sort_values(["Symbol", "Open_time"])
    print(dup_df.groupby("Symbol").size().to_string())
    df = df.sort_values(["Symbol", "Open_time"], kind="stable")
    df = df.drop_duplicates(subset=["Open_time", "Symbol"], keep="first")
    print(f"→ 중복 제거 후 row 수: {len(df)}")

# =====================================================
# 3-1. 시작일 일관성 점검
# =====================================================

print("\n" + "=" * 70)
print("3-1. 시작일 일관성 점검")
print("=" * 70)

start_dates = (
    df.groupby("Symbol")["Open_time"]
      .min()
      .reset_index()
      .rename(columns={"Open_time": "start_date"})
)
print(start_dates.to_string(index=False))

if start_dates["start_date"].nunique() == 1:
    print("\n모든 자산의 시작일이 동일합니다.")
    print("→ 현재 데이터는 동일 기간을 기준으로 수집된 것으로 확인됩니다.")
else:
    print("\n자산별 시작일이 서로 다릅니다.")
    print("→ 공통 학습 구간은 가장 늦은 시작일 이후로 설정해야 합니다.")

# =====================================================
# 3-2. 산출물 저장: 정합성 리포트
# =====================================================

print("\n" + "=" * 70)
print("3-2. 산출물 저장: 정합성 리포트")
print("=" * 70)

coverage.to_csv(os.path.join(output_dir, "coverage_report.csv"), index=False, encoding="utf-8-sig")

if not gap_df.empty:
    gap_df.to_csv(os.path.join(output_dir, "gap_report.csv"), index=False, encoding="utf-8-sig")
else:
    pd.DataFrame(columns=["Symbol", "gap_start", "gap_end", "gap_size"]).to_csv(
        os.path.join(output_dir, "gap_report.csv"), index=False, encoding="utf-8-sig"
    )

start_dates.to_csv(os.path.join(output_dir, "start_date_check.csv"), index=False, encoding="utf-8-sig")

print("\n산출물 저장 완료:")
print(" - coverage_report.csv        (자산별 커버리지)")
print(" - gap_report.csv             (시간 갭 목록)")
print(" - start_date_check.csv       (자산별 시작일 확인)")

# =====================================================
# 4. pivot + dropna → 실제 학습 가능 구간
# =====================================================

print("\n" + "=" * 70)
print("4. Pivot 후 공통 구간 확정")
print("=" * 70)

pivot_close = df.pivot(index="Open_time", columns="Symbol", values="Close")
print(f"Pivot 전 shape: {pivot_close.shape}")

pivot_dropna = pivot_close.dropna()
print(f"dropna 후 shape: {pivot_dropna.shape}")

if len(pivot_dropna) > 0:
    print(f"실제 학습 가능 구간: {pivot_dropna.index.min()} ~ {pivot_dropna.index.max()}")
    print(f"총 시간 수: {len(pivot_dropna)} ({len(pivot_dropna)/24:.0f}일)")

loss_pct = 100 * (1 - len(pivot_dropna) / len(pivot_close))
print(f"pivot 단계 데이터 손실률: {loss_pct:.2f}%")

# =====================================================
# 5. 누적수익률 + 24h rolling vol 계산
# =====================================================

print("\n" + "=" * 70)
print("5. 누적수익률 + 24h rolling vol 계산")
print("=" * 70)

returns = pivot_dropna.pct_change().dropna()
cum_returns = (1 + returns).cumprod() - 1
rolling_vol = returns.rolling(window=24).std() * np.sqrt(24)

print("수익률 데이터 크기:", returns.shape)
print("누적수익률 데이터 크기:", cum_returns.shape)
print("Rolling Vol 데이터 크기:", rolling_vol.shape)

# =====================================================
# 6. 국면 날짜 정의 
# =====================================================

regime_boundaries = {
    "bull_2021":   ("2021-01-01", "2021-11-09 23:59:59"),
    "bear_2022":   ("2021-11-10", "2023-01-01 23:59:59"),
    "side_2023":   ("2023-01-02", "2023-10-15 23:59:59"),
    "bull_2024":   ("2023-10-16", "2025-01-01 23:59:59"),
    "choppy_2025": ("2025-01-02", "2025-12-31 23:59:59"),
}

# 국면별 확정 split (국면 통째로 train/val/test에 배정 — 70/15/15로 쪼개지 않음)
regime_split = {
    "bull_2021": "train",
    "bear_2022": "train",
    "side_2023": "val",
    "bull_2024": "test",
    "choppy_2025": "test",
}

data_start = pivot_dropna.index.min()
data_end = pivot_dropna.index.max()

regime_table = []
for name, (start, end) in regime_boundaries.items():
    start_ts = max(pd.Timestamp(start), data_start)
    end_ts = min(pd.Timestamp(end), data_end)
    if start_ts >= end_ts:
        print(f"[경고] '{name}' 구간이 실제 데이터 범위를 벗어남 -> 스킵")
        continue
    regime_table.append({
        "regime": name,
        "start": start_ts,
        "end": end_ts,
        "hours": len(pivot_dropna.loc[start_ts:end_ts]),
        "split": regime_split.get(name, "unassigned"),
    })

regime_df = pd.DataFrame(regime_table)
print("\n" + "=" * 70)
print("6. 국면 날짜 정의")
print("=" * 70)
print(regime_df.to_string(index=False))

covered_hours = regime_df["hours"].sum() if len(regime_df) else 0
total_hours = len(pivot_dropna)
uncovered_pct = 100 * (1 - covered_hours / total_hours) if total_hours else 0
print(f"\n국면 커버리지: {covered_hours}/{total_hours} 시간 ({100 - uncovered_pct:.1f}%)")
if uncovered_pct > 0.1:
    print(f"⚠️ 전체 데이터의 {uncovered_pct:.1f}%가 어떤 국면에도 속하지 않습니다. "
          f"regime_boundaries 를 데이터 종료일({data_end.date()})까지 채우는 것을 검토하세요.")

# =====================================================
# 7. 누적수익률 + rolling vol 그래프에 국면 경계선 표시
# =====================================================

print("\n" + "=" * 70)
print("7. 누적수익률 + rolling vol 그래프에 국면 경계선 표시")
print("=" * 70)

colors = {
    "bull_2021": "green",
    "bear_2022": "red",
    "side_2023": "gray",
    "bull_2024": "blue",
    "choppy_2025": "purple",
}
symbols = pivot_dropna.columns.tolist()

for symbol in symbols:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(cum_returns.index, cum_returns[symbol], color="steelblue", linewidth=1)
    ax1.set_title(f"{symbol} Cumulative Return with Market Regimes")
    ax1.set_ylabel("Cumulative Return")
    ax1.grid(alpha=0.3)

    ax2.plot(rolling_vol.index, rolling_vol[symbol], color="darkorange", linewidth=1)
    ax2.set_title(f"{symbol} 24h Rolling Volatility")
    ax2.set_ylabel("Rolling Vol")
    ax2.grid(alpha=0.3)

    for _, row in regime_df.iterrows():
        c = colors.get(row["regime"], "black")
        for ax in (ax1, ax2):
            ax.axvline(row["start"], color=c, linestyle="--", alpha=0.6)
        ax1.text(row["start"], ax1.get_ylim()[1] * 0.9, row["regime"],
                  rotation=90, va="top", fontsize=9, color=c)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{symbol}_regime_plot.png"), dpi=120)
    plt.close(fig)

print("\n자산별 국면 그래프 저장 완료")

# =====================================================
# 8. 국면별 train/val/test 분할 (국면 전체를 통째로 split에 배정)
# =====================================================

print("\n" + "=" * 70)
print("8. 국면별 train/val/test 분할 (국면 단위 배정)")
print("=" * 70)

split_df = regime_df[["regime", "split", "start", "end", "hours"]].rename(
    columns={"start": "period_start", "end": "period_end"}
)
print(split_df.to_string(index=False))

# split 별 합계 시간도 참고용으로 출력
split_summary = split_df.groupby("split")["hours"].sum().reset_index()
print("\nSplit별 총 시간 수:")
print(split_summary.to_string(index=False))

# =====================================================
# 9. 왜도(skewness) · 첨도(kurtosis) 자산별 계산 → heavy-tailedness 근거
# =====================================================

log_returns = np.log(pivot_dropna / pivot_dropna.shift(1)).dropna() # 로그수익률

skew_kurt = pd.DataFrame({
    "skewness": log_returns.apply(lambda x: stats.skew(x.dropna())),
    "kurtosis_excess": log_returns.apply(lambda x: stats.kurtosis(x.dropna())),
    "std": log_returns.std(),
})
skew_kurt["heavy_tailed"] = skew_kurt["kurtosis_excess"] > 0
skew_kurt = skew_kurt.sort_values("kurtosis_excess", ascending=False)

print("\n" + "=" * 70)
print("9. 왜도·첨도 자산별 계산 (전체기간, 시간봉 로그수익률 — heavy-tail 근거)")
print("=" * 70)
print(skew_kurt.to_string())

# =====================================================
# 10. 산출물 저장
# =====================================================

print("\n" + "=" * 70)
print("10. 산출물 저장")
print("=" * 70)

regime_df.to_csv(os.path.join(output_dir, "regime_definition_table.csv"), index=False, encoding="utf-8-sig")
split_df.to_csv(os.path.join(output_dir, "train_val_test_split.csv"), index=False, encoding="utf-8-sig")
skew_kurt.to_csv(os.path.join(output_dir, "skew_kurtosis_by_asset.csv"), encoding="utf-8-sig")

print("\n산출물 저장 완료:")
print(" - coverage_report.csv           (자산별 커버리지)")
print(" - gap_report.csv                (시간 갭 목록)")
print(" - start_date_check.csv          (자산별 시작일 확인)")
print(" - regime_definition_table.csv   (국면 정의표, 날짜+split 명시)")
print(" - regime_validation.csv         (국면 날짜 검증 결과, 참고용)")
print(" - train_val_test_split.csv      (국면별 split 배정 결과)")
print(" - skew_kurtosis_by_asset.csv    (자산별 왜도/첨도)")
print(" - {symbol}_regime_plot.png      (자산별 누적수익률+rolling vol 그래프)")

# =====================================================
# 11. 기술지표 계산 (pandas-ta): RSI, MACD, ATR, OBV, SMA, EMA
# =====================================================

import pandas_ta as ta

print("\n" + "=" * 70)
print("11. 기술지표 계산 (RSI, MACD, ATR, OBV, SMA, EMA)")
print("=" * 70)

required_cols = {"Open", "High", "Low", "Close", "Volume"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(
        f"OHLCV 컬럼 중 다음이 없습니다: {missing}\n"
        f"현재 df 컬럼: {list(df.columns)}\n"
        f"-> ohlcv_data 테이블의 실제 컬럼명에 맞게 이 스크립트의 컬럼명을 수정하세요."
    )

common_index = pivot_dropna.index
df_common = df[df["Open_time"].isin(common_index)].copy()
df_common = df_common.sort_values(["Symbol", "Open_time"]).reset_index(drop=True)

print(f"지표 계산 대상 rows: {len(df_common)} (심볼 {df_common['Symbol'].nunique()}개)")

INDICATOR_PARAMS = {
    "rsi_length": 14,
    "atr_length": 14,
    "sma_short": 20,
    "sma_long": 50,
    "ema_short": 12,
    "ema_long": 26,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
}
LOOKBACK_BY_INDICATOR = {
    "RSI": INDICATOR_PARAMS["rsi_length"],
    "ATR": INDICATOR_PARAMS["atr_length"],
    "SMA_short": INDICATOR_PARAMS["sma_short"],
    "SMA_long": INDICATOR_PARAMS["sma_long"],
    "EMA_short": INDICATOR_PARAMS["ema_short"],
    "EMA_long": INDICATOR_PARAMS["ema_long"],
    "MACD": INDICATOR_PARAMS["macd_slow"] + INDICATOR_PARAMS["macd_signal"],
    "OBV": 0,
}
MAX_LOOKBACK = max(LOOKBACK_BY_INDICATOR.values())
print(f"지표별 lookback: {LOOKBACK_BY_INDICATOR}")
print(f"-> 심볼별 앞에서 자를 행 수(MAX_LOOKBACK): {MAX_LOOKBACK}")

feature_frames = []
for symbol, g in df_common.groupby("Symbol"):
    g = g.sort_values("Open_time").reset_index(drop=True)

    close = g["Close"]
    high = g["High"]
    low = g["Low"]
    volume = g["Volume"]

    feat = pd.DataFrame({"Symbol": symbol, "Open_time": g["Open_time"]})
    feat["Open"] = g["Open"]
    feat["High"] = high
    feat["Low"] = low
    feat["Close"] = close
    feat["Volume"] = volume

    feat["RSI_14"] = ta.rsi(close, length=INDICATOR_PARAMS["rsi_length"])

    macd_df = ta.macd(
        close,
        fast=INDICATOR_PARAMS["macd_fast"],
        slow=INDICATOR_PARAMS["macd_slow"],
        signal=INDICATOR_PARAMS["macd_signal"],
    )
    if macd_df is not None:
        macd_cols = macd_df.columns.tolist()
        feat["MACD"] = macd_df[macd_cols[0]]
        feat["MACD_hist"] = macd_df[macd_cols[1]]
        feat["MACD_signal"] = macd_df[macd_cols[2]]
    else:
        feat["MACD"] = np.nan
        feat["MACD_hist"] = np.nan
        feat["MACD_signal"] = np.nan

    feat["ATR_14"] = ta.atr(high, low, close, length=INDICATOR_PARAMS["atr_length"])
    feat["OBV"] = ta.obv(close, volume)
    feat["SMA_20"] = ta.sma(close, length=INDICATOR_PARAMS["sma_short"])
    feat["SMA_50"] = ta.sma(close, length=INDICATOR_PARAMS["sma_long"])
    feat["EMA_12"] = ta.ema(close, length=INDICATOR_PARAMS["ema_short"])
    feat["EMA_26"] = ta.ema(close, length=INDICATOR_PARAMS["ema_long"])

    feature_frames.append(feat)

feature_df = pd.concat(feature_frames, ignore_index=True)
feature_df = feature_df.sort_values(["Symbol", "Open_time"]).reset_index(drop=True)

print(f"\n지표 계산 완료. feature_df shape: {feature_df.shape}")

# =====================================================
# 12. 결측치(NaN) 처리
# =====================================================

print("\n" + "=" * 70)
print("12. 결측치(NaN) 처리")
print("=" * 70)

nan_before = feature_df.isna().sum()
print("자르기 전 컬럼별 NaN 개수:")
print(nan_before[nan_before > 0].to_string())

trimmed_frames = []
for symbol, g in feature_df.groupby("Symbol"):
    g = g.sort_values("Open_time").reset_index(drop=True)
    trimmed_frames.append(g.iloc[MAX_LOOKBACK:])

feature_df_clean = pd.concat(trimmed_frames, ignore_index=True)
feature_df_clean = feature_df_clean.sort_values(["Symbol", "Open_time"]).reset_index(drop=True)

nan_after = feature_df_clean.isna().sum()
remaining_nan_cols = nan_after[nan_after > 0]

print(f"\n심볼별 {MAX_LOOKBACK}행 제거 후 shape: {feature_df_clean.shape}")

if len(remaining_nan_cols) > 0:
    print("\n⚠️ 경고: 자른 뒤에도 NaN이 남아있는 컬럼이 있습니다.")
    print(remaining_nan_cols.to_string())
else:
    print("\n남은 NaN 없음 확인 완료 (스케일링은 적용하지 않음, 원본 값 그대로 저장)")

loss_pct = 100 * (1 - len(feature_df_clean) / len(feature_df))
print(f"NaN 처리로 인한 데이터 손실률: {loss_pct:.2f}%")

# =====================================================
# 12-1. 피처 테이블에 국면(regime)/split 라벨 부여 
# =====================================================

print("\n" + "=" * 70)
print("12-1. 피처 테이블에 국면/split 라벨 부여")
print("=" * 70)

def label_regime(ts: pd.Timestamp):
    for _, row in regime_df.iterrows():
        if row["start"] <= ts <= row["end"]:
            return row["regime"], row["split"]
    return None, None

labels = feature_df_clean["Open_time"].apply(label_regime)
feature_df_clean["regime"] = labels.apply(lambda x: x[0])
feature_df_clean["split"] = labels.apply(lambda x: x[1])

unlabeled = feature_df_clean["regime"].isna().sum()
if unlabeled > 0:
    print(f"⚠️ {unlabeled}개 행이 어떤 국면에도 속하지 않아 regime/split이 NaN입니다.")
else:
    print("모든 행에 regime/split 라벨 부여 완료")

print(feature_df_clean.groupby(["split", "regime"]).size().to_string())

# =====================================================
# 13. 산출물 저장: 피처 목록 확정
# =====================================================
print("\n" + "=" * 70)
print("13. 산출물 저장: 피처 목록 확정")
print("=" * 70)

feature_columns = [c for c in feature_df_clean.columns if c not in ("Symbol", "Open_time", "regime", "split")]

feature_list_df = pd.DataFrame({
    "feature_name": feature_columns,
    "category": [
        "raw_ohlcv" if c in ("Open", "High", "Low", "Close", "Volume")
        else "momentum" if c in ("RSI_14", "MACD", "MACD_hist", "MACD_signal")
        else "volatility" if c in ("ATR_14",)
        else "volume" if c in ("OBV",)
        else "trend"
        for c in feature_columns
    ],
})

feature_list_df.to_csv(
    os.path.join(output_dir, "feature_list.csv"), index=False, encoding="utf-8-sig"
)
feature_df_clean.to_csv(
    os.path.join(output_dir, "feature_engineered_data.csv"), index=False, encoding="utf-8-sig"
)

print("\n확정된 피처 목록:")
print(feature_list_df.to_string(index=False))

print("\n산출물 저장 완료:")
print(" - feature_list.csv              (확정 피처 목록 + 카테고리)")
print(" - feature_engineered_data.csv   (심볼별 NaN 처리 완료 + regime/split 라벨 포함 피처 테이블)")
