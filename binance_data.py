import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

# matplotlib 한글 폰트 설정 (Windows 환경 기준, Mac이면 'AppleGothic'으로 변경)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

conn = sqlite3.connect(r"C:\Users\ilove\Downloads\crypto_market.db") # 사용자 환경에 맞게 수정 필요
df = pd.read_sql("SELECT * FROM ohlcv_data", conn, parse_dates=["Open_time"])
conn.close()
output_dir = r"C:\Users\ilove\OneDrive\바탕 화면\outputs" # 사용자 환경에 맞게 수정 필요
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
coverage = coverage.sort_values("start", ascending=False)  # 늦게 상장한 애들 위로

print("=" * 70)
print("1. 자산별 커버리지")
print("=" * 70)
print(coverage.to_string(index=False))

# 공통 시작일 = 가장 늦게 상장한 자산의 시작일
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
    # 중복 제거 (첫 번째만 유지) — pivot 전에 반드시 처리
    df = df.drop_duplicates(subset=["Open_time", "Symbol"], keep="first")
    print(f"→ 중복 제거 후 row 수: {len(df)}")

# =====================================================
# 3-1. 커버리지 이상 징후 점검 (신규)
#      -> 8종목 시작일이 완전히 동일하게 나오는 것이 실제 상장일 차이를
#         반영하지 못하는 것인지, 초기 구간이 패딩/평탄화된 데이터인지 점검
# =====================================================

print("\n" + "=" * 70)
print("3-1. 커버리지 이상 징후 점검")
print("=" * 70)

ohlc_cols_present = set(["Open", "High", "Low", "Close"]).issubset(df.columns)
vol_col_present = "Volume" in df.columns

anomaly_records = []
for symbol, g in df.groupby("Symbol"):
    g = g.sort_values("Open_time")
    first_rows = g.head(48)  # 시작 후 48시간(2일) 구간 점검

    flat_close = first_rows["Close"].nunique() <= 1
    zero_volume = (first_rows["Volume"] == 0).all() if vol_col_present else None
    ohlc_equal = (
        (first_rows["Open"] == first_rows["Close"])
        & (first_rows["Close"] == first_rows["High"])
        & (first_rows["High"] == first_rows["Low"])
    ).all() if ohlc_cols_present else None

    expected_hours = int((g["Open_time"].max() - g["Open_time"].min()).total_seconds() // 3600) + 1
    actual_rows = len(g)
    missing_hours = expected_hours - actual_rows

    anomaly_records.append({
        "Symbol": symbol,
        "start": g["Open_time"].min(),
        "first_48h_flat_close": flat_close,
        "first_48h_zero_volume": zero_volume,
        "first_48h_ohlc_equal": ohlc_equal,
        "expected_hours": expected_hours,
        "actual_rows": actual_rows,
        "missing_hours": missing_hours,
    })

anomaly_df = pd.DataFrame(anomaly_records)
print(anomaly_df.to_string(index=False))

flat_flag = anomaly_df["first_48h_flat_close"].fillna(False)
zero_vol_flag = anomaly_df["first_48h_zero_volume"].fillna(False) if vol_col_present else pd.Series([False] * len(anomaly_df))
ohlc_flag = anomaly_df["first_48h_ohlc_equal"].fillna(False) if ohlc_cols_present else pd.Series([False] * len(anomaly_df))

suspicious = anomaly_df[flat_flag | zero_vol_flag | ohlc_flag]

if len(suspicious) > 0:
    print(f"\n⚠️ 경고: {len(suspicious)}개 종목에서 시작 직후 구간이 평탄/거래량0/OHLC동일 -> "
          f"실거래가 아닌 패딩 데이터일 가능성: {suspicious['Symbol'].tolist()}")
else:
    print("\n초기 구간 평탄화/패딩 의심 신호는 없음 (첫 48시간 기준)")

if anomaly_df["start"].nunique() == 1:
    print(
        "\n⚠️ 참고: 8개 종목의 시작일이 모두 완전히 동일합니다.\n"
        "   실제 바이낸스 상장일은 종목마다 다릅니다 (예: SOL/AVAX가 BTC/ETH보다 늦게 상장).\n"
        "   이 결과가 그대로 유지된다면 두 가지 가능성 중 하나입니다:\n"
        "     (a) DB를 만들 때 이미 공통 구간으로 잘라서 저장했거나\n"
        "     (b) 데이터 수집기가 상장 이전 구간을 임의의 값으로 채워 넣었을 가능성\n"
        "   -> DB를 생성한 수집 스크립트(크롤러/API 호출 부분)를 열어서\n"
        "      각 심볼의 조회 시작 파라미터가 고정값인지 확인하고,\n"
        "      필요하면 거래소 API에서 실제 상장일을 별도로 조회해 대조하세요."
    )
else:
    print("\n종목별 시작일이 서로 다름 -> 정상적으로 상장일 차이가 반영된 것으로 보임")

# =====================================================
# 3-2. 산출물 저장: 정합성 리포트 (커버리지 + 갭 + 이상징후)
# =====================================================

coverage.to_csv(os.path.join(output_dir, "coverage_report.csv"), index=False, encoding="utf-8-sig")

if not gap_df.empty:
    gap_df.to_csv(os.path.join(output_dir, "gap_report.csv"), index=False, encoding="utf-8-sig")
else:
    pd.DataFrame(columns=["Symbol", "gap_start", "gap_end", "gap_size"]).to_csv(
        os.path.join(output_dir, "gap_report.csv"), index=False, encoding="utf-8-sig"
    )

anomaly_df.to_csv(os.path.join(output_dir, "coverage_anomaly_check.csv"), index=False, encoding="utf-8-sig")

print("\n산출물 저장 완료:")
print(" - coverage_report.csv        (자산별 커버리지)")
print(" - gap_report.csv             (시간 갭 목록)")
print(" - coverage_anomaly_check.csv (커버리지 이상 징후 점검)")

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

# 손실률 확인
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
# 6. 국면 날짜 정의 (후보값 — 실제 커버리지 결과 보고 조정 필수)
# =====================================================

regime_boundaries = {
    "하락": ("2022-01-01", "2022-12-31"),
    "횡보": ("2023-01-01", "2023-09-30"),
    "상승": ("2023-10-01", "2024-03-31"),
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
    })

regime_df = pd.DataFrame(regime_table)
print("\n" + "=" * 70)
print("6. 국면 날짜 정의")
print("=" * 70)
print(regime_df.to_string(index=False))

# =====================================================
# 7. 누적수익률 + rolling vol 그래프에 국면 경계선 표시
# =====================================================

print("\n" + "=" * 70)
print("7. 누적수익률 + rolling vol 그래프에 국면 경계선 표시")
print("=" * 70)

colors = {"하락": "red", "횡보": "gray", "상승": "green"}
symbols = pivot_dropna.columns.tolist()

for symbol in symbols:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(cum_returns.index, cum_returns[symbol], color="steelblue", linewidth=1)
    ax1.set_title(f"{symbol} - 누적수익률")
    ax1.set_ylabel("Cumulative Return")
    ax1.grid(alpha=0.3)

    ax2.plot(rolling_vol.index, rolling_vol[symbol], color="darkorange", linewidth=1)
    ax2.set_title(f"{symbol} - 24h Rolling Volatility")
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
# 7-1. 국면 날짜 검증 (신규)
#      -> 국면 이름(하락/횡보/상승)이 실제 수익률 방향/변동성과 맞는지 정량 확인
# =====================================================

print("\n" + "=" * 70)
print("7-1. 국면 날짜 검증 (실제 데이터와 국면 이름 일치 여부)")
print("=" * 70)

regime_validation_records = []
for _, row in regime_df.iterrows():
    seg_returns = returns.loc[row["start"]:row["end"]]
    seg_total_return = (1 + seg_returns).prod() - 1  # 종목별 구간 전체 수익률
    seg_vol = seg_returns.std() * np.sqrt(24)  # 종목별 구간 변동성

    regime_validation_records.append({
        "regime": row["regime"],
        "start": row["start"],
        "end": row["end"],
        "avg_total_return_pct": seg_total_return.mean() * 100,
        "pct_assets_positive": (seg_total_return > 0).mean() * 100,
        "avg_vol": seg_vol.mean(),
    })

regime_validation_df = pd.DataFrame(regime_validation_records)
print(regime_validation_df.to_string(index=False))

expect_positive = {"하락": False, "횡보": None, "상승": True}
print("\n국면 이름과 실제 수익률 방향 일치 여부 점검:")
mismatch_found = False
for _, r in regime_validation_df.iterrows():
    name = r["regime"]
    actual_positive = r["avg_total_return_pct"] > 0
    exp = expect_positive.get(name)
    if exp is None:
        note = "횡보 구간 -> 평균 수익률 절대값이 하락/상승 구간보다 작은지 육안 비교 필요"
    elif exp == actual_positive:
        note = "이름과 실제 방향 일치 ✅"
    else:
        note = "⚠️ 불일치: 이름과 실제 평균 수익률 방향이 반대 -> 날짜 재조정 필요"
        mismatch_found = True
    print(f"  [{name}] {r['start'].date()} ~ {r['end'].date()}  "
          f"평균 총수익률 {r['avg_total_return_pct']:.2f}%  ({note})")

if mismatch_found:
    print(
        "\n⚠️ 하나 이상의 국면에서 이름과 실제 방향이 불일치합니다.\n"
        "   regime_boundaries 딕셔너리의 시작/종료일을 조정한 뒤\n"
        "   저장된 {symbol}_regime_plot.png를 다시 열어 육안으로도 확인하세요."
    )
else:
    print("\n모든 국면에서 이름과 실제 방향이 일치함 (횡보 구간은 별도 육안 확인 권장)")

regime_validation_df.to_csv(
    os.path.join(output_dir, "regime_validation.csv"), index=False, encoding="utf-8-sig"
)
print("\n -> regime_validation.csv 저장 완료")

# =====================================================
# 8. 국면별 train/val/test 분할 (국면 내 시간순 70/15/15)
# =====================================================

split_ratio = {"train": 0.7, "val": 0.15, "test": 0.15}

split_records = []
for _, row in regime_df.iterrows():
    seg = pivot_dropna.loc[row["start"]:row["end"]]
    n = len(seg)
    n_train = int(n * split_ratio["train"])
    n_val = int(n * split_ratio["val"])

    train_idx = seg.index[:n_train]
    val_idx = seg.index[n_train:n_train + n_val]
    test_idx = seg.index[n_train + n_val:]

    split_records.append({
        "regime": row["regime"],
        "train_start": train_idx.min() if len(train_idx) else None,
        "train_end": train_idx.max() if len(train_idx) else None,
        "val_start": val_idx.min() if len(val_idx) else None,
        "val_end": val_idx.max() if len(val_idx) else None,
        "test_start": test_idx.min() if len(test_idx) else None,
        "test_end": test_idx.max() if len(test_idx) else None,
    })

split_df = pd.DataFrame(split_records)
print("\n" + "=" * 70)
print("8. 국면별 train/val/test 분할")
print("=" * 70)
print(split_df.to_string(index=False))

# =====================================================
# 9. 왜도(skewness) · 첨도(kurtosis) 자산별 계산 → heavy-tailedness 근거
# =====================================================

skew_kurt = pd.DataFrame({
    "skewness": returns.apply(lambda x: stats.skew(x.dropna())),
    "kurtosis_excess": returns.apply(lambda x: stats.kurtosis(x.dropna())),  # 정규분포=0 기준
})
skew_kurt["heavy_tailed"] = skew_kurt["kurtosis_excess"] > 0

print("\n" + "=" * 70)
print("9. 왜도·첨도 자산별 계산")
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
print(" - coverage_anomaly_check.csv    (커버리지 이상 징후 점검)")
print(" - regime_definition_table.csv   (국면 정의표, 날짜 명시)")
print(" - regime_validation.csv         (국면 날짜 검증 결과)")
print(" - train_val_test_split.csv      (국면별 분할 규칙)")
print(" - skew_kurtosis_by_asset.csv    (자산별 왜도/첨도)")
print(" - {symbol}_regime_plot.png      (자산별 누적수익률+rolling vol 그래프)")
