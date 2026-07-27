"""
Week 2: 시장 국면(Regime) 정의 (CryptoAgent)

[1단계] REGIMES를 비워둔 채 실행
        -> regime_explore.png 생성. 누적수익률 + rolling vol 보고 경계 날짜를 눈으로 확정.
[2단계] 확정한 날짜를 REGIMES에 채우고 재실행
        -> 국면 경계가 그려진 그림 + 국면 정의표 + 국면별/자산별 왜도·첨도 + train/val/test 집계.

설정 4줄(DB_PATH, TS_COL, SYM_COL, CLOSE_COL)만 본인 환경에 맞게.
"""
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── 설정 ─────────────────────────────────────────────
DB_PATH   = "crypto_market.db"
TS_COL    = "Open_time"
SYM_COL   = "Symbol"
CLOSE_COL = "Close"

# 국면 확정 전엔 비워두기(→ 탐색 그림만).
# 확정 후 아래 형식으로 채우기: "이름": ("시작", "끝", "train|val|test")
# 참고 후보(그림 보고 조정):
#   "bull_2021":  ("2021-01-01", "2021-11-09", "train"),
#   "bear_2022":  ("2021-11-10", "2022-12-31", "test"),
#   "side_2023":  ("2023-01-01", "2023-10-15", "test"),
#   "bull_2024":  ("2023-10-16", "2024-03-14", "test"),
REGIMES = {
    "bull_2021":  ("2021-01-01", "2021-11-09", "train"),
    "bear_2022":  ("2021-11-10", "2023-01-01", "train"),
    "side_2023":  ("2023-01-02", "2023-10-15", "val"),
    "bull_2024":  ("2023-10-16", "2025-01-01", "test"),
    "choppy_2025": ("2025-01-02", "2025-12-31", "test"),
}

VOL_WINDOWS = {"24h": 24, "7d": 24 * 7}   # rolling vol 창
# ─────────────────────────────────────────────────────


def load_close():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT {TS_COL},{SYM_COL},{CLOSE_COL} FROM ohlcv_data", con)
    con.close()
    ts = df[TS_COL]
    if pd.api.types.is_numeric_dtype(ts):
        df[TS_COL] = pd.to_datetime(ts, unit="ms" if ts.max() > 1e12 else "s")
    else:
        df[TS_COL] = pd.to_datetime(ts)
    df[CLOSE_COL] = pd.to_numeric(df[CLOSE_COL], errors="coerce")
    wide = df.pivot(index=TS_COL, columns=SYM_COL, values=CLOSE_COL).dropna()
    wide = wide.loc[:"2025-12-31"]
    return wide.sort_index()


close = load_close()
ret = close.pct_change().dropna()          # 자산별 시간봉 수익률
logret = np.log(close).diff().dropna()     # 왜도·첨도용 로그수익률

# 동일가중 포트폴리오 기준선
port_ret = ret.mean(axis=1)
port_cum = (1 + port_ret).cumprod()
btc_cum = (1 + ret["BTCUSDT"]).cumprod() if "BTCUSDT" in ret else None

# ── 그림: 누적수익률(log) + rolling vol ────────────────
fig, ax = plt.subplots(2, 1, figsize=(15, 9), sharex=True,
                       gridspec_kw={"height_ratios": [2, 1]})

ax[0].plot(port_cum.index, port_cum, lw=1.2, label="Equal-weight portfolio")
if btc_cum is not None:
    ax[0].plot(btc_cum.index, btc_cum, lw=1.0, alpha=0.6, label="BTC")
ax[0].set_yscale("log")
ax[0].set_ylabel("Cumulative return (log)")
ax[0].legend(loc="upper left")
ax[0].set_title("Regime exploration — cumulative return & rolling volatility")

for name, w in VOL_WINDOWS.items():
    ann = np.sqrt(24 * 365)
    ax[1].plot(port_ret.index, port_ret.rolling(w).std() * ann,
               lw=0.9, label=f"{name} vol (annualized)")
ax[1].set_ylabel("Volatility")
ax[1].legend(loc="upper left")

# REGIMES가 채워져 있으면 경계선/음영
colors = {"train": "#4C78A8", "val": "#F58518", "test": "#54A24B"}
for name, (s, e, split) in REGIMES.items():
    s, e = pd.Timestamp(s), pd.Timestamp(e)
    for a in ax:
        a.axvspan(s, e, color=colors.get(split, "gray"), alpha=0.08)
        a.axvline(s, color="gray", ls="--", lw=0.7)
    ax[0].text(s, ax[0].get_ylim()[1], f" {name}\n ({split})",
               va="top", fontsize=8)

plt.tight_layout()
plt.savefig("regime_explore.png", dpi=130)
print("저장: regime_explore.png")

# ── 왜도·첨도 (heavy-tail 근거) ────────────────────────
def tail_stats(r):
    return pd.DataFrame({
        "skew": r.skew(),
        "excess_kurtosis": r.kurt(),   # pandas는 excess(정규=0) 기준
        "std": r.std(),
    })

print("\n=== 전체기간 자산별 왜도·첨도 (시간봉 로그수익률) ===")
overall = tail_stats(logret)
print(overall.round(3))
overall.to_csv("tail_stats_overall.csv")

# ── REGIMES 채워졌을 때: 국면 정의표 + 국면별 통계 ──────
if REGIMES:
    rows, tail_by_regime = [], []
    for name, (s, e, split) in REGIMES.items():
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        seg = port_ret.loc[s:e]
        seg_close = close.loc[s:e]
        rows.append({
            "regime": name, "split": split,
            "start": s.date(), "end": e.date(),
            "days": (e - s).days,
            "bars": len(seg),
            "port_return(%)": round((seg_close.mean(axis=1).iloc[-1] /
                                     seg_close.mean(axis=1).iloc[0] - 1) * 100, 1),
            "ann_vol": round(seg.std() * np.sqrt(24 * 365), 3),
        })
        t = tail_stats(logret.loc[s:e]).add_prefix("")
        t.insert(0, "regime", name)
        tail_by_regime.append(t.reset_index())

    regime_table = pd.DataFrame(rows)
    print("\n=== 국면 정의표 ===")
    print(regime_table.to_string(index=False))
    regime_table.to_csv("regime_definition.csv", index=False)

    tail_r = pd.concat(tail_by_regime, ignore_index=True)
    tail_r.to_csv("tail_stats_by_regime.csv", index=False)

    print("\n=== train/val/test 분포 ===")
    print(regime_table.groupby("split")[["days", "bars"]].sum())
    print("\n저장: regime_definition.csv, tail_stats_by_regime.csv")
else:
    print("\n[1단계] REGIMES가 비어있음 → 탐색 그림만 생성.")
    print("regime_explore.png 보고 경계 날짜 확정 후 REGIMES 채워서 재실행.")