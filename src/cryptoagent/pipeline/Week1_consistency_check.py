"""
Week 1: 데이터 정합성 점검 (CryptoAgent)
리포 루트에서 실행: python src/cryptoagent/pipeline/Week1_consistency_check.py
설정 4줄(DB_PATH, TS_COL, SYM_COL, CLOSE_COL)만 본인 스키마에 맞게 수정.
산출물: docs/consistency_report.md, data/processed/coverage.csv, data/processed/gaps.csv
"""
import os
import sqlite3
import pandas as pd
import numpy as np

# ── 설정 (본인 환경에 맞게 수정) ─────────────────────
DB_PATH   = "data/raw/binance_ohlcv.db"   # .db 경로
TS_COL    = "Open_time"         # 시각 컬럼명
SYM_COL   = "Symbol"             # 심볼 컬럼명
CLOSE_COL = "Close"              # 종가 컬럼명
EXPECTED  = pd.Timedelta(hours=1)  # 기대 간격 (1h)
# ────────────────────────────────────────────────────


def md_table(df):
    """tabulate 있으면 markdown, 없으면 monospace로 폴백"""
    try:
        return df.to_markdown()
    except Exception:
        return "```\n" + df.to_string() + "\n```"


if not os.path.exists(DB_PATH):
    raise FileNotFoundError(
        f"{DB_PATH}를 찾을 수 없습니다. fetch_binance_data.py를 먼저 실행했는지 확인하세요."
    )
con = sqlite3.connect(DB_PATH)

# 0. 스키마 확인 ─ 위 설정 컬럼명과 다르면 이 출력 보고 고칠 것
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con)
print("=== 테이블 목록 ===")
print(tables, "\n")
TABLE = tables["name"].iloc[0]   # 테이블이 여러 개면 여기 직접 지정
print(f"=== '{TABLE}' 스키마 ===")
print(pd.read_sql(f"PRAGMA table_info({TABLE})", con), "\n")

# 1. 로드 + 타입 정리
df = pd.read_sql(f"SELECT * FROM {TABLE}", con)

# timestamp 파싱 (ms epoch / s epoch / 문자열 자동 판별)
ts = df[TS_COL]
if pd.api.types.is_numeric_dtype(ts):
    unit = "ms" if ts.max() > 1e12 else "s"
    df[TS_COL] = pd.to_datetime(ts, unit=unit)
else:
    df[TS_COL] = pd.to_datetime(ts)

# object로 들어온 가격 컬럼 숫자화 (dtype 이슈 방어)
df[CLOSE_COL] = pd.to_numeric(df[CLOSE_COL], errors="coerce")
df = df.sort_values([SYM_COL, TS_COL]).reset_index(drop=True)

# 2. 자산별 커버리지
cov = (df.groupby(SYM_COL)[TS_COL]
         .agg(start="min", end="max", rows="count"))
cov["expected_rows"] = ((cov["end"] - cov["start"]) / EXPECTED + 1).astype(int)
cov["missing"] = cov["expected_rows"] - cov["rows"]
print("=== 1) 자산별 커버리지 ===")
print(cov, "\n")

# 3. 시간 갭 점검 (1h 아닌 간격)
gap_rows = []
for sym, g in df.groupby(SYM_COL):
    g = g.sort_values(TS_COL).copy()
    g["gap"]  = g[TS_COL].diff()
    g["prev"] = g[TS_COL].shift()
    bad = g[g["gap"].notna() & (g["gap"] != EXPECTED)]
    for _, r in bad.iterrows():
        gap_rows.append({"symbol": sym, "from": r["prev"],
                         "to": r[TS_COL], "gap": r["gap"]})
gaps = pd.DataFrame(gap_rows)
print(f"=== 2) 1h 아닌 간격: 총 {len(gaps)}건 ===")
print(gaps.head(20), "\n")

# 4. 중복 (timestamp, symbol) ─ 있으면 pivot 깨짐
dup_mask = df.duplicated([TS_COL, SYM_COL], keep=False)
n_dup = int(dup_mask.sum())
print(f"=== 3) 중복 (timestamp, symbol): {n_dup}건 ===")
if n_dup:
    print(df[dup_mask].sort_values([SYM_COL, TS_COL]).head(20), "\n")

# 5. pivot().dropna() 후 실제 공통 학습기간
wide   = df.pivot(index=TS_COL, columns=SYM_COL, values=CLOSE_COL)
before = wide.shape
common = wide.dropna()
after  = common.shape
print("=== 4) 공통 학습기간 ===")
print(f"pivot 원본 shape : {before}")
print(f"dropna 후 shape  : {after}")
if len(common):
    print(f"공통 시작 : {common.index.min()}")
    print(f"공통 종료 : {common.index.max()}")
    print(f"공통 길이 : {len(common)} 봉 (~{len(common)/24:.0f}일)\n")

# 6. 리포트 저장
with open("docs/consistency_report.md", "w", encoding="utf-8") as f:
    f.write("# 데이터 정합성 리포트\n\n")
    f.write("## 1. 자산별 커버리지\n\n" + md_table(cov) + "\n\n")
    f.write(f"## 2. 시간 갭 (1h 아닌 간격): 총 {len(gaps)}건\n\n")
    f.write((md_table(gaps) if len(gaps) else "없음") + "\n\n")
    f.write(f"## 3. 중복 (timestamp, symbol): {n_dup}건\n\n")
    f.write("## 4. 공통 학습기간\n\n")
    f.write(f"- pivot 원본: {before}\n- dropna 후: {after}\n")
    if len(common):
        f.write(f"- 공통 시작: {common.index.min()}\n")
        f.write(f"- 공통 종료: {common.index.max()}\n")
        f.write(f"- 길이: {len(common)} 봉 (~{len(common)/24:.0f}일)\n")

cov.to_csv("data/processed/coverage.csv")
gaps.to_csv("data/processed/gaps.csv", index=False)
con.close()
print("저장 완료: docs/consistency_report.md, data/processed/coverage.csv, data/processed/gaps.csv")