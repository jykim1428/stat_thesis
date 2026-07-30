"""
============================================================
Week 2 : 시장 국면 정의 (Regime Definition)
============================================================

[목적]
- BTC 시장 흐름 분석 결과를 기반으로
  최종 시장 국면(Bull / Bear / Side)을 정의한다.
- 이후 데이터 분할 및 시장 국면별 성능 분석의 기준으로 활용한다.

[입력 데이터]
- data/crypto_market.db

[산출물]
- results/regime_definition.csv

[비고]
- 국면 날짜는 BTC 누적수익률, Rolling Volatility,
  주요 고점/저점을 기반으로 사전에 정의한다.
"""

import os
import pandas as pd


# =====================================================
# 설정
# =====================================================

SAVE_DIR = "results"

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


# =====================================================
# 최종 시장 국면 정의
# =====================================================

REGIMES = [

    {
        "Regime": "Bull_2021",
        "Start": "2021-01-01",
        "End": "2021-11-09"
    },

    {
        "Regime": "Bear_2022",
        "Start": "2021-11-10",
        "End": "2022-12-31"
    },

    {
        "Regime": "Side_2023",
        "Start": "2023-01-01",
        "End": "2023-10-31"
    },

    {
        "Regime": "Bull_2023_2025",
        "Start": "2023-11-01",
        "End": "2025-08-31"
    },

    {
        "Regime": "Bear_2025",
        "Start": "2025-09-01",
        "End": "2025-12-31"
    }

]


# =====================================================
# DataFrame 생성
# =====================================================

regime_definition = pd.DataFrame(REGIMES)


regime_definition["Start"] = pd.to_datetime(
    regime_definition["Start"]
)

regime_definition["End"] = pd.to_datetime(
    regime_definition["End"]
)


regime_definition["Days"] = (
    regime_definition["End"]
    -
    regime_definition["Start"]
).dt.days + 1



# =====================================================
# 저장
# =====================================================

save_path = os.path.join(
    SAVE_DIR,
    "regime_definition.csv"
)


regime_definition.to_csv(
    save_path,
    index=False,
    encoding="utf-8-sig"
)


# =====================================================
# 출력
# =====================================================

print("=" * 60)
print("Regime Definition")
print("=" * 60)

print(regime_definition)

print("=" * 60)

print("Saved :", save_path)