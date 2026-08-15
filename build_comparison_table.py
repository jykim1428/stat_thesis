"""
8월 4주차 — 비교표 생성 스크립트
====================================

results/benchmarks/*.csv (4종) + results/ppo_mlp/backtest_test.csv (PPO MLP)를
공용 evaluate()로 채점해서 비교표 1장(csv + 콘솔 출력)으로 완성한다.

실행 위치: 프로젝트 루트 (stat_thesis/)
    python build_comparison_table.py

출력:
    results/comparison_table.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# src/ 를 path에 추가해서 evaluate() import
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cryptoagent.envs.evaluate import evaluate, PERIODS_PER_YEAR  # noqa: E402


# 평가할 대상들: (표시 이름, csv 경로)
TARGETS = [
    ("Buy & Hold BTC",  "results/benchmarks/buy_and_hold_btc.csv"),
    ("Equal-Weight",    "results/benchmarks/equal_weight.csv"),
    ("Markowitz",       "results/benchmarks/markowitz.csv"),
    ("Risk Parity",     "results/benchmarks/risk_parity.csv"),
    ("PPO (MLP)",       "results/ppo_mlp/backtest_test.csv"),
]

OUTPUT_PATH = "results/comparison_table.csv"

# 기준 N periods (PPO 기준). 다른 값이 나오면 데이터 주기 불일치 가능성 -> 경고.
EXPECTED_N_PERIODS = 19320


def load_metrics_df(csv_path: str) -> pd.DataFrame:
    """CSV를 evaluate()가 요구하는 형식(date index, weights=array)으로 로드."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.set_index("date")
    # weights 컬럼이 문자열 "[1.0, 0.0, ...]" 형태(표준 JSON 배열)이므로 json.loads로 파싱
    df["weights"] = df["weights"].apply(json.loads).apply(np.array)
    return df


def main():
    rows = []
    failed = []

    for name, path in TARGETS:
        p = Path(path)
        if not p.exists():
            print(f"[SKIP] {name}: 파일 없음 ({path})")
            failed.append(name)
            continue

        print(f"[채점 중] {name} ...")
        try:
            metrics_df = load_metrics_df(path)
            result = evaluate(metrics_df, periods_per_year=PERIODS_PER_YEAR)
        except Exception as e:
            # 개별 파일 실패가 전체 실행을 막지 않도록 여기서 잡고 다음 파일로 진행
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            failed.append(name)
            continue

        # N periods 검증: PPO(19320) 기준과 다르면 데이터 주기/구간 불일치 가능성 -> 경고만 하고 계속 진행
        if result["n_periods"] != EXPECTED_N_PERIODS:
            print(
                f"  ⚠️  경고: {name}의 N periods={result['n_periods']} "
                f"(기대값={EXPECTED_N_PERIODS}과 불일치 — 데이터 주기/구간이 다를 수 있음, 결과 비교 시 주의)"
            )

        rows.append(
            {
                "전략": name,
                "CAGR": result["cagr"],
                "Sharpe": result["sharpe"],
                "Sortino": result["sortino"],
                "Calmar": result["calmar"],
                "MDD": result["mdd"],
                "Volatility": result["vol"],
                "Avg Turnover": result["avg_turnover"],
                "Total Cost": result["total_cost"],
                "N periods": result["n_periods"],
            }
        )

    if failed:
        print(f"\n[요약] 실패/누락된 항목: {', '.join(failed)}")

    if not rows:
        print("채점된 결과가 없습니다. 파일 경로를 확인하세요.")
        return

    table = pd.DataFrame(rows).set_index("전략")

    # 콘솔에 보기 좋게 출력 (퍼센트 포맷)
    display_table = table.copy()
    for col in ["CAGR", "MDD", "Volatility", "Avg Turnover", "Total Cost"]:
        display_table[col] = display_table[col].map(lambda x: f"{x:.2%}")
    for col in ["Sharpe", "Sortino", "Calmar"]:
        display_table[col] = display_table[col].map(lambda x: f"{x:.4f}")

    print("\n" + "=" * 80)
    print("비교표 (PPO(MLP) + 전통 벤치마크 4종)")
    print("=" * 80)
    print(display_table.to_string())

    # 원본(숫자) 값으로 CSV 저장 — 나중에 추가 분석/plot 할 때 편하도록
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_PATH, encoding="utf-8-sig")
    print(f"\n[OK] 저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()