"""
9월 4주차: PPO 정책 행동 진단 (재학습 없이 저장된 backtest_oos.csv만 분석)
리포 루트에서 실행: python experiments/walk_forward_behavior_diagnostics.py

배경
----
walk_forward_benchmarks.py로 만든 비교표에서 PPO(MLP/Transformer)가 두 OOS
구간(bull_2024, choppy_2025) 모두 전통 벤치마크 대비 일관된 우위를 보이지
못했다. 이 스크립트는 그 원인을 진단하기 위해 이미 저장된 backtest_oos.csv의
weights/target_weights만으로 정책의 실제 행동을 분석한다 - 결과를 바꾸려는
재튜닝이 아니라 "왜 이런 결과가 나왔는가"를 설명하기 위한 사후 분석이며,
locked_candidates.json으로 확정된 후보/seed를 그대로 쓴다 (재탐색 없음).

진단 항목 (코덱스 2차 리뷰에서 제안됨)
------------------------------------------
- 국면별 현금·자산 평균 비중과 집중도(HHI, Herfindahl-Hirschman Index)
  HHI가 1/9(완전균등분산의 이론값)에 가까울수록 정책이 실질적으로
  균등분산과 구분되지 않는 행동을 한다는 뜻이다.
- 현금 비중과 낙폭(drawdown) 사이 상관계수: 0에 가까우면 정책이 손실
  국면에서 현금으로 도피하는 방어적 행동을 학습하지 못했다는 뜻이다.
- Turnover: evaluate.py의 compute_turnover_from_weights()와 동일한 정의
  (target_weights[t] - weights[t-1])/2 사용.
- 15% 이상 낙폭 구간 탐지: peak 대비 낙폭이 임계치를 넘는 연속 구간을
  찾아 시작/저점 날짜와 최대낙폭을 기록.
- MLP vs Transformer 행동 차이: 자산별 비중 표준편차, 스텝당 평균 비중
  변화량.

출력
----
results/walk_forward_behavior_diagnostics/diagnostics.json - Artifact 등
후속 분석에서 재사용 가능한 구조화된 결과.
콘솔에도 정책×fold별 요약을 출력.
"""

from __future__ import annotations

import ast
import json
import os

import numpy as np
import pandas as pd

RESULTS_ROOT = "results/walk_forward"
LOCKED_CANDIDATES_PATH = "configs/walk_forward_locked_candidates.json"
OUTPUT_DIR = "results/walk_forward_behavior_diagnostics"

FOLDS = [
    ("fold1_final", "bull_2024"),
    ("fold2_final", "choppy_2025"),
]

# adapter.py COLUMN_RENAME 및 benchmark_buy_and_hold_equal_weight.py의
# TIC_ORDER와 동일 순서 (현금 + 8자산, env._tic_list 기준 알파벳순)
TIC_ORDER_WITH_CASH = ["CASH", "ADA", "AVAX", "BNB", "BTC", "DOGE", "ETH", "SOL", "XRP"]

DD_THRESHOLD = 0.15  # 15% 이상 낙폭만 "주요 낙폭 구간"으로 집계


def load_locked_candidates() -> dict:
    with open(LOCKED_CANDIDATES_PATH) as f:
        return json.load(f)


def load_backtest(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df["weights"] = df["weights"].apply(ast.literal_eval)
    df["target_weights"] = df["target_weights"].apply(ast.literal_eval)
    return df


def herfindahl(w: np.ndarray) -> float:
    """집중도 지수(HHI). 완전균등분산(N자산 각 1/N)이면 1/N, 단일자산 집중이면 1."""
    return float(np.sum(np.asarray(w) ** 2))


def find_drawdown_periods(portfolio_values: np.ndarray, dates: list, threshold: float) -> list[tuple[str, str, float]]:
    """peak 대비 낙폭이 threshold를 넘는 연속 구간의 (시작일, 저점일, 최대낙폭)."""
    values = np.asarray(portfolio_values)
    peak = np.maximum.accumulate(values)
    dd = (values - peak) / peak
    in_dd = dd < -threshold

    periods = []
    start = None
    for i, flag in enumerate(in_dd):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            trough_idx = start + int(np.argmin(dd[start:i]))
            periods.append((str(dates[start].date()), str(dates[trough_idx].date()), float(dd[trough_idx])))
            start = None
    if start is not None:
        trough_idx = start + int(np.argmin(dd[start:]))
        periods.append((str(dates[start].date()), str(dates[trough_idx].date()), float(dd[trough_idx])))
    return periods


def diagnose_one_run(path: str) -> dict:
    df = load_backtest(path)
    W = np.vstack(df["weights"].to_numpy())  # (T, 9)
    TW = np.vstack(df["target_weights"].to_numpy())

    turnover = np.abs(TW[1:] - W[:-1]).sum(axis=1) / 2.0  # evaluate.py와 동일 정의

    values = df["portfolio_values"].to_numpy()
    peak = np.maximum.accumulate(values)
    dd = (values - peak) / peak
    cash = W[:, 0]
    cash_dd_corr = float(np.corrcoef(cash, dd)[0, 1])

    dd_periods = find_drawdown_periods(values, df.index.tolist(), DD_THRESHOLD)

    return {
        "avg_weights": W.mean(axis=0),
        "weight_std": W.std(axis=0),
        "hhi": float(np.mean([herfindahl(w) for w in W])),
        "avg_turnover": float(turnover.mean()),
        "mean_step_change": float(np.abs(np.diff(W, axis=0)).mean()),
        "cash_dd_correlation": cash_dd_corr,
        "drawdown_periods": dd_periods,
    }


def diagnose_policy_fold(policy: str, fold: str, candidate: str, seeds: list[int]) -> dict:
    per_seed = []
    for seed in seeds:
        path = os.path.join(RESULTS_ROOT, policy, fold, candidate, f"seed{seed}", "backtest_oos.csv")
        per_seed.append(diagnose_one_run(path))

    avg_weights = np.mean([s["avg_weights"] for s in per_seed], axis=0)
    weight_std = np.mean([s["weight_std"] for s in per_seed], axis=0)

    return {
        "policy": policy,
        "fold": fold,
        "candidate": candidate,
        "n_seeds": len(seeds),
        "avg_weights": avg_weights.tolist(),
        "weight_std": weight_std.tolist(),
        "hhi": float(np.mean([s["hhi"] for s in per_seed])),
        "avg_turnover": float(np.mean([s["avg_turnover"] for s in per_seed])),
        "mean_step_change": float(np.mean([s["mean_step_change"] for s in per_seed])),
        "cash_dd_correlation": float(np.mean([s["cash_dd_correlation"] for s in per_seed])),
        # 낙폭 구간은 seed42(대표)만 기록 - seed마다 미세하게 다른 날짜를 전부 나열하면 노이즈만 커짐
        "major_drawdowns": per_seed[0]["drawdown_periods"],
    }


def print_summary(result: dict) -> None:
    w = result["avg_weights"]
    print(f"\n=== {result['policy']} / {result['fold']} (candidate={result['candidate']}, n_seeds={result['n_seeds']}) ===")
    print("  평균 비중:", {tic: round(w[i], 4) for i, tic in enumerate(TIC_ORDER_WITH_CASH)})
    print(f"  HHI(집중도) = {result['hhi']:.4f}  (완전균등분산 기준값 = {1 / len(TIC_ORDER_WITH_CASH):.4f})")
    print(f"  평균 turnover = {result['avg_turnover']:.5f}")
    print(f"  cash-drawdown 상관계수 = {result['cash_dd_correlation']:.4f}  (0에 가까우면 방어행동 없음)")
    print(f"  15%+ 낙폭 구간 수(seed42 기준) = {len(result['major_drawdowns'])}")


def main() -> None:
    locked = load_locked_candidates()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    for policy in ("mlp", "transformer"):
        candidate = locked[policy]["candidate"]
        seeds = locked[policy]["seeds"]
        for fold, fold_label in FOLDS:
            result = diagnose_policy_fold(policy, fold, candidate, seeds)
            result["fold_label"] = fold_label
            results.append(result)
            print_summary(result)

    output_path = os.path.join(OUTPUT_DIR, "diagnostics.json")
    with open(output_path, "w") as f:
        json.dump({"tic_order": TIC_ORDER_WITH_CASH, "rows": results}, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    main()
