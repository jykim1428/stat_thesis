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

target_weights를 분석 대상으로 쓰는 이유 (2026-09-07 코덱스 리뷰 반영, Major)
------------------------------------------------------------------------------
최초 구현은 weights(가격 변동을 반영한 사후 비중)로 HHI/비중변화/cash-drawdown
상관을 계산했다. 하지만 정책이 실제로 "무엇을 하려 했는지"를 보려면
target_weights(그 스텝에서 에이전트가 지시한 목표 비중, env._actions_memory)를
봐야 한다 - weights는 target_weights에 가격 변동이 누적된 결과라 정책이
가만히 있어도 시간에 따라 흔들린다. 코덱스 리뷰가 target_weights를 직접
검산해서 발견한 사실:
    - target_weights의 평균 스텝 변화량이 약 3e-9~1e-8 (float32 정밀도 수준)
    - 자산별 전체 구간 최대 변동폭이 6.1e-7 이하
    - 즉 각 학습된 정책은 OOS 전체 구간에서 사실상 완전히 고정된 목표 비중을
      출력한다 (관측이 바뀌어도 action이 사실상 바뀌지 않음)
    - PPO와 8자산 Equal-Weight 벤치마크의 시간별 순수익률 상관계수가
      0.9991~0.9997
이는 "약간의 잡음이 섞인 균등분산"보다 훨씬 강한 사실이다 - 상태 의존적인
동적 자산배분 행동 자체가 관찰되지 않는다. 실측으로 재확인함(python -c로
target_weights의 np.diff 최대/평균값과 MLP-EW returns 상관계수를 직접 계산).

HHI 비교 기준 불일치 수정 (Major)
--------------------------------------
이전 버전은 "HHI 1/9(0.111)가 완전균등분산 기준값"이라고 표현했는데, 이는
현금을 포함한 9원소 균등분배 기준이다. 실제 Equal-Weight 벤치마크
(benchmark_buy_and_hold_equal_weight.py)는 현금 0%, 8자산 각 1/8이라 HHI가
1/8(0.125)이다. 이 스크립트는 두 기준(HHI_9, HHI_8_ASSETS_ONLY)을 모두
계산해서 혼동하지 않게 한다.

진단 항목
------------
- target_weights 기준 평균 비중, 시간에 따른 변화량(스텝당/전체구간),
  집중도(HHI, 현금 포함 9원소 기준과 8자산만 재정규화한 기준 둘 다)
- target cash 비중과 낙폭(drawdown) 사이 상관계수, 그리고 "낙폭 구간 vs
  평시" target cash 평균 차이 - target 자체가 거의 상수이므로 이 차이가
  방어 행동 부재의 직접적인 증거
- PPO returns와 Equal-Weight 벤치마크 returns의 상관계수 및 tracking error
  (results/walk_forward_benchmarks/에서 로드, 없으면 스킵)
- Turnover: evaluate.py의 compute_turnover_from_weights()와 동일한 정의
  (target_weights[t] - weights[t-1])/2 사용
- 15% 이상 낙폭 구간 탐지 (weights 기준 portfolio_values는 실제 자산가치
  경로이므로 그대로 사용)

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
BENCHMARKS_ROOT = "results/walk_forward_benchmarks"
LOCKED_CANDIDATES_PATH = "configs/walk_forward_locked_candidates.json"
OUTPUT_DIR = "results/walk_forward_behavior_diagnostics"

FOLDS = [
    ("fold1_final", "fold1_oos", "bull_2024"),
    ("fold2_final", "fold2_oos", "choppy_2025"),
]

# adapter.py COLUMN_RENAME 및 benchmark_buy_and_hold_equal_weight.py의
# TIC_ORDER와 동일 순서 (현금 + 8자산, env._tic_list 기준 알파벳순)
TIC_ORDER_WITH_CASH = ["CASH", "ADA", "AVAX", "BNB", "BTC", "DOGE", "ETH", "SOL", "XRP"]
N_ASSETS = 8  # 현금 제외

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


def herfindahl_assets_only(w_with_cash: np.ndarray) -> float:
    """현금을 뺀 8자산만 재정규화한 HHI. Equal-Weight 벤치마크(8자산 각 1/8, HHI=0.125)와
    비교하려면 현금 포함 9원소 HHI(0.111)가 아니라 이 기준을 써야 한다."""
    assets = np.asarray(w_with_cash)[1:]
    total = assets.sum()
    if total <= 0:
        return float("nan")
    normalized = assets / total
    return float(np.sum(normalized ** 2))


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


def diagnose_one_run(path: str, ew_returns: pd.Series | None) -> dict:
    df = load_backtest(path)
    W = np.vstack(df["weights"].to_numpy())  # (T, 9) - 가격변동 반영 사후 비중
    TW = np.vstack(df["target_weights"].to_numpy())  # (T, 9) - 정책이 실제 지시한 목표 비중

    turnover = np.abs(TW[1:] - W[:-1]).sum(axis=1) / 2.0  # evaluate.py와 동일 정의

    values = df["portfolio_values"].to_numpy()
    peak = np.maximum.accumulate(values)
    dd = (values - peak) / peak
    dd_periods = find_drawdown_periods(values, df.index.tolist(), DD_THRESHOLD)

    # target_weights 기준 행동 분석 (정책이 실제로 "하려 한" 것)
    target_step_change = np.abs(np.diff(TW, axis=0))
    target_range = TW.max(axis=0) - TW.min(axis=0)  # 자산별 전체 구간 변동폭
    target_cash = TW[:, 0]
    target_cash_dd_corr = float(np.corrcoef(target_cash, dd)[0, 1])

    in_dd_mask = dd < -DD_THRESHOLD
    target_cash_in_dd = float(target_cash[in_dd_mask].mean()) if in_dd_mask.any() else None
    target_cash_outside_dd = float(target_cash[~in_dd_mask].mean())

    result = {
        "avg_target_weights": TW.mean(axis=0),
        "target_hhi_9": float(np.mean([herfindahl(w) for w in TW])),
        "target_hhi_8assets": float(np.mean([herfindahl_assets_only(w) for w in TW])),
        "target_mean_step_change": float(target_step_change.mean()),
        "target_max_step_change": float(target_step_change.max()),
        "target_range_per_asset": target_range.tolist(),
        "target_cash_dd_correlation": target_cash_dd_corr,
        "target_cash_in_dd": target_cash_in_dd,
        "target_cash_outside_dd": target_cash_outside_dd,
        "avg_turnover": float(turnover.mean()),
        "drawdown_periods": dd_periods,
    }

    if ew_returns is not None:
        common_idx = df.index.intersection(ew_returns.index)
        ppo_r = df.loc[common_idx, "returns"]
        ew_r = ew_returns.loc[common_idx]
        corr = float(np.corrcoef(ppo_r, ew_r)[0, 1])
        tracking_error = float((ppo_r - ew_r).std())
        result["ppo_ew_return_correlation"] = corr
        result["ppo_ew_tracking_error"] = tracking_error

    return result


def load_ew_returns(benchmark_fold_dir: str) -> pd.Series | None:
    path = os.path.join(benchmark_fold_dir, "equal_weight.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df["returns"]


def diagnose_policy_fold(policy: str, fold: str, candidate: str, seeds: list[int], ew_returns: pd.Series | None) -> dict:
    per_seed = [
        diagnose_one_run(os.path.join(RESULTS_ROOT, policy, fold, candidate, f"seed{seed}", "backtest_oos.csv"), ew_returns)
        for seed in seeds
    ]

    avg_target_weights = np.mean([s["avg_target_weights"] for s in per_seed], axis=0)

    result = {
        "policy": policy,
        "fold": fold,
        "candidate": candidate,
        "n_seeds": len(seeds),
        "avg_target_weights": avg_target_weights.tolist(),
        "target_hhi_9": float(np.mean([s["target_hhi_9"] for s in per_seed])),
        "target_hhi_8assets": float(np.mean([s["target_hhi_8assets"] for s in per_seed])),
        "target_mean_step_change": float(np.mean([s["target_mean_step_change"] for s in per_seed])),
        "target_max_step_change": float(np.max([s["target_max_step_change"] for s in per_seed])),
        "avg_turnover": float(np.mean([s["avg_turnover"] for s in per_seed])),
        "target_cash_dd_correlation": float(np.mean([s["target_cash_dd_correlation"] for s in per_seed])),
        "target_cash_in_dd": float(np.mean([s["target_cash_in_dd"] for s in per_seed if s["target_cash_in_dd"] is not None]))
            if any(s["target_cash_in_dd"] is not None for s in per_seed) else None,
        "target_cash_outside_dd": float(np.mean([s["target_cash_outside_dd"] for s in per_seed])),
        # 낙폭 구간은 seed42(대표)만 기록 - seed마다 미세하게 다른 날짜를 전부 나열하면 노이즈만 커짐
        "major_drawdowns": per_seed[0]["drawdown_periods"],
    }

    if "ppo_ew_return_correlation" in per_seed[0]:
        result["ppo_ew_return_correlation"] = float(np.mean([s["ppo_ew_return_correlation"] for s in per_seed]))
        result["ppo_ew_tracking_error"] = float(np.mean([s["ppo_ew_tracking_error"] for s in per_seed]))

    return result


def print_summary(result: dict) -> None:
    w = result["avg_target_weights"]
    print(f"\n=== {result['policy']} / {result['fold']} (candidate={result['candidate']}, n_seeds={result['n_seeds']}) ===")
    print("  평균 target 비중:", {tic: round(w[i], 4) for i, tic in enumerate(TIC_ORDER_WITH_CASH)})
    print(f"  target HHI(현금+8자산, 9원소 기준) = {result['target_hhi_9']:.4f}  (완전균등분산 기준값 = {1/9:.4f})")
    print(f"  target HHI(8자산만 재정규화, EW 벤치마크와 비교용) = {result['target_hhi_8assets']:.4f}  (EW 기준값 = {1/8:.4f})")
    print(f"  target 평균 스텝 변화량 = {result['target_mean_step_change']:.2e}  (거의 0이면 정책 출력이 사실상 상수)")
    print(f"  target 최대 스텝 변화량 = {result['target_max_step_change']:.2e}")
    print(f"  평균 turnover = {result['avg_turnover']:.5f}")
    print(f"  target cash-drawdown 상관계수 = {result['target_cash_dd_correlation']:.4f}")
    if result["target_cash_in_dd"] is not None:
        print(f"  target cash 평균: 낙폭구간={result['target_cash_in_dd']:.5f} vs 평시={result['target_cash_outside_dd']:.5f}"
              f" (거의 동일하면 손실 국면에서도 목표 현금비중을 바꾸지 않았다는 뜻)")
    if "ppo_ew_return_correlation" in result:
        print(f"  PPO-EqualWeight returns 상관계수 = {result['ppo_ew_return_correlation']:.4f}"
              f", tracking error = {result['ppo_ew_tracking_error']:.6f}")
    print(f"  15%+ 낙폭 구간 수(seed42 기준) = {len(result['major_drawdowns'])}")


def main() -> None:
    locked = load_locked_candidates()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    for policy in ("mlp", "transformer"):
        candidate = locked[policy]["candidate"]
        seeds = locked[policy]["seeds"]
        for fold, benchmark_fold, fold_label in FOLDS:
            ew_returns = load_ew_returns(os.path.join(BENCHMARKS_ROOT, benchmark_fold))
            result = diagnose_policy_fold(policy, fold, candidate, seeds, ew_returns)
            result["fold_label"] = fold_label
            results.append(result)
            print_summary(result)

    output_path = os.path.join(OUTPUT_DIR, "diagnostics.json")
    with open(output_path, "w") as f:
        json.dump({"tic_order": TIC_ORDER_WITH_CASH, "rows": results}, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    main()
