"""
10월 1주차: PPO vs 벤치마크 / MLP vs Transformer 통계 검정
리포 루트에서 실행: python experiments/walk_forward_statistical_tests.py

배경
----
walk_forward_benchmarks.py의 비교표는 PPO(seed 5개 반복)의 mean Sharpe와
벤치마크(결정론적, 반복 없음)의 point estimate를 나란히 놓은 "기술통계
순위"일 뿐 가설검정이 아니다 (2026-09-07 코덱스 3차 리뷰). 엄밀한 통계
추론이 필요하면 같은 시간 인덱스에서 PPO-벤치마크 수익률 차이를 block
bootstrap으로 리샘플해 Sharpe 차이의 신뢰구간을 구해야 한다.

방법론
------
- arch.bootstrap.StationaryBootstrap으로 net_returns 차이 시계열을 리샘플
  (block_size=24 - 1시간봉 데이터의 하루 주기성을 고려한 블록 길이).
- PPO vs 벤치마크: 각 PPO seed의 returns에서 벤치마크 returns를 뺀 차이
  시계열의 Sharpe를 통계량으로 잡아 95% CI 계산. CI가 0을 포함하면
  "이 두 국면에서는 통계적으로 구분되지 않는다"로 해석.
- MLP vs Transformer: paired seed(같은 seed 인덱스끼리)의 수익률 차이로
  동일하게 계산.
- 비교가 여러 개(정책 2 x 벤치마크 4 x fold 2 = 16개)이므로 Holm-Bonferroni
  보정을 적용해 p-value 부풀림을 막는다.
- OOS 국면이 bull_2024/choppy_2025 2개뿐이므로 "모든 시장 국면에
  일반화된다"는 주장은 하지 않는다 - 이 두 국면 한정 결론이다.

주의
----
재학습이나 재탐색이 아니라, 이미 저장된 backtest_oos.csv/benchmark csv로
사후 통계 검정만 수행한다. 이 결과를 보고 후보나 하이퍼파라미터를 다시
조정하지 않는다 (OOS 확인 후 재조정 금지 원칙).

출력
----
results/walk_forward_statistical_tests/bootstrap_ci.json
콘솔에 정책×벤치마크×fold별 CI와 Holm 보정 후 유의성 여부 출력.
"""

from __future__ import annotations

import ast
import json
import os

import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap

RESULTS_ROOT = "results/walk_forward"
BENCHMARKS_ROOT = "results/walk_forward_benchmarks"
LOCKED_CANDIDATES_PATH = "configs/walk_forward_locked_candidates.json"
OUTPUT_DIR = "results/walk_forward_statistical_tests"

FOLDS = [
    ("fold1_final", "fold1_oos", "bull_2024"),
    ("fold2_final", "fold2_oos", "choppy_2025"),
]

BENCHMARK_STRATEGIES = ["buy_and_hold_btc", "equal_weight", "markowitz", "risk_parity"]

BLOCK_SIZE = 24  # 1시간봉 데이터의 하루 주기성을 고려
N_BOOTSTRAP_REPS = 2000
PERIODS_PER_YEAR = 24 * 365
CI_LEVEL = 0.95


def load_locked_candidates() -> dict:
    with open(LOCKED_CANDIDATES_PATH) as f:
        return json.load(f)


def load_returns(path: str) -> pd.Series:
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df["returns"]


def load_net_returns(path: str, cost_rate: float) -> pd.Series:
    """evaluate.py와 동일하게 turnover 비용을 차감한 net returns."""
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df["weights"] = df["weights"].apply(ast.literal_eval)
    df["target_weights"] = df["target_weights"].apply(ast.literal_eval)

    W = np.vstack(df["weights"].to_numpy())
    TW = np.vstack(df["target_weights"].to_numpy())
    turnover = np.zeros(len(df))
    turnover[1:] = np.abs(TW[1:] - W[:-1]).sum(axis=1) / 2.0

    net = df["returns"] - turnover * cost_rate
    net.index = df.index
    return net


def sharpe_of_diff(diff_returns: np.ndarray) -> float:
    std = diff_returns.std()
    if std == 0 or not np.isfinite(std):
        return 0.0
    return float(diff_returns.mean() / std * np.sqrt(PERIODS_PER_YEAR))


def bootstrap_sharpe_ci(diff_returns: pd.Series, seed: int = 0) -> tuple[float, float, float]:
    """diff_returns(PPO net returns - 상대방 returns) 시계열의 Sharpe 통계량에 대한
    stationary block bootstrap 95% CI. 반환: (point_estimate, ci_low, ci_high)."""
    values = diff_returns.dropna().to_numpy()
    point = sharpe_of_diff(values)
    bs = StationaryBootstrap(BLOCK_SIZE, values, seed=seed)
    ci = bs.conf_int(sharpe_of_diff, reps=N_BOOTSTRAP_REPS, method="percentile", size=CI_LEVEL)
    return point, float(ci[0, 0]), float(ci[1, 0])


def holm_bonferroni(p_values: list[float]) -> list[bool]:
    """Holm-Bonferroni 보정. 반환: 각 비교가 alpha=0.05에서 유의한지 여부(원래 순서 유지)."""
    alpha = 0.05
    order = np.argsort(p_values)
    m = len(p_values)
    reject = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if p_values[idx] < threshold:
            reject[idx] = True
        else:
            break  # Holm: 한 번 기각 실패하면 이후(더 큰 p-value)는 전부 채택
    return reject


def ci_excludes_zero_as_pvalue_proxy(ci_low: float, ci_high: float) -> float:
    """CI가 0을 포함하는지를 이용한 근사 p-value (0이면 유의, 1이면 완전히 겹침).
    정확한 bootstrap p-value 대신 Holm 보정 순위 매기기 용도의 근사치."""
    if ci_low > 0 or ci_high < 0:
        return 0.01  # 유의: CI가 0을 완전히 배제
    span = ci_high - ci_low
    if span == 0:
        return 1.0
    dist_to_zero = min(abs(ci_low), abs(ci_high))
    return float(min(1.0, dist_to_zero / (span / 2) + 0.5))


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    locked = load_locked_candidates()

    ppo_vs_benchmark_results = []
    for policy_key, candidate_field in [("mlp", "candidate03"), ("transformer", "candidate04")]:
        candidate = locked[policy_key]["candidate"]
        seeds = locked[policy_key]["seeds"]
        cost_rate = locked.get("cost_rate", 0.001)

        for fold, benchmark_fold, fold_label in FOLDS:
            for bench_strategy in BENCHMARK_STRATEGIES:
                bench_path = os.path.join(BENCHMARKS_ROOT, benchmark_fold, f"{bench_strategy}.csv")
                bench_returns = load_returns(bench_path)

                seed_cis = []
                for seed in seeds:
                    ppo_path = os.path.join(RESULTS_ROOT, policy_key, fold, candidate, f"seed{seed}", "backtest_oos.csv")
                    ppo_net_returns = load_net_returns(ppo_path, cost_rate)

                    common_idx = ppo_net_returns.index.intersection(bench_returns.index)
                    diff = ppo_net_returns.loc[common_idx] - bench_returns.loc[common_idx]
                    point, lo, hi = bootstrap_sharpe_ci(diff, seed=seed)
                    seed_cis.append({"seed": seed, "point": point, "ci_low": lo, "ci_high": hi})

                mean_point = float(np.mean([s["point"] for s in seed_cis]))
                mean_lo = float(np.mean([s["ci_low"] for s in seed_cis]))
                mean_hi = float(np.mean([s["ci_high"] for s in seed_cis]))

                ppo_vs_benchmark_results.append({
                    "policy": policy_key,
                    "fold": fold,
                    "fold_label": fold_label,
                    "benchmark": bench_strategy,
                    "sharpe_diff_mean": mean_point,
                    "sharpe_diff_ci_low": mean_lo,
                    "sharpe_diff_ci_high": mean_hi,
                    "excludes_zero": mean_lo > 0 or mean_hi < 0,
                    "per_seed": seed_cis,
                })

    p_proxies = [ci_excludes_zero_as_pvalue_proxy(r["sharpe_diff_ci_low"], r["sharpe_diff_ci_high"]) for r in ppo_vs_benchmark_results]
    significant = holm_bonferroni(p_proxies)
    for r, sig in zip(ppo_vs_benchmark_results, significant):
        r["significant_after_holm"] = bool(sig)

    print("=== PPO vs 벤치마크: Sharpe 차이 95% block bootstrap CI (Holm 보정 후 유의성) ===")
    for r in ppo_vs_benchmark_results:
        sig_mark = "유의함" if r["significant_after_holm"] else "유의하지 않음"
        print(f"  {r['policy']:12s} vs {r['benchmark']:18s} [{r['fold_label']}]: "
              f"diff={r['sharpe_diff_mean']:+.3f}  CI=[{r['sharpe_diff_ci_low']:+.3f}, {r['sharpe_diff_ci_high']:+.3f}]  -> {sig_mark}")

    # MLP vs Transformer paired seed 차이 (2026-09-07 코덱스 리뷰에서 이미 제시된 값 재현+검증)
    mlp_seeds = locked["mlp"]["seeds"]
    tf_seeds = locked["transformer"]["seeds"]
    assert mlp_seeds == tf_seeds, "MLP/Transformer seed 목록이 달라 paired 비교 불가"

    mlp_vs_tf_results = []
    for fold, _, fold_label in FOLDS:
        sharpe_diffs = []
        for seed in mlp_seeds:
            mlp_metrics_path = os.path.join(RESULTS_ROOT, "mlp", fold, locked["mlp"]["candidate"], f"seed{seed}", "metrics.json")
            tf_metrics_path = os.path.join(RESULTS_ROOT, "transformer", fold, locked["transformer"]["candidate"], f"seed{seed}", "metrics.json")
            with open(mlp_metrics_path) as f:
                mlp_sharpe = json.load(f)["sharpe"]
            with open(tf_metrics_path) as f:
                tf_sharpe = json.load(f)["sharpe"]
            sharpe_diffs.append(tf_sharpe - mlp_sharpe)

        sharpe_diffs = np.array(sharpe_diffs)
        mean_diff = float(sharpe_diffs.mean())
        se = float(sharpe_diffs.std(ddof=1) / np.sqrt(len(sharpe_diffs)))
        # n=5라 t분포(df=4) 사용 - 코덱스가 제시한 값과 동일 방법
        from scipy import stats
        t_crit = float(stats.t.ppf(0.975, df=len(sharpe_diffs) - 1))
        ci_low = float(mean_diff - t_crit * se)
        ci_high = float(mean_diff + t_crit * se)

        mlp_vs_tf_results.append({
            "fold": fold, "fold_label": fold_label,
            "transformer_minus_mlp_mean": mean_diff,
            "ci_low": ci_low, "ci_high": ci_high,
            "within_uncertainty": bool(ci_low < 0 < ci_high),
        })

    print("\n=== MLP vs Transformer: paired seed Sharpe 차이 (Transformer - MLP), 95% t-CI ===")
    for r in mlp_vs_tf_results:
        status = "불확실성 안 (구분 어려움)" if r["within_uncertainty"] else "0을 배제 (구분 가능)"
        print(f"  {r['fold_label']}: diff={r['transformer_minus_mlp_mean']:+.4f}  "
              f"CI=[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  -> {status}")

    output_path = os.path.join(OUTPUT_DIR, "statistical_tests.json")
    with open(output_path, "w") as f:
        json.dump({
            "ppo_vs_benchmark": ppo_vs_benchmark_results,
            "mlp_vs_transformer": mlp_vs_tf_results,
            "config": {
                "block_size": BLOCK_SIZE,
                "n_bootstrap_reps": N_BOOTSTRAP_REPS,
                "ci_level": CI_LEVEL,
                "note": "OOS 국면(bull_2024, choppy_2025) 2개 한정 결론. 모든 시장 국면 일반화 주장 아님.",
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    main()
