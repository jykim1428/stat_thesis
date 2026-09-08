"""
10월 1주차: PPO vs 벤치마크 / MLP vs Transformer 통계 검정
리포 루트에서 실행: python experiments/walk_forward_statistical_tests.py

배경
----
walk_forward_benchmarks.py의 비교표는 PPO(seed 5개 반복)의 mean Sharpe와
벤치마크(결정론적, 반복 없음)의 point estimate를 나란히 놓은 "기술통계
순위"일 뿐 가설검정이 아니다 (2026-09-07 코덱스 3차 리뷰). 이 스크립트는
그 통계적 엄밀성을 확보하려는 시도다.

첫 구현의 결함과 수정 (2026-09-07 코덱스 4차 리뷰)
----------------------------------------------------------
최초 구현은 아래 네 가지가 틀렸었다:

1. 통계량 자체가 틀림: Sharpe(PPO) - Sharpe(benchmark)를 계산해야 하는데
   Sharpe(PPO returns - benchmark returns)를 계산했다. 이 둘은 Sharpe가
   비선형(평균/표준편차 비율) 통계량이라 전혀 다른 값이다. 실측: MLP-EW
   Fold1의 실제 Sharpe 차이는 +0.005인데 잘못된 계산은 -2.447을 냈다.
2. 벤치마크는 gross returns, PPO는 net(비용 차감) returns를 섞어 썼다 -
   비교 조건 자체가 안 맞았다.
3. seed별로 각각 구한 CI의 하한/상한을 단순 평균했는데, 이건 통계적으로
   의미 있는 "전체 평균 효과의 CI"가 아니다.
4. p-value proxy가 CI가 0을 배제하는 모든 경우에 상수 0.01을 반환했다.
   Holm 첫 임계값(0.05/16=0.003125)보다 항상 크므로, 코드 구조상
   "16개 전부 Holm 보정 후 비유의"가 나올 수밖에 없었다 - 실제 검정이
   아니라 검정을 흉내 낸 것이었다.

수정된 방법론
----------------
- 벤치마크도 evaluate.py와 동일한 정의(turnover*cost_rate 차감)의 net
  returns를 사용해 PPO와 조건을 맞춘다.
- 통계량을 Sharpe(PPO) - Sharpe(benchmark)로 정확히 정의한다.
- seed 불확실성과 시계열(bootstrap) 불확실성을 함께 반영하기 위해, PPO
  5-seed net returns와 벤치마크 net returns, 총 6개 시계열을
  arch.bootstrap.StationaryBootstrap에 동시에 넣는다 - 매 리샘플
  replicate마다 "동일한" 블록 인덱스로 6개 시계열이 함께 리샘플되므로
  seed 간·PPO-벤치마크 간 시장 충격 상관관계가 보존된 채로,
  mean_seed[Sharpe(PPO_seed) - Sharpe(bench)]라는 통계량(추정하려는
  모수 그 자체)의 replicate별 값을 얻는다. 이 replicate 값들의 분포가
  올바른 표본분포다 (자세한 내용은 multi_seed_bootstrap_distribution()
  참고. 최초 구현은 seed마다 "독립적으로" bootstrap을 돌려 그 분포들을
  단순히 이어붙였는데, 이는 "seed 하나만 있었다면 나왔을 결과"의
  분포에 가깝고 CI 폭을 약 sqrt(5)배 부풀렸다 - 2026-09-08 리뷰에서
  시뮬레이션으로 확인 후 지금 방식으로 교체함).
- 실제 bootstrap p-value: replicate 분포가 0을 기준으로 얼마나 치우쳐
  있는지로 계산하는 양측검정 (2*min(P(dist<=0), P(dist>=0))).
- 비교가 여러 개(정책 2 x 벤치마크 4 x fold 2 = 16개)이므로
  Holm-Bonferroni 보정을 적용한다.
- block size는 24(하루 주기성)와 168(주 단위 주기성)을 모두 계산해
  민감도를 확인한다 - 결과가 크게 달라지면 결론에 그 사실을 명시해야
  한다.
- bootstrap의 rng seed는 (policy, fold, benchmark, block_size) 조합에서
  SHA-256으로 결정론적으로 파생한다 (2026-09-08 리뷰 - Major: 이전에는
  Python 내장 hash()를 썼는데, PYTHONHASHSEED가 프로세스마다 랜덤이라
  hash(문자열) 값이 실행마다 달라져 같은 명령을 재실행해도 CI/p-value가
  달라질 수 있었다 - 재현성이 없는 "재현 가능한 검정"이었던 셈. 파생
  규칙은 derive_seed() 참고, MASTER_SEED와 함께 결과 JSON에 기록한다).
- OOS 국면이 bull_2024/choppy_2025 2개뿐이므로 "모든 시장 국면에
  일반화된다"는 주장은 하지 않는다 - 이 두 국면 한정 결론이다.
- 결과 해석 문구: 유의하지 않다고 "PPO와 벤치마크가 통계적으로 동등하다"
  고 쓰면 안 된다 (비유의는 동등성의 증거가 아니다 - equivalence test가
  아닌 한). 정확한 문구는 "이 두 OOS 국면에서 PPO와 전통 벤치마크 간
  Sharpe 차이가 0과 다르다는 통계적 증거를 얻지 못했다"이다.

주의
----
재학습이나 재탐색이 아니라, 이미 저장된 backtest_oos.csv/benchmark csv로
사후 통계 검정만 수행한다. 이 결과를 보고 후보나 하이퍼파라미터를 다시
조정하지 않는다 (OOS 확인 후 재조정 금지 원칙).

출력
----
results/walk_forward_statistical_tests/statistical_tests.json
콘솔에 정책×벤치마크×fold별 point estimate, CI, bootstrap p-value,
Holm 보정 후 유의성, block size 민감도를 출력.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os

import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap
from scipy import stats

RESULTS_ROOT = "results/walk_forward"
BENCHMARKS_ROOT = "results/walk_forward_benchmarks"
LOCKED_CANDIDATES_PATH = "configs/walk_forward_locked_candidates.json"
OUTPUT_DIR = "results/walk_forward_statistical_tests"

FOLDS = [
    ("fold1_final", "fold1_oos", "bull_2024"),
    ("fold2_final", "fold2_oos", "choppy_2025"),
]

BENCHMARK_STRATEGIES = ["buy_and_hold_btc", "equal_weight", "markowitz", "risk_parity"]

BLOCK_SIZES = [24, 168]  # 하루/주 단위 주기성 - 민감도 확인용
PRIMARY_BLOCK_SIZE = 24
N_BOOTSTRAP_REPS = 2000
PERIODS_PER_YEAR = 24 * 365
CI_LEVEL = 0.95

# bootstrap rng seed 파생용 master seed. 2026-09-08 리뷰 - Major: 이전에는
# Python 내장 hash(bench_strategy)를 그대로 rng_seed로 썼는데, PYTHONHASHSEED가
# 프로세스마다 랜덤이라 같은 문자열의 hash() 값이 실행마다 달라진다 (실측:
# 동일 문자열이 프로세스 재실행 시 1058216092 / 1901723970으로 다르게 나옴).
# 즉 통계 검정을 재실행하면 CI/p-value가 매번 바뀔 수 있어 "재현 가능한
# 검정"이 아니었다. derive_seed()로 (policy, fold, benchmark, block_size)
# 조합에서 SHA-256 기반 결정론적 정수를 파생해 이 문제를 없앤다.
MASTER_SEED = 20261007  # 임의의 고정값 - 이 스크립트를 다시 실행해도 항상 동일


def derive_seed(*parts: str) -> int:
    """(policy, fold, benchmark, block_size) 등 문자열 조합에서 결정론적
    정수 seed를 파생. hashlib.sha256은 PYTHONHASHSEED와 무관하게 항상
    동일한 값을 내므로, 이 함수는 프로세스·환경에 관계없이 재현 가능하다."""
    key = f"{MASTER_SEED}|" + "|".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31)


def load_locked_candidates() -> dict:
    with open(LOCKED_CANDIDATES_PATH) as f:
        return json.load(f)


def load_net_returns(path: str, cost_rate: float) -> pd.Series:
    """evaluate.py와 동일하게 turnover 비용을 차감한 net returns.

    PPO와 벤치마크 모두 이 함수로 net returns를 계산해서 비교 조건을
    맞춘다 (최초 구현은 벤치마크에 gross returns를 그대로 썼음 - Major #2).
    """
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


def sharpe(returns: np.ndarray) -> float:
    std = returns.std()
    if std == 0 or not np.isfinite(std):
        return 0.0
    return float(returns.mean() / std * np.sqrt(PERIODS_PER_YEAR))


def sharpe_diff_statistic(ppo_returns: np.ndarray, bench_returns: np.ndarray) -> float:
    """통계량: Sharpe(PPO) - Sharpe(benchmark) - Sharpe(PPO-benchmark)가 아니다.

    Sharpe는 평균/표준편차 비율이라 비선형 통계량이므로, 두 시계열 각각의
    Sharpe를 구한 뒤 빼는 것과 차이 시계열의 Sharpe를 구하는 것은 전혀
    다른 값이다 (2026-09-07 코덱스 4차 리뷰 - Major #1, 실측: MLP-EW
    Fold1 실제 차이 +0.005 vs 잘못된 계산 -2.447).
    """
    return sharpe(ppo_returns) - sharpe(bench_returns)


def mean_seed_sharpe_diff_statistic(*arrays: np.ndarray) -> float:
    """arrays = (ppo_seed1, ppo_seed2, ..., ppo_seedN, bench) 순서.

    매 bootstrap replicate에서 이 함수가 호출되며, 그 시점에 6개 시계열이
    전부 "동일한" 블록 인덱스로 함께 리샘플된 상태다. 여기서 계산하는
    mean_seed[Sharpe(PPO_seed) - Sharpe(bench)]가 바로 우리가 추정하려는
    모수(5-seed 평균 효과)의 추정량이므로, 이 값의 replicate별 분포가
    올바른 표본분포다.
    """
    bench_arr = arrays[-1]
    ppo_arrays = arrays[:-1]
    bench_sharpe = sharpe(bench_arr)
    diffs = [sharpe(ppo_arr) - bench_sharpe for ppo_arr in ppo_arrays]
    return float(np.mean(diffs))


def multi_seed_bootstrap_distribution(
    seed_ppo_returns: dict[int, pd.Series], bench_returns: pd.Series, block_size: int, seeds: list[int], rng_seed: int
) -> np.ndarray:
    """PPO 5-seed + benchmark, 총 6개 시계열을 함께 block bootstrap 리샘플한다.

    2026-09-08 리뷰 - Major #2 수정: 이전 구현은 seed마다 "독립적으로"
    bootstrap을 돌려 그 리샘플 분포들을 단순히 이어붙였다. 이건
    "seed 하나만 있었다면 나왔을 결과"의 분포에 가깝고, 우리가 실제로
    추정해야 하는 "5-seed 평균 효과"의 표본분포가 아니다 (시뮬레이션
    검증: 잘못된 방식은 CI 폭을 약 sqrt(5)배 부풀림).

    StationaryBootstrap에 6개 시계열을 한꺼번에 넣으면 매 replicate에서
    "동일한" 블록 인덱스로 6개를 같이 리샘플한다 - 같은 시점의 시장
    충격 상관관계(PPO 5-seed끼리도, PPO와 벤치마크 사이도)가 보존된
    채로, mean_seed[Sharpe(PPO_seed)-Sharpe(bench)]라는 통계량 자체의
    표본분포를 얻는다.
    """
    common_idx = bench_returns.index
    for seed in seeds:
        common_idx = common_idx.intersection(seed_ppo_returns[seed].index)

    arrays = [seed_ppo_returns[seed].loc[common_idx].to_numpy() for seed in seeds]
    arrays.append(bench_returns.loc[common_idx].to_numpy())

    bs = StationaryBootstrap(block_size, *arrays, seed=rng_seed)
    dist = bs.apply(mean_seed_sharpe_diff_statistic, reps=N_BOOTSTRAP_REPS)
    return dist.flatten()


def ci_and_pvalue(dist: np.ndarray, point_estimate: float) -> dict:
    ci_low, ci_high = np.percentile(dist, [(1 - CI_LEVEL) / 2 * 100, (1 + CI_LEVEL) / 2 * 100])
    p_below = float((dist <= 0).mean())
    p_above = float((dist >= 0).mean())
    p_value = float(min(1.0, 2 * min(p_below, p_above)))
    return {
        "point_estimate": float(point_estimate),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "bootstrap_p_value": p_value,
    }


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni 보정. 반환: 각 비교가 alpha에서 유의한지 여부(원래 순서 유지)."""
    order = np.argsort(p_values)
    m = len(p_values)
    reject = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if p_values[idx] < threshold:
            reject[int(idx)] = True
        else:
            break  # Holm: 한 번 기각 실패하면 이후(더 큰 p-value)는 전부 채택
    return reject


def compute_ppo_vs_benchmark(locked: dict) -> list[dict]:
    results = []
    for policy_key in ("mlp", "transformer"):
        candidate = locked[policy_key]["candidate"]
        seeds = locked[policy_key]["seeds"]
        cost_rate = locked.get("cost_rate", 0.001)

        for fold, benchmark_fold, fold_label in FOLDS:
            # seed별 PPO net returns는 이 fold 안에서 벤치마크 4종 전부에 재사용 - 한 번만 로드
            seed_ppo_returns = {
                seed: load_net_returns(
                    os.path.join(RESULTS_ROOT, policy_key, fold, candidate, f"seed{seed}", "backtest_oos.csv"), cost_rate
                )
                for seed in seeds
            }

            for bench_strategy in BENCHMARK_STRATEGIES:
                bench_path = os.path.join(BENCHMARKS_ROOT, benchmark_fold, f"{bench_strategy}.csv")
                bench_net_returns = load_net_returns(bench_path, cost_rate)

                # point estimate: block size와 무관, seed별 Sharpe(PPO)-Sharpe(bench)의 평균
                point_estimates = []
                for seed in seeds:
                    ppo_net_returns = seed_ppo_returns[seed]
                    common_idx = ppo_net_returns.index.intersection(bench_net_returns.index)
                    point_estimates.append(sharpe_diff_statistic(
                        ppo_net_returns.loc[common_idx].to_numpy(),
                        bench_net_returns.loc[common_idx].to_numpy(),
                    ))
                mean_point = float(np.mean(point_estimates))

                block_size_results = {}
                for block_size in BLOCK_SIZES:
                    rng_seed = derive_seed(policy_key, fold, bench_strategy, block_size)
                    dist = multi_seed_bootstrap_distribution(
                        seed_ppo_returns, bench_net_returns, block_size, seeds, rng_seed=rng_seed
                    )
                    block_size_results[block_size] = ci_and_pvalue(dist, mean_point)

                primary = block_size_results[PRIMARY_BLOCK_SIZE]
                results.append({
                    "policy": policy_key,
                    "fold": fold,
                    "fold_label": fold_label,
                    "benchmark": bench_strategy,
                    **primary,
                    "block_size_sensitivity": {str(bs): block_size_results[bs] for bs in BLOCK_SIZES},
                })

    return results


def compute_mlp_vs_transformer(locked: dict) -> list[dict]:
    """MLP vs Transformer paired seed Sharpe 차이. n=5 seed 변동성 한정 결과.

    이 비교는 2026-09-07 코덱스 3차 리뷰에서 이미 올바르게 계산되어
    (Sharpe끼리 직접 뺀 값), 그대로 재사용한다 - PPO-벤치마크 비교와 달리
    "차이 시계열의 Sharpe"를 계산하는 오류가 없었다.
    """
    mlp_seeds = locked["mlp"]["seeds"]
    tf_seeds = locked["transformer"]["seeds"]
    assert mlp_seeds == tf_seeds, "MLP/Transformer seed 목록이 달라 paired 비교 불가"

    results = []
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
        t_crit = float(stats.t.ppf(0.975, df=len(sharpe_diffs) - 1))
        ci_low = float(mean_diff - t_crit * se)
        ci_high = float(mean_diff + t_crit * se)

        results.append({
            "fold": fold, "fold_label": fold_label,
            "n_seeds": len(sharpe_diffs),
            "transformer_minus_mlp_mean": mean_diff,
            "ci_low": ci_low, "ci_high": ci_high,
            "within_uncertainty": bool(ci_low < 0 < ci_high),
            "note": "n=5 seed 변동성 한정 결과 - seed 수를 늘리면 CI가 달라질 수 있음",
        })

    return results


def print_ppo_vs_benchmark(results: list[dict], significant: list[bool]) -> None:
    print(f"=== PPO vs 벤치마크: Sharpe(PPO)-Sharpe(benchmark), pooled bootstrap (block={PRIMARY_BLOCK_SIZE}) ===")
    for r, sig in zip(results, significant):
        sig_mark = "유의함" if sig else "유의하지 않음"
        bs168 = r["block_size_sensitivity"]["168"]
        print(f"  {r['policy']:12s} vs {r['benchmark']:18s} [{r['fold_label']}]: "
              f"diff={r['point_estimate']:+.4f}  CI=[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  "
              f"p={r['bootstrap_p_value']:.4f}  -> {sig_mark}  "
              f"(block=168 CI=[{bs168['ci_low']:+.4f}, {bs168['ci_high']:+.4f}])")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    locked = load_locked_candidates()

    ppo_vs_benchmark_results = compute_ppo_vs_benchmark(locked)
    p_values = [r["bootstrap_p_value"] for r in ppo_vs_benchmark_results]
    significant = holm_bonferroni(p_values)
    for r, sig in zip(ppo_vs_benchmark_results, significant):
        r["significant_after_holm"] = bool(sig)

    print_ppo_vs_benchmark(ppo_vs_benchmark_results, significant)

    mlp_vs_tf_results = compute_mlp_vs_transformer(locked)
    print("\n=== MLP vs Transformer: paired seed Sharpe 차이 (Transformer - MLP), 95% t-CI, n=5 한정 ===")
    for r in mlp_vs_tf_results:
        status = "불확실성 안 (구분 어려움)" if r["within_uncertainty"] else "0을 배제 (구분 가능)"
        print(f"  {r['fold_label']}: diff={r['transformer_minus_mlp_mean']:+.4f}  "
              f"CI=[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  -> {status}")

    n_significant = sum(significant)
    print(f"\n결론: 16개 비교 중 {n_significant}개가 Holm 보정 후 유의함.")
    print("주의: 유의하지 않다고 해서 '통계적으로 동등하다'를 의미하지 않는다 (equivalence test가 아님).")
    print("정확한 문구: '이 두 OOS 국면에서 PPO와 전통 벤치마크 간 Sharpe 차이가")
    print("0과 다르다는 통계적 증거를 얻지 못했다.'")

    output_path = os.path.join(OUTPUT_DIR, "statistical_tests.json")
    with open(output_path, "w") as f:
        json.dump({
            "ppo_vs_benchmark": ppo_vs_benchmark_results,
            "mlp_vs_transformer": mlp_vs_tf_results,
            "config": {
                "primary_block_size": PRIMARY_BLOCK_SIZE,
                "block_size_sensitivity_checked": BLOCK_SIZES,
                "n_bootstrap_reps": N_BOOTSTRAP_REPS,
                "ci_level": CI_LEVEL,
                "n_comparisons_holm": len(ppo_vs_benchmark_results),
                "pooling_method": (
                    "PPO 5-seed + benchmark, 총 6개 net returns 시계열을 "
                    "StationaryBootstrap에 동시에 넣어 매 replicate마다 동일한 "
                    "블록 인덱스로 함께 리샘플하고, 그 replicate에서 "
                    "mean_seed[Sharpe(PPO_seed) - Sharpe(benchmark)]를 계산한 "
                    "표본분포 (multi_seed_bootstrap_distribution 참고)"
                ),
                "rng_seed_derivation": (
                    f"MASTER_SEED={MASTER_SEED}에서 derive_seed(policy, fold, "
                    "benchmark, block_size)로 SHA-256 기반 결정론적 파생 - "
                    "Python 내장 hash()는 PYTHONHASHSEED 랜덤화로 프로세스마다 "
                    "달라져 사용하지 않음"
                ),
                "interpretation_note": (
                    "유의하지 않음은 동등성의 증거가 아니다. 정확한 해석: "
                    "이 두 OOS 국면에서 PPO와 전통 벤치마크 간 Sharpe 차이가 "
                    "0과 다르다는 통계적 증거를 얻지 못했다."
                ),
                "generalization_note": "OOS 국면(bull_2024, choppy_2025) 2개 한정 결론. 모든 시장 국면 일반화 주장 아님.",
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    main()
