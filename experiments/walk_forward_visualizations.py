"""
10월 2주차: Walk-forward OOS 결과 시각화 (재학습 없이 저장된 결과만 사용)
리포 루트에서 실행: python experiments/walk_forward_visualizations.py

배경
----
docs/project_milestone_plan.md 10월 2주차 계획: 누적수익률 곡선, drawdown
곡선, target allocation 변화, quantstats 리포트. 새 실험이나 재탐색이
아니라 이미 확정된 결과(results/walk_forward/, results/walk_forward_
benchmarks/)를 논문용 그림으로 옮기는 작업이다.

PPO는 net returns(거래비용 차감) 기준으로 그린다 - gross를 쓰면 evaluate()
가 계산한 Sharpe/CAGR/MDD와 그래프가 서로 다른 기준을 섞게 된다
(2026-09-07 리뷰에서 gross/net 혼용을 이미 한 번 지적받았음).

Seed 표현 방식
--------------
PPO는 seed 5개를 개별 실행한 결과라, 대표 seed 하나만 그리면 seed
불확실성이 그림에서 사라진다. 각 시점에서 5-seed의 min-max 범위를
음영 밴드로, 평균을 실선으로 그린다 - "이 정책이 이 정도 안정성으로
이 경로를 걸었다"는 것을 있는 그대로 보여준다.

Fold 1과 Fold 2는 국면이 다르므로(bull_2024 vs choppy_2025) 이어붙여
그리지 않고 별도 그림으로 유지한다 (docs/walk_forward_design.md: 두
국면 성과를 단순 평균하지 않는다는 원칙을 시각화에도 동일 적용).

target allocation 그래프는 실험적으로 확인된 사실(정책이 사실상 고정된
목표 비중을 출력함, walk_forward_behavior_diagnostics.py)을 시각적으로
보여주기 위한 것 - 그래프가 평평한 것 자체가 발견이므로 그대로 그린다.

출력
----
results/walk_forward_figures/{fold}/{policy}_cumulative_returns.png
results/walk_forward_figures/{fold}/{policy}_drawdown.png
results/walk_forward_figures/{fold}/{policy}_target_allocation.png
results/walk_forward_figures/{fold}/quantstats_{policy}_vs_{benchmark}.html
"""

from __future__ import annotations

import ast
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

RESULTS_ROOT = "results/walk_forward"
BENCHMARKS_ROOT = "results/walk_forward_benchmarks"
LOCKED_CANDIDATES_PATH = "configs/walk_forward_locked_candidates.json"
OUTPUT_DIR = "results/walk_forward_figures"

FOLDS = [
    ("fold1_final", "fold1_oos", "bull_2024"),
    ("fold2_final", "fold2_oos", "choppy_2025"),
]

BENCHMARK_STRATEGIES = {
    "buy_and_hold_btc": "Buy & Hold BTC",
    "equal_weight": "Equal-Weight",
    "markowitz": "Markowitz (MinVar)",
    "risk_parity": "Risk Parity (ERC)",
}

TIC_ORDER_WITH_CASH = ["CASH", "ADA", "AVAX", "BNB", "BTC", "DOGE", "ETH", "SOL", "XRP"]
POLICY_COLORS = {"mlp": "#6b7280", "transformer": "#b8763e"}
BENCHMARK_COLORS = {
    "buy_and_hold_btc": "#4a7c9e",
    "equal_weight": "#7a5c9e",
    "markowitz": "#2f7a4f",
    "risk_parity": "#a8433a",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#1e1c1a",
    "text.color": "#1e1c1a",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "font.size": 10,
    "figure.dpi": 150,
})


def load_locked_candidates() -> dict:
    with open(LOCKED_CANDIDATES_PATH) as f:
        return json.load(f)


def load_backtest(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df["weights"] = df["weights"].apply(ast.literal_eval)
    df["target_weights"] = df["target_weights"].apply(ast.literal_eval)
    return df


def compute_net_returns(df: pd.DataFrame, cost_rate: float) -> pd.Series:
    W = np.vstack(df["weights"].to_numpy())
    TW = np.vstack(df["target_weights"].to_numpy())
    turnover = np.zeros(len(df))
    turnover[1:] = np.abs(TW[1:] - W[:-1]).sum(axis=1) / 2.0
    net = df["returns"] - turnover * cost_rate
    net.index = df.index
    return net


def cumulative_from_returns(returns: pd.Series, initial: float = 100_000) -> pd.Series:
    return initial * (1 + returns).cumprod()


def drawdown_from_cumulative(cumulative: pd.Series) -> pd.Series:
    peak = cumulative.cummax()
    return (cumulative - peak) / peak


def plot_cumulative_and_drawdown(
    policy: str, candidate: str, seeds: list[int], fold: str, benchmark_fold: str, fold_label: str,
    cost_rate: float, output_dir: str,
) -> None:
    # PPO: seed별 net returns -> 누적곡선/drawdown, 5-seed min/max/mean
    ppo_cumulatives = []
    ppo_drawdowns = []
    common_index = None
    for seed in seeds:
        path = os.path.join(RESULTS_ROOT, policy, fold, candidate, f"seed{seed}", "backtest_oos.csv")
        df = load_backtest(path)
        net_returns = compute_net_returns(df, cost_rate)
        cum = cumulative_from_returns(net_returns)
        dd = drawdown_from_cumulative(cum)
        ppo_cumulatives.append(cum)
        ppo_drawdowns.append(dd)
        if common_index is None:
            common_index = cum.index

    ppo_cum_df = pd.DataFrame({s: c.values for s, c in zip(seeds, ppo_cumulatives)}, index=common_index)
    ppo_dd_df = pd.DataFrame({s: d.values for s, d in zip(seeds, ppo_drawdowns)}, index=common_index)

    # 벤치마크: 결정론적이라 seed 없음
    bench_cumulatives = {}
    bench_drawdowns = {}
    for bench_key, bench_label in BENCHMARK_STRATEGIES.items():
        bench_path = os.path.join(BENCHMARKS_ROOT, benchmark_fold, f"{bench_key}.csv")
        bench_df = load_backtest(bench_path)
        bench_net_returns = compute_net_returns(bench_df, cost_rate)
        bench_cum = cumulative_from_returns(bench_net_returns)
        bench_cumulatives[bench_key] = bench_cum
        bench_drawdowns[bench_key] = drawdown_from_cumulative(bench_cum)

    # === 누적수익률 곡선 ===
    fig, ax = plt.subplots(figsize=(10, 5.5))
    color = POLICY_COLORS[policy]
    ax.fill_between(ppo_cum_df.index, ppo_cum_df.min(axis=1), ppo_cum_df.max(axis=1),
                     color=color, alpha=0.18, label=f"{policy.upper()} (5-seed range)")
    ax.plot(ppo_cum_df.index, ppo_cum_df.mean(axis=1), color=color, linewidth=2, label=f"{policy.upper()} (mean)")
    for bench_key, bench_label in BENCHMARK_STRATEGIES.items():
        ax.plot(bench_cumulatives[bench_key].index, bench_cumulatives[bench_key].values,
                color=BENCHMARK_COLORS[bench_key], linewidth=1.2, linestyle="--", alpha=0.85, label=bench_label)
    ax.set_title(f"{policy.upper()} vs Benchmarks — Cumulative Portfolio Value ({fold_label})")
    ax.set_ylabel("Portfolio Value ($, net of costs)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{policy}_cumulative_returns.png"))
    plt.close(fig)

    # === Drawdown 곡선 ===
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(ppo_dd_df.index, ppo_dd_df.min(axis=1) * 100, ppo_dd_df.max(axis=1) * 100,
                     color=color, alpha=0.18, label=f"{policy.upper()} (5-seed range)")
    ax.plot(ppo_dd_df.index, ppo_dd_df.mean(axis=1) * 100, color=color, linewidth=2, label=f"{policy.upper()} (mean)")
    for bench_key, bench_label in BENCHMARK_STRATEGIES.items():
        ax.plot(bench_drawdowns[bench_key].index, bench_drawdowns[bench_key].values * 100,
                color=BENCHMARK_COLORS[bench_key], linewidth=1.2, linestyle="--", alpha=0.85, label=bench_label)
    ax.set_title(f"{policy.upper()} vs Benchmarks — Drawdown ({fold_label})")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{policy}_drawdown.png"))
    plt.close(fig)


def plot_target_allocation(policy: str, candidate: str, seeds: list[int], fold: str, fold_label: str, output_dir: str) -> None:
    """target_weights(정책이 실제 지시한 목표 비중)의 시간별 변화.

    2026-09-07 리뷰에서 확인된 사실(target_weights 스텝 변화량이
    float32 정밀도 수준)을 시각적으로 보여주기 위한 그래프 - 평평하게
    나오는 것 자체가 발견이므로 그대로 그린다.
    """
    # seed 평균 target_weights 시계열 (스텝 변화가 거의 없어 seed 간 차이도 미미함)
    all_tw = []
    common_index = None
    for seed in seeds:
        path = os.path.join(RESULTS_ROOT, policy, fold, candidate, f"seed{seed}", "backtest_oos.csv")
        df = load_backtest(path)
        TW = np.vstack(df["target_weights"].to_numpy())
        all_tw.append(TW)
        if common_index is None:
            common_index = df.index
    mean_tw = np.mean(all_tw, axis=0)  # (T, 9)

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = plt.cm.tab10(np.linspace(0, 1, len(TIC_ORDER_WITH_CASH)))
    ax.stackplot(common_index, mean_tw.T * 100, labels=TIC_ORDER_WITH_CASH, colors=palette, alpha=0.85)
    ax.set_title(f"{policy.upper()} — Target Allocation over Time ({fold_label})")
    ax.set_ylabel("Target Weight (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=9, fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{policy}_target_allocation.png"))
    plt.close(fig)


def generate_quantstats_report(policy: str, candidate: str, seed: int, fold: str, benchmark_fold: str, cost_rate: float, output_dir: str) -> None:
    """quantstats 리포트는 단일 시계열이 필요하므로 대표 seed(첫 번째, 보통 42)를 사용.
    파일명에 seed 번호를 명시해 대표값임을 분명히 한다."""
    import quantstats as qs

    ppo_path = os.path.join(RESULTS_ROOT, policy, fold, candidate, f"seed{seed}", "backtest_oos.csv")
    ppo_df = load_backtest(ppo_path)
    ppo_net_returns = compute_net_returns(ppo_df, cost_rate)
    ppo_net_returns.index = ppo_net_returns.index.tz_localize(None)

    ew_path = os.path.join(BENCHMARKS_ROOT, benchmark_fold, "equal_weight.csv")
    ew_df = load_backtest(ew_path)
    ew_net_returns = compute_net_returns(ew_df, cost_rate)
    ew_net_returns.index = ew_net_returns.index.tz_localize(None)

    output_path = os.path.join(output_dir, f"quantstats_{policy}_seed{seed}_vs_equal_weight.html")
    qs.reports.html(
        ppo_net_returns, benchmark=ew_net_returns, output=output_path,
        title=f"{policy.upper()} (seed{seed}) vs Equal-Weight",
    )
    print(f"  quantstats 리포트 저장: {output_path}")


def main() -> None:
    locked = load_locked_candidates()
    cost_rate = locked.get("cost_rate", 0.001)

    for fold, benchmark_fold, fold_label in FOLDS:
        fold_output_dir = os.path.join(OUTPUT_DIR, fold)
        os.makedirs(fold_output_dir, exist_ok=True)
        print(f"=== {fold_label} ({fold}) ===")

        for policy_key in ("mlp", "transformer"):
            candidate = locked[policy_key]["candidate"]
            seeds = locked[policy_key]["seeds"]

            print(f"  [{policy_key}] 누적수익률/drawdown 곡선 생성 중...")
            plot_cumulative_and_drawdown(policy_key, candidate, seeds, fold, benchmark_fold, fold_label, cost_rate, fold_output_dir)

            print(f"  [{policy_key}] target allocation 그래프 생성 중...")
            plot_target_allocation(policy_key, candidate, seeds, fold, fold_label, fold_output_dir)

            print(f"  [{policy_key}] quantstats 리포트 생성 중 (대표 seed={seeds[0]})...")
            generate_quantstats_report(policy_key, candidate, seeds[0], fold, benchmark_fold, cost_rate, fold_output_dir)

    print(f"\n모든 그림/리포트 저장 완료: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
