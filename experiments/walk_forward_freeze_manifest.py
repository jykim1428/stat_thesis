"""
10월 3주차: Walk-forward 결과 동결(freeze) manifest 생성
리포 루트에서 실행: python experiments/walk_forward_freeze_manifest.py

배경
----
docs/project_milestone_plan.md 10월 3주차: "모든 숫자 freeze - 11월엔
이 숫자로만 씀". Codex 리뷰에서 요청된 필수 항목(commit SHA, worktree
상태, 연율화 상수, 후보/seed 목록, 3->5 seed 확장 이력, 평가 시작·종료
시각과 행 수, Fold2 실제 데이터 종료 시각, 원시 CSV/frozen summary의
SHA-256, bootstrap 설정, 패키지 버전, quantstats canonical 여부)을 모두
담은 manifest.json을 생성한다.

재학습이나 재계산 없이, 이미 확정된 결과 파일들의 메타데이터만 수집한다.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

RESULTS_ROOT = "results/walk_forward"
BENCHMARKS_ROOT = "results/walk_forward_benchmarks"
FROZEN_SUMMARIES_ROOT = "results/frozen_summaries"
LOCKED_CANDIDATES_PATH = "configs/walk_forward_locked_candidates.json"
OUTPUT_PATH = "results/frozen_summaries/manifest.json"

FOLDS = [
    ("fold1_final", "fold1_oos", "bull_2024"),
    ("fold2_final", "fold2_oos", "choppy_2025"),
]
BENCHMARK_STRATEGIES = ["buy_and_hold_btc", "equal_weight", "markowitz", "risk_parity"]

KEY_PACKAGES = [
    "torch", "stable-baselines3", "gym", "gymnasium", "shimmy",
    "pandas", "numpy", "empyrical-reloaded", "arch", "quantstats",
    "scipy", "PyPortfolioOpt",
]


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_info() -> dict:
    def run(cmd: list[str]) -> str:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()

    commit_sha = run(["git", "rev-parse", "HEAD"])
    status_output = run(["git", "status", "--short"])
    # 이 스크립트 실행 자체로 만들어지는 결과물(manifest.json, __pycache__ 등)은
    # "untracked 상태"로 잡혀도 dirty worktree의 증거가 아니므로 제외하지 않고
    # 그대로 기록한다 - 정직하게 무엇이 untracked였는지 남기는 게 목적.
    is_clean = status_output == ""
    return {
        # source_commit_sha: 이 manifest가 서술하는 숫자/코드가 확정된 커밋
        # (= manifest 생성 시점의 HEAD). manifest.json 자체는 이 커밋에는
        # 없고 "다음" 커밋에 저장되므로, manifest를 담은 커밋 SHA는 생성
        # 시점에 알 수 없다 - 커밋 후 manifest_commit_note를 참고할 것
        # (2026-09-08 리뷰 - Moderate: "commit X 기준 freeze"라는 표현이
        # 'X를 checkout하면 manifest도 있다'로 오독될 수 있어 명칭 구분).
        "source_commit_sha": commit_sha,
        "manifest_commit_note": (
            "manifest.json 자체는 source_commit_sha 다음 커밋에 저장된다. "
            "git log --follow results/frozen_summaries/manifest.json 으로 확인 가능."
        ),
        "worktree_clean": is_clean,
        "untracked_or_modified": status_output.splitlines() if status_output else [],
    }


def get_package_versions() -> dict:
    versions = {"python": sys.version.split()[0]}
    for pkg in KEY_PACKAGES:
        try:
            import importlib.metadata as md
            versions[pkg] = md.version(pkg)
        except Exception:
            versions[pkg] = "not found"
    return versions


def collect_ppo_backtest_manifest(locked: dict) -> dict:
    info = {}
    for fold, _, fold_label in FOLDS:
        for policy in ("mlp", "transformer"):
            candidate = locked[policy]["candidate"]
            for seed in locked[policy]["seeds"]:
                path = os.path.join(RESULTS_ROOT, policy, fold, candidate, f"seed{seed}", "backtest_oos.csv")
                df = pd.read_csv(path, index_col="date", parse_dates=True)
                key = f"{policy}_{fold}_seed{seed}"
                info[key] = {
                    "policy": policy, "fold": fold, "fold_label": fold_label,
                    "candidate": candidate, "seed": seed,
                    "eval_start": str(df.index.min()), "eval_end": str(df.index.max()),
                    "n_rows": len(df),
                    "sha256": sha256_of_file(path),
                    "model_zip_sha256": sha256_of_file(
                        os.path.join(RESULTS_ROOT, policy, fold, candidate, f"seed{seed}", "model.zip")
                    ),
                }
    return info


def collect_benchmark_manifest() -> dict:
    info = {}
    for fold_key, fold_label in [("fold1_oos", "bull_2024"), ("fold2_oos", "choppy_2025")]:
        for bench in BENCHMARK_STRATEGIES:
            path = os.path.join(BENCHMARKS_ROOT, fold_key, f"{bench}.csv")
            df = pd.read_csv(path, index_col="date", parse_dates=True)
            info[f"{fold_key}_{bench}"] = {
                "fold": fold_key, "fold_label": fold_label, "strategy": bench,
                "eval_start": str(df.index.min()), "eval_end": str(df.index.max()),
                "n_rows": len(df),
                "sha256": sha256_of_file(path),
            }
    return info


def collect_frozen_summaries_manifest() -> dict:
    info = {}
    for fname in sorted(os.listdir(FROZEN_SUMMARIES_ROOT)):
        path = os.path.join(FROZEN_SUMMARIES_ROOT, fname)
        if os.path.isfile(path) and fname != "manifest.json":
            info[fname] = sha256_of_file(path)
    return info


def collect_raw_input_manifest() -> dict:
    """processed input(feature_table DB)의 체크섬 - 재현성 확인용."""
    db_path = "data/raw/crypto_market_features.db"
    if not os.path.exists(db_path):
        return {"note": "data/raw/crypto_market_features.db not found on this machine (gitignored, *.db)"}
    return {"crypto_market_features.db": sha256_of_file(db_path)}


def main() -> None:
    with open(LOCKED_CANDIDATES_PATH) as f:
        locked = json.load(f)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_declaration": (
            "10월 3주차 결과 동결. 이 시점 이후 여기 기록된 수치/설정은 "
            "재조정하지 않는다 (docs/project_milestone_plan.md 10월 게이트)."
        ),
        "git": get_git_info(),
        "package_versions": get_package_versions(),
        "evaluation_config": {
            "cost_rate": locked.get("cost_rate", 0.001),
            "periods_per_year": 24 * 365,
            "periods_per_year_note": "1시간봉 데이터 연율화 상수 (evaluate.py, walk_forward_visualizations.py 공통)",
            "fold2_actual_end_note": (
                "Fold2(choppy_2025) 명목 종료는 2026-01-01(반개구간 상한)이지만 "
                "원천 DB(crypto_market_features.db)의 실제 마지막 시점은 "
                "2025-12-31 00:00:00 - 모든 전략(PPO, 벤치마크)에 동일하게 적용됨."
            ),
        },
        "locked_candidates": locked,
        "seed_protocol_history": {
            "initial_protocol": "3-seed (42, 43, 44), 2026-09-02 완료",
            "extension": "5-seed (42~46), 2026-09-07 실행 - seeds 45,46 post-hoc 추가",
            "extension_type": "post-hoc robustness extension (사전 계획된 프로토콜 아님, locked_candidates.json의 seeds_note_protocol_status 참고)",
            "candidate_reselection": "없음 - 후보/하이퍼파라미터 변경 없이 seed만 추가",
            "original_3seed_preserved_at": [
                "results/frozen_summaries/mlp_fold1_final_summary_3seed_original.csv",
                "results/frozen_summaries/mlp_fold2_final_summary_3seed_original.csv",
                "results/frozen_summaries/transformer_fold1_final_summary_3seed_original.csv",
                "results/frozen_summaries/transformer_fold2_final_summary_3seed_original.csv",
            ],
        },
        "statistical_test_config": {
            "method": "arch.bootstrap.StationaryBootstrap, pooled across 5 seeds + benchmark (6 series resampled jointly per replicate)",
            "primary_block_size": 24,
            "sensitivity_block_size": 168,
            "n_bootstrap_reps": 2000,
            "ci_level": 0.95,
            "n_comparisons_holm": 16,
            "statistic": "mean_seed[Sharpe(PPO_seed) - Sharpe(benchmark)] per bootstrap replicate",
        },
        "quantstats_reports_status": (
            "참고용(non-canonical). 대표 seed 1개(42)만 사용한 단일 시계열 "
            "tearsheet이며, 본문 수치의 근거는 results/walk_forward/*/metrics.json "
            "및 results/frozen_summaries/*.csv이다."
        ),
        "ppo_backtest_files": collect_ppo_backtest_manifest(locked),
        "benchmark_files": collect_benchmark_manifest(),
        "frozen_summary_files": collect_frozen_summaries_manifest(),
        "raw_input_files": collect_raw_input_manifest(),
    }

    os.makedirs(FROZEN_SUMMARIES_ROOT, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Manifest 저장: {OUTPUT_PATH}")
    print(f"  source_commit_sha: {manifest['git']['source_commit_sha']}")
    print(f"  worktree_clean: {manifest['git']['worktree_clean']}")
    if not manifest["git"]["worktree_clean"]:
        print(f"  untracked/modified: {manifest['git']['untracked_or_modified']}")
    print(f"  PPO backtest 파일 수: {len(manifest['ppo_backtest_files'])}")
    print(f"  벤치마크 파일 수: {len(manifest['benchmark_files'])}")


if __name__ == "__main__":
    main()
