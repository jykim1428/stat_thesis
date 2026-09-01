"""
9월 4주차: Walk-forward validation 오케스트레이터 (CryptoAgent)
리포 루트에서 실행: python experiments/walk_forward_orchestrator.py --policy mlp --stage fold1_search

docs/walk_forward_design.md, docs/colab_compute_budget.md에서 확정된 설계를 그대로
구현한다. 이 스크립트는 train_ppo_mlp.py / train_ppo_transformer.py를 대체하지
않는다 - 그 둘은 기존 split(train/val/test) 기준 단발 실행용으로 그대로 둔다.
여기서는 fold별 날짜 범위(make_env_by_date), 후보 하이퍼파라미터, seed 조합을
순회하며 학습하는 역할만 담당한다.

Fold 구성 (docs/walk_forward_design.md)
----------------------------------------
Fold 1
    하이퍼파라미터 탐색: train=[bull_2021+bear_2022], val=side_2023
        train: 2021-01-01 ~ 2023-01-02 (반개구간)
        val:   2023-01-02 ~ 2023-10-16 (반개구간)
    최종 재학습: 2021-01-01 ~ 2023-10-16
    OOS:        bull_2024 = 2023-10-16 ~ 2025-01-02

Fold 2
    최종 학습: 2021-01-01 ~ 2025-01-02 (Fold1 OOS 구간까지 누적)
    OOS:      choppy_2025 = 2025-01-02 ~ 2026-01-01

날짜 경계는 반개구간([start, end_exclusive))으로 이어 붙인다 - adapter.py의
end 폐구간 해석 버그(날짜만 있는 문자열이 00:00으로 파싱되어 그날 나머지가
누락되는 문제)를 이 스크립트 레벨에서도 반복하지 않기 위함.

Validation/OOS 워밍업 (코덱스 리뷰 반영 - Major #1)
------------------------------------------------------
최초 구현에서는 val 평가(fold1_search)에만 enable_warmup=False를 썼는데, 이러면
PortfolioOptimizationEnv가 reset 후 첫 관측을 만드는 데 TIME_WINDOW(50)개 과거
시점이 필요하다는 사실을 그대로 노출시켜 val 구간 첫 49시간이 거래 기록에서
통째로 누락된다 (실측: side_2023 val 요청 시작 2023-01-02 00:00인데 실제
backtest_df 첫 행이 2023-01-04 01:00으로 49시간 뒤였음). train/val/OOS 평가
전부 동일하게 enable_warmup=True + trim_warmup_rows(oos_start=구간 시작)를
적용해야 한다 - "구간 시작 시점의 성과만 워밍업으로 소모돼 빠진다"는 문제는
val이든 OOS든 다르지 않다.

OOS 실행 잠금 (코덱스 리뷰 반영 - Major #2)
-----------------------------------------------
fold1_final/fold2_final은 자리표시 후보로도 바로 실행할 수 있으면 안 된다.
docs/colab_compute_budget.md의 "OOS 결과를 본 뒤 후보/seed 수를 바꾸지 않는다"
원칙을 지키려면, Fold 1 validation 결과로 후보를 확정한 후 그 결정을
LOCKED_CANDIDATES_PATH(JSON)에 명시적으로 기록해야만 final 단계 실행을
허용해야 한다. 파일이 없거나 요청한 policy/candidate/seed 조합이 그 안에
없으면 즉시 거부한다.

Resume 검증 강화 (코덱스 리뷰 반영 - Major #3)
--------------------------------------------------
status.json == completed만으로 skip하면: (a) 같은 candidate 이름으로 hparams를
바꿔도 낡은 결과를 그대로 재사용, (b) model.zip/backtest csv가 실제로는 없는데
skip, (c) status.json이 손상되면 예외로 전체 중단, (d) status 저장 후
metrics 저장 전에 프로세스가 죽으면 그 사실이 어디에도 안 남는 문제가 생긴다.
지금은 저장된 config.json이 현재 spec과 완전히 같은지, 그리고 model.zip/
backtest csv/config.json이 모두 실존하는지까지 확인해야 skip한다. status.json은
임시 파일에 쓴 뒤 os.replace로 원자적으로 교체해서, 쓰다 만 파일이 completed로
오인되는 일이 없게 한다.

summary 재생성 방식 (코덱스 리뷰 반영 - Major #4)
------------------------------------------------------
여러 프로세스가 병렬로 같은 summary.csv에 append하면 헤더 중복/행 충돌이 날 수
있다 (run별 cwd 분리는 PNG 충돌만 해결하지, 공용 CSV 쓰기 충돌은 해결 못 함).
각 run은 자기 디렉토리 안에 metrics.json만 원자적으로 저장하고, summary.csv는
aggregate_summary()가 그 시점까지 존재하는 모든 metrics.json을 다시 읽어
매번 통째로 재생성한다 - 어느 프로세스가 마지막에 쓰든 최종 상태는 디스크에 있는
completed run 전체를 반영하므로 안전하다.

PPO 하이퍼파라미터 allowlist
--------------------------------
정식 후보 탐색에서는 learning_rate/gamma/batch_size 등을 후보마다 다르게
지정해야 하므로, hparams 딕셔너리에서 PPO_HPARAM_KEYS에 있는 키만 뽑아
PPO(...) 생성자에 그대로 전달한다. total_timesteps과 policy별 전용 키
(d_model 등)는 여기서 제외하고 build_model()에서 별도 처리한다.

지표 (코덱스 리뷰 반영)
--------------------------
metrics.json에는 최종 자산가치뿐 아니라 evaluate.evaluate()가 계산하는
sharpe/sortino/calmar/mdd/vol/cagr/avg_turnover/total_cost까지 저장한다 -
Fold 1 후보 선택 기준(및 그 동률 처리 기준)은 OOS를 보기 전에 별도로
문서화하고 고정해야 하며, 이 스크립트는 그 판단에 필요한 지표를 전부
남기는 역할만 한다.

디렉토리 구조 (docs/colab_compute_budget.md 8절)
------------------------------------------------------
results/walk_forward/{policy}/{fold}/{candidate}/{seed}/
    config.json       실행에 사용된 모든 설정(날짜 범위, 하이퍼파라미터, seed 등)
    model.zip         학습된 SB3 모델
    backtest_{split}.csv   공용 스펙 백테스트 결과 (val 또는 oos, 워밍업 trim 적용됨)
    metrics.json      학습/평가 소요 시간 + evaluate() 지표
    status.json       {"status": "completed"} (원자적으로 기록됨)

summary.csv는 results/walk_forward/{policy}/{fold}/summary.csv에 위치하며
aggregate_summary()가 매 실행 종료 후 그 시점의 모든 metrics.json으로부터
다시 생성한다 (9절 resume 원칙에서 쓰는 결과물이지 사람이 직접 append하는
로그가 아님).

정책망 비교 원칙 (docs/walk_forward_design.md)
------------------------------------------------
MLP/Transformer만 대상 - LSTM은 연구 범위 밖으로 확정됨(코덱스 검토에서도
동의). 두 정책망 모두 동일 조건(fold 경계/거래비용/초기자본/TIME_WINDOW/
평가지표)을 강제로 공유하도록 이 스크립트 하나에서 관리한다.

오늘(9월 4주차) 구현 범위
--------------------------
스켈레톤(디렉토리 구조/resume/config 저장 전체) + Fold 1 하이퍼파라미터
탐색(fold1_search) 단계의 실제 실행 검증까지. Fold 1 최종 재학습
(fold1_final)과 Fold 2(fold2_final) 단계도 동일 인프라로 구현되어 있으나,
LOCKED_CANDIDATES_PATH가 없으면 실행 자체가 거부된다 - 후보 확정 전에는
의도적으로 막아둔 것.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from glob import glob

import shimmy
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from cryptoagent.envs.evaluate import evaluate as compute_eval_metrics
from cryptoagent.policies.transformer_extractor import TransformerFeaturesExtractor
from cryptoagent.training.common import (
    backtest,
    make_env_by_date,
    sanity_check,
    trim_warmup_rows,
)

# ── 공통 조건 (모든 정책망/fold에 동일 적용, docs/colab_compute_budget.md 10절) ──
FEATURES = ["close", "high", "low"]
INITIAL_AMOUNT = 100_000
TIME_WINDOW = 50
RESULTS_ROOT = "results/walk_forward"
LOCKED_CANDIDATES_PATH = os.path.join(RESULTS_ROOT, "locked_candidates.json")

# ── Fold 날짜 경계 (docs/walk_forward_design.md, 반개구간) ──
FOLD1_TRAIN = ("2021-01-01", "2023-01-02")
FOLD1_VAL = ("2023-01-02", "2023-10-16")
FOLD1_FINAL_TRAIN = ("2021-01-01", "2023-10-16")
FOLD1_OOS = ("2023-10-16", "2025-01-02")
FOLD2_FINAL_TRAIN = ("2021-01-01", "2025-01-02")
FOLD2_OOS = ("2025-01-02", "2026-01-01")

# PPO(...) 생성자에 그대로 전달할 수 있는 하이퍼파라미터 allowlist.
# hparams에 이 목록 밖의 키가 있어도(total_timesteps, d_model 등 policy 전용
# 설정) 조용히 무시되지 않도록 build_model()에서 명시적으로만 꺼내 쓴다.
PPO_HPARAM_KEYS = ("learning_rate", "gamma", "batch_size", "n_steps", "n_epochs", "ent_coef", "clip_range")

# ── Fold 1 하이퍼파라미터 후보 (파일럿/스켈레톤 검증용 초기값) ──
# N, S는 docs/colab_compute_budget.md 11절에 따라 파일럿 완료 후 확정한다.
# 여기 있는 값은 오케스트레이터 인프라 자체를 검증하기 위한 자리표시 후보이며,
# 정식 실행 전 실제 후보 목록으로 교체한다.
MLP_CANDIDATES: dict[str, dict] = {
    "candidate01": {"total_timesteps": 10_240},
}

TRANSFORMER_CANDIDATES: dict[str, dict] = {
    "candidate01": {"total_timesteps": 10_240, "d_model": 32, "n_heads": 4, "n_layers": 2},
}

SEEDS = [42]


@dataclass
class RunSpec:
    policy: str  # "mlp" | "transformer"
    fold: str  # "fold1_search" | "fold1_final" | "fold2_final"
    candidate: str
    seed: int
    train_range: tuple[str, str]
    eval_range: tuple[str, str]
    eval_split_name: str  # 저장 파일명에 쓰일 이름 ("val" 또는 "oos")
    hparams: dict = field(default_factory=dict)

    @property
    def run_dir(self) -> str:
        return os.path.join(RESULTS_ROOT, self.policy, self.fold, self.candidate, f"seed{self.seed}")

    def as_config_dict(self) -> dict:
        return {
            "policy": self.policy,
            "fold": self.fold,
            "candidate": self.candidate,
            "seed": self.seed,
            "train_range": list(self.train_range),
            "eval_range": list(self.eval_range),
            "eval_split_name": self.eval_split_name,
            "hparams": self.hparams,
            "time_window": TIME_WINDOW,
            "features": FEATURES,
            "initial_amount": INITIAL_AMOUNT,
        }


def build_model(policy: str, vec_env, seed: int, hparams: dict, device: str = "auto") -> PPO:
    """device: "auto"(SB3 기본, CUDA 있으면 GPU) / "cpu" / "cuda".

    파일럿 실측 결과(2026-09-01, Colab T4) - MLP와 우리 규모의 Transformer
    (d_model=32) 둘 다 파라미터 수가 작고 PortfolioOptimizationEnv.step()
    자체가 pandas 기반 CPU 바운드라, GPU로 보내는 텐서 전송 오버헤드가
    연산 이득보다 커서 오히려 CPU보다 느렸다 (MLP: CPU 292fps vs GPU 54fps,
    Transformer: CPU 29fps vs GPU 45fps - Transformer만 근소하게 GPU가
    유리했고 MLP는 GPU가 5배 이상 느림). SB3 자체도 MlpPolicy를 GPU에서
    돌리면 이런 경고를 낸다 (DLR-RM/stable-baselines3#1245). 그래서 이
    함수 시그니처에 device를 노출해 실행할 때 명시적으로 고를 수 있게
    했다 - 학습 결과(가중치)는 device와 무관하게 동일해야 하므로
    RunSpec.as_config_dict()에는 포함하지 않는다 (resume 판단에 영향을
    주면 안 되는 순수 실행 환경 설정).
    """
    ppo_kwargs = {k: hparams[k] for k in PPO_HPARAM_KEYS if k in hparams}

    if policy == "mlp":
        return PPO("MlpPolicy", vec_env, verbose=1, seed=seed, device=device, **ppo_kwargs)

    if policy == "transformer":
        policy_kwargs = dict(
            features_extractor_class=TransformerFeaturesExtractor,
            features_extractor_kwargs=dict(
                d_model=hparams["d_model"],
                n_heads=hparams["n_heads"],
                n_layers=hparams["n_layers"],
            ),
        )
        return PPO(
            "MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1, seed=seed, device=device, **ppo_kwargs
        )

    raise ValueError(f"지원하지 않는 policy: {policy} (mlp/transformer만 walk-forward 대상)")


def make_train_env(spec: RunSpec):
    start, end_exclusive = spec.train_range
    return make_env_by_date(
        start,
        end_exclusive,
        features=FEATURES,
        initial_amount=INITIAL_AMOUNT,
        time_window=TIME_WINDOW,
        enable_warmup=False,  # train은 fold 경계 자체가 국면 경계라 워밍업 대상 없음
        cwd=spec.run_dir,
    )


def make_eval_env(spec: RunSpec):
    """val이든 OOS든 평가 구간은 항상 워밍업을 켠다 (Major #1 수정).

    구간 시작 직전 TIME_WINDOW시간을 관측용으로만 포함하고,
    trim_warmup_rows()로 실제 거래 기록에서는 제외한다 - 그렇지 않으면
    구간 시작 첫 TIME_WINDOW-1시간이 통째로 backtest 결과에서 사라진다.
    """
    start, end_exclusive = spec.eval_range
    return make_env_by_date(
        start,
        end_exclusive,
        features=FEATURES,
        initial_amount=INITIAL_AMOUNT,
        time_window=TIME_WINDOW,
        enable_warmup=True,
        cwd=spec.run_dir,
    )


def _write_json_atomic(path: str, payload: dict) -> None:
    """쓰다 만 파일이 완료 상태로 오인되지 않도록 임시 파일 후 os.replace로 교체."""
    directory = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def is_completed(spec: RunSpec) -> bool:
    """status.json뿐 아니라 config 일치와 필수 산출물 실존까지 확인 (Major #3 수정).

    hparams를 바꿔 같은 candidate 이름을 재실행하면 config가 달라지므로 재실행
    하도록 강제하고, model.zip/backtest csv/metrics.json 중 하나라도 없으면
    (예: status 저장 후 프로세스가 죽은 경우) 미완료로 간주해 재실행한다.
    """
    run_dir = spec.run_dir
    status_path = os.path.join(run_dir, "status.json")
    config_path = os.path.join(run_dir, "config.json")
    metrics_path = os.path.join(run_dir, "metrics.json")
    model_path = os.path.join(run_dir, "model.zip")
    backtest_path = os.path.join(run_dir, f"backtest_{spec.eval_split_name}.csv")

    required_paths = [status_path, config_path, metrics_path, model_path, backtest_path]
    if not all(os.path.exists(p) for p in required_paths):
        return False

    try:
        with open(status_path) as f:
            status = json.load(f)
        with open(config_path) as f:
            saved_config = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    if status.get("status") != "completed":
        return False

    return saved_config == spec.as_config_dict()


def aggregate_summary(policy: str, fold: str) -> None:
    """{policy}/{fold} 아래 모든 completed run의 metrics.json을 모아 summary.csv를 통째로 재생성.

    병렬 실행 시 공용 CSV에 동시에 append하면 헤더 중복/행 충돌이 날 수 있어
    (Major #4), run별로는 자기 디렉토리 안 metrics.json만 원자적으로 쓰고
    summary.csv는 매 실행 종료 후 이 함수가 그 시점의 전체 상태로부터 새로
    만든다 - 어느 프로세스가 마지막에 끝나든 결과가 누락/충돌되지 않는다.
    """
    import csv

    fold_dir = os.path.join(RESULTS_ROOT, policy, fold)
    rows = []
    for metrics_path in sorted(glob(os.path.join(fold_dir, "*", "seed*", "metrics.json"))):
        run_dir = os.path.dirname(metrics_path)
        status_path = os.path.join(run_dir, "status.json")
        if not os.path.exists(status_path):
            continue
        with open(status_path) as f:
            if json.load(f).get("status") != "completed":
                continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        candidate = os.path.basename(os.path.dirname(run_dir))
        seed = os.path.basename(run_dir).replace("seed", "")
        rows.append({"policy": policy, "fold": fold, "candidate": candidate, "seed": seed, **metrics})

    summary_path = os.path.join(fold_dir, "summary.csv")
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_one(spec: RunSpec, device: str = "auto") -> None:
    if is_completed(spec):
        print(f"[skip] {spec.run_dir} 이미 completed (config 일치, 산출물 실존 확인됨) - resume 원칙에 따라 건너뜀")
        return

    os.makedirs(spec.run_dir, exist_ok=True)
    config_path = os.path.join(spec.run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(spec.as_config_dict(), f, indent=2)

    print(f"=== [{spec.run_dir}] 학습 시작 (train {spec.train_range}, device={device}) ===")
    start_time = time.time()

    train_env = make_train_env(spec)
    gym_env = shimmy.GymV21CompatibilityV0(env=train_env)
    vec_env = DummyVecEnv([lambda: Monitor(gym_env)])

    model = build_model(spec.policy, vec_env, spec.seed, spec.hparams, device=device)
    model.learn(total_timesteps=spec.hparams["total_timesteps"])

    model_path = os.path.join(spec.run_dir, "model.zip")
    model.save(model_path)
    train_elapsed = time.time() - start_time
    print(f"모델 저장: {model_path} (학습 {train_elapsed:.1f}s)")

    print(f"=== [{spec.run_dir}] {spec.eval_split_name} 백테스트 ===")
    eval_start = time.time()
    eval_env = make_eval_env(spec)
    backtest_df = backtest(model, eval_env)
    backtest_df = trim_warmup_rows(backtest_df, oos_start=spec.eval_range[0])
    sanity_check(backtest_df)
    eval_elapsed = time.time() - eval_start

    backtest_path = os.path.join(spec.run_dir, f"backtest_{spec.eval_split_name}.csv")
    backtest_df.to_csv(backtest_path)
    print(f"백테스트 결과 저장: {backtest_path} shape={backtest_df.shape}")

    eval_metrics = compute_eval_metrics(backtest_df)

    metrics = {
        "total_timesteps": spec.hparams["total_timesteps"],
        "train_elapsed_sec": round(train_elapsed, 1),
        "eval_elapsed_sec": round(eval_elapsed, 1),
        "final_portfolio_value": round(float(backtest_df["portfolio_values"].iloc[-1]), 2),
        "n_eval_rows": len(backtest_df),
        **eval_metrics,
    }
    metrics_path = os.path.join(spec.run_dir, "metrics.json")
    _write_json_atomic(metrics_path, metrics)

    status_path = os.path.join(spec.run_dir, "status.json")
    _write_json_atomic(status_path, {"status": "completed"})

    aggregate_summary(spec.policy, spec.fold)
    print(f"[done] {spec.run_dir}")


def build_fold1_search_specs(policy: str) -> list[RunSpec]:
    candidates = MLP_CANDIDATES if policy == "mlp" else TRANSFORMER_CANDIDATES
    specs = []
    for candidate, hparams in candidates.items():
        for seed in SEEDS:
            specs.append(
                RunSpec(
                    policy=policy,
                    fold="fold1_search",
                    candidate=candidate,
                    seed=seed,
                    train_range=FOLD1_TRAIN,
                    eval_range=FOLD1_VAL,
                    eval_split_name="val",
                    hparams=hparams,
                )
            )
    return specs


def _load_locked_candidates() -> dict:
    """fold1_final/fold2_final 실행 잠금 (Major #2 수정).

    후보/seed 선택이 side_2023 validation 결과로 확정된 뒤에만 이 파일을
    직접 만들어서 final 단계 실행을 허용한다. 파일이 없으면 자리표시
    candidate01로 최종/OOS 학습이 실수로 실행되는 것을 원천 차단한다.

    형식:
        {
          "mlp": {"candidate": "candidate03", "seed": 42},
          "transformer": {"candidate": "candidate02", "seed": 42}
        }
    """
    if not os.path.exists(LOCKED_CANDIDATES_PATH):
        raise SystemExit(
            f"{LOCKED_CANDIDATES_PATH}가 없습니다. fold1_final/fold2_final은 side_2023 "
            f"validation으로 후보를 확정한 뒤 이 파일을 직접 만들어야 실행할 수 있습니다 "
            f"(자리표시 후보로 최종/OOS 학습이 실수로 실행되는 것을 막기 위함)."
        )
    with open(LOCKED_CANDIDATES_PATH) as f:
        return json.load(f)


def build_fold1_final_spec(policy: str, candidate: str, seed: int, hparams: dict) -> RunSpec:
    return RunSpec(
        policy=policy,
        fold="fold1_final",
        candidate=candidate,
        seed=seed,
        train_range=FOLD1_FINAL_TRAIN,
        eval_range=FOLD1_OOS,
        eval_split_name="oos",
        hparams=hparams,
    )


def build_fold2_final_spec(policy: str, candidate: str, seed: int, hparams: dict) -> RunSpec:
    return RunSpec(
        policy=policy,
        fold="fold2_final",
        candidate=candidate,
        seed=seed,
        train_range=FOLD2_FINAL_TRAIN,
        eval_range=FOLD2_OOS,
        eval_split_name="oos",
        hparams=hparams,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("mlp", "transformer"), required=True)
    parser.add_argument(
        "--stage",
        choices=("fold1_search", "fold1_final", "fold2_final"),
        required=True,
        help=(
            "fold1_search: Fold1 train/val로 후보 하이퍼파라미터 탐색 (잠금 없음). "
            "fold1_final/fold2_final: locked_candidates.json에 등록된 확정 후보만 실행 가능."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help=(
            "PPO 학습에 쓸 디바이스. 기본 auto는 SB3가 CUDA 가용 여부로 자동 선택 - "
            "MLP와 우리 규모의 Transformer(d_model=32)는 파라미터가 작고 env.step()이 "
            "pandas 기반 CPU 바운드라 GPU 전송 오버헤드가 더 클 수 있다 (2026-09-01 "
            "Colab T4 파일럿 실측: MLP는 CPU가 GPU보다 5배 이상 빠름). 병렬로 로컬/"
            "Colab을 같이 쓸 때 이 옵션으로 각 환경에 맞는 디바이스를 명시할 것."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.stage == "fold1_search":
        specs = build_fold1_search_specs(args.policy)
    else:
        locked = _load_locked_candidates()
        if args.policy not in locked:
            raise SystemExit(
                f"{LOCKED_CANDIDATES_PATH}에 policy={args.policy}에 대한 확정 후보가 없습니다."
            )
        candidate = locked[args.policy]["candidate"]
        seed = locked[args.policy]["seed"]
        candidates = MLP_CANDIDATES if args.policy == "mlp" else TRANSFORMER_CANDIDATES
        if candidate not in candidates:
            raise SystemExit(
                f"locked_candidates.json이 가리키는 candidate={candidate}가 "
                f"{'MLP_CANDIDATES' if args.policy == 'mlp' else 'TRANSFORMER_CANDIDATES'}에 없습니다."
            )
        hparams = candidates[candidate]
        builder = build_fold1_final_spec if args.stage == "fold1_final" else build_fold2_final_spec
        specs = [builder(args.policy, candidate, seed, hparams)]

    for spec in specs:
        run_one(spec, device=args.device)


if __name__ == "__main__":
    main()
