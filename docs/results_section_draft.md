# Results/Discussion 초안 문장 — 은아·유지 인수인계용

> [!warning] **2026-10-19 행동 진단 수치 갱신됨 (4절)**
> 행동 진단 분석 코드의 정의 오류를 수정하고 진단만 재생성했다
> (학습·성과지표·통계검정은 불변). 4절의 스텝 변화량과 시드 이질성
> 수치가 바뀌었으니 **이 문서의 현재 값을 쓸 것**. 이전 값
> (3×10⁻⁹~1×10⁻⁸, 1~3%p 등)이 다른 곳에 남아 있다면 무시한다.
> 상세: `docs/thesis_notation_glossary.md` §8.4

**🚩 2026-09-08부로 아래 모든 수치는 동결(freeze)됐다.** **실험 동결 커밋**
`10892a1`(학습 결과·성과지표·통계검정이 확정된 커밋) 기준이며,
`results/frozen_summaries/manifest.json`에 재현에 필요한 모든 메타데이터
(체크섬 61건, 패키지 버전, 평가 설정, seed 확장 이력)가 기록되어 있다.
manifest의 `freeze_lineage.experiment_freeze_commit`이 이 동결 커밋이고,
`git.source_commit_sha`는 명세가 생성된 시점의 HEAD이므로 사후 분석
코드가 수정되면 갱신된다 — 두 값은 다를 수 있다.
**11월 논문 작성 기간 동안 이 문서와 manifest에 있는 숫자를 바꾸지 말 것**
— 새로운 계산이나 재실행이 필요하다고 느껴지면 먼저 준영에게 확인할 것.
"조금만 더 확인해보자"는 이유로 스크립트를 다시 돌리면 freeze 원칙이
깨진다 (`docs/project_milestone_plan.md`의 10월 게이트 참고).

이 문서는 9월 4주차~10월 3주차 walk-forward validation 실험에서 확정된
검증 결과를 논문 Results/Discussion 절에 바로 쓸 수 있는 문장 단위로
정리한 것이다. 여기 있는 문장은 실제 코드 실행으로 검증됐고, 여러 차례
AI 보조 코드 검토(총 6라운드)와 재실행 검증을 거쳐 정확성을 확인했다 — **수치나 통계적
주장을 바꾸지 말고, 학술적 문체로 다듬거나 문단 흐름을 만드는 데만
활용할 것.** 숫자를 수정해야 할 필요가 있으면 준영에게 먼저 확인할 것
(재계산 없이 문구만 바꾸는 게 아니라면).

## 0. 이 문서를 읽기 전에 (은아·유지 필독)

- **왜 이렇게 여러 번 고쳐졌나**: 아래 결과는 처음부터 한 번에 나온 게
  아니라, 통계 검정·시각화 코드에서 발견된 여러 번의 버그(gross/net
  혼용, 잘못된 통계량 정의, bootstrap seed 비결정성 등)를 재현 검증하고
  고친 뒤의 최종본이다. 이전 버전(예: "PPO가 벤치마크와 통계적으로
  구분되지 않는다"는 표현, "6전략 겹침"이라던 그림에 실제로는 5개
  전략만 있었던 것 등)이 다른 곳(슬랙, 이전 초안 등)에 남아있다면 이
  문서의 최신 버전을 기준으로 삼을 것.
- **재현이 필요하면**: 아래 6절(인용 근거 파일)과 7절(재현 방법)을 그대로
  따라가면 된다 (bootstrap 통계 검정까지 결정론적으로 재현됨, 3회 이상
  별도 프로세스로 검증함).
- **"PPO가 이겼다/졌다"고 단순화해서 쓰지 말 것**: 아래 3절에 있는
  통계적 엄밀성 문단이 이 연구에서 가장 중요한 방법론적 포인트다.
  "유의하지 않음"을 "동등함"으로 바꿔 쓰면 통계적으로 틀린 문장이 된다.

## 1. 실험 설계 (Methods에 들어갈 내용, 참고용)

- Regime-preserving expanding-window 2-fold walk-forward validation.
  Fold 1: train(bull_2021+bear_2022) → validation(side_2023)으로
  하이퍼파라미터 후보 선택 → 확정 설정으로 2021-01-01~2023-10-15 전체
  재학습 → bull_2024(2023-10-16~2025-01-02) OOS 평가. Fold 2: 확정
  설정 그대로 2021-01-01~2025-01-01 재학습 → choppy_2025
  (2025-01-02~2026-01-01) OOS 평가.
- MLP/Transformer 각각 4개 후보(learning_rate, 구조 변형)를 3-seed로
  탐색해 side_2023 validation net Sharpe 평균이 최고인 후보를 선택:
  MLP는 candidate03(learning_rate=1e-3), Transformer는
  candidate04(n_layers=1, 기본 2에서 축소).
- 최종 OOS 평가는 seed 5개(42~46)로 반복. **주의(각주 필수)**: 최초
  프로토콜은 3-seed(42~44)였고, "최소 5시드" 원칙에 맞춰 OOS 확인 후
  후보나 하이퍼파라미터 변경 없이 seed 45·46을 두 정책·두 fold에
  동일하게 추가한 post-hoc robustness extension이다. 원래 3-seed
  결과도 별도 보존되어 있으며, 5-seed로 확장해도 결론은 동일했다.
- 거래비용 0.1%(cost_rate)를 turnover에 곱해 차감한 net returns를
  Sharpe/Sortino/Calmar/MDD/CAGR 계산에 사용.

## 2. Fold별 OOS 성과 (문장 후보)

> MLP(candidate03)와 Transformer(candidate04)는 각각 5개 seed로
> bull_2024 OOS 구간에서 평균 Sharpe 2.203(±0.017)과 2.206(±0.018)을,
> choppy_2025 OOS 구간에서 평균 Sharpe -0.194(±0.029)와 -0.222(±0.031)을
> 기록했다. 두 정책 모두 상승 국면에서는 높은 위험조정 성과를 보였으나
> 횡보·조정 국면에서는 손실을 기록했다.

> MLP와 Transformer 간 Sharpe 차이(Transformer - MLP)의 95% 신뢰구간은
> bull_2024에서 [-0.031, +0.037], choppy_2025에서 [-0.069, +0.013]으로
> 두 구간 모두 0을 포함했다. n=5 seed 표본에 한정된 결과이지만, 두
> feature extractor 구조 간 성과 차이가 seed 변동성 안에 있다는 것을
> 시사한다.

## 3. 전통 벤치마크와의 비교 (문장 후보 — 정정된 버전)

> 동일한 OOS 날짜 범위로 재실행한 4개 전통 벤치마크(Buy & Hold BTC,
> Equal-Weight, Minimum-Variance, Risk Parity)와 비교했을 때, PPO 두
> 모델은 어느 국면에서도 일관되게 우위를 보이지 않았다. bull_2024에서는
> Buy & Hold BTC(Sharpe 2.272)와 Equal-Weight(2.218)가 PPO(2.20~2.21)
> 보다 근소하게 높았고, choppy_2025에서는 Minimum-Variance(0.171)와
> Buy & Hold BTC(0.066)만 플러스 Sharpe를 유지한 반면 PPO 두 모델과
> Equal-Weight, Risk Parity는 모두 손실을 기록했다.

> **(통계적 엄밀성 — 반드시 이 문장 또는 동등한 취지로 넣을 것)** 같은
> 시간 인덱스의 PPO 5-seed와 벤치마크 net returns를 함께(6개 시계열을
> 동일한 블록 인덱스로) stationary block bootstrap(block size 24, 민감도
> 확인용 168에서도 유사)으로 리샘플해, 매 리샘플에서 5-seed 평균
> Sharpe(PPO) - Sharpe(benchmark)를 계산하는 방식으로 95% 신뢰구간과
> bootstrap p-value를 구했다. 정책 2종 × 벤치마크 4종 × fold 2개의 16개
> 비교 전부 Holm-Bonferroni 보정 후 유의하지 않았다(최소 p=0.118,
> Transformer vs Minimum-Variance, choppy_2025). **이 두 OOS 국면에서
> PPO와 전통 벤치마크 간 Sharpe 차이가 0과 다르다는 통계적 증거를 얻지
> 못했다.**

> 위 결과를 "PPO와 벤치마크가 통계적으로 동등하다"고 서술해서는 안 된다
> — 비유의는 동등성의 증거가 아니며, 동등성을 주장하려면 사전에 정의된
> 실질적 동등성 범위(예: Sharpe 차이 ±δ)에 대한 equivalence test가
> 별도로 필요하다. 본 연구는 그러한 검정을 수행하지 않았다.

> Buy & Hold BTC(Sharpe 0.066)와 Minimum-Variance(0.171)는 choppy_2025
> 구간에서 Sharpe가 양수였지만 CAGR은 각각 -6.7%, -5.0%로 실제로는
> 손실이었다 — Sharpe(산술평균 기반)와 CAGR(기하평균 기반 복리) 계산
> 방식의 차이에서 비롯된 결과이며, 이 국면에서는 6개 전략 모두 손실을
> 기록했고 Minimum-Variance가 손실을 가장 잘 방어했다고 해석하는 것이
> 정확하다.

## 4. 정책 행동 진단 (가장 강한 문장 — 논문의 핵심 기여 중 하나가 될 수 있음)

> 저성과의 원인을 진단하기 위해 각 정책이 실제로 지시한 목표 비중
> (target_weights — 환경이 PPO raw action을 softmax로 정규화해 저장한
> 값, 가격 변동이 반영되기 전의 리밸런싱 목표)을 직접 분석했다. 두
> 정책 모두 OOS 전체 구간에서 목표 비중의 시점별 변화량(9개 성분의
> 절대값 합)이 평균 3.8×10⁻⁸~7.4×10⁻⁸ 수준으로, 이는 float32 연산
> 정밀도에 해당하는 값이다.
> 즉 학습된 정책은 관측(observation)이 매 시점 달라져도 사실상 동일한
> action을 출력했으며, 상태 의존적인 동적 자산배분 행동은 관찰되지
> 않았다.
>
> 다만 이는 "모든 seed가 동일한 배분을 학습했다"는 뜻이 아니다. seed
> 간에는 시간 평균 목표 비중이 자산별로 표준편차 0.63~3.43%p, 최대 범위
> 3.70~8.07%p(현금 비중은 7.70%~14.23%) 정도의 차이를 보였다 — 즉 각 seed는
> 서로 다른 정적(static) 포트폴리오를 학습했고, 그 정적 배분이 각 seed
> 내부에서는 시간에 따라 변하지 않은 것이다.

> 이 결과와 일관되게, PPO의 시간별 순수익률은 Equal-Weight 벤치마크의
> 순수익률과 상관계수 0.999 이상을 기록했다. 목표 현금 비중 역시 15%
> 이상의 낙폭(drawdown) 구간과 평시 구간에서 소수점 5자리까지 동일한
> 평균값을 보여, 손실 국면에서도 정책이 현금으로 자산을 이동시키는
> 방어적 행동을 학습하지 못했음을 시사한다.

> 정책이 사실상 고정된 배분에 수렴한 정확한 원인 — 학습 과정에서의
> 국면 과적합, reward 함수 설계, 제한된 feature 집합, 최적화 절차의
> 한계 중 어느 것이 지배적인지 — 은 본 실험 설계만으로는 식별할 수
> 없으며, 향후 연구 과제로 남긴다.

## 5. 논문 결론 문단 (종합, 초안)

> OOS 평가 전에 고정된 후보 설정과, 사후 강건성 확장을 투명하게 보고한
> 5-seed 결과를 사용한 regime-preserving expanding-window walk-forward
> 평가에서, MLP 및 Transformer 기반 PPO 정책은 전통적인 포트폴리오
> 전략 대비 일관된 위험조정 성과 우위를
> 보이지 않았으며, 두 OOS 국면에서 PPO와 벤치마크 간 Sharpe 차이가
> 0과 다르다는 통계적 증거도 확인되지 않았다. 두 정책망(MLP,
> Transformer) 간 성과 차이 역시 seed 변동성 범위 안에 있었다. 사후
> 행동 진단 결과, 두 정책 모두 관측에 실질적으로 반응하지 않는 고정된
> 목표 비중을 학습했으며 그 수익 경로는 단순 Equal-Weight 전략과
> 거의 동일했다. 이는 본 연구의 설정(데이터 규모, feature 집합, reward
> 설계, 학습 예산)에서 state-dependent한 동적 자산배분이 유의미하게
> 학습되지 않았음을 보여주며, 그 근본 원인의 규명은 향후 연구로 남긴다.

## 6. 인용 근거 파일 (필요시 재확인용)

- **freeze manifest(가장 먼저 확인할 것)**: `results/frozen_summaries/manifest.json`
  — 실험 동결 커밋·분석 계보, 패키지 버전, cost_rate/연율화 상수, seed 확장
  이력, 결과 파일 41개와 모형 아카이브 20개의 SHA-256(총 61건) 전부 포함
- 후보 선택 및 확정 설정: `configs/walk_forward_locked_candidates.json`
  (selection_rule, seeds_note_protocol_status에 5-seed가 post-hoc
  extension이라는 사실이 정확히 기록되어 있음 — Methods 절 작성 시
  이 문구를 그대로 참고할 것)
- Fold별 OOS 성과(5-seed): `results/frozen_summaries/{mlp,transformer}_{fold1,fold2}_final_summary.csv`
- 원래 3-seed 결과(참고용, post-hoc 확장 전): `results/frozen_summaries/{mlp,transformer}_{fold1,fold2}_final_summary_3seed_original.csv`
- 벤치마크 비교(gross/net 모두 포함): `results/frozen_summaries/benchmarks_summary.csv`
- 통계 검정(bootstrap CI, p-value, Holm 보정 결과): `results/frozen_summaries/statistical_tests.json`
- 행동 진단: `results/walk_forward_behavior_diagnostics/diagnostics.json`
  (`experiments/walk_forward_behavior_diagnostics.py`로 재생성 가능)
- 논문용 그림: `results/walk_forward_figures/{fold}/main_{cumulative_returns,drawdown}.png`
  (본문용, MLP+Transformer+벤치마크 4종 통합), `{policy}_*.png`(부록용,
  정책별 상세 + target allocation + seed 이질성 range plot). quantstats
  HTML은 대표 seed 1개짜리 참고용 tearsheet이며 본문 수치의 근거가 아님
  (`experiments/walk_forward_visualizations.py`로 재생성 가능)

## 7. 재현 방법 (숫자를 의심하게 되면 이 순서로 직접 확인)

목적에 따라 절차가 다르다. **대부분의 경우 A만 하면 된다.**

### A. 동결 파일의 무결성 검증 (가장 먼저, 재실행 불필요)

`results/frozen_summaries/`의 파일이 손상·교체되지 않았는지만 확인한다.

```
manifest.json의 체크섬과 실제 파일을 대조
  - ppo_backtest_files: 20개 항목 (백테스트 CSV 20 + 모형 .zip 20 = 40)
  - benchmark_files 8, frozen_summary_files 12, raw_input_files 1
  - 합계 = 결과 파일 41개 + 모형 아카이브 20개 = 체크섬 61건
```

체크섬이 전부 일치하면 **이 문서의 숫자는 그대로 유효하다.** 분석을
재실행할 필요가 없다.

### B. 저장된 백테스트로 분석만 재실행

통계검정·행동진단·그림을 다시 만들어 보고 싶을 때. **PPO 재학습은
하지 않는다.**

1. manifest의 `analysis_lineage.scripts[*].last_commit` 확인 —
   **스크립트마다 커밋이 다르다** (동결 이후 분석 코드가 수정된 이력이
   있기 때문. 3.7.1 참고)
2. 각 스크립트를 그 커밋 기준으로 맞춘다. 현재 main의 파일 체크섬이
   manifest와 일치한다면 main을 그대로 써도 된다 — 다만 이는 "동결 이후
   변경이 없었기 때문"이 아니라 "지금 상태가 기록과 같기 때문"이다
3. **프로젝트 가상환경**으로 실행한다 (`.venv/bin/python`).
   다른 인터프리터로 돌리면 패키지 버전이 기록되지 않는다
4. 실행:

   | 스크립트 | 출력 위치 |
   |---|---|
   | `walk_forward_statistical_tests.py` | `results/walk_forward_statistical_tests/` |
   | `walk_forward_behavior_diagnostics.py` | `results/walk_forward_behavior_diagnostics/` |
   | `walk_forward_visualizations.py` | `results/walk_forward_figures/` |
   | `walk_forward_freeze_manifest.py` | `results/frozen_summaries/manifest.json` |

> [!warning] 이 스크립트들은 동결 폴더를 갱신하지 않는다
> `frozen_summaries/`의 요약 CSV 11개는 **어떤 스크립트도 재생성하지
> 않는다**(동결 시점에 수동 보존한 사본). 같은 폴더의
> `statistical_tests.json`도 동결 사본이며, 통계검정 스크립트는
> 자기 출력 폴더에 쓰고 이 사본을 갱신하지 않는다. 따라서 B를 실행해도
> 동결 파일은 그대로이며, 비교하려면 새 출력과 동결 사본을 **직접
> 대조**해야 한다.
>
> `manifest.json`은 생성 시각과 당시 HEAD를 기록하므로 재생성본이
> 바이트 단위로 같지 않다. 체크섬·계보 항목만 비교할 것.

### C. 원천 데이터부터 전체 재현

PPO 재학습을 포함한 완전 재현. `walk_forward_orchestrator.py`가 필요하다.

- 원천 DB(`data/raw/crypto_market_features.db`)와 학습된 모형 파일은
  **저장소에 포함되어 있지 않다**(용량). manifest의 체크섬은 무결성
  확인용이지 파일을 복원하지 못한다
- 학습 시드는 고정되어 있으나 결정론적 연산 모드를 강제하지 않았으므로,
  다른 하드웨어·라이브러리에서는 통계적으로 유사하되 비트 단위로 동일한
  결과가 나오지 않을 수 있다

### 결과가 이 문서의 숫자와 다를 때

재계산하지 말고 **원인부터 확인**한다. 다음 순서로 좁힌다.

1. **입력** — 백테스트 CSV와 원천 DB의 체크섬이 manifest와 일치하는가
2. **코드** — 실행한 스크립트의 체크섬이 `analysis_lineage` 기록과 같은가
3. **환경** — `.venv`로 실행했는가, 패키지 버전이 manifest와 같은가
4. **경로** — 동결 사본을 본 것인지 새 출력을 본 것인지 (B의 경고 참고)

위를 전부 확인해도 다르면 준영에게 알릴 것. "freeze 이후 코드가 바뀌었다"는
여러 가능한 원인 중 하나일 뿐이다.
