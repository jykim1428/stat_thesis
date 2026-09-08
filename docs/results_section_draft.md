# Results/Discussion 초안 문장 — 은아·유지 인수인계용

이 문서는 9월 4주차~10월 1주차 walk-forward validation 실험에서 확정된
검증 결과를 논문 Results/Discussion 절에 바로 쓸 수 있는 문장 단위로
정리한 것이다. 여기 있는 문장은 실제 코드 실행으로 검증됐고, 여러 차례
독립 리뷰(코덱스)를 거쳐 정확성을 확인했다 — **수치나 통계적 주장을
바꾸지 말고, 학술적 문체로 다듬거나 문단 흐름을 만드는 데만 활용할 것.**
숫자를 수정해야 할 필요가 있으면 준영에게 먼저 확인할 것 (재계산 없이
문구만 바꾸는 게 아니라면).

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
> 시간 인덱스의 PPO와 벤치마크 net returns 차이를 stationary block
> bootstrap(block size 24, 민감도 확인용 168에서도 유사)으로 리샘플해
> Sharpe(PPO) - Sharpe(benchmark)의 95% 신뢰구간과 bootstrap p-value를
> 계산한 결과, 정책 2종 × 벤치마크 4종 × fold 2개의 16개 비교 전부
> Holm-Bonferroni 보정 후 유의하지 않았다(최소 p=0.138, Transformer vs
> Minimum-Variance, choppy_2025). **이 두 OOS 국면에서 PPO와 전통
> 벤치마크 간 Sharpe 차이가 0과 다르다는 통계적 증거를 얻지 못했다.**

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
> (target actions, 가격 변동이 반영되기 전의 raw policy output)을
> 직접 분석했다. 두 정책 모두 OOS 전체 구간에서 목표 비중의 스텝 간
> 평균 변화량이 3×10⁻⁹~1×10⁻⁸ 수준으로, 이는 float32 연산 정밀도에
> 해당하는 값이다. 즉 학습된 정책은 관측(observation)이 매 시점 달라
> 져도 사실상 동일한 action을 출력했으며, 상태 의존적인 동적 자산배분
> 행동은 관찰되지 않았다.

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

> 사전에 고정된 후보와 seed 프로토콜을 사용한 regime-preserving
> expanding-window walk-forward 평가에서, MLP 및 Transformer 기반 PPO
> 정책은 전통적인 포트폴리오 전략 대비 일관된 위험조정 성과 우위를
> 보이지 않았으며, 두 OOS 국면에서 PPO와 벤치마크 간 Sharpe 차이가
> 0과 다르다는 통계적 증거도 확인되지 않았다. 두 정책망(MLP,
> Transformer) 간 성과 차이 역시 seed 변동성 범위 안에 있었다. 사후
> 행동 진단 결과, 두 정책 모두 관측에 실질적으로 반응하지 않는 고정된
> 목표 비중을 학습했으며 그 수익 경로는 단순 Equal-Weight 전략과
> 거의 동일했다. 이는 본 연구의 설정(데이터 규모, feature 집합, reward
> 설계, 학습 예산)에서 state-dependent한 동적 자산배분이 유의미하게
> 학습되지 않았음을 보여주며, 그 근본 원인의 규명은 향후 연구로 남긴다.

## 인용 근거 파일 (필요시 재확인용)

- 후보 선택: `configs/walk_forward_locked_candidates.json`
- Fold별 OOS 성과: `results/frozen_summaries/{mlp,transformer}_{fold1,fold2}_final_summary.csv`
- 벤치마크 비교: `results/frozen_summaries/benchmarks_summary.csv`
- 통계 검정: `results/frozen_summaries/statistical_tests.json`
- 행동 진단: `results/walk_forward_behavior_diagnostics/diagnostics.json`
  (`experiments/walk_forward_behavior_diagnostics.py`로 재생성 가능)
