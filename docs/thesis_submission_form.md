# 졸업논문 제출 양식 — 제목 · 목차 · 연구방법 및 참고서

**제출자:** 김준영 (공동연구: 유지, 은아) · **작성일:** 2026-09-28
**근거 자료:** `docs/thesis_notation_glossary.md`(용어·기호 사전),
`results/frozen_summaries/manifest.json`(2026-09-08 동결, source_commit_sha
`10892a1`), `docs/thesis_writing_schedule.md`, `paper/related_work_draft.md`

---

# 1. 논문 제목

## 국문

> **시장 국면을 보존한 전진 평가에서의 심층강화학습 기반 암호자산 포트폴리오 최적화:
> PPO 정책망 구조 비교와 전통 전략 대비 성과 검증**

## 영문

> **Deep Reinforcement Learning for Cryptocurrency Portfolio Optimization
> under Regime-Preserving Forward Evaluation: A Comparison of PPO Policy
> Network Architectures against Traditional Strategies**

### 제목 선정 근거

- 본 연구의 최종 결과는 "PPO가 전통 전략을 능가했다"가 **아니라**,
  두 표본외(OOS) 국면에서 **일관된 우위를 확인하지 못했다**는 것이다.
  따라서 제목에 성과 주장("우수한", "향상된")을 넣지 않고 **"성과 검증"**
  으로 서술했다.
- **"시장 국면을 보존한 전진 평가"**는 본 연구의 방법론적 차별점
  (regime-preserving expanding-window 2-fold forward evaluation)을 드러낸다.
  단일 기간 백테스트가 아니라는 점이 이 논문의 핵심 기여이기 때문이다.
- **"정책망 구조 비교"**는 MLP와 Transformer feature extractor를 동일 조건에서
  비교했다는 실험 설계를 반영한다.

### 대안 제목 (심사 의견에 따라 선택 가능)

| 안 | 제목 | 성격 |
|---|---|---|
| A (권장) | 위 확정안 | 방법론 + 결과 중립 |
| B | 암호자산 포트폴리오 최적화를 위한 PPO 정책망 구조 비교: Transformer는 MLP를 능가하는가 | 질문형, 서브가설 2 중심 |
| C | 심층강화학습 기반 암호자산 포트폴리오 전략의 국면 간 일반화 검증: 정적 배분 수렴 현상을 중심으로 | 행동 진단(4.5) 중심 |

> ⚠️ 제목은 용어사전 §11의 **미확정 4건 중 하나**다. 위 A안을 기준으로 제출하되,
> 지도교수 의견에 따라 확정한다.

---

# 2. 목차 및 개요

## 2.1 연구 개요 (Abstract 초안)

본 연구는 암호자산 8종(ADA, AVAX, BNB, BTC, DOGE, ETH, SOL, XRP)의 1시간봉
데이터를 대상으로, Proximal Policy Optimization(PPO) 기반 포트폴리오 배분 정책이
전통적 포트폴리오 전략 대비 위험조정 성과 우위를 갖는지, 그리고 정책망의
feature extractor 구조(MLP vs Transformer)가 성과 차이를 만드는지를 검증한다.

평가는 시장 국면의 시간 순서를 보존하는 **국면 보존 확장 윈도우 2-fold 전진
평가**로 수행하며, 하이퍼파라미터는 표본외(OOS) 구간을 확인하기 **전에** 동결한다.
최종 평가는 시드 5개로 반복하고, 거래비용 0.1%를 차감한 순수익률 기준으로
모든 모델과 벤치마크를 동일한 평가 모듈로 채점한다. 성과 차이는 stationary
block bootstrap으로 신뢰구간을 구하고 Holm-Bonferroni로 다중비교를 보정한다.

분석 결과, 두 표본외 국면에서 PPO는 전통 벤치마크 대비 일관된 우위를 보이지
않았고, MLP와 Transformer의 차이도 시드 변동성 범위 안에 있었다. 사후 행동
진단 결과 두 정책 모두 관측 변화에 반응하지 않는 고정된 목표 비중을 학습한
것으로 나타났다. 본 연구의 기여는 성과 향상이 아니라, **다중 시드·다중비교
보정·결과 동결을 갖춘 엄밀한 평가 설계 하에서 부정적 결과를 투명하게 보고**하고
그 원인을 정책 행동 수준에서 진단한 데 있다.

**주제어:** 강화학습, PPO, 포트폴리오 최적화, 암호자산, Transformer,
전진 평가, 시장 국면

---

## 2.2 목차

```
1. 서론
   1.1 연구 배경 및 필요성
   1.2 연구 목적 및 연구 질문
   1.3 논문의 구성

2. 이론적 배경 및 선행연구
   2.1 포트폴리오 최적화와 강화학습
   2.2 PPO 기반 연속 비중 결정
   2.3 Transformer를 이용한 시계열 표현
   2.4 암호자산 백테스트와 시간 순서 기반 검증
   2.5 본 연구의 위치

3. 연구방법
   3.1 데이터 및 전처리                              [은아, 5쪽]
       3.1.1 데이터 수집 및 자산 구성
       3.1.2 정합성 점검 및 데이터 동결
       3.1.3 시장 국면 정의 및 구간 배정
       3.1.4 특징 생성 및 train-only 표준화
   3.2 강화학습 환경 설계                            [준영, 3쪽]
       3.2.1 상태·행동·보상 정의
       3.2.2 거래비용 및 turnover 처리
   3.3 정책 네트워크                                 [준영, 3쪽]
       3.3.1 PPO 학습 구조
       3.3.2 MLP / Transformer feature extractor
   3.4 전통 벤치마크 모형                            [은아, 3쪽]
   3.5 Walk-Forward 평가 설계                        [준영, 4쪽]
   3.6 평가 지표 및 통계적 유의성 검정               [준영, 3쪽]
       3.6.1 성과 평가 지표
       3.6.2 Stationary block bootstrap 기반 Sharpe 차이 검정
       3.6.3 다중비교 보정(Holm-Bonferroni)
       3.6.4 정책망 간 성과 차이의 검정              ← 신설
       3.6.5 정책 행동 진단 방법                     ← 신설 확정(아래 주 참조)
   3.7 실험 재현성 확보 절차                         [준영, 2쪽]

4. 연구결과                                          [준영, 10쪽]
   4.1 하이퍼파라미터 탐색 결과
   4.2 국면별 표본외(OOS) 성과
   4.3 PPO와 전통 벤치마크 비교 및 통계적 유의성
   4.4 정책망 구조 비교 (MLP vs Transformer)
   4.5 정책 행동 진단
   4.6 결과 종합

5. 논의 및 결론                                      [준영, 6쪽]
   5.1 결과 요약
   5.2 연구 질문별 논의
       5.2.1 PPO는 전통 전략을 능가했는가
       5.2.2 정책망 구조 차이는 성과 차이를 만드는가
       5.2.3 왜 정적 배분으로 수렴했는가
   5.3 방법론적 시사점
   5.4 연구의 한계
   5.5 향후 연구 과제

참고문헌
부록 A. 실험 환경 및 패키지 버전                     [은아, 1쪽]
부록 B. 하이퍼파라미터 탐색 전체 표                  [은아, 2쪽]
부록 C. 재현 절차 및 결과 동결 manifest
부록 D. 정책별 상세 그림                             [유지, 6~8그림]
표 목차 / 그림 목차                                  [유지, 2쪽]
```

> **주 (3.6.5):** 4.5 정책 행동 진단의 측정 방법(분석 대상을 `weights`가 아닌
> `target_weights`로 삼은 이유 포함)을 서술할 자리를 3장에 확보하기 위해
> 신설한 절이다(`docs/proposal_3_6_4.md`). 이 절이 없으면 4.5가
> 사후 변명으로 읽힐 위험이 있고 3장-4장 대칭도 깨진다. 추가 실험 비용은
> 없으며, 분량은 1~1.5쪽이다. **2026-10-04 회의에서 신설 확정됨.**

**예상 분량:** 본문 약 42쪽 + 부록 약 6쪽 (표 8~10개, 그림 16~22개)

---

## 2.3 연구 질문 및 가설

| 구분 | 내용 | 검증 절 | 결과 |
|---|---|---|---|
| 연구질문 1 | PPO 기반 동적 배분이 전통 포트폴리오 전략보다 우수한 위험조정 성과를 내는가 | 4.3 | 일관된 우위 없음. 16개 비교 전부 보정 후 비유의 |
| 연구질문 2 | Transformer feature extractor가 MLP보다 우수한가 | 4.4 | 차이의 95% CI가 두 국면 모두 0을 포함 |
| 연구질문 3 | 학습된 정책은 서로 다른 시장 국면에 일반화되는가 | 4.2, 4.5 | 상승 국면 양호·조정 국면 손실. 상태 의존적 배분 미학습 |

> 🚩 본 연구의 가설은 "이긴다"가 아니라 **"이기는지 검증한다"**로 설정되어 있다
> (`docs/project_milestone_plan.md` 운영 원칙). 따라서 부정적 결과도 유효한
> 연구 결과로 보고한다.

---

# 3. 연구방법

## 3.1 연구 설계 요약

| 항목 | 내용 |
|---|---|
| **데이터** | Binance 현물 USDT 마켓, 암호자산 8종(ADA, AVAX, BNB, BTC, DOGE, ETH, SOL, XRP) **1시간봉**, 2021-01-01 ~ 2025-12-31. 피처 15개(OHLCV 5 + 기술지표 10: RSI, MACD 3종, ATR, OBV, SMA 20/50, EMA 20/50) |
| **배분 대상** | 자산 8종 + 현금 1 = **9개** |
| **강화학습 환경** | FinRL `PortfolioOptimizationEnv` 벤더링. 상태 **(3, 8, 50)** 텐서(가격 피처 3 × 자산 8 × lookback 50시간), 행동 **(9,)**, 보상 = 로그 수익률 `log(V_t/V_{t-1})`, 거래비용 **편도 0.1%** |
| **학습 알고리즘** | **PPO**(Stable-Baselines3). feature extractor만 **MLP / Transformer**로 교체하고 피처·lookback·학습량·평가를 동일하게 통제 |
| **비교 기준** | 전통 벤치마크 4종 — Buy & Hold BTC, Equal-Weight, Minimum-Variance(Ledoit-Wolf shrinkage, rolling 720h), Risk Parity(ERC) |
| **평가 설계** | **국면 보존 확장 윈도우 2-fold 전진 평가**(regime-preserving expanding-window forward evaluation). k-fold 교차검증이 아님 |
| **성과 지표** | Sharpe, Sortino, Calmar, MDD, Volatility, CAGR, Turnover — **모든 모델·벤치마크를 동일 평가 모듈**로, **거래비용 차감 후(net)** 기준으로 산출. 연율화 상수 8,760(=24×365) |
| **통계 검정** | **Stationary block bootstrap**(블록 24h, 2,000회)으로 Sharpe 차이의 95% 신뢰구간 산출 + **Holm-Bonferroni** 다중비교 보정(비교 16개) |
| **반복 실험** | 최종 표본외 평가를 **시드 5개**로 반복하여 mean ± std 보고 |

## 3.2 시장 국면 구분 및 전진 평가 구조

국면 단위로 통째 배정한다(시계열 특성상 무작위 셔플 불가).

| 국면 | 기간 | 용도 |
|---|---|---|
| 상승(2021) · 하락(2022) | 2021-01-01 ~ 2023-01-01 | 학습 |
| 횡보(2023) | 2023-01-02 ~ 2023-10-15 | 검증 (Fold 1) |
| **상승(2024)** | 2023-10-16 ~ 2025-01-01 | **표본외 평가 (Fold 1)** |
| **조정(2025)** | 2025-01-02 ~ 2025-12-31 | **표본외 평가 (Fold 2)** |

- Fold 1에서 검증 구간으로 하이퍼파라미터를 선택하고, **표본외 구간을 확인하기
  전에** 설정을 동결한다. Fold 2는 학습 구간만 확장하고 동일 설정을 그대로 쓴다.
- 데이터 누수 방지: 표준화 통계량을 **학습 구간에서만 추정**하고 검증·평가
  구간에 적용한다. 표본외 결과를 본 뒤 후보·하이퍼파라미터를 재조정하지 않는다.
- 두 표본외 국면은 **단순 평균하지 않고** Fold별로 분리해 보고한다.

## 3.3 결과 동결 및 재현성

2026-09-08 기준으로 모든 수치를 동결했다(source_commit_sha `10892a1`).
원시 산출물 61개 파일의 SHA-256 체크섬과 패키지 버전 전체를
`results/frozen_summaries/manifest.json`에 기록했으며, 논문 집필 단계에서는
이 동결 수치만 사용한다.

---

# 4. 참고서

## 4.1 주요 참고문헌

**선행연구 (3편)**

1. Jiang, Z., & Liang, J. (2017). *Cryptocurrency Portfolio Management with Deep
   Reinforcement Learning*. arXiv:1612.01277.
2. Jiang, Z., Xu, D., & Liang, J. (2017). *A Deep Reinforcement Learning Framework
   for the Financial Portfolio Management Problem*. arXiv:1706.10059.
3. Gort, B. J. D., Liu, X.-Y., Sun, X., Gao, J., Chen, S., & Wang, C. D. (2022).
   *Deep Reinforcement Learning for Cryptocurrency Trading: Practical Approach to
   Address Backtest Overfitting*. arXiv:2209.05559.

**방법론 근거**

4. Markowitz, H. (1952). *Portfolio Selection*. The Journal of Finance, 7(1), 77–91.
5. Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*.
   arXiv:1707.06347.
6. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS 30.
7. Lim, B., Arik, S. O., Loeff, N., & Pfister, T. (2021). *Temporal Fusion
   Transformers for Interpretable Multi-horizon Time Series Forecasting*.
   International Journal of Forecasting, 37(4), 1748–1764.
8. Liu, X.-Y., et al. (2021). *FinRL: Deep Reinforcement Learning Framework to
   Automate Trading in Quantitative Finance*. arXiv:2111.09395.
9. Politis, D. N., & Romano, J. P. (1994). *The Stationary Bootstrap*. JASA,
   89(428), 1303–1313.
10. Holm, S. (1979). *A Simple Sequentially Rejective Multiple Test Procedure*.
    Scandinavian Journal of Statistics, 6(2), 65–70.
11. Ledoit, O., & Wolf, M. (2004). *A Well-Conditioned Estimator for
    Large-Dimensional Covariance Matrices*. Journal of Multivariate Analysis,
    88(2), 365–411.
12. Maillard, S., Roncalli, T., & Teïletche, J. (2010). *The Properties of Equally
    Weighted Risk Contribution Portfolios*. The Journal of Portfolio Management,
    36(4), 60–70.
13. Henderson, P., et al. (2018). *Deep Reinforcement Learning That Matters*. AAAI 2018.
14. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
    (2nd ed.). MIT Press.

> 1~8번은 관련연구 초안에서 확정된 목록이며, 9~14번은 평가·검정 절차
> (block bootstrap, 다중비교 보정, 공분산 추정, 다중 시드 보고)의 근거로
> 추가한 것이다. 서식은 학과 지정 양식에 맞춰 최종 통일한다.

## 4.2 주요 도구

Python 3.13 · PyTorch · Stable-Baselines3(PPO) · Gymnasium · FinRL
`PortfolioOptimizationEnv` · pandas-ta(기술지표) · empyrical-reloaded(성과 지표) ·
arch(block bootstrap) · PyPortfolioOpt(Minimum-Variance) · Weights & Biases(학습 로그)

---

> **상세본:** 데이터 정합성 수치, 환경 파라미터, 후보 선택·시드 프로토콜,
> 동결 성과 매트릭스 등 전체 내용은 `docs/thesis_submission_form_full.md` 참조.
