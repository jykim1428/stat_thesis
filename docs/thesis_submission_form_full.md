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
데이터를 대상으로, Proximal Policy Optimization(PPO) 기반 포트폴리오 배분
정책이 전통적 포트폴리오 전략 대비 위험조정 성과 우위를 갖는지, 그리고
정책망의 feature extractor 구조(MLP vs Transformer)가 성과 차이를 만드는지를
검증한다.

평가는 시장 국면의 시간 순서를 보존하는 **국면 보존 확장 윈도우 2-fold 전진
평가(regime-preserving expanding-window 2-fold forward evaluation)**로 수행했으며,
하이퍼파라미터 후보는 표본외(OOS) 구간을 확인하기 **전에** 동결했다. 최종 OOS
평가는 5개 시드로 반복하고, 거래비용 0.1%를 차감한 순수익률(net) 기준으로
Sharpe·Sortino·Calmar·MDD·CAGR·turnover를 **모든 모델과 벤치마크에 동일한 평가
모듈**로 산출했다. PPO와 벤치마크의 Sharpe 차이는 stationary block bootstrap
(블록 24시간, 2,000회 복원추출)으로 95% 신뢰구간을 구하고 Holm-Bonferroni로
다중비교를 보정했다.

분석 결과, 두 OOS 국면에서 PPO는 전통 벤치마크 대비 **일관된 우위를 보이지
않았고**, 16개 비교 전부 보정 후 유의하지 않았다(최소 p=0.118). MLP와
Transformer 간 Sharpe 차이의 95% 신뢰구간도 두 국면 모두 0을 포함했다.
사후 행동 진단 결과 두 정책 모두 관측 변화에 실질적으로 반응하지 않는
**고정된 목표 비중**을 학습했으며(스텝 간 평균 변화량 4.3×10⁻⁹~8.2×10⁻⁹),
그 수익 경로는 Equal-Weight 전략과 상관계수 0.999 이상으로 거의 동일했다.

본 연구의 기여는 성과 향상이 아니라, **다중 시드·다중 비교 보정·결과 동결을
갖춘 엄밀한 평가 설계 하에서 부정적 결과를 투명하게 보고**하고, 그 원인을
정책 행동 수준에서 진단한 데 있다.

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
       3.6.4 정책 행동 진단 방법                     ← 신설 확정(아래 주 참조)
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

> **주 (3.6.4):** 4.5 정책 행동 진단의 측정 방법(분석 대상을 `weights`가 아닌
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

## 3.1 데이터

| 항목 | 내용 |
|---|---|
| 출처 | Binance 현물, USDT 마켓 |
| 자산 | 8종 + 현금 1 = **9개 배분 대상**<br>ADA, AVAX, BNB, BTC, DOGE, ETH, SOL, XRP (알파벳순) |
| 주기 | **1시간봉** |
| 전체 기간 | 2021-01-01 ~ 2026-07-31 (분석 사용 구간 2021-01-01 ~ 2025-12-31) |
| 자산별 행 수 | 48,897 (기대 48,911, 결측 14) |
| 최종 피처테이블 | (349,896, 17) |
| 피처 | 원본 OHLCV 5 + 기술지표 10 = **15개**<br>RSI_14, MACD(12,26,9) 3종, ATR_14, OBV, SMA_20/50, EMA_20/50 |
| 결측 처리 | 종목별 warm-up 50봉 선행 절단 후 `dropna` |
| 표준화 | **train-only 표준화** — 평균·표준편차를 학습 구간에서만 추정해 검증·평가 구간에 적용 (데이터 누수 방지) |
| 제외 | 거시지표(DXY, VIX)는 본 연구 범위에서 제외 |

### 시장 국면 정의

국면 단위로 통째 배정한다(시계열 특성상 무작위 셔플 불가).

| 국면 코드 | 표기 | 기간 | 용도 |
|---|---|---|---|
| `bull_2021` | 상승 국면(2021) | 2021-01-01 ~ 2021-11-09 | 학습 |
| `bear_2022` | 하락 국면(2022) | 2021-11-10 ~ 2023-01-01 | 학습 |
| `side_2023` | 횡보 국면(2023) | 2023-01-02 ~ 2023-10-15 | 검증(Fold 1) |
| `bull_2024` | 상승 국면(2024) | 2023-10-16 ~ 2025-01-01 | **OOS (Fold 1)** |
| `choppy_2025` | 조정 국면(2025) | 2025-01-02 ~ 2025-12-31 | **OOS (Fold 2)** |

**데이터 동결:** 2026-07 `crypto_ppo_feature_dataset v1.0` 동결 선언 이후
`features.parquet`, `regime_definition.csv`는 변경하지 않았다.

---

## 3.2 강화학습 환경

FinRL의 `PortfolioOptimizationEnv`를 벤더링해 본 연구 데이터에 연결했다.

| 요소 | 정의 |
|---|---|
| 상태(State) | **(3, 8, 50)** 텐서 = (가격 피처 3, 자산 8, lookback 50시간), `TIME_WINDOW = 50` |
| 행동(Action) | shape **(9,)** = 자산 8 + 현금 1, `Box(low=-10.0, high=10.0)`<br>(softmax 정규화 **전** raw action이므로 [0,1] 제약을 두지 않음) |
| 목표 비중 | `target_weights` = raw action의 softmax 정규화 값, **가격 변동 반영 전** 리밸런싱 목표 |
| 사후 비중 | `weights` = 가격 변동이 반영된 기말 실제 비중 |
| 보상(Reward) | 로그 수익률 `r_t = log(V_t / V_{t-1})`, `reward_scaling = 1` |
| 거래비용 | `cost_rate = 0.001` (편도 0.1%) |
| Turnover | `turnover_t = Σ|target_weights_t − weights_{t−1}| / 2`, `t=0`은 0 |
| 순수익률 | `net_return_t = gross_return_t − turnover_t × cost_rate` |

> turnover를 `Σ|weights_t − weights_{t−1}|/2`로 정의하면 가격 변동에 의한 비중
> 변화까지 거래량으로 계상되어 거래비용이 약 1.2~1.45배 과대계상된다. 본 연구는
> 정책이 실제로 지시한 리밸런싱량만 계상하는 위 정의를 사용한다.

---

## 3.3 정책 네트워크 및 학습

- **알고리즘:** PPO (Stable-Baselines3 2.9.0)
- **학습량:** `total_timesteps = 51,200` (모든 최종 학습 동일)
- **비교 대상 2종** — 피처·lookback·평가·학습량을 **동일하게** 통제하고
  feature extractor만 교체

| 정책망 | 구조 | 확정 후보 | 1회 학습 시간 |
|---|---|---|---|
| **MLP** | SB3 기본 구조 | `candidate03` (learning_rate = 1e-3) | 약 320~410초 |
| **Transformer** | `BaseFeaturesExtractor` 상속, 관측을 시간축 시퀀스로 변환 후 self-attention (초기 2 layer / 4 head / d_model=32) | `candidate04` (n_layers = 1) | 약 640~1,310초 |

> LSTM 정책망은 팀 합의로 연구 범위에서 제외했다(5.4 한계에 명시).

---

## 3.4 전통 벤치마크 모형 (4종)

| 모형 | 구성 |
|---|---|
| **Buy & Hold BTC** | BTC 100% 매수 후 보유. 리밸런싱 없음 → turnover 0 |
| **Equal-Weight** | 자산 8종 각 1/8, 현금 비중 0 |
| **Minimum-Variance** | rolling 720시간(30일), Ledoit-Wolf shrinkage 공분산, 개별 자산 상한 30%, μ 추정 없이 분산만 최소화 (PyPortfolioOpt 1.6.0) |
| **Risk Parity (ERC)** | 자산별 위험 기여도 균등화. Minimum-Variance와 동일한 window·shrinkage·weight_bounds, `scipy.optimize` SLSQP |

- 네 벤치마크 모두 PPO와 **동일한 OOS 날짜 범위**로 재실행한 값을 사용한다.
- 거래비용 0.1%를 PPO와 **동일한 공용 평가 모듈**로 적용한다.

---

## 3.5 Walk-Forward 평가 설계

**정식 명칭:** regime-preserving expanding-window 2-fold forward evaluation
(국면 보존 확장 윈도우 2-fold 전진 평가)

> ⚠️ 일반적인 k-fold cross-validation이 **아니다**. 시간 순서를 보존하므로
> "교차검증"으로 서술하지 않는다.

| | **Fold 1** | **Fold 2** |
|---|---|---|
| 하이퍼파라미터 선택 | train: `bull_2021`+`bear_2022`<br>val: `side_2023` | Fold 1 확정 설정 그대로 사용 |
| 최종 학습 구간 | 2021-01-01 ~ 2023-10-15 | 2021-01-01 ~ 2025-01-01 |
| OOS 평가 구간 | `bull_2024` (2023-10-16 ~ 2025-01-01) | `choppy_2025` (2025-01-02 ~ 2025-12-31) |
| OOS 스텝 수 | **10,656** | **8,713** |

- **워밍업:** 각 평가 구간 시작 직전 50시간을 관측 구성에만 사용하고,
  거래·보상·성과 기록은 OOS 시작 시점부터 한다.
- **Fold 2의 누수 통제:** `bull_2024`를 학습에 포함하는 것은 ① `choppy_2025`가
  시간상 이후이고 ② Fold 1의 OOS **성과**를 Fold 2 설정 결정에 사용하지
  않았으므로 누수가 아니다.
- **두 국면을 단순 평균하지 않는다** — Fold별로 분리해 보고한다.

### 후보 선택 및 시드 프로토콜

- **선택 기준:** 고정된 3개 시드(42/43/44)의 net validation Sharpe 산술평균
  최대. 동률 시 (1) 최저 시드 Sharpe → (2) 평균 Calmar → (3) 더 작은 구조 순.
  - ⚠️ 이 기준은 **validation 결과 검토 후 확정하고 OOS 확인 전 동결한** 것이다.
    "사전 정의된 기준"으로 서술하지 않는다.
- **최종 OOS 평가 시드:** 42, 43, 44, 45, 46 (5개)
  - ⚠️ 최초 프로토콜은 3-seed(42~44)였고, "최소 5시드" 원칙에 맞춰 **OOS 확인 후**
    후보·하이퍼파라미터 변경 없이 시드 45·46을 두 정책·두 fold에 동일하게
    추가한 **post-hoc robustness extension**이다. 원 3-seed 결과는 별도 보존되어
    있으며 5-seed 확장 후에도 결론은 동일했다. **각주로 반드시 공시한다.**

---

## 3.6 평가 지표 및 통계적 검정

### 성과 지표

- **계산 모듈:** `src/cryptoagent/envs/evaluate.py` —
  **모든 모델·벤치마크가 동일 함수로 평가**된다(공정 비교의 전제).
- **라이브러리:** empyrical-reloaded 0.5.12
- **연율화 상수:** `periods_per_year = 8760` (= 24 × 365, 1시간봉이므로 252 아님)
- **무위험 수익률:** `risk_free = 0`
- **지표 8종:** Sharpe, Sortino, Calmar, MDD, Volatility, CAGR,
  Average Turnover, Total Transaction Cost
- **모든 지표는 순수익률(net, 거래비용 0.1% 차감 후) 기준으로 보고**한다.

### 통계적 유의성 검정

| 항목 | 설정 |
|---|---|
| 방법 | **stationary block bootstrap** (`arch.bootstrap.StationaryBootstrap`) |
| 블록 크기 | 주 분석 **24시간**, 민감도 확인 **168시간** |
| 복원추출 | **2,000회**, 신뢰수준 95% |
| 통계량 | `mean_seed[ Sharpe(PPO_seed) − Sharpe(benchmark) ]` |
| 풀링 | PPO 5-seed + 벤치마크 1개 = **6개 시계열을 동일한 블록 인덱스로 공동 리샘플** |
| 다중비교 보정 | **Holm-Bonferroni, 비교 수 16** (정책 2 × 벤치마크 4 × fold 2) |
| 재현성 | SHA-256 기반 결정론적 rng seed 파생 |

> 🚩 **해석 원칙:** 비유의는 동등성의 증거가 아니다. 결과는
> "Sharpe 차이가 **0과 다르다는 통계적 증거를 얻지 못했다**"로 서술하며,
> "통계적으로 동등하다"는 표현은 쓰지 않는다(사전 정의된 실질적 동등성 범위에
> 대한 equivalence test를 수행하지 않았기 때문).

### 정책 행동 진단 방법 (3.6.4)

재학습·재탐색 없이 **저장된 backtest 결과만** 분석한다. 분석 대상은
`target_weights`(가격 변동 반영 전 목표 비중)이며, `weights`를 쓰면 정책이
가만히 있어도 가격 변동 때문에 비중이 흔들려 정책의 의도를 관측할 수 없다.

| 지표 | 측정 내용 |
|---|---|
| 시간 불변성 | 목표 비중의 스텝 간 변화량 (평균·최대) |
| 집중도 | HHI — 현금 포함 9원소 기준(1/9)과 8자산 재정규화 기준(1/8) 병기 |
| 시드 간 이질성 | 시드별 시간평균 목표 배분의 분산 |
| 벤치마크 유사성 | Equal-Weight 대비 순수익률 상관계수 및 tracking error |
| 방어 행동 | 15% 이상 낙폭 구간과 평시의 목표 현금 비중 비교 |

---

## 3.7 재현성 확보 절차

- **결과 동결(freeze):** 2026-09-08, source_commit_sha `10892a1`
  (manifest.json 자체는 다음 커밋 `f55b2d3`에 저장), 원시 산출물 **61개 파일의
  SHA-256** 및 패키지 버전 전체를 `results/frozen_summaries/manifest.json`에 기록.
- **OOS 확인 후 후보·하이퍼파라미터 재조정 금지.** Optuna 자동탐색은
  OOS 결과를 이미 본 상태에서 수행하면 탐색 자체가 오염되므로 본 연구 범위에서
  제외하고 향후 연구로 이관했다.
- **독립 리뷰를 통한 교차 검증:** 통계량 정의, gross/net 일관성,
  bootstrap 결정성 등을 반복 검증했다.
- **공용 출력 스펙:** 모든 전략이 동일한 backtest 출력 형식과 평가 모듈을 사용한다.

---

## 3.8 실험 환경

| 항목 | 버전 | 항목 | 버전 |
|---|---|---|---|
| Python | 3.13.7 | empyrical-reloaded | 0.5.12 |
| PyTorch | 2.13.0 | arch | 8.0.0 |
| Stable-Baselines3 | 2.9.0 | scipy | 1.18.0 |
| Gymnasium | 1.3.0 | PyPortfolioOpt | 1.6.0 |
| pandas | 3.0.5 | quantstats | 0.0.81 (참고용, 본문 수치 근거 아님) |
| numpy | 2.2.6 | 학습 환경 | Google Colab |

---

## 3.9 주요 결과 요약 (동결 수치)

> 모든 값은 2026-09-08 동결된 수치이며, 거래비용 0.1% 차감 후(net) 기준이다.

**PPO OOS 성과 (5-seed 평균 ± 표준편차)**

| 정책망 | 상승 국면(`bull_2024`) Sharpe | 조정 국면(`choppy_2025`) Sharpe |
|---|---|---|
| MLP | **2.203 ± 0.017** | **−0.194 ± 0.029** |
| Transformer | **2.206 ± 0.018** | **−0.222 ± 0.031** |

**전통 벤치마크 OOS 성과**

| 전략 | 상승 국면 Sharpe | 조정 국면 Sharpe | 조정 국면 CAGR |
|---|---|---|---|
| Buy & Hold BTC | 2.272 | 0.066 | −6.7% |
| Equal-Weight | 2.218 | −0.213 | −32.7% |
| Minimum-Variance | 1.818 | 0.171 | −5.0% |
| Risk Parity | 2.141 | −0.172 | −26.8% |

- 조정 국면에서는 **6개 전략 모두 손실**이며, Minimum-Variance가 손실을 가장 잘
  방어했다. Sharpe가 양수여도 CAGR이 음수일 수 있다(산술평균 대 기하평균의 차이).
- **정책망 간 Sharpe 차이(Transformer − MLP) 95% CI:**
  상승 국면 **[−0.031, +0.037]**, 조정 국면 **[−0.069, +0.013]** — 두 구간 모두 0 포함.
- **통계 검정:** 16개 비교 전부 Holm 보정 후 비유의, **최소 p = 0.118**
  (Transformer vs Minimum-Variance, 조정 국면).
- **행동 진단:** 목표 비중의 스텝 간 평균 변화량 **4.3×10⁻⁹ ~ 8.2×10⁻⁹**
  (float32 연산 정밀도 수준) → 사실상 불변. Equal-Weight 대비 순수익률 상관계수
  **0.999 이상**. 드로다운 구간과 평시의 목표 현금 비중이 소수점 6자리까지 동일
  → 방어적 현금 이동을 학습하지 못함.
  - 단, 각 시드는 **서로 다른 정적(static) 포트폴리오**를 학습했다.
    "모든 시드가 동일한 배분을 학습했다"는 서술은 부정확하다.

---

# 4. 참고서 (참고문헌)

## 4.1 핵심 참고문헌

1. Markowitz, H. (1952). *Portfolio Selection*. **The Journal of Finance**, 7(1),
   77–91. https://doi.org/10.2307/2975974
   → 평균-분산 최적화의 원형. 3.4 Minimum-Variance 벤치마크의 이론적 배경.

2. Jiang, Z., & Liang, J. (2017). *Cryptocurrency Portfolio Management with Deep
   Reinforcement Learning*. arXiv:1612.01277. https://arxiv.org/abs/1612.01277
   → 암호자산 포트폴리오에 심층강화학습을 적용한 초기 연구. 선행연구 3편 중 1편.

3. Jiang, Z., Xu, D., & Liang, J. (2017). *A Deep Reinforcement Learning Framework
   for the Financial Portfolio Management Problem*. arXiv:1706.10059.
   https://arxiv.org/abs/1706.10059
   → CNN/RNN/LSTM 정책을 포함하는 model-free 프레임워크. 선행연구 3편 중 2편.

4. Liu, X.-Y., Yang, H., Gao, J., Wang, C. D., et al. (2021). *FinRL: Deep
   Reinforcement Learning Framework to Automate Trading in Quantitative Finance*.
   arXiv:2111.09395. https://arxiv.org/abs/2111.09395
   → 본 연구가 `PortfolioOptimizationEnv`를 벤더링해 사용한 프레임워크. 3.2의 직접 근거.

5. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
   *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
   https://arxiv.org/abs/1707.06347
   → PPO 원논문. 2.2 및 3.3의 알고리즘 근거.

6. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
   Kaiser, L., & Polosukhin, I. (2017). *Attention Is All You Need*. **NeurIPS 30**.
   https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html
   → Transformer 원논문. 3.3.2 feature extractor 구조의 근거.

7. Lim, B., Arik, S. O., Loeff, N., & Pfister, T. (2021). *Temporal Fusion
   Transformers for Interpretable Multi-horizon Time Series Forecasting*.
   **International Journal of Forecasting**, 37(4), 1748–1764.
   https://arxiv.org/abs/1912.09363
   → Transformer 계열의 다변량 시계열 적용 사례. 2.3의 근거.

8. Gort, B. J. D., Liu, X.-Y., Sun, X., Gao, J., Chen, S., & Wang, C. D. (2022).
   *Deep Reinforcement Learning for Cryptocurrency Trading: Practical Approach to
   Address Backtest Overfitting*. arXiv:2209.05559. https://arxiv.org/abs/2209.05559
   → 암호자산 DRL의 백테스트 과적합 문제 제기. 선행연구 3편 중 3편이자
     3.5 전진 평가 설계의 직접적 동기.

## 4.2 방법론·통계 참고문헌 (3.6 검정 절차 근거)

9. Politis, D. N., & Romano, J. P. (1994). *The Stationary Bootstrap*. **Journal of
   the American Statistical Association**, 89(428), 1303–1313.
   → 3.6.2 stationary block bootstrap의 원방법론.

10. Holm, S. (1979). *A Simple Sequentially Rejective Multiple Test Procedure*.
    **Scandinavian Journal of Statistics**, 6(2), 65–70.
    → 3.6.3 다중비교 보정(16개 비교)의 근거.

11. Ledoit, O., & Wolf, M. (2004). *A Well-Conditioned Estimator for
    Large-Dimensional Covariance Matrices*. **Journal of Multivariate Analysis**,
    88(2), 365–411.
    → 3.4 Minimum-Variance·Risk Parity 공분산 추정(shrinkage)의 근거.

12. Maillard, S., Roncalli, T., & Teïletche, J. (2010). *The Properties of Equally
    Weighted Risk Contribution Portfolios*. **The Journal of Portfolio Management**,
    36(4), 60–70.
    → 3.4 Risk Parity(ERC)의 엄밀한 정의.

13. Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014).
    *The Probability of Backtest Overfitting*. **Journal of Computational Finance**.
    → 5.3 방법론적 시사점(다중 검정·백테스트 과적합)의 보강 근거.

14. Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D. (2018).
    *Deep Reinforcement Learning That Matters*. **AAAI 2018**.
    → 5.3 다중 시드 보고 필요성의 근거(단일 시드 결과의 위험성).

15. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
    (2nd ed.). MIT Press.
    → 2장 강화학습 일반 이론의 교과서적 근거.

> ⚠️ 9~15번은 3~5장 집필에 필요한 방법론 인용으로 **추가 제안한 항목**이다.
> 1~8번은 `paper/related_work_draft.md`에서 이미 확정된 목록이다.
> 서식 통일(학과 지정 양식)과 본문-목록 상호 누락 확인은 은아 담당(10/21).

## 4.3 소프트웨어·도구 참고

| 도구 | 용도 | 버전 |
|---|---|---|
| Stable-Baselines3 | PPO 구현 | 2.9.0 |
| FinRL `PortfolioOptimizationEnv` | 포트폴리오 환경 (벤더링) | — |
| Gymnasium | 강화학습 환경 인터페이스 | 1.3.0 |
| empyrical-reloaded | 성과 지표 계산 | 0.5.12 |
| arch | stationary block bootstrap | 8.0.0 |
| PyPortfolioOpt | Minimum-Variance 최적화 | 1.6.0 |
| pandas-ta | 기술지표 생성 | — |
| Weights & Biases | 학습 로그 추적 | — |

## 4.4 내부 근거 자료 (프로젝트 산출물)

| 문서 | 내용 |
|---|---|
| `docs/thesis_notation_glossary.md` | 3장 용어·기호 사전 (집필 단일 기준) |
| `docs/walk_forward_design.md` | 3.5 전진 평가 설계 원문 |
| `docs/results_section_draft.md` | 4~5장 결과 문장 및 재현 절차 |
| `docs/consistency_report.md` | 3.1.2 데이터 정합성 점검 |
| `docs/feature_list.md` | 3.1.4 피처 목록 |
| `results/frozen_summaries/manifest.json` | 동결 manifest (체크섬 61개) |
| `results/frozen_summaries/statistical_tests.json` | 통계 검정 결과 |
| `results/walk_forward_behavior_diagnostics/diagnostics.json` | 4.5 행동 진단 |
| `configs/walk_forward_locked_candidates.json` | 확정 후보 및 선택 기준 |

---

# 5. 제출 전 확인 필요 사항

| # | 항목 | 현황 |
|---|---|---|
| 1 | **논문 제목 확정** | 위 A안 제출, 지도교수 확인 필요 |
| 2 | ~~3.6.4 신설 여부~~ | **확정 (2026-10-04)** — A안 신설 |
| 3 | 4.2 turnover·거래비용 표 포함 여부 | 미정 (포함 시 4.5의 선행 증거) |
| 4 | LSTM 언급 범위 | 5.4 한계에 한 줄 포함으로 잠정 결정 |
| 5 | 참고문헌 서식 | 학과 지정 양식 확인 필요 |
| 6 | 4.5 시드 간 이질성 수치 | `diagnostics.json`에 5-seed 평균만 저장 → 그림 기반 서술로 대체 검토 |
