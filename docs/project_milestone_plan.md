# 11월까지 끝내는 플랜 (8월 말 중간 기점, 11월 논문까지)

**목표 데드라인:** 11월 말 논문 완성
**가용 기간:** 약 4.5개월 (2026-07 ~ 2026-11)
**운영 원칙:**
① 시드 1개 결과는 안 씀(멀티시드 mean±std)
② 데이터 누수 금지(scaler·튜닝 전부 train-only)
③ 월말 게이트 미달 시 범위 축소, 일정은 안 밀기
④ 11월 이후 확장 항목은 백로그로 격리

---

## 🎯 월별 마일스톤

| 월 | 테마 | 월말 게이트(통과 조건) |
| --- | --- | --- |
| 7월 | 데이터 확정 + 파이프라인 | 데이터 "동결" 선언 (더 안 건드림), 피처테이블 완성 |
| 8월 | 베이스라인 완주 | **PPO(MLP) vs 전통모델 비교표 1장** 완성 |
| 9월 | 트랜스포머 + LSTM 정책망 | 세 정책망 모두 학습 붕괴 없이 end-to-end 돌아감 |
| 10월 | 본 실험 + 통계 처리 | **모든 숫자 freeze** (모델×국면×멀티시드 매트릭스) |
| 11월 | 논문 + 발표 | 논문 초고 + 발표덱 완성 |

---

## 📅 7월 — 데이터 확정 + 파이프라인

> EDA는 거의 끝난 상태. 이번 달의 유일한 목표는 데이터를 두 번 다시 안 건드려도 되는 상태로 못 박는 것.

### 7월 1주차 — 데이터 정합성 점검

- [ ] 자산별 커버리지 확인: 8종목 각각 시작일·종료일·행 수 출력 (SOL·AVAX 늦은 상장 → 공통기간 얼마나 짧아지는지)
- [ ] 시간 갭 점검: 1h 아닌 간격이 몇 개인지, 어느 구간인지 (거래소 점검 등)
- [ ] 중복 (timestamp, symbol) 쌍 존재 여부 확인 → 있으면 pivot 깨짐
- [ ] `pivot().dropna()` 후 최종 shape 기록 → **실제 학습 가능 공통기간 확정**
- **산출물:** 정합성 리포트(커버리지 표 + 갭 목록)

### 7월 2주차 — Regime(시장국면) 날짜 확정

- [ ] 누적수익률 곡선 + 24h rolling vol 위에 국면 경계선 긋기
- [ ] 상승/하락/횡보 3구간 **구체 날짜로** 확정 (후보: 하락 2022, 횡보 2023 상반기, 상승 2023 하반기~2024 초 — 실제 확보기간에 맞게 조정)
- [ ] 각 국면을 train/val/test로 어떻게 나눌지 규칙 정의
- [ ] 왜도·첨도 자산별 계산 → 문서의 "heavy-tailedness" 근거로 남기기
- **산출물:** 국면 정의표(날짜 명시)

### 7월 3주차 — 피처 생성

- [ ] pandas-ta로 기술지표 계산: RSI, MACD, ATR, OBV, SMA, EMA
- [ ] (선택) 거시지표 보조: DXY, VIX 등 — 넣을지 말지 이번 주에 결론
- [ ] 결측치·NaN 처리 규칙 정하고 적용 (지표 초기 구간 NaN 등)
- [ ] 피처 스케일링은 **여기서 하지 않음** — train-only fit 원칙 때문에 학습 단계로 미룸 (지금은 원본 피처만 저장)
- **산출물:** 피처 목록 확정

### 7월 4주차 — SQLite 테이블 마무리 + 데이터 동결

- [ ] 최종 피처테이블을 SQLite에 저장 (재현성 확보)
- [ ] 데이터 딕셔너리 작성 (컬럼명·의미·단위)
- [ ] GitHub + (가능하면) DVC로 데이터 버전 고정
- [ ] **🚩 "데이터 동결" 선언** — 이후 데이터 변경 금지
- **담당:** 준영 · **산출물:** 최종 SQLite DB + 데이터 딕셔너리

> **✅ 7월 게이트:** 피처테이블 완성 + 국면 날짜 확정 + 데이터 동결 선언

---

## 📅 8월 — 베이스라인 완주

> 트랜스포머는 아직 손도 안 댐. **파이프라인이 숫자를 뱉게** 만드는 게 목표. 여기서 나오는 비교표가 트랜스포머 없이도 발표 가능한 최소 결과.

### 8월 1주차 — FinRL 환경 세팅 (의존성 지옥 처리)

- [ ] Python 3.10 고정, torch·SB3·gymnasium·FinRL 버전 `requirements.txt`에 핀
- [ ] Colab에서 재현 확인 (팀원 전원 같은 환경 재현되는지)
- [ ] FinRL_Crypto의 `PortfolioOptimizationEnv`에 우리 데이터 물리기
- [ ] env가 주는 observation shape 정확히 파악 (9월 트랜스포머 reshape의 전제)
- **산출물:** 재현 가능한 환경 + requirements.txt

### 8월 2주차 — Vanilla PPO(MLP) 학습·추론

- [ ] MLP PPO 한 사이클 end-to-end 학습 → 백테스트까지 완주
- [ ] 학습 로그를 W&B에 연결
- [ ] 결과가 말이 되는지 sanity check (수익률·비중 합=1 등)
- **산출물:** PPO(MLP) 첫 백테스트 결과

### 8월 3주차 — 공용 평가 모듈 (전 모델 재사용)

- [ ] empyrical로 Sharpe·Sortino·Calmar·MDD·Volatility·CAGR 계산 함수화
- [ ] 거래비용 0.1% 반영 + Turnover 계산
- [ ] **모든 모델·벤치마크가 동일 함수·동일 수수료·동일 기간**으로 평가되게 인터페이스 통일
- **산출물:** `evaluate()` 공용 모듈

### 8월 4주차 — 전통 벤치마크 4종

- [ ] Buy & Hold BTC
- [ ] Equal-Weight (1/N)
- [ ] Markowitz Mean-Variance (PyPortfolioOpt) — **추정은 train 구간/rolling만** (full-sample 금지)
- [ ] Risk Parity
- [ ] 전부 공용 평가 모듈로 채점 → **PPO vs 전통모델 비교표** 작성
- **담당:** 은아(벤치마크) + 준영(PPO) · **산출물:** 비교표 1장
- **병행:** 유지 — 선행연구 3편 심화 정독, 방법론 초안 작성 시작

> **✅ 8월 게이트:** PPO(MLP) vs 전통모델 비교표 완성 = 서브가설 1의 1차 답

---

## 📅 9월 — 트랜스포머 + LSTM 정책망 (기술 코어)

> 제일 잘 터지는 구간. **9월 한 달을 통째로 여기 쓴다고 각오.** 학습 붕괴·NaN 디버깅에 2~3주 날아가는 게 정상.

### 9월 1주차 — 커스텀 트랜스포머 정책망 구현

- [x] SB3 `BaseFeaturesExtractor` 상속해서 트랜스포머 인코더 작성
- [x] observation을 `(batch, lookback, feature_dim)`으로 reshape (env shape 기준)
- [x] lookback window 하이퍼파라미터로 노출 (예: 50~168h)
- [x] **작게 시작**: 2 layer / 2~4 head (파라미터 많으면 이 데이터 크기에서 오버피팅 직행)
- **산출물:** 돌아가는 트랜스포머 extractor

### 9월 2주차 — 트랜스포머 학습 안정화 (디버깅 주간)

- [x] `policy_kwargs`로 끼워서 PPO 학습 시도
- [x] 학습 붕괴/NaN/entropy collapse 디버깅 (gradient clip, lr 조정, 정규화)
- [x] 스케일링은 이 단계에서 **train-only fit** 적용
- [x] 오버피팅 조짐 모니터링 (train↑ val↓)
- **산출물:** 안정적으로 수렴하는 트랜스포머 PPO 1개
- ⚠️ **버퍼 지점:** 여기가 늘어지면 7월에 벌어둔 여유로 흡수. 3주차로 넘어가되 병렬 진행.

### 9월 3주차 — LSTM 정책망 (비교군)

- [x] 같은 방식으로 LSTM feature extractor 구현
- [x] 트랜스포머와 동일 조건(피처·lookback·평가)으로 세팅 → 공정 비교 담보
- **산출물:** LSTM 정책망 → 삼자비교(MLP/LSTM/Transformer) 성립

### 9월 4주차 — 실험 자동화 셋업

- [x] 모델 × 국면 × 시드 조합을 스크립트로 일괄 실행되게 오케스트레이션
- [ ] Optuna 탐색 파이프라인 준비 (lr·gamma·batch size, **val 기준만**) —
      **미구현, 의도적 범위 제외.** 4개 후보 수동 탐색(`MLP_CANDIDATES`/
      `TRANSFORMER_CANDIDATES`)으로 대체. 10월 계획에서 "OOS 확인 후
      Optuna 재개는 탐색 자체를 오염시킨다"는 이유로 이번 사이클에서는
      하지 않기로 확정(Future Work로 이동, 새 holdout 생길 때만 진행)
- [x] Walk-forward validation 세팅
- [x] Colab 컴퓨트 예산 점검 (10월 대량 실행 대비)
- **산출물:** "버튼 하나로 전체 실험" 스크립트
- **병행:** 논문 관련연구(Related Work) 섹션 초안 완성

> **✅ 9월 게이트:** MLP·LSTM·Transformer 세 정책망 모두 붕괴 없이 end-to-end 작동
>
> **실제 진행 메모 (2026-09):** LSTM은 팀 합의로 연구 범위 밖으로 확정(코덱스 검토 동의) — MLP/Transformer만 walk-forward 대상. 9월 4주차 오케스트레이터(`experiments/walk_forward_orchestrator.py`)로 Fold1 탐색(N=4 후보×S=3 seed, 24회) → 후보 확정(`configs/walk_forward_locked_candidates.json`) → Fold1/Fold2 최종 재학습+OOS 완료. 전통 벤치마크 4종도 동일 OOS 날짜로 재실행 비교 완료(`experiments/walk_forward_benchmarks.py`). 정책 행동 진단 스크립트(`experiments/walk_forward_behavior_diagnostics.py`)도 추가. **결과: PPO(MLP/Transformer) 모두 두 OOS 구간에서 전통 벤치마크 대비 일관된 우위를 입증하지 못함** — 행동 진단(비중 HHI, 현금-낙폭 상관계수)상 두 정책 모두 균등분산에 가깝게 수렴, 하락 국면 방어 행동 학습 안 됨. 이 결과를 뒤집으려는 재탐색은 하지 않기로 함(OOS 확인 후 재조정 금지 원칙).
>
> **5-seed 확장 (2026-09-07):** 운영 원칙의 "최소 5시드"에 맞춰 Fold1_final/Fold2_final의 OOS 재학습을 seed 3개(42~44)에서 5개(42~46)로 확장(`locked_candidates.json` 갱신, resume으로 기존 3개는 skip하고 45·46만 신규 실행). Fold1_search(후보 선택 근거)는 3-seed 그대로 유지 — 이미 확정한 candidate03(MLP)/candidate04(Transformer) 선택 자체는 바뀌지 않음. 5-seed 결과도 3-seed와 동일한 결론(Fold1 두 정책 동률, Fold2 둘 다 손실이고 MLP가 근소하게 덜 나쁨, 벤치마크 대비 일관된 우위 없음)을 유지.

---

## 📅 10월 — 통계 처리 + 시각화 + 결과 동결

> 새 코드는 최소화. **이미 만든 걸 통계적으로 정리하고 그림으로 남기는** 달.
>
> **원안 대비 순서 변경 (2026-09-07, 코덱스 3차 리뷰 반영):** 원안은
> "1주차 멀티시드 → 2주차 Optuna → 3주차 통계처리 → 4주차 시각화/동결"
> 순이었으나, 멀티시드 대량 실행(5-seed)은 9월 4주차 walk-forward
> validation 실행 중 이미 완료됐다 (`configs/walk_forward_locked_
> candidates.json`, `results/walk_forward/`). Optuna 자동탐색은 이번
> 달에 하지 않는다 - OOS(bull_2024/choppy_2025) 결과를 이미 확인한
> 상태에서 하이퍼파라미터 탐색을 재개하면, 목적함수가 val 기준이어도
> 탐색 범위·중단 시점 결정이 이미 본 OOS 결과의 영향을 받아 암묵적으로
> 오염된다 (코덱스 리뷰: "같은 OOS에서 새 모델을 다시 평가해 개선됐다고
> 주장하면 그 OOS는 더 이상 깨끗한 최종 검증 구간이 아니다"). Optuna·
> reward 변경·feature 추가·LSTM 재고는 Future Work로 이동 - 꼭 한다면
> 아직 보지 않은 새 holdout(예: 2026년 이후 구간)이 생겼을 때만 확증적
> 평가로 진행한다. 대신 이번 달은 통계 검정 → 시각화 → freeze → 인수인계
> 순으로 진행한다.

### 10월 1주차 — 통계 검정

- [x] PPO(seed 반복 있음) vs 전통 벤치마크(결정론적, 반복 없음) 비교의
      통계적 엄밀성 확보: 같은 시간 인덱스의 PPO-벤치마크 수익률 차이를
      moving/stationary block bootstrap으로 리샘플해 Sharpe 차이의 신뢰
      구간을 계산 (비교가 여러 개면 Holm 보정 병행). OOS 국면이 2개뿐이라
      "모든 시장 국면에 일반화된다"는 검정은 하지 않는다 - 이 두 국면
      한정 결론임을 명시.
- [x] MLP vs Transformer paired seed 차이 CI 최종 확정 (2026-09-07
      코덱스 3차 리뷰에서 이미 계산: Fold1 Transformer-MLP 0.0030
      [-0.0312, 0.0372], Fold2 -0.0278 [-0.0689, 0.0133] - 둘 다
      불확실성 안이라는 결론 유지되는지 재확인)
- [x] target_weights 행동 진단(정책이 사실상 고정된 목표비중을 출력,
      PPO-EW 수익률 상관 0.999+)을 논문 문장으로 정리
- **산출물:** 통계 검증된 성능 매트릭스 + 신뢰구간

### 10월 2주차 — 시각화

- [x] 누적수익률 곡선(PPO 2종 + 벤치마크 4종 겹쳐서, net 기준 - gross
      아님, 2026-09-07 리뷰에서 gross/net 분리 완료)
- [x] 자산별 목표 비중(target_weights) 변화 Area chart - 사실상 flat
      line이라는 사실 자체가 시각적 증거가 되므로 그대로 보여줄 것
- [x] Drawdown 곡선
- [x] 국면별 성능표 (Fold1/Fold2 별도 유지, 단순 평균 금지 -
      docs/walk_forward_design.md 원칙)
- [x] quantstats 리포트 자동 생성
- **산출물:** 논문용 그림·표 전체

### 10월 3주차 — 결과 동결 (freeze)

- [x] 결과 manifest 작성: commit SHA, cost_rate, 확정 후보/seed 설정,
      정확한 평가 시작·종료 시각, 원시 backtest CSV 체크섬(SHA-256)
      (2026-09-07 리뷰 권고 - Moderate: fresh clone에서 재현 가능하도록)
- [x] **🚩 모든 숫자 freeze** — 11월엔 이 숫자로만 씀
- **산출물:** `results/frozen_summaries/manifest.json` + 확정 성능 매트릭스
- **완료 (2026-09-08):** `experiments/walk_forward_freeze_manifest.py`로
  생성. source_commit_sha `10892a1`(숫자·코드가 확정된 커밋 — manifest.json
  자체는 이 커밋에는 없고 다음 커밋에 저장됨. "commit X 기준 freeze"를
  "X를 checkout하면 manifest도 있다"로 오독하지 않도록 source_commit_sha/
  manifest_commit_note로 구분해 기록함, 2026-09-08 리뷰 반영) 기준,
  PPO backtest 20개 + model.zip 20개 + 벤치마크 8개 + frozen_summaries
  12개 + 원천 DB 1개, 총 61개 SHA-256 기록. Python 3.13.7, torch 2.13.0,
  scipy 1.18.0(arch 설치로 1.16.3에서 자동 업그레이드, requirements.txt
  갱신) 등 패키지 버전 고정. cost_rate=0.001, periods_per_year=8760,
  Fold2 실제 데이터 종료(2025-12-31 00:00), 3→5 seed post-hoc 확장
  이력과 원본 3-seed 보존 위치, bootstrap 설정(block=24/168, reps=2000,
  6-시계열 공동 리샘플 pooling 정의, SHA-256 기반 결정론적 rng seed
  파생) 전부 포함. 이 시점 이후
  숫자·설정 재조정 금지 — 11월엔 이 manifest 기준으로만 논문 작성.

### 10월 4주차 — 인수인계 + 버퍼

- [ ] 은아·유지에게 넘길 방법론 요약 문서 작성 (freeze 후에 작성하기로
      합의됨 - 미리 만들면 3주차 freeze 전까지 계속 고쳐야 함): 최종
      수치, 각 수치가 나온 방법(fold 경계, seed 수와 근거, 벤치마크 날짜
      재조정 이유 등), 해석 시 주의사항(Sharpe/CAGR 계산방식 차이,
      gross/net 구분, 5-seed가 post-hoc extension이라는 점) 포함
- [ ] 밀린 것 흡수, 최종 제출본 확정
- **산출물:** 인수인계 문서 + 확정된 논문용 자료 일체

> **✅ 10월 게이트:** 통계 검증된 성능 매트릭스 + 시각화 + 숫자 동결 +
> 인수인계 문서 완성. **이후 "조금만 더 튜닝" 금지.**

---

## 📅 11월 — 논문 + 발표

> 새 실험 없음. 10월 말 숫자로 **쓰기만** 함.

### 11월 1주차 — 논문 본문 (선행연구·방법론)

- [ ] Abstract / Introduction
- [ ] Related Work (3논문 요약 + 우리 차별점: PPO+Transformer, 다국면 일반화 검증)
- [ ] Methodology (데이터·환경·정책망·평가체계)
- **산출물:** 논문 전반부 초고

### 11월 2주차 — 논문 본문 (실험·결과·해석)

- [ ] Experiments / Results (표·그림 삽입)
- [ ] Discussion (가설별 결론 — 이겼든 졌든 정직하게)
- [ ] Limitations & Future Work (여기에 11월 이후 확장 항목 명시)
- [ ] Conclusion
- **산출물:** 논문 초고 완성

### 11월 3주차 — 발표덱 + 리뷰

- [ ] 학술제 발표 자료 (역할분배: 제작/발표)
- [ ] 논문 내부 교차 리뷰·교정
- [ ] 그림·표 최종 다듬기
- **산출물:** 발표덱 + 교정된 논문

### 11월 4주차 — 버퍼 + 마무리

- [ ] 밀린 것 흡수, 최종 제출본 확정
- [ ] 11월 이후 디벨롭 백로그 정리

> **✅ 11월 게이트:** 논문 완성 + 발표덱 완성 → **프로젝트 1차 종료**

---

## 🔒 11월 이후 확장 백로그 (지금은 손대지 말 것)

> 아래는 논문이 끝난 *다음* 세계의 일. 4.5개월 필수경로에 넣는 순간 프로젝트가 안 끝남. 목록만 유지.

- 실시간 모의투자 1~2달 트래킹 (달력 시간을 강제 소모)
- 온체인 지표 (MVRV / NVM — 수집 난이도 높음)
- 두 번째 타임프레임 (5분봉·30분봉)
- Fear & Greed Index 보상함수 반영
- 현금 자산 비중 정교화 / 리스크 패리티 고도화
- 학회·저널 제출 및 리비전 사이클
- (2차 프로젝트) 실시간 데이터 + ccxt + PostgreSQL 스택

---

## ⚠️ 프로젝트를 죽일 수 있는 4가지 (상시 점검)

1. **시드 1개 결과 보고** → RL 논문 신뢰도 즉사. 최소 5시드 mean±std.
2. **데이터 누수** → scaler·튜닝을 test로 하면 결과 전부 무효. train-only 고정.
3. **트랜스포머 오버피팅** → 8자산 시간봉은 신호가 적음. MLP를 못 이겨도 정당한 결론. 가설은 "이긴다"가 아니라 "이기는지 검증한다".
4. **평가 함수 불일치** → 8월 3주차 공용 모듈로 전 모델 동일 채점 강제.
