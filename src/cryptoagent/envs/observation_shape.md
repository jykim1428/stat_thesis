# PortfolioOptimizationEnv Observation Shape

## 1. Observation 정의

본 프로젝트에서는 FinRL 기반 `PortfolioOptimizationEnv`를 사용하여 암호화폐 포트폴리오 최적화 강화학습 환경을 구성한다.

환경이 Agent(PPO 등)에 제공하는 observation은 다음 형태의 3차원 Tensor이다.

```
Observation = (Feature, Asset, Time)
```

즉, observation shape은 다음과 같다.

```
(feature 수, 자산 수, 시간 window)
```

---

## 2. 현재 Observation Shape

현재 환경 설정:

```python
FEATURES = ["close", "high", "low"]
TIME_WINDOW = 50
```

데이터 기준:

* Feature 개수: 3개
* Asset 개수: 8개
* Time window: 50 timestep

따라서 최종 observation shape:

```
(3, 8, 50)
```

---

## 3. Action Space

FinRL `PortfolioOptimizationEnv`는 portfolio weight vector를 action으로 사용한다.

현재 자산 수:

```
N = 8
```

이므로 action dimension은:

```
action_space.shape = (N + 1)
               = (9,)
```

이다.

추가된 1개의 dimension은 현금(cash) weight이다.

Action index 구조:

| Index | Asset    |
| ----- | -------- |
| 0     | Cash     |
| 1     | ADAUSDT  |
| 2     | AVAXUSDT |
| 3     | BNBUSDT  |
| 4     | BTCUSDT  |
| 5     | DOGEUSDT |
| 6     | ETHUSDT  |
| 7     | SOLUSDT  |
| 8     | XRPUSDT  |

검증 방법:

* `enumerate_portfolio()` 함수에서 `Index: 0. Tic: Cash` 확인
* 초기 action은 `[1,0,0,...,0]` 형태로 현금 100% 상태에서 시작
* `tic_list`는 정렬된 dataframe 기준 생성

따라서 현재 action 구조는:

```
(Cash, ADA, AVAX, BNB, BTC, DOGE, ETH, SOL, XRP)
```

순서이다.

---

## 4. Observation Dimension 의미

### 4.1 Feature Dimension

첫 번째 dimension은 feature channel이다.

```
Feature = 3
```

구성:

| Index | Feature |
| ----- | ------- |
| 0     | close   |
| 1     | high    |
| 2     | low     |

---

### 4.2 Asset Dimension

두 번째 dimension은 portfolio 구성 자산이다.

현재 사용 자산:

```
ADAUSDT
AVAXUSDT
BNBUSDT
BTCUSDT
DOGEUSDT
ETHUSDT
SOLUSDT
XRPUSDT
```

총:

```
Asset = 8
```

---

### 4.3 Time Dimension

세 번째 dimension은 과거 관측 window이다.

현재 데이터는 1시간봉(hourly candle)을 사용한다.

```
TIME_WINDOW = 50
```

따라서 Agent는:

```
최근 50시간의 시장 정보
```

를 보고 portfolio action을 결정한다.

---

## 5. Observation 생성 과정

데이터 흐름:

```
crypto_market_features.db
        |
        v
adapter.load_env_ready_df()
        |
        v
Environment DataFrame
(139,848 rows, 8 assets)
        |
        v
PortfolioOptimizationEnv
        |
        v
reset()
        |
        v
Observation
(3, 8, 50)
```

데이터 검증:

```
episode_length = 17,432
time_window = 50
```

따라서:

```
전체 timestep
= 17,432 + 50 - 1
= 17,481
```

이고,

```
17,481 × 8 assets
= 139,848 rows
```

로 실제 dataframe 크기와 일치한다.

---

## 6. Normalization

환경 생성 시:

```python
normalize_df="by_previous_time"
```

설정을 사용한다.

FinRL 소스코드 검증 결과, 이전 timestep 대비 변화율로 변환한다.

공식:

```
X_t = Price_t / Price_(t-1)
```

예:

```
현재 close = 60,500
이전 close = 60,000

60,500 / 60,000 = 1.0083
```

즉:

```
약 +0.83% 상승
```

을 의미한다.

---

## 7. Observation Shape 검증 결과

실행:

```bash
python check_env.py
```

결과:

```
실제 obs.shape : (3, 8, 50)
기대 shape     : (3, 8, 50)

✅ 공식과 일치
✅ USAGE.md 문서값과 일치
```

최종 검증 결과:

| 항목                | 값        |
| ----------------- | -------- |
| Feature 수         | 3        |
| Asset 수           | 8        |
| Time Window       | 50       |
| Observation Shape | (3,8,50) |
| Action Space      | (9,)     |
| Episode Length    | 17,432   |

현재 FinRL PortfolioOptimizationEnv 데이터 연결 및 observation 구조 검증 완료.

---

## 8. Transformer 확장 기준

현재 FinRL 출력:

```
(feature, asset, time)

(3,8,50)
```

Transformer Actor-Critic 모델 적용 시 입력 변환 기준:

```
(batch, asset, time, feature)

(batch,8,50,3)
```

현재 observation shape `(3,8,50)`은 향후 Transformer Encoder 입력 설계의 기준값으로 사용한다.