# Colab 컴퓨트 예산 점검

## 1. 목적

Walk-forward validation을 수행하기 전에 PPO 학습에 필요한 Colab 컴퓨트 예산을 사전에 추정하고, 장시간 실행으로 인한 세션 종료 및 재실행 비용을 최소화한다.

현재 시간 추정치는 2026-08-21에 로컬 CPU 환경에서 수행한 `train_ppo_mlp.py` 실행 결과를 기준으로 한 **예비 추정치**이다. 따라서 실제 Colab 런타임에서 MLP와 Transformer 각각의 파일럿을 수행한 후 최종 예산을 확정한다.

---

## 2. 기준 실행 기록

2026-08-21에 `train_ppo_mlp.py`를 `TOTAL_TIMESTEPS=50_000`으로 실행한 결과를 기준 실행으로 사용한다.

| 항목                   |              관측값 |
| -------------------- | ---------------: |
| 설정 `TOTAL_TIMESTEPS` |           50,000 |
| 실제 PPO 학습 step 수     |           51,200 |
| 학습 wall-clock 시간     | 약 952초 (약 15.9분) |
| 종료 시 처리량             |         약 53 fps |
| 장치                   |        Local CPU |
| Test 백테스트            |            정상 완료 |
| Sanity check         |            정상 완료 |

### 실제 학습 step 수가 51,200인 이유

Stable-Baselines3 PPO는 rollout을 `n_steps=2048` 단위로 수집한다.

따라서 `TOTAL_TIMESTEPS=50,000`을 지정하더라도 실제 학습은 rollout 단위로 진행되며, 50,000을 처음으로 초과하는 지점인

```text
2048 × 25 = 51,200
```

step에서 종료된다.

따라서 이후의 시간 추정에서는 설정값인 50,000이 아니라 **실제로 학습된 51,200 step**을 기준으로 한다.

기준 실행의 step당 평균 학습 시간은 다음과 같다.

```text
952초 / 51,200 step
≈ 0.0186초/step
```

이는 약 53.8 step/s이며, 실행 로그의 약 53 fps와 일치한다.

> **주의:** 위 값은 Local CPU에서 측정한 값이며 Colab의 T4 GPU 등 다른 런타임에서 동일하게 재현된다고 가정할 수 없다.

---

## 3. Walk-forward validation의 학습 횟수

정책망 수를 `M`, Fold 1에서 탐색할 하이퍼파라미터 후보 수를 `N`, seed 반복 수를 `S`라고 한다.

각 후보는 **Fold 1의 train 구간에서 학습하고 Fold 1 validation 구간에서 평가**한다.

정책망 하나당 필요한 학습 횟수는 다음과 같다.

```text
N + 2
```

구성은 다음과 같다.

```text
N       : Fold 1 하이퍼파라미터 후보 학습
+ 1     : Fold 1에서 선택된 설정으로 2021~2023 전체 구간 재학습
+ 1     : 동일한 선택 설정으로 2021~2024 전체 구간을 사용한 Fold 2 학습
```

따라서 모든 정책망을 동일한 후보 수와 seed 수로 실행하는 경우:

```text
총 PPO 학습 횟수 = M × S × (N + 2)
```

MLP와 Transformer를 모두 비교하는 경우 `M=2`이다.

단, 정책망별 후보 수가 다르면 다음과 같이 각각 계산한다.

```text
총 학습 횟수
= Σ [정책망별 학습 횟수]
= Σ [S × (N_policy + 2)]
```

### 중요한 원칙

Fold 1 validation 결과를 이용해 후보 설정을 선택한 후에는 **Fold 1/2 OOS 결과를 보고 하이퍼파라미터를 추가로 조정하지 않는다.**

즉, 최종적인 OOS 성능을 확인한 뒤 다시 후보 수나 하이퍼파라미터를 변경하면 데이터 누수 및 평가 편향이 발생할 수 있으므로 금지한다.

---

## 4. 현재 기준 MLP-equivalent 컴퓨트 예산

아래 계산은 2026-08-21 Local CPU의 MLP 기준 실행 결과인 **51,200 step / 약 15.9분**을 이용한 예비 추정이다.

실제 실험에서는 W&B 동기화, 모델 저장, 백테스트, 파일 저장 및 기타 런타임 오버헤드가 발생할 수 있으므로 순수 학습 시간에 최소 25%의 여유를 적용한다.

### MLP 기준 예비 예산

| 목적             |  M |  N |  S | 총 학습 횟수 | 순수 학습 추정 | 25% 여유 포함 |
| -------------- | -: | -: | -: | ------: | -------: | --------: |
| 설정 1개, seed 1개 |  2 |  1 |  1 |       6 |  약 1.6시간 |   약 2.0시간 |
| 후보 4개, seed 1개 |  2 |  4 |  1 |      12 |  약 3.2시간 |   약 4.0시간 |
| 후보 4개, seed 3개 |  2 |  4 |  3 |      36 |  약 9.5시간 |  약 12.0시간 |

위 표는 **MLP와 Transformer의 실행 시간이 동일하다고 가정한 MLP-equivalent 예산**이다.

따라서 이 값은 실제 Transformer 실험의 확정 예산이 아니라 **예비적인 하한선**으로 취급한다.

---

## 5. 실행 전 파일럿 측정

전체 Walk-forward validation을 시작하기 전에 MLP와 Transformer 각각에 대해 **동일한 Colab 런타임 환경**에서 파일럿을 수행한다.

### 파일럿 설정

각 정책망에 대해:

```text
TOTAL_TIMESTEPS = 10,240
```

으로 실행한다.

PPO의 `n_steps=2048`인 경우:

```text
10,240 = 2,048 × 5
```

이므로 정확히 5개의 PPO rollout에 해당한다.

파일럿 종료 후에는 반드시 실제 로그의 `time_elapsed`와 `total_timesteps`를 확인한다.

측정 항목은 다음과 같다.

| 측정 항목             | 내용                |
| ----------------- | ----------------- |
| Policy            | MLP / Transformer |
| Runtime           | 실제 Colab 런타임 종류   |
| `total_timesteps` | 실제 종료 step        |
| `time_elapsed`    | 실제 학습 시간          |
| fps               | 종료 시 처리량          |
| Model save time   | 모델 저장에 걸린 시간      |
| Backtest time     | test/fold 백테스트 시간 |
| W&B overhead      | 로그 동기화 및 관련 오버헤드  |
| Total wall-clock  | 전체 실행에 걸린 시간      |

---

## 6. 파일럿 기반 1회 실행 시간 추정

정책망별 파일럿 결과를 이용해 51,200-step 학습 시간을 추정한다.

### 학습 시간

```text
예상 51,200-step 학습 시간
=
파일럿 학습 시간
×
(51,200 / 파일럿 실제 step 수)
```

또는 seconds/step을 이용하면:

```text
예상 학습 시간
=
(파일럿 학습 시간 / 파일럿 실제 step 수)
× 51,200
```

### 전체 1회 실행 시간

학습 이외에 모델 저장, W&B 동기화, 백테스트 및 결과 저장 시간이 필요하므로:

```text
1회 전체 실행 예상 시간
=
예상 학습 시간
+ 모델 저장 시간
+ 백테스트 시간
+ 결과 저장/W&B 오버헤드
```

여기에 예측 오차를 고려하여 최소 25%의 여유 시간을 적용한다.

```text
예산 산정 시간
=
1회 전체 실행 예상 시간 × 1.25
```

---

## 7. 전체 Walk-forward 컴퓨트 예산

정책망별 1회 실행 시간을 `T_policy`, 정책망별 학습 횟수를 `K_policy`라고 하면:

```text
총 예상 실행 시간
=
Σ (T_policy × K_policy)
```

예를 들어 MLP와 Transformer를 각각 후보 4개, seed 3개로 수행한다면:

```text
정책망 하나당 학습 횟수
= 3 × (4 + 2)
= 18회
```

따라서:

```text
MLP 총 실행 시간
= MLP 1회 실행 예상 시간 × 18

Transformer 총 실행 시간
= Transformer 1회 실행 예상 시간 × 18
```

전체 예산은 두 값을 합산한다.

```text
전체 예상 시간
=
MLP 예상 총 시간
+
Transformer 예상 총 시간
```

단, 실제 Transformer 학습 시간이 MLP와 다를 수 있으므로 **Transformer 파일럿 이전에는 이 값을 확정 예산으로 사용하지 않는다.**

---

## 8. Colab 실행 단위 및 중단 대응

Colab 세션의 런타임 종료 또는 연결 해제에 대비하여 모든 조합을 하나의 장시간 실행으로 묶지 않는다.

각 실험은 다음 단위로 독립적으로 실행한다.

```text
{policy}/{fold}/{candidate}/{seed}
```

예:

```text
MLP/Fold1/candidate01/seed42
MLP/Fold1/candidate02/seed42
Transformer/Fold1/candidate01/seed42
Transformer/Fold2/final/seed42
```

각 실행이 완료되면 다음 파일을 영구 저장소에 남긴다.

* 학습된 `.zip` 모델
* Fold별 OOS 결과 CSV
* 실행 설정 JSON 또는 YAML
* 실제 `total_timesteps`
* 실제 wall-clock time
* seed
* policy
* fold 날짜 범위
* candidate 설정
* 주요 평가 지표
* 완료 여부를 나타내는 summary CSV

예시:

```text
results/
└── walk_forward/
    ├── mlp/
    │   └── fold1/
    │       ├── candidate01/
    │       │   └── seed42/
    │       ├── candidate02/
    │       │   └── seed42/
    │       └── summary.csv
    │
    └── transformer/
        └── fold1/
            └── candidate01/
                └── seed42/
```

---

## 9. Resume 원칙

Colab 세션이 중단되더라도 이미 완료된 학습을 다시 수행하지 않도록 한다.

실행 시작 시 summary 파일을 확인하여 다음 조건을 만족하는 조합은 건너뛴다.

```text
{policy, fold, candidate, seed}
```

에 해당하는 결과가 존재하고,

```text
status = "completed"
```

인 경우 재학습하지 않는다.

반대로 다음 경우에는 해당 조합을 다시 실행한다.

* 모델 파일이 없음
* 결과 CSV가 없음
* summary가 없음
* `status != "completed"`
* 실행 중 오류가 발생하여 결과가 불완전함

이 방식으로 Colab 세션이 중간에 종료되더라도 **완료된 조합은 보존하고 미완료 조합만 재개**한다.

---

## 10. 공정한 비교를 위한 고정 조건

MLP와 Transformer 및 각 벤치마크를 비교할 때 다음 조건은 가능한 한 동일하게 유지한다.

| 조건                      | 적용 원칙 |
| ----------------------- | ----- |
| Fold 경계                 | 동일    |
| Train/Validation/OOS 구간 | 동일    |
| Initial capital         | 동일    |
| 거래비용                    | 동일    |
| 평가 지표                   | 동일    |
| 데이터 빈도                  | 동일    |
| 자산 universe             | 동일    |
| Time window             | 동일    |
| Seed 정책                 | 사전 확정 |
| 평가 방법                   | 동일    |

특히 Fold 1 validation은 **하이퍼파라미터 선택에만 사용**하고, 최종 OOS 결과를 확인한 뒤 설정을 변경하지 않는다.

---

## 11. 실행 결정 기준

파일럿 결과를 확인한 뒤 다음 기준으로 전체 실험 실행 여부를 결정한다.

### ① 1회 실행 시간이 Colab 세션에서 안정적으로 완료되는가?

파일럿을 기반으로 추정한 1회 전체 실행 시간이 단일 Colab 세션에서 충분히 완료 가능한 수준인지 확인한다.

장시간 실행이 불안정한 경우 실험을 더 작은 독립 실행 단위로 분할한다.

### ② 정책망별 실제 학습 속도를 확인했는가?

MLP와 Transformer의 실제 `seconds/step`을 각각 측정한다.

특히 Transformer는 feature extractor와 attention 연산으로 인해 MLP보다 느릴 수 있으므로 MLP 기준 시간을 그대로 적용하지 않는다.

### ③ 후보 수와 seed 수를 사전에 확정했는가?

전체 Walk-forward 실행 전에:

```text
N = 후보 수
S = seed 반복 수
```

를 확정한다.

OOS 결과를 본 후 후보나 seed 수를 변경하지 않는다.

### ④ 완료된 실험을 재사용할 수 있는가?

summary와 결과 파일을 기준으로 이미 완료된 조합을 자동으로 건너뛸 수 있어야 한다.

---

## 12. 현재 컴퓨트 예산 결론

현재 확보된 기준 실행 결과는 다음과 같다.

```text
MLP
50,000 설정
→ 실제 51,200 step
→ 약 952초
→ 약 15.9분
→ Local CPU
```

따라서 현재의 51,200-step MLP 실행 시간은 Walk-forward validation의 **예비적인 기준값**으로 사용할 수 있다.

그러나 이 값을 그대로 Colab 전체 예산으로 확정해서는 안 된다.

실제 Colab 환경에서는 하드웨어, 런타임 상태, W&B 동기화, 모델 저장 및 백테스트에 따라 실행 시간이 달라질 수 있으며, 특히 Transformer는 MLP와 학습 시간이 다를 가능성이 있다.

따라서 전체 Walk-forward validation에 앞서 다음 순서로 진행한다.

```text
1. Colab 런타임 확정
        ↓
2. MLP 10,240-step 파일럿
        ↓
3. Transformer 10,240-step 파일럿
        ↓
4. 실제 total_timesteps / time_elapsed 기록
        ↓
5. 정책망별 51,200-step 학습 시간 추정
        ↓
6. 모델 저장 + W&B + 백테스트 시간 측정
        ↓
7. 최소 25% 여유를 포함한 1회 실행 예산 계산
        ↓
8. 후보 수 N 및 seed 수 S를 적용하여
   전체 Walk-forward 컴퓨트 예산 산정
        ↓
9. 독립적인 {policy}/{fold}/{candidate}/{seed}
   단위로 전체 실험 실행
```

### 최종 판단

현재 문서의 **15.9분 및 2~12시간 등의 수치는 확정 예산이 아니라 Local CPU 기반 MLP의 예비 추정치**로 유지한다.

최종 컴퓨트 예산은 **동일한 Colab 런타임에서 MLP와 Transformer 각각 10,240-step 파일럿을 완료한 후 실제 측정값으로 갱신한다.**

파일럿 결과를 확인하기 전에는 Transformer의 전체 실행 시간이나 최종 Walk-forward 소요 시간을 단정하지 않는다.
