# 부록 B. 하이퍼파라미터 탐색 전체 표

> **집필: 은아** (원문 그대로 수록)

탐색은 Fold 1의 검증 구간인 횡보 국면(side_2023)에서 정책망별 4개 후보 × 3개 시드(42, 43, 44), 총 12회씩 수행하였다. 모든 값은 거래비용(0.1%) 차감 후(net) validation Sharpe이며 소수점 3자리로 표기한다. 후보별 평균 ± 표준편차는 시드별 값에서 산출하였다. 모든 후보의 학습량은 51,200으로 동일하다.

## B.1 MLP 탐색 결과

| 후보 | 시드 | validation Sharpe | 평균 ± 표준편차 |
|---|---|---|---|
| candidate01 | 42 | 0.762 | 0.736 ± 0.039 |
| candidate01 | 43 | 0.756 | |
| candidate01 | 44 | 0.692 | |
| candidate02 | 42 | 0.759 | 0.736 ± 0.046 |
| candidate02 | 43 | 0.766 | |
| candidate02 | 44 | 0.683 | |
| **candidate03 (선택)** | 42 | 0.775 | **0.762 ± 0.045** |
| **candidate03 (선택)** | 43 | 0.799 | |
| **candidate03 (선택)** | 44 | 0.712 | |
| candidate04 | 42 | 0.772 | 0.744 ± 0.055 |
| candidate04 | 43 | 0.779 | |
| candidate04 | 44 | 0.680 | |

**최종 선택 (MLP):** candidate03 (`learning_rate` = 1e-3, net validation Sharpe 산술평균 0.762 ± 0.045, 최저 시드 0.712)

## B.2 Transformer 탐색 결과

| 후보 | 시드 | validation Sharpe | 평균 ± 표준편차 |
|---|---|---|---|
| candidate01 | 42 | 0.781 | 0.807 ± 0.030 |
| candidate01 | 43 | 0.798 | |
| candidate01 | 44 | 0.840 | |
| candidate02 | 42 | 0.776 | 0.807 ± 0.031 |
| candidate02 | 43 | 0.805 | |
| candidate02 | 44 | 0.839 | |
| candidate03 | 42 | 0.817 | 0.809 ± 0.021 |
| candidate03 | 43 | 0.785 | |
| candidate03 | 44 | 0.824 | |
| **candidate04 (선택)** | 42 | 0.907 | **0.859 ± 0.058** |
| **candidate04 (선택)** | 43 | 0.875 | |
| **candidate04 (선택)** | 44 | 0.794 | |

**최종 선택 (Transformer):** candidate04 (`n_layers` = 1, net validation Sharpe 산술평균 0.859 ± 0.058, 최저 시드 0.794)

## B.3 후보 선택 기준

정책망별로 3개 시드의 net validation Sharpe 산술평균이 최대인 후보를 선택한다. 소수점 셋째 자리까지 동률이면 (1) 최저 시드 Sharpe, (2) 평균 Calmar, (3) 더 작은 구조(낮은 계산비용) 순으로 정한다. 표준편차는 선택 점수에 포함하지 않고 불확실성으로 별도 보고한다.

이 선택 기준은 validation 결과 검토 후 확정하고 OOS 확인 전 동결한 기준이다. 후보와 시드 조합 자체는 탐색 실행 전에 고정하였다.

확정된 후보는 표본 외(OOS) 평가(Fold 1, Fold 2)에서 재조정하지 않았다.

최종 OOS 평가의 시드는 42~46(5개)이며, 시드 45, 46은 OOS 확인 후 후보·하이퍼파라미터 변경 없이 일괄 추가한 post-hoc robustness extension이다. 위 탐색 표의 validation 수치는 확장 전 3-seed(42, 43, 44) 기준 그대로이다.
