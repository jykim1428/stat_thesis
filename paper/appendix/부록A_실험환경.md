# 부록 A. 실험 환경 및 패키지 버전

> **집필: 은아** (원문 그대로 수록)

## A.1 패키지 버전

| 패키지 | 버전 | 패키지 | 버전 |
|---|---|---|---|
| Python | 3.13.7 | empyrical-reloaded | 0.5.12 |
| PyTorch | 2.13.0 | arch | 8.0.0 |
| Stable-Baselines3 | 2.9.0 | scipy | 1.18.0 |
| Gymnasium | 1.3.0 | PyPortfolioOpt | 1.6.0 |
| pandas | 3.0.5 | quantstats | 0.0.81 |
| numpy | 2.2.6 | | |

## A.2 재현성 확보 기록

| 항목 | 내용 |
|---|---|
| 실험 동결 커밋 (experiment freeze commit) | `10892a1` (2026-09-08) |
| 동결 범위 | 모형 가중치, 백테스트 산출물, 성과지표, 통계검정 결과 |
| 체크섬 기록 파일 수 | 61개 (백테스트 20 + 모형 .zip 20 + 벤치마크 8 + 요약 12 + 원천 DB 1) |

## A.3 학습 환경 및 소요 시간

| 정책망 | 학습 환경 | 1회 학습 소요 시간 |
|---|---|---|
| MLP | Google Colab | 약 336~412초 |
| Transformer | Google Colab | 약 1,024~1,314초 |

> ⚠️ **수정 이력(2026-11-06).** 초안의 "MLP 320~410초, Transformer
> 640~1,310초"는 **Fold 1 탐색 단계**의 값이었다. 위 표는 본문 표
> 4-6과 동일한 **최종 학습(fold1_final·fold2_final, 5-seed)** 기준으로
> 정정한 것이다. 동결 산출물의 `train_elapsed_sec` 실측값이다.
