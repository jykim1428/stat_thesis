# stat_thesis

암호화폐(BTC, ETH, BNB, XRP, ADA, DOGE, SOL, AVAX 8종목) 시간봉 데이터를 이용한 시장 국면(Regime) 분석 및 피처 엔지니어링 통계 졸업 논문 프로젝트.

- **데이터 소스:** Binance 1시간봉 OHLCV (USDT 페어)
- **분석 기간:** 2021-01-01 ~ 2025-12-31

## 진행 현황

| 주차 | 내용 | 스크립트 | 주요 산출물 |
|---|---|---|---|
| 1주차 | 데이터 정합성 점검 | [Week1_consistency_check.py](Week1_consistency_check.py) | [consistency_report.md](consistency_report.md), [coverage.csv](coverage.csv), [gaps.csv](gaps.csv) |
| 2주차 | 시장 국면(Regime) 정의 | [Week2_regime_definition.py](Week2_regime_definition.py) | [regime_definition.csv](regime_definition.csv), [regime_explore.png](regime_explore.png), [tail_stats_overall.csv](tail_stats_overall.csv), [tail_stats_by_regime.csv](tail_stats_by_regime.csv) |
| 3주차 | 피처 생성 (기술지표) | [Week3_feature_generation.py](Week3_feature_generation.py) | [features.parquet](features.parquet), [feature_list.md](feature_list.md) |
| 4주차 | 최종 데이터셋 구축 및 동결 | [Week4_data_freeze.py](Week4_data_freeze.py) | `data/crypto_market.db`(SQLite, 로컬 생성), [results/data_dictionary.csv](results/data_dictionary.csv), [results/freeze_metadata.json](results/freeze_metadata.json) |

자세한 내용은 [meeting_report_7월.md](meeting_report_7월.md) 참고.

## 데이터 파이프라인

1. [fetch_binance_data.py](fetch_binance_data.py) — Binance API로 OHLCV 수집 후 `crypto_market.db`(SQLite)에 저장
2. `Week1_consistency_check.py` — 커버리지/시간 갭/중복 점검
3. `Week2_regime_definition.py` — 누적수익률 및 rolling 변동성 기반 국면 확정, train/val/test 분할
4. `Week3_feature_generation.py` — pandas-ta로 기술지표 계산 (RSI, MACD, ATR, OBV, SMA, EMA)
5. `Week4_data_freeze.py` — features.parquet에 국면/split 라벨 조인 후 SQLite 저장 + 데이터 딕셔너리·동결 메타데이터 생성

## 국면(Regime) 및 Train/Val/Test 분할

국면 단위로 통째 배정(시계열 특성상 랜덤 셔플 불가). 기준 파일은 루트의 [regime_definition.csv](regime_definition.csv) (아래 표와 동일).

| 국면 | 기간 | Split |
|---|---|---|
| bull_2021 | 2021-01-01 ~ 2021-11-09 | train |
| bear_2022 | 2021-11-10 ~ 2023-01-01 | train |
| side_2023 | 2023-01-02 ~ 2023-10-15 | val |
| bull_2024 | 2023-10-16 ~ 2025-01-01 | test |
| choppy_2025 | 2025-01-02 ~ 2025-12-31 | test |

## 🚩 데이터 동결 (Data Freeze)

`Week4_data_freeze.py` 실행 결과([results/freeze_metadata.json](results/freeze_metadata.json)) 기준으로 `crypto_ppo_feature_dataset v1.0`을 동결한다. **이후 `features.parquet`, `regime_definition.csv`는 변경하지 않는다.** 데이터 수정이 필요하면 버전을 올리고(`v1.1` 등) 새 freeze_metadata를 생성한다.

- SQLite DB(`data/crypto_market.db`)는 용량 문제로 `.gitignore` 처리되어 있음 — 필요 시 `Week4_data_freeze.py`를 로컬에서 재실행해 생성 (입력이 동일하면 항상 동일한 결과, idempotent).

## 환경 설정

```bash
pip install python-binance pandas pandas-ta python-dotenv matplotlib
```

`.env` 파일에 Binance API 키 설정 필요:

```
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
```

## 다음 단계

- 8월: FinRL 환경 세팅 (베이스라인 파이프라인 완주, 트랜스포머 이전 단계)
- 학습 단계에서 train 구간 기준 스케일러 fit → val/test 적용 (data leakage 방지)
- 거시지표(DXY, VIX) 포함 여부 결정 및 외부 데이터 연동
