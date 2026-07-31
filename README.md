# stat_thesis

암호화폐(BTC, ETH, BNB, XRP, ADA, DOGE, SOL, AVAX 8종목) 시간봉 데이터를 이용한 시장 국면(Regime) 분석 및
PPO 기반 포트폴리오 최적화 통계 졸업 논문 프로젝트.

- **데이터 소스:** Binance 1시간봉 OHLCV (USDT 페어)
- **분석 기간:** 2021-01-01 ~ 2025-12-31

## 프로젝트 구조

```
src/cryptoagent/
  pipeline/     Week1~4 데이터 파이프라인 스크립트 (fetch_binance_data.py 포함)
  envs/         PPO 학습용 환경 (FinRL PortfolioOptimizationEnv 벤더링 + adapter)
data/
  raw/          로컬에서 생성되는 SQLite DB (.gitignore, 스크립트 재실행으로 복구)
  processed/    Week1~3 산출물 (features.parquet, regime_definition.csv 등, git 추적)
results/        Week4 최종 산출물 (data_dictionary.csv, freeze_metadata.json)
docs/           리포트/회의록 문서
requirements.txt, pyproject.toml
```

모든 파이프라인 스크립트는 **리포 루트에서 실행**하는 것을 전제로 상대경로를 씁니다
(예: `python src/cryptoagent/pipeline/Week1_consistency_check.py`).

## 진행 현황

| 주차 | 내용 | 스크립트 | 주요 산출물 |
|---|---|---|---|
| 1주차 | 데이터 정합성 점검 | [Week1_consistency_check.py](src/cryptoagent/pipeline/Week1_consistency_check.py) | [docs/consistency_report.md](docs/consistency_report.md), [data/processed/coverage.csv](data/processed/coverage.csv), [data/processed/gaps.csv](data/processed/gaps.csv) |
| 2주차 | 시장 국면(Regime) 정의 | [Week2_regime_definition.py](src/cryptoagent/pipeline/Week2_regime_definition.py) | [data/processed/regime_definition.csv](data/processed/regime_definition.csv), [data/processed/regime_explore.png](data/processed/regime_explore.png), [data/processed/tail_stats_overall.csv](data/processed/tail_stats_overall.csv), [data/processed/tail_stats_by_regime.csv](data/processed/tail_stats_by_regime.csv) |
| 3주차 | 피처 생성 (기술지표) | [Week3_feature_generation.py](src/cryptoagent/pipeline/Week3_feature_generation.py) | [data/processed/features.parquet](data/processed/features.parquet), [docs/feature_list.md](docs/feature_list.md) |
| 4주차 | 최종 데이터셋 구축 및 동결 | [Week4_data_freeze.py](src/cryptoagent/pipeline/Week4_data_freeze.py) | `data/raw/crypto_market_features.db`(SQLite, 로컬 생성), [results/data_dictionary.csv](results/data_dictionary.csv), [results/freeze_metadata.json](results/freeze_metadata.json) |
| 8월 1주차 | FinRL 환경 세팅 | [envs/](src/cryptoagent/envs/) | `PortfolioOptimizationEnv` 벤더링 + adapter, observation shape `(3, 8, 50)` 확정 |

자세한 내용은 [docs/meeting_report_7월.md](docs/meeting_report_7월.md) 참고.

## 데이터 파이프라인

1. `fetch_binance_data.py` — Binance API로 OHLCV 수집 후 `data/raw/binance_ohlcv.db`(SQLite)에 저장
2. `Week1_consistency_check.py` — 커버리지/시간 갭/중복 점검
3. `Week2_regime_definition.py` — 누적수익률 및 rolling 변동성 기반 국면 확정, train/val/test 분할
4. `Week3_feature_generation.py` — pandas-ta로 기술지표 계산 (RSI, MACD, ATR, OBV, SMA, EMA)
5. `Week4_data_freeze.py` — features.parquet에 국면/split 라벨 조인 후 `data/raw/crypto_market_features.db`에 저장 + 데이터 딕셔너리·동결 메타데이터 생성

## 국면(Regime) 및 Train/Val/Test 분할

국면 단위로 통째 배정(시계열 특성상 랜덤 셔플 불가). 기준 파일은 [data/processed/regime_definition.csv](data/processed/regime_definition.csv) (아래 표와 동일).

| 국면 | 기간 | Split |
|---|---|---|
| bull_2021 | 2021-01-01 ~ 2021-11-09 | train |
| bear_2022 | 2021-11-10 ~ 2023-01-01 | train |
| side_2023 | 2023-01-02 ~ 2023-10-15 | val |
| bull_2024 | 2023-10-16 ~ 2025-01-01 | test |
| choppy_2025 | 2025-01-02 ~ 2025-12-31 | test |

## 🚩 데이터 동결 (Data Freeze)

`Week4_data_freeze.py` 실행 결과([results/freeze_metadata.json](results/freeze_metadata.json)) 기준으로 `crypto_ppo_feature_dataset v1.0`을 동결한다. **이후 `data/processed/features.parquet`, `data/processed/regime_definition.csv`는 변경하지 않는다.** 데이터 수정이 필요하면 버전을 올리고(`v1.1` 등) 새 freeze_metadata를 생성한다.

- SQLite DB(`data/raw/*.db`)는 용량 문제로 `.gitignore` 처리되어 있음 — 필요 시 해당 파이프라인 스크립트를 로컬에서 재실행해 생성 (입력이 동일하면 항상 동일한 결과, idempotent).

## PPO 환경 (FinRL)

`src/cryptoagent/envs/`에 [AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL)의
`PortfolioOptimizationEnv`를 벤더링(해당 파일만 복사, pip `finrl` 패키지의 불필요한 의존성 배제)해서
Week4 데이터에 연결했다. 사용법·확인된 observation/action shape·알려진 이슈는
[src/cryptoagent/envs/USAGE.md](src/cryptoagent/envs/USAGE.md) 참고.

## 환경 설정

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .   # src/cryptoagent를 어디서든 import 가능하게
```

`.env` 파일에 Binance API 키 설정 필요:

```bash
cp .env.example .env
# .env 열어서 BINANCE_API_KEY / BINANCE_API_SECRET 채우기
```

## 다음 단계

- 8월 2주차: MLP PPO end-to-end 학습·백테스트, W&B 연동
- 8월 3주차: 공용 평가 모듈(`evaluate()`) — Sharpe·Sortino·Calmar·MDD·Turnover
- 학습 단계에서 train 구간 기준 스케일러 fit → val/test 적용 (data leakage 방지)
- 거시지표(DXY, VIX) 포함 여부 결정 및 외부 데이터 연동
