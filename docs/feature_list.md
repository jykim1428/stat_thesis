# Week 3 피처 목록

- 원본 OHLCV: ['Open', 'High', 'Low', 'Close', 'Volume']
- 기술지표 (10개): ['RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'ATR_14', 'OBV', 'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50']

## 결측치 처리 규칙

- 지표 계산에 필요한 최대 warm-up 구간(50봉)을 종목별로 앞에서 잘라냄
- 이후 남은 NaN은 dropna로 제거
- 최종 shape: (349896, 17)

## 스케일링

- 이번 단계에서는 하지 않음. train-only fit 원칙에 따라 학습 단계로 이관.
- 거시지표(DXY, VIX)는 이번 주 범위에서 제외.
