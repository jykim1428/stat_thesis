"""
공용 스펙 DataFrame 독립 검증 스크립트 
사용법: python experiments/verify_backtest_spec.py results/ppo_mlp/backtest_test.csv

train_ppo_mlp.py의 sanity_check()와는 별개로, 저장된 CSV 결과물 자체를
다시 읽어서 독립적으로 검증한다. 9월 트랜스포머/LSTM 결과물도
동일 스펙이면 그대로 재사용 가능.
"""
from __future__ import annotations

import ast
import sys

import numpy as np
import pandas as pd

N_ASSETS_PLUS_CASH = 9  # 자산 8종목 + 현금
WEIGHT_SUM_TOL = 1e-3
RETURN_JITTER_TOL = 1e-3  # portfolio_values 정합성 체크 허용 오차


def load_backtest_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    # CSV로 저장되며 weights/target_weights가 문자열이 되므로 복원 필수
    df["weights"] = df["weights"].apply(ast.literal_eval)
    if "target_weights" in df.columns:
        df["target_weights"] = df["target_weights"].apply(ast.literal_eval)
    return df


def _verify_weight_column(df: pd.DataFrame, col: str) -> list[str]:
    """weights/target_weights 컬럼 하나에 대한 길이/합/범위 검증."""
    errors = []

    bad_len = df[col].apply(len) != N_ASSETS_PLUS_CASH
    if bad_len.any():
        errors.append(f"{col} 길이가 {N_ASSETS_PLUS_CASH}이 아닌 행 {bad_len.sum()}개")

    weight_sums = df[col].apply(sum)
    max_dev = (weight_sums - 1.0).abs().max()
    if max_dev >= WEIGHT_SUM_TOL:
        errors.append(f"{col} 합이 1에서 {max_dev:.2e}만큼 벗어남")

    min_w = df[col].apply(min).min()
    max_w = df[col].apply(max).max()
    if min_w < -1e-6 or max_w > 1 + 1e-6:
        errors.append(f"{col}의 개별 값이 [0,1] 범위 밖: min={min_w}, max={max_w}")

    return errors


def verify(df: pd.DataFrame) -> list[str]:
    errors = []

    # 1. NaN / inf 체크
    if df["returns"].isna().any():
        errors.append("returns에 NaN 존재")
    if df["portfolio_values"].isna().any():
        errors.append("portfolio_values에 NaN 존재")
    if not np.isfinite(df["portfolio_values"]).all():
        errors.append("portfolio_values에 inf 존재")

    # 2~4. weights 벡터 길이/합/범위
    errors.extend(_verify_weight_column(df, "weights"))

    # target_weights도 동일 검증 (공용 스펙의 핵심 컬럼 - turnover 계산에 직접 쓰임)
    if "target_weights" in df.columns:
        errors.extend(_verify_weight_column(df, "target_weights"))
    else:
        errors.append(
            "target_weights 컬럼 없음 - turnover가 weights 간 차이로만 근사되어 "
            "과대계상됨 (evaluate.py의 allow_legacy_turnover 참고)"
        )

    # 5. date 인덱스 정렬 + 중복
    if not df.index.is_monotonic_increasing:
        errors.append("date 인덱스가 오름차순이 아님")
    if df.index.duplicated().any():
        errors.append(f"중복된 date 인덱스 {df.index.duplicated().sum()}개")

    # 6. portfolio_values와 returns 정합성
    #    portfolio_values[i] ≈ portfolio_values[i-1] * (1 + returns[i])
    pv = df["portfolio_values"].values
    ret = df["returns"].values
    expected = pv[:-1] * (1 + ret[1:])
    actual = pv[1:]
    rel_diff = np.abs(expected - actual) / np.maximum(np.abs(actual), 1e-8)
    if (rel_diff > RETURN_JITTER_TOL).any():
        n_bad = (rel_diff > RETURN_JITTER_TOL).sum()
        errors.append(
            f"portfolio_values <-> returns 불일치 {n_bad}행 "
            f"(최대 상대오차 {rel_diff.max():.2e}) - t=0 초기값 처리 방식 확인 필요"
        )

    return errors


def main():
    if len(sys.argv) < 2:
        print("사용법: python scripts/verify_backtest_spec.py <backtest_csv_path>")
        sys.exit(1)

    path = sys.argv[1]
    df = load_backtest_csv(path)
    print(f"로드 완료: {path}  shape={df.shape}")

    errors = verify(df)
    if errors:
        print(f"\n❌ 검증 실패 ({len(errors)}건):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\n✅ 모든 검증 통과")
        print(f"   최종 포트폴리오 가치: {df['portfolio_values'].iloc[-1]:,.2f}")
        print(f"   기간: {df.index.min()} ~ {df.index.max()}  ({len(df)}행)")


if __name__ == "__main__":
    main()
