"""Đọc, kiểm tra và chia dữ liệu giao dịch."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from fraud.config import MERCHANT_CATEGORIES, RAW_COLUMNS, TARGET


def load_transactions(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def validate(df: pd.DataFrame, require_target: bool = True) -> list[str]:
    """Hợp đồng dữ liệu: danh sách vấn đề (rỗng = hợp lệ)."""
    required = RAW_COLUMNS + ([TARGET] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        return [f"thiếu cột {missing}"]
    issues = []
    if df[required].isna().any().any():
        issues.append("có giá trị thiếu")
    if (df.amount_vnd <= 0).any():
        issues.append("amount_vnd phải dương")
    if not df.hour.between(0, 23).all():
        issues.append("hour ngoài [0, 23]")
    if (df.distance_from_home_km < 0).any() or (df.card_age_days < 0).any() or (df.n_tx_last_24h < 0).any():
        issues.append("giá trị âm không hợp lệ")
    unknown = set(df.merchant_category) - set(MERCHANT_CATEGORIES)
    if unknown:
        issues.append(f"merchant_category lạ: {sorted(unknown)}")
    if require_target:
        if not set(df[TARGET].unique()) <= {0, 1}:
            issues.append("nhãn không phải 0/1")
        elif not 0.001 <= df[TARGET].mean() <= 0.2:
            issues.append(f"tỉ lệ gian lận bất thường: {df[TARGET].mean():.4f}")
    return issues


def split(df: pd.DataFrame, seed: int = 0):
    """train 49% / validation 21% / test 30%, phân tầng theo nhãn."""
    rest, test = train_test_split(df, test_size=0.3, random_state=seed, stratify=df[TARGET])
    train, valid = train_test_split(rest, test_size=0.3, random_state=seed, stratify=rest[TARGET])
    return train, valid, test
