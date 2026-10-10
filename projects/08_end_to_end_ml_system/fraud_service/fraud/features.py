"""Đặc trưng dùng chung cho huấn luyện và phục vụ (import được → pickle an toàn giữa các tiến trình)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_amount"] = np.log(out["amount_vnd"].astype(float).clip(lower=1_000))
    out["is_night"] = ((out["hour"] <= 4) | (out["hour"] >= 23)).astype(int)
    out["is_foreign"] = out["is_foreign"].astype(bool)
    return out
