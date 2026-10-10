"""Mô hình, chính sách quyết định theo tổn thất kỳ vọng, và chi phí."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from fraud.config import CATEGORICAL, NUMERIC, REVIEW_COST_VND, TARGET
from fraud.features import add_features


def build_model():
    """Hồi quy logistic KHÔNG cân bằng lớp — giữ xác suất hiệu chỉnh tốt (quyết định dùng trực tiếp xác suất)."""
    columns = ColumnTransformer([("num", StandardScaler(), NUMERIC), ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return make_pipeline(FunctionTransformer(add_features), columns, LogisticRegression(max_iter=2000))


def decide(p_fraud, amount_vnd, review_cost: float = REVIEW_COST_VND) -> np.ndarray:
    """Kiểm duyệt khi tổn thất kỳ vọng nếu duyệt thẳng (p × số tiền) lớn hơn chi phí kiểm duyệt."""
    return np.asarray(p_fraud, float) * np.asarray(amount_vnd, float) > review_cost


def total_cost(df: pd.DataFrame, review, review_cost: float = REVIEW_COST_VND) -> float:
    """Chi phí thực tế (đồng): kiểm duyệt → review_cost; duyệt thẳng giao dịch gian lận → mất toàn bộ số tiền."""
    review = np.asarray(review, bool)
    fraud = df[TARGET].to_numpy() == 1
    return float(np.where(review, review_cost, np.where(fraud, df.amount_vnd.to_numpy(), 0.0)).sum())


def baseline_policies(df: pd.DataFrame) -> dict[str, np.ndarray]:
    feats = add_features(df)
    return {
        "duyệt tất cả": np.zeros(len(df), bool),
        "kiểm duyệt tất cả": np.ones(len(df), bool),
        "luật: nước ngoài hoặc (đêm và > 1 triệu)": (feats.is_foreign | ((feats.is_night == 1) & (feats.amount_vnd > 1_000_000))).to_numpy(),
    }
