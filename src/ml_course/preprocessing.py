"""Preprocessing viết from scratch — cùng API ``fit`` / ``transform`` như scikit-learn.

Nguyên tắc vàng (chương 04, 05): **chỉ fit trên train**, rồi transform cả train/val/test.
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np


def train_test_split(*arrays, test_size: float = 0.2, seed: int | None = 0, stratify=None, shuffle: bool = True):
    """Chia các array theo cùng một hoán vị. ``stratify``: giữ tỉ lệ lớp ở cả 2 tập."""
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    if stratify is not None:
        stratify = np.asarray(stratify)
        test_idx = []
        for label in np.unique(stratify):
            idx = np.flatnonzero(stratify == label)
            rng.shuffle(idx)
            test_idx.extend(idx[: int(round(len(idx) * test_size))])
        test_idx = np.array(sorted(test_idx))
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        rng.shuffle(train_idx)
    else:
        idx = rng.permutation(n) if shuffle else np.arange(n)
        n_test = int(round(n * test_size))
        test_idx, train_idx = idx[n - n_test :], idx[: n - n_test]
    out = []
    for arr in arrays:
        if hasattr(arr, "iloc"):
            out.extend([arr.iloc[train_idx], arr.iloc[test_idx]])
        else:
            arr = np.asarray(arr)
            out.extend([arr[train_idx], arr[test_idx]])
    return out


class StandardScalerScratch:
    """z = (x − μ) / σ, với μ, σ học từ train."""

    def fit(self, X):
        X = np.asarray(X, float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0  # cột hằng: tránh chia 0
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        return np.asarray(Z, float) * self.scale_ + self.mean_


class MinMaxScalerScratch:
    """x' = (x − min) / (max − min) → [0, 1] trên train (test có thể vượt ra ngoài!)."""

    def fit(self, X):
        X = np.asarray(X, float)
        self.min_, self.max_ = X.min(axis=0), X.max(axis=0)
        self.range_ = np.where(self.max_ - self.min_ == 0, 1.0, self.max_ - self.min_)
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.min_) / self.range_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class RobustScalerScratch:
    """x' = (x − median) / IQR — ít bị outlier kéo lệch."""

    def fit(self, X):
        X = np.asarray(X, float)
        self.center_ = np.median(X, axis=0)
        q1, q3 = np.percentile(X, [25, 75], axis=0)
        self.scale_ = np.where(q3 - q1 == 0, 1.0, q3 - q1)
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.center_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class OneHotEncoderScratch:
    """One-hot cho 1 cột phân loại. ``handle_unknown='ignore'`` → category lạ thành vector 0."""

    def __init__(self, handle_unknown: str = "ignore"):
        self.handle_unknown = handle_unknown

    def fit(self, values):
        self.categories_ = np.unique(np.asarray(values).astype(str))
        self._index = {c: i for i, c in enumerate(self.categories_)}
        return self

    def transform(self, values):
        values = np.asarray(values).astype(str)
        out = np.zeros((len(values), len(self.categories_)))
        for row, v in enumerate(values):
            if v in self._index:
                out[row, self._index[v]] = 1.0
            elif self.handle_unknown == "error":
                raise ValueError(f"Category chưa gặp khi fit: {v!r}")
        return out

    def fit_transform(self, values):
        return self.fit(values).transform(values)


def add_bias(X) -> np.ndarray:
    """Thêm cột 1 vào đầu ma trận: X → [1, X] (để gộp bias b vào w)."""
    X = np.asarray(X, float)
    return np.hstack([np.ones((X.shape[0], 1)), X])


def polynomial_features(X, degree: int = 2, include_bias: bool = False) -> np.ndarray:
    """Mọi tích các feature có bậc ≤ degree (giống sklearn PolynomialFeatures)."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    n, d = X.shape
    cols = [np.ones(n)] if include_bias else []
    for deg in range(1, degree + 1):
        for combo in combinations_with_replacement(range(d), deg):
            cols.append(np.prod(X[:, combo], axis=1))
    return np.column_stack(cols)


__all__ = [
    "train_test_split",
    "StandardScalerScratch",
    "MinMaxScalerScratch",
    "RobustScalerScratch",
    "OneHotEncoderScratch",
    "add_bias",
    "polynomial_features",
]
