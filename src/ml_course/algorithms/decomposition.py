"""PCA from scratch (qua SVD của dữ liệu đã center)."""

from __future__ import annotations

import numpy as np


class PCAScratch:
    def __init__(self, n_components: int | float | None = None):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, float)
        n = len(X)
        self.mean_ = X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        # Cố định dấu để kết quả xác định (giống sklearn svd_flip theo U)
        signs = np.sign(U[np.argmax(np.abs(U), axis=0), np.arange(U.shape[1])])
        signs[signs == 0] = 1
        Vt = Vt * signs[:, None]
        var = S**2 / (n - 1)
        ratio = var / var.sum()
        k = self.n_components
        if k is None:
            k = len(S)
        elif isinstance(k, float) and 0 < k < 1:
            k = int(np.searchsorted(np.cumsum(ratio), k) + 1)
        self.components_ = Vt[:k]
        self.explained_variance_ = var[:k]
        self.explained_variance_ratio_ = ratio[:k]
        self.singular_values_ = S[:k]
        self.n_components_ = k
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        return np.asarray(Z, float) @ self.components_ + self.mean_
