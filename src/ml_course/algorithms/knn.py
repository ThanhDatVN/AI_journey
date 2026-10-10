"""K-Nearest Neighbors from scratch (vectorized distance)."""

from __future__ import annotations

import numpy as np


def pairwise_distances(A, B, metric: str = "euclidean", p: float = 2) -> np.ndarray:
    """Ma trận khoảng cách (n_A, n_B). Euclid dùng ‖a‖² + ‖b‖² − 2a·b (không vòng lặp)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    if metric == "euclidean":
        sq = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T
        return np.sqrt(np.maximum(sq, 0))
    if metric == "manhattan":
        return np.abs(A[:, None, :] - B[None, :, :]).sum(-1)
    if metric == "minkowski":
        return (np.abs(A[:, None, :] - B[None, :, :]) ** p).sum(-1) ** (1 / p)
    if metric == "cosine":
        An = A / np.linalg.norm(A, axis=1, keepdims=True).clip(1e-12)
        Bn = B / np.linalg.norm(B, axis=1, keepdims=True).clip(1e-12)
        return 1 - An @ Bn.T
    raise ValueError(metric)


class _KNNBase:
    def __init__(self, k: int = 5, metric: str = "euclidean", weights: str = "uniform"):
        self.k, self.metric, self.weights = k, metric, weights

    def fit(self, X, y):
        # "Training" của KNN chỉ là ghi nhớ dữ liệu → fit O(1), predict O(n·d).
        self.X_, self.y_ = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(self.y_)
        return self

    def kneighbors(self, X):
        D = pairwise_distances(X, self.X_, self.metric)
        idx = np.argsort(D, axis=1, kind="stable")[:, : self.k]
        return np.take_along_axis(D, idx, axis=1), idx

    def _weights(self, dist):
        if self.weights == "distance":
            return 1.0 / np.maximum(dist, 1e-12)
        return np.ones_like(dist)


class KNNClassifierScratch(_KNNBase):
    def predict_proba(self, X):
        dist, idx = self.kneighbors(X)
        w = self._weights(dist)
        labels = self.y_[idx]
        proba = np.stack([(w * (labels == c)).sum(1) for c in self.classes_], axis=1)
        return proba / proba.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class KNNRegressorScratch(_KNNBase):
    def predict(self, X):
        dist, idx = self.kneighbors(X)
        w = self._weights(dist)
        return (w * self.y_[idx]).sum(1) / w.sum(1)
