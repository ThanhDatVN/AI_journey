"""Clustering from scratch: K-Means (Lloyd + k-means++), DBSCAN."""

from __future__ import annotations

from collections import deque

import numpy as np

from .knn import pairwise_distances


class KMeansScratch:
    """min Σᵢ ‖xᵢ − μ_{cᵢ}‖² bằng thuật toán Lloyd; chạy ``n_init`` lần, giữ lần có inertia nhỏ nhất."""

    def __init__(self, n_clusters: int = 3, init: str = "k-means++", n_init: int = 10,
                 max_iter: int = 300, tol: float = 1e-6, seed: int = 0):
        self.n_clusters, self.init, self.n_init = n_clusters, init, n_init
        self.max_iter, self.tol, self.seed = max_iter, tol, seed

    def _init_centers(self, X, rng):
        n = len(X)
        if self.init == "random":
            return X[rng.choice(n, self.n_clusters, replace=False)].copy()
        centers = [X[rng.integers(n)]]
        for _ in range(1, self.n_clusters):
            d2 = np.min(pairwise_distances(X, np.array(centers)) ** 2, axis=1)
            centers.append(X[rng.choice(n, p=d2 / d2.sum())])  # xa tâm cũ → xác suất được chọn cao
        return np.array(centers)

    def _run(self, X, rng):
        centers = self._init_centers(X, rng)
        history = [centers.copy()]
        for _ in range(self.max_iter):
            labels = np.argmin(pairwise_distances(X, centers), axis=1)
            new = np.array([X[labels == k].mean(0) if np.any(labels == k) else centers[k]
                            for k in range(self.n_clusters)])
            shift = np.linalg.norm(new - centers)
            centers = new
            history.append(centers.copy())
            if shift < self.tol:
                break
        labels = np.argmin(pairwise_distances(X, centers), axis=1)
        inertia = float(np.sum((X - centers[labels]) ** 2))
        return centers, labels, inertia, len(history) - 1, history  # mỗi vòng lặp thêm đúng một phần tử

    def fit(self, X):
        X = np.asarray(X, float)
        rng = np.random.default_rng(self.seed)
        best = None
        for _ in range(self.n_init):
            run = self._run(X, rng)
            if best is None or run[2] < best[2]:
                best = run
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_, self.history_ = best
        return self

    def predict(self, X):
        return np.argmin(pairwise_distances(np.asarray(X, float), self.cluster_centers_), axis=1)

    def fit_predict(self, X):
        return self.fit(X).labels_


class DBSCANScratch:
    """Core point: có ≥ ``min_samples`` điểm (kể cả chính nó) trong bán kính ``eps``. Noise = −1."""

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps, self.min_samples = eps, min_samples

    def fit(self, X):
        X = np.asarray(X, float)
        n = len(X)
        D = pairwise_distances(X, X)
        neighbors = [np.flatnonzero(D[i] <= self.eps) for i in range(n)]
        core = np.array([len(nb) >= self.min_samples for nb in neighbors])
        labels = np.full(n, -1)
        cluster = 0
        for i in range(n):
            if labels[i] != -1 or not core[i]:
                continue
            labels[i] = cluster
            queue = deque(neighbors[i])
            while queue:  # BFS mở rộng cụm từ các core point
                j = queue.popleft()
                if labels[j] == -1:
                    labels[j] = cluster
                    if core[j]:
                        queue.extend(neighbors[j])
            cluster += 1
        self.labels_, self.core_sample_indices_ = labels, np.flatnonzero(core)
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_
