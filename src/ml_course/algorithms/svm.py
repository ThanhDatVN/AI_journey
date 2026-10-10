"""SVM from scratch: Linear SVM (subgradient trên hinge loss) và Kernel SVM (simplified SMO)."""

from __future__ import annotations

import numpy as np


class LinearSVMScratch:
    """min ½‖w‖² + C Σ max(0, 1 − yᵢ(wᵀxᵢ + b)), yᵢ ∈ {−1, +1}. Subgradient descent (full batch)."""

    def __init__(self, C: float = 1.0, lr: float = 0.01, n_iters: int = 2000):
        self.C, self.lr, self.n_iters = C, lr, n_iters

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        ys = np.where(y == self.classes_[1], 1.0, -1.0)
        n, d = X.shape
        self.coef_, self.intercept_ = np.zeros(d), 0.0
        self.loss_history_ = []
        for t in range(self.n_iters):
            margins = ys * (X @ self.coef_ + self.intercept_)
            viol = margins < 1
            loss = 0.5 * self.coef_ @ self.coef_ + self.C * np.maximum(0, 1 - margins).sum()
            self.loss_history_.append(float(loss))
            grad_w = self.coef_ - self.C * (ys[viol, None] * X[viol]).sum(0)
            grad_b = -self.C * ys[viol].sum()
            step = self.lr / np.sqrt(1 + t / 100)
            self.coef_ -= step * grad_w / n
            self.intercept_ -= step * grad_b / n
        return self

    def decision_function(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])


def rbf_kernel(A, B, gamma: float = 1.0):
    sq = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-gamma * np.maximum(sq, 0))


def poly_kernel(A, B, degree: int = 3, coef0: float = 1.0, gamma: float = 1.0):
    return (gamma * A @ B.T + coef0) ** degree


class KernelSVMScratch:
    """Bài toán đối ngẫu giải bằng *simplified SMO* (Platt / CS229). Phù hợp n ≤ vài nghìn."""

    def __init__(self, C: float = 1.0, kernel: str = "rbf", gamma: float = 1.0, degree: int = 3,
                 tol: float = 1e-3, max_passes: int = 10, seed: int = 0):
        self.C, self.kernel, self.gamma, self.degree = C, kernel, gamma, degree
        self.tol, self.max_passes, self.seed = tol, max_passes, seed

    def _K(self, A, B):
        if self.kernel == "rbf":
            return rbf_kernel(A, B, self.gamma)
        if self.kernel == "poly":
            return poly_kernel(A, B, self.degree, gamma=self.gamma)
        return A @ B.T

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        ys = np.where(y == self.classes_[1], 1.0, -1.0)
        n = len(ys)
        K = self._K(X, X)
        alpha, b = np.zeros(n), 0.0
        rng = np.random.default_rng(self.seed)
        passes, it = 0, 0
        while passes < self.max_passes and it < 200:
            changed = 0
            for i in range(n):
                Ei = (alpha * ys) @ K[:, i] + b - ys[i]
                if (ys[i] * Ei < -self.tol and alpha[i] < self.C) or (ys[i] * Ei > self.tol and alpha[i] > 0):
                    j = int(rng.integers(n - 1))
                    j = j + 1 if j >= i else j
                    Ej = (alpha * ys) @ K[:, j] + b - ys[j]
                    ai, aj = alpha[i], alpha[j]
                    if ys[i] != ys[j]:
                        L, H = max(0, aj - ai), min(self.C, self.C + aj - ai)
                    else:
                        L, H = max(0, ai + aj - self.C), min(self.C, ai + aj)
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if L == H or eta >= 0:
                        continue
                    alpha[j] = np.clip(aj - ys[j] * (Ei - Ej) / eta, L, H)
                    if abs(alpha[j] - aj) < 1e-5:
                        continue
                    alpha[i] = ai + ys[i] * ys[j] * (aj - alpha[j])
                    b1 = b - Ei - ys[i] * (alpha[i] - ai) * K[i, i] - ys[j] * (alpha[j] - aj) * K[i, j]
                    b2 = b - Ej - ys[i] * (alpha[i] - ai) * K[i, j] - ys[j] * (alpha[j] - aj) * K[j, j]
                    b = b1 if 0 < alpha[i] < self.C else (b2 if 0 < alpha[j] < self.C else (b1 + b2) / 2)
                    changed += 1
            passes = passes + 1 if changed == 0 else 0
            it += 1
        sv = alpha > 1e-6
        self.support_ = np.flatnonzero(sv)
        self.support_vectors_, self.dual_coef_ = X[sv], (alpha * ys)[sv]
        self.intercept_ = b
        return self

    def decision_function(self, X):
        return self._K(np.asarray(X, float), self.support_vectors_) @ self.dual_coef_ + self.intercept_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])
