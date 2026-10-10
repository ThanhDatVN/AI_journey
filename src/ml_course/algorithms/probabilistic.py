"""Mô hình xác suất from scratch: Gaussian Mixture Model huấn luyện bằng EM."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


class GaussianMixtureScratch:
    """EM cho GMM full covariance.

    E-step: responsibility r_ik = π_k N(x_i|μ_k,Σ_k) / Σ_j π_j N(x_i|μ_j,Σ_j)
    M-step: cập nhật π, μ, Σ bằng trung bình có trọng số r_ik.
    """

    def __init__(self, n_components: int = 3, max_iter: int = 200, tol: float = 1e-6,
                 reg_covar: float = 1e-6, seed: int = 0):
        self.n_components, self.max_iter, self.tol = n_components, max_iter, tol
        self.reg_covar, self.seed = reg_covar, seed

    def _log_prob(self, X):
        return np.column_stack([
            np.log(self.weights_[k]) + multivariate_normal(self.means_[k], self.covariances_[k]).logpdf(X)
            for k in range(self.n_components)
        ])

    def fit(self, X):
        X = np.asarray(X, float)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        self.means_ = X[rng.choice(n, self.n_components, replace=False)]
        self.covariances_ = np.array([np.cov(X.T) + self.reg_covar * np.eye(d)] * self.n_components)
        self.weights_ = np.full(self.n_components, 1 / self.n_components)
        self.log_likelihood_history_ = []
        prev = -np.inf
        for _ in range(self.max_iter):
            lp = self._log_prob(X)
            ll = logsumexp(lp, axis=1)
            R = np.exp(lp - ll[:, None])  # E-step
            Nk = R.sum(0) + 1e-12  # M-step
            self.weights_ = Nk / n
            self.means_ = (R.T @ X) / Nk[:, None]
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_[k] = (R[:, k, None] * diff).T @ diff / Nk[k] + self.reg_covar * np.eye(d)
            total = float(ll.sum())
            self.log_likelihood_history_.append(total)
            if abs(total - prev) < self.tol * abs(total):
                break
            prev = total
        return self

    def predict_proba(self, X):
        lp = self._log_prob(np.asarray(X, float))
        return np.exp(lp - logsumexp(lp, axis=1, keepdims=True))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score_samples(self, X):
        return logsumexp(self._log_prob(np.asarray(X, float)), axis=1)
