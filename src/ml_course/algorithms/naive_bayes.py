"""Naive Bayes from scratch — tính toán trong không gian log để tránh underflow."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


class GaussianNBScratch:
    """P(x_j | y=c) = N(μ_cj, σ²_cj). ``var_smoothing`` giống sklearn (cộng ε·max var)."""

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        eps = self.var_smoothing * X.var(axis=0).max()
        self.theta_ = np.array([X[y == c].mean(0) for c in self.classes_])
        self.var_ = np.array([X[y == c].var(0) for c in self.classes_]) + eps
        self.class_log_prior_ = np.log(np.array([np.mean(y == c) for c in self.classes_]))
        return self

    def joint_log_likelihood(self, X):
        X = np.asarray(X, float)
        ll = -0.5 * (np.log(2 * np.pi * self.var_)[None].sum(-1)
                     + (((X[:, None, :] - self.theta_[None]) ** 2) / self.var_[None]).sum(-1))
        return ll + self.class_log_prior_

    def predict_log_proba(self, X):
        jll = self.joint_log_likelihood(X)
        return jll - logsumexp(jll, axis=1, keepdims=True)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.joint_log_likelihood(X), axis=1)]


class MultinomialNBScratch:
    """Cho dữ liệu đếm (bag-of-words). Laplace/Lidstone smoothing với ``alpha``."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        counts = np.array([X[y == c].sum(0) for c in self.classes_]) + self.alpha
        self.feature_log_prob_ = np.log(counts / counts.sum(1, keepdims=True))
        self.class_log_prior_ = np.log(np.array([np.mean(y == c) for c in self.classes_]))
        return self

    def joint_log_likelihood(self, X):
        return np.asarray(X, float) @ self.feature_log_prob_.T + self.class_log_prior_

    def predict_proba(self, X):
        jll = self.joint_log_likelihood(X)
        return np.exp(jll - logsumexp(jll, axis=1, keepdims=True))

    def predict(self, X):
        return self.classes_[np.argmax(self.joint_log_likelihood(X), axis=1)]
