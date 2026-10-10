"""Linear models from scratch: Linear/Ridge/Lasso regression, Logistic & Softmax regression."""

from __future__ import annotations

import numpy as np

from ..utils import sigmoid, softmax


class LinearRegressionScratch:
    """ŷ = Xw + b.

    ``method="normal"``: nghiệm đóng w = (XᵀX)⁻¹Xᵀy (dùng lstsq cho ổn định số học).
    ``method="gd"``: batch gradient descent trên MSE; lưu ``loss_history_``.
    """

    def __init__(self, method: str = "normal", lr: float = 0.01, n_iters: int = 1000):
        self.method, self.lr, self.n_iters = method, lr, n_iters

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        n, d = X.shape
        if self.method == "normal":
            Xb = np.hstack([np.ones((n, 1)), X])
            theta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
            self.intercept_, self.coef_ = theta[0], theta[1:]
            return self
        self.coef_, self.intercept_ = np.zeros(d), 0.0
        self.loss_history_ = []
        for _ in range(self.n_iters):
            err = X @ self.coef_ + self.intercept_ - y
            self.loss_history_.append(float(np.mean(err**2)))
            self.coef_ -= self.lr * (2 / n) * (X.T @ err)
            self.intercept_ -= self.lr * (2 / n) * err.sum()
        return self

    def predict(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_


class RidgeScratch:
    """Ridge: min ‖y − Xw − b‖² + α‖w‖² (không phạt bias). Nghiệm đóng trên dữ liệu đã center."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - x_mean, y - y_mean
        d = X.shape[1]
        self.coef_ = np.linalg.solve(Xc.T @ Xc + self.alpha * np.eye(d), Xc.T @ yc)
        self.intercept_ = y_mean - x_mean @ self.coef_
        return self

    def predict(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_


class LassoScratch:
    """Lasso bằng coordinate descent: min (1/2n)‖y − Xw − b‖² + α‖w‖₁ (giống sklearn).

    Mỗi bước cập nhật một w_j bằng soft-thresholding → nhiều w_j bằng đúng 0 (feature selection).
    """

    def __init__(self, alpha: float = 1.0, n_iters: int = 1000, tol: float = 1e-6):
        self.alpha, self.n_iters, self.tol = alpha, n_iters, tol

    @staticmethod
    def soft_threshold(rho: float, lam: float) -> float:
        return np.sign(rho) * max(abs(rho) - lam, 0.0)

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        n, d = X.shape
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - x_mean, y - y_mean
        col_sq = (Xc**2).sum(axis=0) / n
        w = np.zeros(d)
        resid = yc.copy()
        for _ in range(self.n_iters):
            w_max_change = 0.0
            for j in range(d):
                if col_sq[j] == 0:
                    continue
                resid += Xc[:, j] * w[j]
                rho = Xc[:, j] @ resid / n
                new = self.soft_threshold(rho, self.alpha) / col_sq[j]
                w_max_change = max(w_max_change, abs(new - w[j]))
                w[j] = new
                resid -= Xc[:, j] * w[j]
            if w_max_change < self.tol:
                break
        self.coef_ = w
        self.intercept_ = y_mean - x_mean @ w
        return self

    def predict(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_


class LogisticRegressionScratch:
    """Binary logistic regression: P(y=1|x) = σ(wᵀx + b), tối ưu BCE + (λ/2n)‖w‖² bằng batch GD."""

    def __init__(self, lr: float = 0.1, n_iters: int = 2000, l2: float = 0.0, tol: float = 1e-8):
        self.lr, self.n_iters, self.l2, self.tol = lr, n_iters, l2, tol

    def _loss(self, X, y):
        p = np.clip(sigmoid(X @ self.coef_ + self.intercept_), 1e-15, 1 - 1e-15)
        n = len(y)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + self.l2 / (2 * n) * self.coef_ @ self.coef_)

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        n, d = X.shape
        self.coef_, self.intercept_ = np.zeros(d), 0.0
        self.loss_history_ = []
        prev = np.inf
        for _ in range(self.n_iters):
            p = sigmoid(X @ self.coef_ + self.intercept_)
            grad_w = X.T @ (p - y) / n + self.l2 / n * self.coef_
            grad_b = np.mean(p - y)
            self.coef_ -= self.lr * grad_w
            self.intercept_ -= self.lr * grad_b
            loss = self._loss(X, y)
            self.loss_history_.append(loss)
            if abs(prev - loss) < self.tol:
                break
            prev = loss
        return self

    def decision_function(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class SoftmaxRegressionScratch:
    """Multinomial logistic regression: P(y=k|x) = softmax(xW + b)_k, tối ưu cross-entropy bằng GD."""

    def __init__(self, lr: float = 0.5, n_iters: int = 1000, l2: float = 0.0):
        self.lr, self.n_iters, self.l2 = lr, n_iters, l2

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        n, d = X.shape
        self.classes_ = np.unique(y)
        idx = np.searchsorted(self.classes_, y)
        K = len(self.classes_)
        Y = np.zeros((n, K))
        Y[np.arange(n), idx] = 1
        self.W_, self.b_ = np.zeros((d, K)), np.zeros(K)
        self.loss_history_ = []
        for _ in range(self.n_iters):
            P = softmax(X @ self.W_ + self.b_)
            self.loss_history_.append(float(-np.mean(np.log(P[np.arange(n), idx] + 1e-15))))
            G = (P - Y) / n
            self.W_ -= self.lr * (X.T @ G + self.l2 / n * self.W_)
            self.b_ -= self.lr * G.sum(axis=0)
        return self

    def predict_proba(self, X):
        return softmax(np.asarray(X, float) @ self.W_ + self.b_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
