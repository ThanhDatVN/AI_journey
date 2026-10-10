"""Multi-layer perceptron from scratch (NumPy): forward, backprop, mini-batch SGD/Adam."""

from __future__ import annotations

import numpy as np

from ..utils import softmax

ACTIVATIONS = {
    "relu": (lambda z: np.maximum(0, z), lambda z, a: (z > 0).astype(float)),
    "tanh": (np.tanh, lambda z, a: 1 - a**2),
    "sigmoid": (lambda z: 1 / (1 + np.exp(-z)), lambda z, a: a * (1 - a)),
}


class MLPScratch:
    """Classifier: [Linear → activation] × L → Linear → softmax, loss cross-entropy.

    Khởi tạo He (ReLU) / Xavier (tanh, sigmoid). Lưu ``history_`` = loss mỗi epoch.
    """

    def __init__(self, hidden=(32,), activation: str = "relu", lr: float = 0.01, epochs: int = 100,
                 batch_size: int = 32, optimizer: str = "adam", l2: float = 0.0, seed: int = 0):
        self.hidden, self.activation, self.lr, self.epochs = tuple(hidden), activation, lr, epochs
        self.batch_size, self.optimizer, self.l2, self.seed = batch_size, optimizer, l2, seed

    def _init(self, d, K, rng):
        sizes = [d, *self.hidden, K]
        self.params_ = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            scale = np.sqrt(2 / fan_in) if self.activation == "relu" else np.sqrt(1 / fan_in)
            self.params_.append([rng.normal(0, scale, (fan_in, fan_out)), np.zeros(fan_out)])
        self._m = [[np.zeros_like(W), np.zeros_like(b)] for W, b in self.params_]
        self._v = [[np.zeros_like(W), np.zeros_like(b)] for W, b in self.params_]
        self._t = 0

    def _forward(self, X):
        act, _ = ACTIVATIONS[self.activation]
        cache, a = [], X
        for i, (W, b) in enumerate(self.params_):
            z = a @ W + b
            cache.append((a, z))
            a = softmax(z) if i == len(self.params_) - 1 else act(z)
        return a, cache

    def _backward(self, P, Y, cache):
        _, dact = ACTIVATIONS[self.activation]
        n = len(Y)
        grads = [None] * len(self.params_)
        delta = (P - Y) / n  # ∂L/∂z của lớp cuối (softmax + CE)
        for i in reversed(range(len(self.params_))):
            a_prev, _ = cache[i]
            W, _ = self.params_[i]
            grads[i] = [a_prev.T @ delta + self.l2 * W, delta.sum(0)]
            if i > 0:
                _, z_prev = cache[i - 1]
                a_act = ACTIVATIONS[self.activation][0](z_prev)
                delta = (delta @ W.T) * dact(z_prev, a_act)
        return grads

    def _step(self, grads):
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for (p, g, m, v) in zip(self.params_, grads, self._m, self._v):
            for j in range(2):
                if self.optimizer == "sgd":
                    p[j] -= self.lr * g[j]
                else:
                    m[j] = b1 * m[j] + (1 - b1) * g[j]
                    v[j] = b2 * v[j] + (1 - b2) * g[j] ** 2
                    m_hat, v_hat = m[j] / (1 - b1**self._t), v[j] / (1 - b2**self._t)
                    p[j] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        rng = np.random.default_rng(self.seed)
        self.classes_ = np.unique(y)
        idx = np.searchsorted(self.classes_, y)
        Y = np.eye(len(self.classes_))[idx]
        self._init(X.shape[1], len(self.classes_), rng)
        self.history_ = []
        for _ in range(self.epochs):
            perm = rng.permutation(len(X))
            for s in range(0, len(X), self.batch_size):
                bidx = perm[s : s + self.batch_size]
                P, cache = self._forward(X[bidx])
                self._step(self._backward(P, Y[bidx], cache))
            P, _ = self._forward(X)
            self.history_.append(float(-np.mean(np.log(P[np.arange(len(X)), idx] + 1e-12))))
        return self

    def predict_proba(self, X):
        return self._forward(np.asarray(X, float))[0]

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
