"""Ensemble from scratch: Random Forest (bagging), AdaBoost, Gradient Boosting."""

from __future__ import annotations

import numpy as np

from ..utils import sigmoid
from .tree import DecisionTreeScratch


class RandomForestScratch:
    """Bagging + random feature subset ở mỗi split → giảm tương quan giữa các cây → giảm variance."""

    def __init__(
        self,
        n_estimators: int = 100,
        task: str = "classification",
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        max_features: int | float | str | None = "sqrt",
        bootstrap: bool = True,
        oob_score: bool = False,
        seed: int = 0,
    ):
        self.n_estimators, self.task = n_estimators, task
        self.max_depth, self.min_samples_leaf, self.max_features = max_depth, min_samples_leaf, max_features
        self.bootstrap, self.oob_score, self.seed = bootstrap, oob_score, seed

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        n = len(y)
        rng = np.random.default_rng(self.seed)
        self.estimators_, self.samples_ = [], []
        if self.task == "classification":
            self.classes_ = np.unique(y)
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, n) if self.bootstrap else np.arange(n)
            tree = DecisionTreeScratch(
                task=self.task, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features, seed=int(rng.integers(1 << 31)),
            ).fit(X[idx], y[idx])
            self.estimators_.append(tree)
            self.samples_.append(idx)
        self.feature_importances_ = np.mean([t.feature_importances_ for t in self.estimators_], axis=0)
        if self.oob_score and self.bootstrap:
            self.oob_score_ = self._compute_oob(X, y)
        return self

    def _tree_proba(self, tree, X):
        """Xác suất theo đúng thứ tự self.classes_ (cây có thể thiếu lớp trong bootstrap)."""
        p = tree.predict_proba(X)
        out = np.zeros((len(X), len(self.classes_)))
        out[:, np.searchsorted(self.classes_, tree.classes_)] = p
        return out

    def _compute_oob(self, X, y):
        n = len(y)
        if self.task == "classification":
            votes = np.zeros((n, len(self.classes_)))
        else:
            sums, counts = np.zeros(n), np.zeros(n)
        for tree, idx in zip(self.estimators_, self.samples_):
            oob = np.setdiff1d(np.arange(n), idx)
            if oob.size == 0:
                continue
            if self.task == "classification":
                votes[oob] += self._tree_proba(tree, X[oob])
            else:
                sums[oob] += tree.predict(X[oob])
                counts[oob] += 1
        if self.task == "classification":
            has = votes.sum(1) > 0
            return float(np.mean(self.classes_[np.argmax(votes[has], 1)] == y[has]))
        has = counts > 0
        pred = sums[has] / counts[has]
        return float(1 - np.sum((y[has] - pred) ** 2) / np.sum((y[has] - y[has].mean()) ** 2))

    def predict_proba(self, X):
        X = np.asarray(X, float)
        return np.mean([self._tree_proba(t, X) for t in self.estimators_], axis=0)

    def predict(self, X):
        X = np.asarray(X, float)
        if self.task == "classification":
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        return np.mean([t.predict(X) for t in self.estimators_], axis=0)


class AdaBoostScratch:
    """AdaBoost (SAMME, 2 lớp) với decision stump. Nhãn nội bộ ∈ {−1, +1}."""

    def __init__(self, n_estimators: int = 50, max_depth: int = 1, learning_rate: float = 1.0):
        self.n_estimators, self.max_depth, self.learning_rate = n_estimators, max_depth, learning_rate

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        assert len(self.classes_) == 2, "AdaBoostScratch chỉ hỗ trợ 2 lớp"
        ys = np.where(y == self.classes_[1], 1, -1)
        n = len(y)
        w = np.full(n, 1 / n)
        rng = np.random.default_rng(0)
        self.estimators_, self.alphas_, self.errors_ = [], [], []
        for _ in range(self.n_estimators):
            # Cây scratch không nhận sample_weight → resample theo trọng số (boosting by resampling)
            idx = rng.choice(n, n, p=w)
            stump = DecisionTreeScratch(max_depth=self.max_depth).fit(X[idx], ys[idx])
            pred = stump.predict(X)
            err = np.clip(np.sum(w * (pred != ys)), 1e-10, 1 - 1e-10)
            if err >= 0.5:
                continue
            alpha = self.learning_rate * 0.5 * np.log((1 - err) / err)
            w = w * np.exp(-alpha * ys * pred)
            w /= w.sum()
            self.estimators_.append(stump)
            self.alphas_.append(alpha)
            self.errors_.append(err)
        return self

    def decision_function(self, X):
        X = np.asarray(X, float)
        return sum(a * est.predict(X) for a, est in zip(self.alphas_, self.estimators_))

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])


class GradientBoostingRegressorScratch:
    """F_m(x) = F_{m−1}(x) + η·h_m(x), với h_m fit vào *negative gradient* (= residual cho MSE)."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3,
                 subsample: float = 1.0, seed: int = 0):
        self.n_estimators, self.learning_rate, self.max_depth = n_estimators, learning_rate, max_depth
        self.subsample, self.seed = subsample, seed

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        rng = np.random.default_rng(self.seed)
        n = len(y)
        self.init_ = float(y.mean())
        F = np.full(n, self.init_)
        self.estimators_, self.train_loss_ = [], []
        for _ in range(self.n_estimators):
            residual = y - F  # = −∂L/∂F với L = ½(y − F)²
            idx = rng.choice(n, int(self.subsample * n), replace=False) if self.subsample < 1 else np.arange(n)
            tree = DecisionTreeScratch(task="regression", max_depth=self.max_depth).fit(X[idx], residual[idx])
            F += self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)
            self.train_loss_.append(float(np.mean((y - F) ** 2)))
        return self

    def staged_predict(self, X):
        X = np.asarray(X, float)
        F = np.full(len(X), self.init_)
        for tree in self.estimators_:
            F = F + self.learning_rate * tree.predict(X)
            yield F.copy()

    def predict(self, X):
        X = np.asarray(X, float)
        return self.init_ + self.learning_rate * sum(t.predict(X) for t in self.estimators_)


class GradientBoostingClassifierScratch:
    """Binary log-loss: F là log-odds; negative gradient = y − σ(F); leaf value bước Newton."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3):
        self.n_estimators, self.learning_rate, self.max_depth = n_estimators, learning_rate, max_depth

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        yb = (y == self.classes_[1]).astype(float)
        p0 = np.clip(yb.mean(), 1e-6, 1 - 1e-6)
        self.init_ = float(np.log(p0 / (1 - p0)))
        F = np.full(len(y), self.init_)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            p = sigmoid(F)
            residual = yb - p
            tree = DecisionTreeScratch(task="regression", max_depth=self.max_depth).fit(X, residual)
            # Thay giá trị lá bằng bước Newton: Σ residual / Σ p(1−p) trên các mẫu của lá
            leaves = [tree._leaf_for(x) for x in X]
            groups: dict[int, list[int]] = {}
            for i, leaf in enumerate(leaves):
                groups.setdefault(id(leaf), []).append(i)
            for members in groups.values():
                m = np.array(members)
                leaves[m[0]].value = float(residual[m].sum() / max(np.sum(p[m] * (1 - p[m])), 1e-12))
            F += self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)
        return self

    def decision_function(self, X):
        X = np.asarray(X, float)
        return self.init_ + self.learning_rate * sum(t.predict(X) for t in self.estimators_)

    def predict_proba(self, X):
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return np.where(self.predict_proba(X)[:, 1] >= 0.5, self.classes_[1], self.classes_[0])
