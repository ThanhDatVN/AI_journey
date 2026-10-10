"""Decision Tree (CART) from scratch cho classification (gini/entropy) và regression (mse).

Thuật toán tham lam: tại mỗi node, thử mọi feature × mọi ngưỡng (trung điểm giữa 2 giá trị
liên tiếp), chọn split giảm impurity nhiều nhất. Tìm ngưỡng bằng sắp xếp + cumulative sum
→ O(n log n) mỗi feature thay vì O(n²).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def gini(y) -> float:
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(1 - np.sum(p**2))


def entropy(y) -> float:
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


@dataclass
class Node:
    value: np.ndarray | float | None = None  # phân phối lớp (classification) hoặc mean (regression)
    feature: int | None = None
    threshold: float | None = None
    left: Node | None = None
    right: Node | None = None
    n_samples: int = 0
    impurity: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class DecisionTreeScratch:
    def __init__(
        self,
        task: str = "classification",
        criterion: str | None = None,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | str | None = None,
        seed: int | None = 0,
    ):
        self.task = task
        self.criterion = criterion or ("gini" if task == "classification" else "mse")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.n_features_ = X.shape[1]
        if self.task == "classification":
            self.classes_ = np.unique(y)
            y_enc = np.searchsorted(self.classes_, y)
        else:
            y_enc = y.astype(float)
        self.feature_importances_ = np.zeros(self.n_features_)
        self.root_ = self._grow(X, y_enc, depth=0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _leaf_value(self, y):
        if self.task == "classification":
            return np.bincount(y, minlength=len(self.classes_)) / len(y)
        return float(np.mean(y))

    def _impurity(self, y) -> float:
        if self.task == "regression":
            return float(np.var(y))
        return entropy(y) if self.criterion == "entropy" else gini(y)

    def _n_features_to_try(self) -> int:
        d, mf = self.n_features_, self.max_features
        if mf is None:
            return d
        if mf == "sqrt":
            return max(1, int(np.sqrt(d)))
        if mf == "log2":
            return max(1, int(np.log2(d)))
        if isinstance(mf, float):
            return max(1, int(mf * d))
        return min(d, int(mf))

    def _grow(self, X, y, depth: int) -> Node:
        node = Node(value=self._leaf_value(y), n_samples=len(y), impurity=self._impurity(y))
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or len(y) < self.min_samples_split
            or node.impurity <= 1e-12
        ):
            return node
        feature, threshold, gain = self._best_split(X, y, node.impurity)
        if feature is None:
            return node
        mask = X[:, feature] <= threshold
        self.feature_importances_[feature] += gain * len(y)
        node.feature, node.threshold = feature, threshold
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def _best_split(self, X, y, parent_impurity):
        n = len(y)
        best = (None, None, 0.0)
        features = self.rng.choice(self.n_features_, self._n_features_to_try(), replace=False)
        leaf = self.min_samples_leaf
        for j in features:
            order = np.argsort(X[:, j], kind="mergesort")
            xs, ys = X[order, j], y[order]
            valid = np.flatnonzero(xs[1:] != xs[:-1])  # chỉ cắt giữa 2 giá trị khác nhau
            valid = valid[(valid + 1 >= leaf) & (n - valid - 1 >= leaf)]
            if valid.size == 0:
                continue
            n_left = valid + 1
            n_right = n - n_left
            if self.task == "classification":
                onehot = np.eye(len(self.classes_))[ys]
                left_counts = np.cumsum(onehot, axis=0)[valid]
                right_counts = onehot.sum(0) - left_counts
                pl = left_counts / n_left[:, None]
                pr = right_counts / n_right[:, None]
                if self.criterion == "entropy":
                    with np.errstate(divide="ignore", invalid="ignore"):
                        il = -np.nansum(np.where(pl > 0, pl * np.log2(pl), 0), axis=1)
                        ir = -np.nansum(np.where(pr > 0, pr * np.log2(pr), 0), axis=1)
                else:
                    il, ir = 1 - (pl**2).sum(1), 1 - (pr**2).sum(1)
            else:
                csum, csq = np.cumsum(ys), np.cumsum(ys**2)
                sl, sql = csum[valid], csq[valid]
                sr, sqr = csum[-1] - sl, csq[-1] - sql
                il = sql / n_left - (sl / n_left) ** 2
                ir = sqr / n_right - (sr / n_right) ** 2
            child = (n_left * il + n_right * ir) / n
            gains = parent_impurity - child
            k = int(np.argmax(gains))
            if gains[k] > best[2] + 1e-12:
                i = valid[k]
                best = (int(j), float((xs[i] + xs[i + 1]) / 2), float(gains[k]))
        return best

    # ------------------------------------------------------------------ predict
    def _leaf_for(self, x) -> Node:
        node = self.root_
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def predict_proba(self, X):
        return np.array([self._leaf_for(x).value for x in np.asarray(X, float)])

    def predict(self, X):
        X = np.asarray(X, float)
        if self.task == "classification":
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        return np.array([self._leaf_for(x).value for x in X])

    # ------------------------------------------------------------------ inspection
    def depth(self, node: Node | None = None) -> int:
        node = node or self.root_
        return 0 if node.is_leaf else 1 + max(self.depth(node.left), self.depth(node.right))

    def n_leaves(self, node: Node | None = None) -> int:
        node = node or self.root_
        return 1 if node.is_leaf else self.n_leaves(node.left) + self.n_leaves(node.right)

    def export_text(self, feature_names=None, node: Node | None = None, indent: str = "") -> str:
        node = node or self.root_
        names = feature_names or [f"x{j}" for j in range(self.n_features_)]
        if node.is_leaf:
            if self.task == "classification":
                return f"{indent}→ class {self.classes_[np.argmax(node.value)]} (n={node.n_samples})\n"
            return f"{indent}→ value {node.value:.3f} (n={node.n_samples})\n"
        text = f"{indent}if {names[node.feature]} <= {node.threshold:.3f}:\n"
        text += self.export_text(feature_names, node.left, indent + "    ")
        text += f"{indent}else:\n"
        text += self.export_text(feature_names, node.right, indent + "    ")
        return text
