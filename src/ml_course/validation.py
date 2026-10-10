"""Chiến lược validation viết from scratch (chương 15) + learning curve + bootstrap."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np


class KFoldScratch:
    def __init__(self, n_splits: int = 5, shuffle: bool = True, seed: int | None = 0):
        self.n_splits, self.shuffle, self.seed = n_splits, shuffle, seed

    def split(self, X, y=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        idx = np.random.default_rng(self.seed).permutation(n) if self.shuffle else np.arange(n)
        fold_sizes = np.full(self.n_splits, n // self.n_splits)
        fold_sizes[: n % self.n_splits] += 1
        start = 0
        for size in fold_sizes:
            val = idx[start : start + size]
            train = np.concatenate([idx[:start], idx[start + size :]])
            yield train, val
            start += size


class StratifiedKFoldScratch:
    """Chia từng lớp riêng thành k phần rồi ghép lại → mỗi fold giữ tỉ lệ lớp."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, seed: int | None = 0):
        self.n_splits, self.shuffle, self.seed = n_splits, shuffle, seed

    def split(self, X, y) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        y = np.asarray(y)
        rng = np.random.default_rng(self.seed)
        folds: list[list[int]] = [[] for _ in range(self.n_splits)]
        for label in np.unique(y):
            idx = np.flatnonzero(y == label)
            if self.shuffle:
                rng.shuffle(idx)
            for i, chunk in enumerate(np.array_split(idx, self.n_splits)):
                folds[i].extend(chunk.tolist())
        all_idx = np.arange(len(y))
        for fold in folds:
            val = np.array(sorted(fold))
            yield np.setdiff1d(all_idx, val), val


class TimeSeriesSplitScratch:
    """Expanding window: train luôn nằm *trước* validation theo thời gian."""

    def __init__(self, n_splits: int = 5, test_size: int | None = None, gap: int = 0):
        self.n_splits, self.test_size, self.gap = n_splits, test_size, gap

    def split(self, X, y=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        test_size = self.test_size or n // (self.n_splits + 1)
        for i in range(self.n_splits):
            val_start = n - (self.n_splits - i) * test_size
            train_end = val_start - self.gap
            yield np.arange(0, train_end), np.arange(val_start, val_start + test_size)


class GroupKFoldScratch:
    """Mọi mẫu của cùng một group (vd: cùng bệnh nhân) nằm trọn trong một fold."""

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        groups = np.asarray(groups)
        unique, counts = np.unique(groups, return_counts=True)
        order = np.argsort(-counts)  # nhét group lớn trước vào fold đang nhẹ nhất
        fold_of_group, load = {}, np.zeros(self.n_splits)
        for g in unique[order]:
            f = int(np.argmin(load))
            fold_of_group[g] = f
            load[f] += np.sum(groups == g)
        fold_ids = np.array([fold_of_group[g] for g in groups])
        for f in range(self.n_splits):
            yield np.flatnonzero(fold_ids != f), np.flatnonzero(fold_ids == f)


def cross_val_score_scratch(make_model: Callable, X, y, scorer: Callable, cv=None) -> np.ndarray:
    """``make_model()`` trả về model *mới* mỗi fold (tránh rò rỉ trạng thái giữa các fold)."""
    X, y = np.asarray(X), np.asarray(y)
    cv = cv or KFoldScratch(5)
    scores = []
    for train, val in cv.split(X, y):
        model = make_model().fit(X[train], y[train])
        scores.append(scorer(y[val], model.predict(X[val])))
    return np.array(scores)


def learning_curve_data(
    make_model: Callable, X_train, y_train, X_val, y_val, scorer: Callable, sizes=None, seed: int = 0
) -> dict[str, np.ndarray]:
    """Train trên m mẫu đầu (m tăng dần) → điểm train & validation. Dùng để chẩn đoán bias/variance."""
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    n = len(X_train)
    sizes = np.unique(np.linspace(max(5, n // 20), n, 10).astype(int)) if sizes is None else np.asarray(sizes)
    perm = np.random.default_rng(seed).permutation(n)
    train_scores, val_scores = [], []
    for m in sizes:
        idx = perm[:m]
        model = make_model().fit(X_train[idx], y_train[idx])
        train_scores.append(scorer(y_train[idx], model.predict(X_train[idx])))
        val_scores.append(scorer(y_val, model.predict(X_val)))
    return {"sizes": sizes, "train": np.array(train_scores), "val": np.array(val_scores)}


def bootstrap_ci(
    values, statistic: Callable = np.mean, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Khoảng tin cậy percentile bootstrap: trả về (estimate, low, high)."""
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    stats = np.array([statistic(values[rng.integers(0, len(values), len(values))]) for _ in range(n_boot)])
    low, high = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(statistic(values)), float(low), float(high)


__all__ = [
    "KFoldScratch",
    "StratifiedKFoldScratch",
    "TimeSeriesSplitScratch",
    "GroupKFoldScratch",
    "cross_val_score_scratch",
    "learning_curve_data",
    "bootstrap_ci",
]
