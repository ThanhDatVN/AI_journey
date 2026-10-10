"""Metric viết from scratch bằng NumPy (được đối chiếu với scikit-learn trong ``tests/``).

Đọc code ở đây để hiểu *chính xác* mỗi metric tính gì — đặc biệt là các trường hợp biên
(chia cho 0, threshold trùng nhau, ...).
"""

from __future__ import annotations

import numpy as np

# ============================================================================ regression


def mean_squared_error(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred) -> float:
    """R² = 1 − SS_res / SS_tot. Có thể âm nếu model tệ hơn dự đoán bằng trung bình."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def mean_absolute_percentage_error(y_true, y_pred, eps: float = 1e-12) -> float:
    """MAPE (dạng tỉ lệ, 0.1 = 10%). Bùng nổ khi y_true ≈ 0 — xem chương 14."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), eps)))


# ============================================================================ classification


def accuracy_score(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    """Hàng = nhãn thật, cột = nhãn dự đoán (giống sklearn)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred])) if labels is None else np.asarray(labels)
    index = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            cm[index[t], index[p]] += 1
    return cm


def _binary_counts(y_true, y_pred, positive=1) -> tuple[int, int, int, int]:
    y_true, y_pred = np.asarray(y_true) == positive, np.asarray(y_pred) == positive
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    return tp, fp, fn, tn


def precision_score(y_true, y_pred, positive=1) -> float:
    tp, fp, _, _ = _binary_counts(y_true, y_pred, positive)
    return tp / (tp + fp) if tp + fp else 0.0


def recall_score(y_true, y_pred, positive=1) -> float:
    tp, _, fn, _ = _binary_counts(y_true, y_pred, positive)
    return tp / (tp + fn) if tp + fn else 0.0


def specificity_score(y_true, y_pred, positive=1) -> float:
    _, fp, _, tn = _binary_counts(y_true, y_pred, positive)
    return tn / (tn + fp) if tn + fp else 0.0


def f1_score(y_true, y_pred, positive=1) -> float:
    p, r = precision_score(y_true, y_pred, positive), recall_score(y_true, y_pred, positive)
    return 2 * p * r / (p + r) if p + r else 0.0


def fbeta_score(y_true, y_pred, beta: float, positive=1) -> float:
    p, r = precision_score(y_true, y_pred, positive), recall_score(y_true, y_pred, positive)
    b2 = beta**2
    return (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) else 0.0


def roc_curve(y_true, scores) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trả về (fpr, tpr, thresholds). Duyệt threshold từ cao xuống thấp; gộp các score trùng nhau."""
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, float)
    order = np.argsort(-scores, kind="mergesort")
    scores, y_sorted = scores[order], y_true[order]
    distinct = np.where(np.diff(scores))[0]
    idx = np.r_[distinct, y_sorted.size - 1]
    tps = np.cumsum(y_sorted)[idx]
    fps = (idx + 1) - tps
    tps, fps = np.r_[0, tps], np.r_[0, fps]
    thresholds = np.r_[np.inf, scores[idx]]
    P, N = y_true.sum(), (~y_true).sum()
    tpr = tps / P if P else np.zeros_like(tps, dtype=float)
    fpr = fps / N if N else np.zeros_like(fps, dtype=float)
    return fpr.astype(float), tpr.astype(float), thresholds


def auc(x, y) -> float:
    """Diện tích dưới đường cong bằng quy tắc hình thang."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))


def roc_auc_score(y_true, scores) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    return auc(fpr, tpr)


def precision_recall_curve(y_true, scores) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trả về (precision, recall, thresholds), sắp theo threshold giảm dần (recall tăng dần)."""
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, float)
    order = np.argsort(-scores, kind="mergesort")
    scores, y_sorted = scores[order], y_true[order]
    distinct = np.where(np.diff(scores))[0]
    idx = np.r_[distinct, y_sorted.size - 1]
    tps = np.cumsum(y_sorted)[idx]
    predicted_pos = idx + 1
    precision = tps / predicted_pos
    recall = tps / max(y_true.sum(), 1)
    return precision, recall, scores[idx]


def average_precision_score(y_true, scores) -> float:
    """AP = Σ (R_n − R_{n−1}) P_n — định nghĩa giống sklearn (không nội suy)."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def log_loss(y_true, proba, eps: float = 1e-15) -> float:
    """Binary cross-entropy (proba = P(y=1)) hoặc multiclass (proba shape (n, K))."""
    y_true = np.asarray(y_true)
    proba = np.clip(np.asarray(proba, float), eps, 1 - eps)
    if proba.ndim == 1:
        return float(-np.mean(y_true * np.log(proba) + (1 - y_true) * np.log(1 - proba)))
    proba = proba / proba.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(proba[np.arange(len(y_true)), y_true.astype(int)])))


def brier_score(y_true, proba) -> float:
    return float(np.mean((np.asarray(proba, float) - np.asarray(y_true, float)) ** 2))


# ============================================================================ ranking (recommender)


def precision_at_k(recommended, relevant, k: int) -> float:
    rec_k = list(recommended)[:k]
    relevant = set(relevant)
    return sum(item in relevant for item in rec_k) / k if k else 0.0


def recall_at_k(recommended, relevant, k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return sum(item in relevant for item in list(recommended)[:k]) / len(relevant)


def average_precision_at_k(recommended, relevant, k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits, score = 0, 0.0
    for i, item in enumerate(list(recommended)[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k)


def ndcg_at_k(recommended, relevance: dict, k: int) -> float:
    """``relevance``: dict item → độ liên quan (graded). Gain = 2^rel − 1, discount = log2(i+1)."""
    gains = [relevance.get(item, 0.0) for item in list(recommended)[:k]]
    dcg = sum((2**g - 1) / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / np.log2(i + 2) for i, g in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


__all__ = [name for name in dir() if not name.startswith("_") and name != "np" and name != "annotations"]
