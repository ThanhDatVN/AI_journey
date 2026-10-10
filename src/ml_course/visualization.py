"""Các hàm vẽ dùng lại nhiều lần trong khoá học.

Triết lý: *mỗi model phải nhìn thấy được* — regression line, decision boundary,
loss surface, learning curve, cluster, ...
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#be185d", "#4d7c0f"]


def set_style() -> None:
    """Style thống nhất cho toàn khoá học."""
    plt.rcParams.update(
        {
            "figure.figsize": (7, 4.5),
            "figure.dpi": 100,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": plt.cycler(color=PALETTE),
            "font.size": 11,
        }
    )


def _predict_fn(model_or_fn) -> Callable[[np.ndarray], np.ndarray]:
    if callable(model_or_fn) and not hasattr(model_or_fn, "predict"):
        return model_or_fn
    return model_or_fn.predict


def plot_decision_boundary(
    model_or_fn,
    X: np.ndarray,
    y: np.ndarray,
    ax=None,
    resolution: int = 250,
    proba: bool = False,
    title: str | None = None,
    padding: float = 0.5,
):
    """Vẽ vùng quyết định của một classifier 2D.

    ``model_or_fn``: model có ``predict`` (hoặc ``predict_proba`` nếu ``proba=True``) hay một hàm X → nhãn.
    """
    ax = ax or plt.gca()
    X = np.asarray(X)
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if proba and hasattr(model_or_fn, "predict_proba"):
        zz = model_or_fn.predict_proba(grid)[:, 1].reshape(xx.shape)
        cs = ax.contourf(xx, yy, zz, levels=20, cmap="RdBu_r", alpha=0.6, vmin=0, vmax=1)
        ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1.5)
        plt.colorbar(cs, ax=ax, label="P(y=1)")
    else:
        zz = np.asarray(_predict_fn(model_or_fn)(grid)).reshape(xx.shape)
        classes = np.unique(zz)
        ax.contourf(xx, yy, zz, levels=np.r_[classes - 0.5, classes.max() + 0.5], cmap="tab10", alpha=0.25)
        ax.contour(xx, yy, zz, levels=classes[:-1] + 0.5, colors="k", linewidths=1)
    for i, label in enumerate(np.unique(y)):
        mask = np.asarray(y) == label
        ax.scatter(X[mask, 0], X[mask, 1], s=18, edgecolor="white", linewidth=0.4,
                   color=PALETTE[i % len(PALETTE)], label=f"class {label}")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc="best", fontsize=9)
    if title:
        ax.set_title(title)
    return ax


def plot_regression_fit(X, y, model_or_fn, ax=None, title: str | None = None, n_points: int = 300):
    """Scatter dữ liệu 1D + đường dự đoán của model."""
    ax = ax or plt.gca()
    X = np.asarray(X).reshape(len(X), -1)
    grid = np.linspace(X[:, 0].min(), X[:, 0].max(), n_points).reshape(-1, 1)
    ax.scatter(X[:, 0], y, s=15, alpha=0.7, label="dữ liệu")
    ax.plot(grid[:, 0], _predict_fn(model_or_fn)(grid), color=PALETTE[1], lw=2, label="dự đoán")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    if title:
        ax.set_title(title)
    return ax


def plot_loss_contour(
    loss_fn: Callable[[float, float], float],
    w_range: tuple[float, float],
    b_range: tuple[float, float],
    path: Sequence[Sequence[float]] | None = None,
    ax=None,
    levels: int = 30,
    log: bool = True,
    labels: tuple[str, str] = ("w", "b"),
    resolution: int = 120,
):
    """Contour của loss theo 2 tham số + (tuỳ chọn) quỹ đạo của optimizer."""
    ax = ax or plt.gca()
    ws = np.linspace(*w_range, resolution)
    bs = np.linspace(*b_range, resolution)
    W, B = np.meshgrid(ws, bs)
    Z = np.vectorize(loss_fn)(W, B)
    Zp = np.log10(Z + 1e-12) if log else Z
    cs = ax.contourf(W, B, Zp, levels=levels, cmap="viridis")
    plt.colorbar(cs, ax=ax, label="log10(loss)" if log else "loss")
    if path is not None:
        path = np.asarray(path)
        ax.plot(path[:, 0], path[:, 1], "o-", color="white", ms=3, lw=1.2)
        ax.plot(path[-1, 0], path[-1, 1], "*", color="red", ms=14)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    return ax


def plot_learning_curve(sizes, train_scores, val_scores, ax=None, metric: str = "score", title: str | None = None):
    ax = ax or plt.gca()
    ax.plot(sizes, train_scores, "o-", label="train")
    ax.plot(sizes, val_scores, "s-", label="validation")
    ax.set_xlabel("số mẫu train (m)")
    ax.set_ylabel(metric)
    ax.legend()
    ax.set_title(title or "Learning curve")
    return ax


def plot_confusion_matrix(cm, labels=None, ax=None, normalize: bool = False, title: str | None = None):
    ax = ax or plt.gca()
    cm = np.asarray(cm, float)
    shown = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    im = ax.imshow(shown, cmap="Blues")
    plt.colorbar(im, ax=ax)
    n = cm.shape[0]
    labels = labels if labels is not None else list(range(n))
    ax.set_xticks(range(n), labels)
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("dự đoán")
    ax.set_ylabel("thực tế")
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            text = f"{shown[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if shown[i, j] > shown.max() / 2 else "black")
    ax.set_title(title or "Confusion matrix")
    return ax


def plot_residuals(y_true, y_pred, title: str | None = None):
    """3 hình chẩn đoán: residual vs fitted, histogram residual, Q-Q plot."""
    from scipy import stats

    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    resid = y_true - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].scatter(y_pred, resid, s=12, alpha=0.6)
    axes[0].axhline(0, color="k", lw=1)
    axes[0].set_xlabel("fitted (ŷ)")
    axes[0].set_ylabel("residual (y − ŷ)")
    axes[0].set_title("Residual vs fitted")
    axes[1].hist(resid, bins=30, alpha=0.8)
    axes[1].set_title("Phân phối residual")
    stats.probplot(resid, dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q plot")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_clusters(X, labels, centers=None, ax=None, title: str | None = None):
    ax = ax or plt.gca()
    X, labels = np.asarray(X), np.asarray(labels)
    for i, label in enumerate(np.unique(labels)):
        mask = labels == label
        color = "lightgray" if label == -1 else PALETTE[i % len(PALETTE)]
        ax.scatter(X[mask, 0], X[mask, 1], s=15, color=color, label="noise" if label == -1 else f"cluster {label}")
    if centers is not None:
        centers = np.asarray(centers)
        ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=200, c="black", label="centroid")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    return ax


def plot_history(history: dict[str, Sequence[float]], ax=None, title: str | None = None, logy: bool = False):
    """Vẽ các đường trong ``history`` (vd: {"train_loss": [...], "val_loss": [...]})."""
    ax = ax or plt.gca()
    for name, values in history.items():
        ax.plot(values, label=name)
    ax.set_xlabel("epoch / iteration")
    if logy:
        ax.set_yscale("log")
    ax.legend()
    if title:
        ax.set_title(title)
    return ax


def plot_images(images, labels=None, n: int = 10, ncols: int = 10, cmap: str = "gray_r", size: float = 1.2):
    """Lưới ảnh nhỏ (vd: digits 8×8)."""
    images = np.asarray(images)[:n]
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * size * 1.15))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i < len(images):
            ax.imshow(images[i], cmap=cmap)
            if labels is not None:
                ax.set_title(str(np.asarray(labels)[i]), fontsize=9)
    fig.tight_layout()
    return fig


__all__ = [
    "PALETTE",
    "set_style",
    "plot_decision_boundary",
    "plot_regression_fit",
    "plot_loss_contour",
    "plot_learning_curve",
    "plot_confusion_matrix",
    "plot_residuals",
    "plot_clusters",
    "plot_history",
    "plot_images",
]
