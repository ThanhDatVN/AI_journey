"""Tiện ích chung: seed, đo thời gian, gradient số, đường dẫn trong repo."""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .progress import repo_root


def set_seed(seed: int = 42) -> np.random.Generator:
    """Cố định seed cho random, NumPy và (nếu có) PyTorch. Trả về một ``np.random.Generator``."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # PyTorch là tuỳ chọn
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    return np.random.default_rng(seed)


@contextmanager
def timer(label: str = "elapsed") -> Iterator[dict]:
    """``with timer("fit") as t: ...`` → in thời gian chạy, ``t["seconds"]`` chứa kết quả."""
    out: dict = {}
    start = time.perf_counter()
    yield out
    out["seconds"] = time.perf_counter() - start
    print(f"⏱ {label}: {out['seconds'] * 1000:.1f} ms")


def numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Gradient bằng sai phân trung tâm: (f(x+h) − f(x−h)) / 2h. Dùng để *gradient check*."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        f_plus = f(x)
        x[idx] = old - eps
        f_minus = f(x)
        x[idx] = old
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """‖a − b‖ / (‖a‖ + ‖b‖) — < 1e-7 là rất tốt, > 1e-3 là gần như chắc có bug."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    denom = np.linalg.norm(a) + np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / denom) if denom > 0 else 0.0


def describe_array(name: str, arr: np.ndarray) -> None:
    """In nhanh shape / dtype / min / max / mean — thói quen tốt khi debug."""
    arr = np.asarray(arr)
    stats = ""
    if arr.size and np.issubdtype(arr.dtype, np.number):
        stats = f" min={arr.min():.4g} max={arr.max():.4g} mean={arr.mean():.4g}"
    print(f"{name}: shape={arr.shape} dtype={arr.dtype}{stats}")


def repo_path(*parts: str) -> Path:
    """Đường dẫn tuyệt đối tới một file trong repo, ví dụ ``repo_path("data", "sample", "churn.csv")``."""
    return repo_root().joinpath(*parts)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid ổn định số học (không overflow khi z rất âm)."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax ổn định số học: trừ max trước khi exp."""
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def one_hot(y: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    n_classes = n_classes or int(y.max()) + 1
    out = np.zeros((y.size, n_classes))
    out[np.arange(y.size), y] = 1.0
    return out


__all__ = [
    "set_seed",
    "timer",
    "numerical_gradient",
    "relative_error",
    "describe_array",
    "repo_path",
    "sigmoid",
    "softmax",
    "one_hot",
]
