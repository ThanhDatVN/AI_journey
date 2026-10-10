"""Các hàm kiểm tra nhỏ dùng trong bài tập và checkpoint.

Mỗi hàm in ✅ khi đúng và raise ``AssertionError`` (kèm gợi ý) khi sai, để:
- trong notebook bài tập: bạn thấy ngay kết quả từng câu;
- trong checkpoint: ``cp.task(...)`` bắt lỗi và ghi FAIL.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def _ok(message: str) -> bool:
    print(f"✅ {message}")
    return True


def check_true(condition: bool, message: str = "điều kiện", hint: str = "") -> bool:
    if not bool(condition):
        raise AssertionError(f"❌ {message} chưa đúng." + (f" Gợi ý: {hint}" if hint else ""))
    return _ok(message)


def check_close(
    actual: Any,
    expected: Any,
    tol: float = 1e-6,
    message: str | None = None,
    hint: str = "",
) -> bool:
    """So sánh số (hoặc array) với sai số tuyệt đối/tương đối ``tol``."""
    label = message or f"giá trị ≈ {np.round(expected, 6)}"
    if actual is None:
        raise AssertionError(f"❌ {label}: bạn chưa trả về giá trị (None).")
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    if a.shape != e.shape:
        raise AssertionError(f"❌ {label}: shape {a.shape} ≠ kỳ vọng {e.shape}.")
    if not np.allclose(a, e, atol=tol, rtol=tol, equal_nan=True):
        shown = np.round(a, 6) if a.size <= 10 else f"array shape {a.shape}"
        raise AssertionError(
            f"❌ {label}: nhận {shown}." + (f" Gợi ý: {hint}" if hint else "")
        )
    return _ok(label)


def check_equal(actual: Any, expected: Any, message: str | None = None, hint: str = "") -> bool:
    label = message or f"giá trị = {expected!r}"
    if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
        same = np.array_equal(np.asarray(actual), np.asarray(expected))
    else:
        same = actual == expected
    if not same:
        raise AssertionError(f"❌ {label}: nhận {actual!r}." + (f" Gợi ý: {hint}" if hint else ""))
    return _ok(label)


def check_shape(array: Any, expected_shape: tuple[int, ...], message: str | None = None) -> bool:
    shape = np.shape(array)
    label = message or f"shape = {expected_shape}"
    if tuple(shape) != tuple(expected_shape):
        raise AssertionError(f"❌ {label}: nhận {shape}.")
    return _ok(label)


def check_less(value: float, bound: float, message: str | None = None, hint: str = "") -> bool:
    label = message or f"giá trị < {bound}"
    if value is None or not float(value) < bound:
        raise AssertionError(f"❌ {label}: nhận {value}." + (f" Gợi ý: {hint}" if hint else ""))
    return _ok(f"{label} (nhận {float(value):.4g})")


def check_greater(value: float, bound: float, message: str | None = None, hint: str = "") -> bool:
    label = message or f"giá trị > {bound}"
    if value is None or not float(value) > bound:
        raise AssertionError(f"❌ {label}: nhận {value}." + (f" Gợi ý: {hint}" if hint else ""))
    return _ok(f"{label} (nhận {float(value):.4g})")


def check_in(value: Any, allowed: Iterable[Any], message: str | None = None) -> bool:
    allowed = list(allowed)
    label = message or f"giá trị thuộc {allowed}"
    if value not in allowed:
        raise AssertionError(f"❌ {label}: nhận {value!r}.")
    return _ok(label)


__all__ = [
    "check_true",
    "check_close",
    "check_equal",
    "check_shape",
    "check_less",
    "check_greater",
    "check_in",
]
