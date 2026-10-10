"""Bộ dữ liệu ảnh thật tải từ Internet — tải một lần, cache trong ``data/raw/`` (đã gitignore).

    from ml_course.real_datasets import load_fashion_mnist, load_cifar10
    X_train, y_train, X_test, y_test = load_fashion_mnist()   # ~30 MB, ảnh xám 28×28
    X_train, y_train, X_test, y_test = load_cifar10()         # ~170 MB, ảnh màu 32×32

Khác với ``ml_course.datasets`` (dữ liệu tổng hợp, sinh offline), các bộ này cần mạng ở lần đầu.
Thư mục cache: biến môi trường ``ML_COURSE_DATA_DIR`` nếu có, ngược lại ``<repo>/data/raw``.
Mọi file tải về được kiểm tra MD5 với giá trị công bố — file hỏng hoặc tải dở sẽ bị tải lại.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import pickle
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

from .progress import repo_root

FASHION_MNIST_CLASSES = ("áo phông", "quần dài", "áo len", "váy", "áo khoác",
                         "dép xăng-đan", "áo sơ mi", "giày thể thao", "túi xách", "bốt")
CIFAR10_CLASSES = ("máy bay", "ô tô", "chim", "mèo", "hươu", "chó", "ếch", "ngựa", "tàu thuỷ", "xe tải")

_FASHION_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
_FASHION_FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
_CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_CIFAR_MD5 = "c58f30108f718f92721af3b95e74349a"


def data_dir() -> Path:
    """Thư mục cache dữ liệu tải về."""
    env = os.environ.get("ML_COURSE_DATA_DIR")
    return Path(env).expanduser().resolve() if env else repo_root() / "data" / "raw"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, md5: str) -> Path:
    """Tải ``url`` về ``dest`` (bỏ qua nếu đã có file đúng MD5)."""
    if dest.exists() and _md5(dest) == md5:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f"Đang tải {url} → {dest} (chỉ lần đầu) …", flush=True)
    with urllib.request.urlopen(url, timeout=60) as response, open(part, "wb") as f:
        shutil.copyfileobj(response, f, length=1 << 20)
    if _md5(part) != md5:
        part.unlink(missing_ok=True)
        raise OSError(f"MD5 không khớp khi tải {url} — thử lại, hoặc tải thủ công và đặt vào {dest}")
    part.replace(dest)
    return dest


def _read_idx(path: Path) -> np.ndarray:
    """Đọc file IDX (định dạng của MNIST / Fashion-MNIST), nén gzip, kiểu uint8."""
    with gzip.open(path, "rb") as f:
        data = f.read()
    if data[:2] != b"\x00\x00" or data[2] != 0x08:
        raise ValueError(f"{path} không phải file IDX uint8")
    ndim = data[3]
    shape = tuple(int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load_fashion_mnist(cache_dir: str | Path | None = None):
    """Fashion-MNIST (Zalando, MIT): 60 000 ảnh train + 10 000 ảnh test, xám 28×28, 10 loại quần áo.

    Trả về ``(X_train, y_train, X_test, y_test)`` — ảnh ``uint8`` shape (n, 28, 28), nhãn ``int64``.
    Tên lớp: ``FASHION_MNIST_CLASSES``.
    """
    folder = Path(cache_dir) if cache_dir else data_dir() / "fashion-mnist"
    arrays = {name: _read_idx(_download(_FASHION_URL + name, folder / name, md5)) for name, md5 in _FASHION_FILES.items()}
    return (arrays["train-images-idx3-ubyte.gz"], arrays["train-labels-idx1-ubyte.gz"].astype(np.int64),
            arrays["t10k-images-idx3-ubyte.gz"], arrays["t10k-labels-idx1-ubyte.gz"].astype(np.int64))


def _load_cifar_batch(path: Path):
    # File chính thức đã kiểm tra MD5 — pickle chỉ an toàn với file đáng tin cậy.
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    X = np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return X, np.asarray(batch[b"labels"], dtype=np.int64)


def _load_cifar_folder(folder: Path):
    parts = [_load_cifar_batch(folder / f"data_batch_{i}") for i in range(1, 6)]
    X_test, y_test = _load_cifar_batch(folder / "test_batch")
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), X_test, y_test


def load_cifar10(cache_dir: str | Path | None = None):
    """CIFAR-10 (Krizhevsky, 2009): 50 000 ảnh train + 10 000 ảnh test, màu 32×32, 10 lớp.

    Trả về ``(X_train, y_train, X_test, y_test)`` — ảnh ``uint8`` shape (n, 32, 32, 3), nhãn ``int64``.
    Tên lớp: ``CIFAR10_CLASSES``.
    """
    root = Path(cache_dir) if cache_dir else data_dir()
    folder = root / "cifar-10-batches-py"
    if not (folder / "test_batch").exists():
        archive = _download(_CIFAR_URL, root / "cifar-10-python.tar.gz", _CIFAR_MD5)
        with tarfile.open(archive, "r:gz") as tar:
            members = [m for m in tar.getmembers()
                       if m.isfile() and m.name.startswith("cifar-10-batches-py/") and ".." not in Path(m.name).parts]
            safe = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}   # Python ≥ 3.10.12 / 3.11.4
            tar.extractall(root, members=members, **safe)
    return _load_cifar_folder(folder)


__all__ = ["load_fashion_mnist", "load_cifar10", "data_dir", "FASHION_MNIST_CLASSES", "CIFAR10_CLASSES"]
