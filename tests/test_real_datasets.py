"""Bộ đọc dữ liệu ảnh thật — kiểm tra offline bằng file giả lập (không tải mạng)."""

import gzip
import hashlib
import pickle

import numpy as np
import pytest

from ml_course import real_datasets as rd


def write_idx(path, array):
    header = bytes([0, 0, 0x08, array.ndim]) + b"".join(int(d).to_bytes(4, "big") for d in array.shape)
    with gzip.open(path, "wb") as f:
        f.write(header + array.astype(np.uint8).tobytes())


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_read_idx_roundtrip(tmp_path):
    images = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    write_idx(tmp_path / "x.gz", images)
    np.testing.assert_array_equal(rd._read_idx(tmp_path / "x.gz"), images)


def test_read_idx_rejects_non_idx(tmp_path):
    with gzip.open(tmp_path / "bad.gz", "wb") as f:
        f.write(b"not an idx file")
    with pytest.raises(ValueError):
        rd._read_idx(tmp_path / "bad.gz")


def test_load_fashion_mnist_from_cache_without_network(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    files = {
        "train-images-idx3-ubyte.gz": rng.integers(0, 256, (6, 28, 28)),
        "train-labels-idx1-ubyte.gz": rng.integers(0, 10, 6),
        "t10k-images-idx3-ubyte.gz": rng.integers(0, 256, (2, 28, 28)),
        "t10k-labels-idx1-ubyte.gz": rng.integers(0, 10, 2),
    }
    for name, array in files.items():
        write_idx(tmp_path / name, array)
    monkeypatch.setattr(rd, "_FASHION_FILES", {name: md5(tmp_path / name) for name in files})
    monkeypatch.setattr(rd.urllib.request, "urlopen", lambda *a, **k: pytest.fail("không được tải mạng khi cache đúng MD5"))
    X_train, y_train, X_test, y_test = rd.load_fashion_mnist(cache_dir=tmp_path)
    assert X_train.shape == (6, 28, 28) and X_test.shape == (2, 28, 28)
    assert y_train.dtype == np.int64
    np.testing.assert_array_equal(y_test, files["t10k-labels-idx1-ubyte.gz"])


def test_cifar_batches_are_channel_last(tmp_path):
    def batch(n, label):
        data = np.concatenate([np.full((n, 1024), c, np.uint8) for c in (10, 20, 30)], axis=1)   # R, G, B liền khối
        return {b"data": data, b"labels": [label] * n}

    for i in range(1, 6):
        with open(tmp_path / f"data_batch_{i}", "wb") as f:
            pickle.dump(batch(3, i), f)
    with open(tmp_path / "test_batch", "wb") as f:
        pickle.dump(batch(2, 0), f)
    X_train, y_train, X_test, y_test = rd._load_cifar_folder(tmp_path)
    assert X_train.shape == (15, 32, 32, 3) and X_test.shape == (2, 32, 32, 3)
    assert X_train[0, 5, 7].tolist() == [10, 20, 30]
    assert y_train.tolist() == [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3 + [5] * 3


def test_corrupted_download_is_rejected(tmp_path, monkeypatch):
    class FakeResponse:
        def __init__(self):
            self.data = [b"corrupted"]

        def read(self, *_):
            return self.data.pop() if self.data else b""

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(rd.urllib.request, "urlopen", lambda *a, **k: FakeResponse())
    with pytest.raises(OSError, match="MD5"):
        rd._download("https://example.invalid/file.gz", tmp_path / "file.gz", md5="0" * 32)
    assert not (tmp_path / "file.gz").exists() and not (tmp_path / "file.gz.part").exists()


def test_data_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_COURSE_DATA_DIR", str(tmp_path))
    assert rd.data_dir() == tmp_path.resolve()
