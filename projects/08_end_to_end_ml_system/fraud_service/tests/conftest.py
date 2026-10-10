import json

import joblib
import pytest

from fraud.config import DEFAULT_DATA
from fraud.data import load_transactions, split
from fraud.train import main as train_main


@pytest.fixture(scope="session")
def data():
    return load_transactions(DEFAULT_DATA)


@pytest.fixture(scope="session")
def splits(data):
    return split(data)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    """Huấn luyện một lần cho cả phiên test, vào thư mục tạm — test không phụ thuộc mô hình có sẵn."""
    out = tmp_path_factory.mktemp("model")
    assert train_main(["--data", str(DEFAULT_DATA), "--out", str(out)]) == 0
    return out


@pytest.fixture(scope="session")
def model(model_dir):
    return joblib.load(model_dir / "model.joblib")


@pytest.fixture(scope="session")
def metadata(model_dir):
    return json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
