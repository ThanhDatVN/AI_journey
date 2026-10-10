import json
import shutil
from pathlib import PurePosixPath

import pytest

from fraud.api import load
from fraud.config import DEFAULT_DATA, default_data_path


def test_config_works_in_shallow_container_path():
    # Trong image Docker, mã nằm ở /app — không có ba cấp thư mục cha như trong repo khoá học.
    assert default_data_path(PurePosixPath("/app")) == PurePosixPath("/app/data/sample/fraud_transactions.csv")
    assert DEFAULT_DATA.exists()


@pytest.fixture
def model_copy(model_dir, tmp_path):
    target = tmp_path / "model"
    shutil.copytree(model_dir, target)
    return target


def test_load_accepts_intact_package(model_copy):
    model, meta = load(model_copy)
    assert hasattr(model, "predict_proba") and meta["version"]


def test_load_rejects_tampered_model(model_copy):
    with open(model_copy / "model.joblib", "ab") as f:
        f.write(b"\x00")
    with pytest.raises(RuntimeError, match="hash"):
        load(model_copy)


def test_load_rejects_other_sklearn_version(model_copy):
    meta_path = model_copy / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["sklearn"] = "0.0.0"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(RuntimeError, match="scikit-learn 0.0.0"):
        load(model_copy)
