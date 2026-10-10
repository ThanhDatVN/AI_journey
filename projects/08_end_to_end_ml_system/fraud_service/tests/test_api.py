import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraud.config import RAW_COLUMNS


@pytest.fixture(scope="module")
def client(model_dir, monkeypatch_module):
    monkeypatch_module.setenv("FRAUD_MODEL_DIR", str(model_dir))
    from fraud.api import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def records(df, n):
    return json.loads(df[RAW_COLUMNS].head(n).to_json(orient="records"))


def test_health(client, metadata):
    assert client.get("/health").json() == {"status": "ok", "model_version": metadata["version"]}


def test_parity_with_model(client, model, splits):
    _, _, test = splits
    response = client.post("/v1/score", json=records(test, 200))
    assert response.status_code == 200
    api = np.array([r["fraud_probability"] for r in response.json()])
    assert np.allclose(api, model.predict_proba(test.head(200))[:, 1])


def test_decision_consistent_with_expected_loss(client, splits):
    _, _, test = splits
    for r in client.post("/v1/score", json=records(test, 100)).json():
        assert (r["decision"] == "review") == (r["expected_loss_vnd"] > 15_000)


def test_rejects_invalid_input(client, splits):
    _, _, test = splits
    good = records(test, 1)[0]
    assert client.post("/v1/score", json=[{**good, "hour": 24}]).status_code == 422
    assert client.post("/v1/score", json=[{**good, "merchant_category": "casino"}]).status_code == 422
    assert client.post("/v1/score", json=[{**good, "extra": 1}]).status_code == 422
    assert client.post("/v1/score", json=[]).status_code == 422
