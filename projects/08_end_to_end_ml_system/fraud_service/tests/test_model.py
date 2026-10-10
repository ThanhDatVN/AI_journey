import numpy as np
from sklearn.metrics import average_precision_score

from fraud.model import baseline_policies, decide, total_cost


def test_decision_rule():
    assert decide([0.01, 0.01, 0.5], [1e6, 2e6, 2e4]).tolist() == [False, True, False]


def test_minimum_quality_on_test(model, splits):
    _, _, test = splits
    p = model.predict_proba(test)[:, 1]
    assert average_precision_score(test.is_fraud, p) > 0.06          # ~6× tỉ lệ nền 1%
    assert abs(p.mean() / test.is_fraud.mean() - 1) < 0.5             # xác suất không lệch mức nền


def test_policy_beats_baselines(model, splits):
    _, _, test = splits
    cost_model = total_cost(test, decide(model.predict_proba(test)[:, 1], test.amount_vnd))
    for name, review in baseline_policies(test).items():
        assert cost_model < total_cost(test, review), name


def test_foreign_transactions_are_riskier(model, splits):
    _, _, test = splits
    domestic = model.predict_proba(test.assign(is_foreign=False))[:, 1]
    foreign = model.predict_proba(test.assign(is_foreign=True))[:, 1]
    assert (foreign > domestic).all()


def test_invariance_to_extra_columns(model, splits):
    _, _, test = splits
    base = model.predict_proba(test)[:, 1]
    assert np.allclose(model.predict_proba(test.assign(customer_note="xyz"))[:, 1], base)


def test_metadata_contains_monitoring_reference(metadata):
    assert {"validation", "test", "reference_profile", "model_sha256"} <= set(metadata)
    assert metadata["test"]["cost_model_vnd"] < metadata["test"]["cost_baselines_vnd"]["duyệt tất cả"]
