import numpy as np
import pytest
from sklearn import metrics as skm

from ml_course import metrics as m

rng = np.random.default_rng(0)
Y_REG = rng.normal(10, 3, 200)
P_REG = Y_REG + rng.normal(0, 1, 200)
Y_CLS = rng.integers(0, 2, 300)
SCORES = np.clip(Y_CLS * 0.3 + rng.uniform(0, 0.8, 300), 0, 1)
SCORES[:20] = 0.5  # ties
P_CLS = (SCORES > 0.5).astype(int)


@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        (m.mean_squared_error, skm.mean_squared_error),
        (m.mean_absolute_error, skm.mean_absolute_error),
        (m.r2_score, skm.r2_score),
        (m.mean_absolute_percentage_error, skm.mean_absolute_percentage_error),
    ],
)
def test_regression_metrics(ours, theirs):
    assert ours(Y_REG, P_REG) == pytest.approx(theirs(Y_REG, P_REG))


def test_rmse():
    assert m.root_mean_squared_error(Y_REG, P_REG) == pytest.approx(np.sqrt(skm.mean_squared_error(Y_REG, P_REG)))


@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        (m.accuracy_score, skm.accuracy_score),
        (m.precision_score, skm.precision_score),
        (m.recall_score, skm.recall_score),
        (m.f1_score, skm.f1_score),
    ],
)
def test_classification_metrics(ours, theirs):
    assert ours(Y_CLS, P_CLS) == pytest.approx(theirs(Y_CLS, P_CLS))


def test_fbeta_and_specificity():
    assert m.fbeta_score(Y_CLS, P_CLS, beta=2) == pytest.approx(skm.fbeta_score(Y_CLS, P_CLS, beta=2))
    tn, fp, fn, tp = skm.confusion_matrix(Y_CLS, P_CLS).ravel()
    assert m.specificity_score(Y_CLS, P_CLS) == pytest.approx(tn / (tn + fp))


def test_confusion_matrix():
    np.testing.assert_array_equal(m.confusion_matrix(Y_CLS, P_CLS), skm.confusion_matrix(Y_CLS, P_CLS))


def test_roc_auc_and_curve():
    assert m.roc_auc_score(Y_CLS, SCORES) == pytest.approx(skm.roc_auc_score(Y_CLS, SCORES))
    fpr, tpr, _ = m.roc_curve(Y_CLS, SCORES)
    sk_fpr, sk_tpr, _ = skm.roc_curve(Y_CLS, SCORES, drop_intermediate=False)
    np.testing.assert_allclose(fpr, sk_fpr)
    np.testing.assert_allclose(tpr, sk_tpr)


def test_average_precision():
    assert m.average_precision_score(Y_CLS, SCORES) == pytest.approx(skm.average_precision_score(Y_CLS, SCORES))


def test_log_loss_and_brier():
    assert m.log_loss(Y_CLS, SCORES) == pytest.approx(skm.log_loss(Y_CLS, SCORES))
    assert m.brier_score(Y_CLS, SCORES) == pytest.approx(skm.brier_score_loss(Y_CLS, SCORES))
    proba = rng.dirichlet(np.ones(3), 50)
    y3 = rng.integers(0, 3, 50)
    assert m.log_loss(y3, proba) == pytest.approx(skm.log_loss(y3, proba, labels=[0, 1, 2]))


def test_ranking_metrics():
    rec = ["a", "b", "c", "d", "e"]
    rel = {"a", "c", "x"}
    assert m.precision_at_k(rec, rel, 3) == pytest.approx(2 / 3)
    assert m.recall_at_k(rec, rel, 3) == pytest.approx(2 / 3)
    assert m.average_precision_at_k(rec, rel, 5) == pytest.approx((1 + 2 / 3) / 3)
    graded = {"a": 3, "c": 2, "x": 1}
    dcg = 7 / np.log2(2) + 3 / np.log2(4)
    idcg = 7 / np.log2(2) + 3 / np.log2(3) + 1 / np.log2(4)
    assert m.ndcg_at_k(rec, graded, 5) == pytest.approx(dcg / idcg)
    assert m.ndcg_at_k(["a", "c", "x"], graded, 3) == pytest.approx(1.0)
