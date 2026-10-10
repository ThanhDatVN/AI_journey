"""So sánh thuật toán from scratch với scikit-learn trên dữ liệu nhỏ."""

import numpy as np
import pytest
from sklearn import cluster, decomposition, linear_model, naive_bayes, neighbors, tree
from sklearn.datasets import load_breast_cancer, load_iris, make_blobs, make_regression
from sklearn.preprocessing import StandardScaler

from ml_course import algorithms as alg
from ml_course.utils import numerical_gradient, relative_error, sigmoid, softmax

X_REG, Y_REG = make_regression(200, 5, noise=5, random_state=0)
IRIS_X, IRIS_Y = load_iris(return_X_y=True)
BC_X, BC_Y = load_breast_cancer(return_X_y=True)
BC_XS = StandardScaler().fit_transform(BC_X)


def test_linear_regression_normal_and_gd():
    sk = linear_model.LinearRegression().fit(X_REG, Y_REG)
    ours = alg.LinearRegressionScratch().fit(X_REG, Y_REG)
    np.testing.assert_allclose(ours.coef_, sk.coef_, rtol=1e-6)
    gd = alg.LinearRegressionScratch(method="gd", lr=0.1, n_iters=3000).fit(X_REG, Y_REG)
    np.testing.assert_allclose(gd.coef_, sk.coef_, rtol=1e-3)
    assert gd.loss_history_[0] > gd.loss_history_[-1]


def test_ridge_and_lasso():
    np.testing.assert_allclose(
        alg.RidgeScratch(alpha=10).fit(X_REG, Y_REG).coef_, linear_model.Ridge(alpha=10).fit(X_REG, Y_REG).coef_, rtol=1e-6
    )
    ours = alg.LassoScratch(alpha=5).fit(X_REG, Y_REG)
    sk = linear_model.Lasso(alpha=5).fit(X_REG, Y_REG)
    np.testing.assert_allclose(ours.coef_, sk.coef_, rtol=1e-3, atol=1e-3)


def test_logistic_regression_matches_sklearn():
    ours = alg.LogisticRegressionScratch(lr=0.5, n_iters=5000, l2=1.0).fit(BC_XS, BC_Y)
    sk = linear_model.LogisticRegression(C=1.0, max_iter=5000).fit(BC_XS, BC_Y)
    # sklearn phạt ½‖w‖² · (1/C) trên tổng loss; ours phạt (λ/2n)‖w‖² trên loss trung bình → cùng nghiệm
    assert np.mean(ours.predict(BC_XS) == sk.predict(BC_XS)) > 0.98
    assert ours.loss_history_[-1] < 0.1


def test_logistic_gradient_matches_numerical():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(20, 3)), rng.integers(0, 2, 20)
    w = rng.normal(size=3)

    def loss(w):
        p = sigmoid(X @ w)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    analytic = X.T @ (sigmoid(X @ w) - y) / len(y)
    assert relative_error(analytic, numerical_gradient(loss, w.copy())) < 1e-7


def test_softmax_regression():
    Xs = StandardScaler().fit_transform(IRIS_X)
    ours = alg.SoftmaxRegressionScratch(lr=0.5, n_iters=2000).fit(Xs, IRIS_Y)
    assert np.mean(ours.predict(Xs) == IRIS_Y) > 0.95
    np.testing.assert_allclose(ours.predict_proba(Xs).sum(1), 1)
    np.testing.assert_allclose(softmax(np.array([[1000.0, 1000.0]])), [[0.5, 0.5]])


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_knn(weights):
    ours = alg.KNNClassifierScratch(k=5, weights=weights).fit(IRIS_X, IRIS_Y).predict(IRIS_X)
    sk = neighbors.KNeighborsClassifier(5, weights=weights).fit(IRIS_X, IRIS_Y).predict(IRIS_X)
    assert np.mean(ours == sk) > 0.98
    reg = alg.KNNRegressorScratch(k=3).fit(X_REG, Y_REG).predict(X_REG[:20])
    np.testing.assert_allclose(reg, neighbors.KNeighborsRegressor(3).fit(X_REG, Y_REG).predict(X_REG[:20]))


def test_naive_bayes():
    ours = alg.GaussianNBScratch().fit(IRIS_X, IRIS_Y)
    sk = naive_bayes.GaussianNB().fit(IRIS_X, IRIS_Y)
    np.testing.assert_allclose(ours.predict_proba(IRIS_X), sk.predict_proba(IRIS_X), atol=1e-6)
    counts = np.random.default_rng(0).poisson(2, (60, 8))
    yc = np.repeat([0, 1, 2], 20)
    np.testing.assert_allclose(
        alg.MultinomialNBScratch().fit(counts, yc).predict_proba(counts),
        naive_bayes.MultinomialNB().fit(counts, yc).predict_proba(counts),
    )


def test_decision_tree_classifier_and_regressor():
    ours = alg.DecisionTreeScratch(max_depth=3).fit(IRIS_X, IRIS_Y)
    sk = tree.DecisionTreeClassifier(max_depth=3, random_state=0).fit(IRIS_X, IRIS_Y)
    assert np.mean(ours.predict(IRIS_X) == sk.predict(IRIS_X)) > 0.97
    full = alg.DecisionTreeScratch().fit(IRIS_X, IRIS_Y)
    assert np.mean(full.predict(IRIS_X) == IRIS_Y) == 1.0
    reg = alg.DecisionTreeScratch(task="regression", max_depth=4).fit(X_REG, Y_REG)
    skr = tree.DecisionTreeRegressor(max_depth=4, random_state=0).fit(X_REG, Y_REG)
    np.testing.assert_allclose(reg.predict(X_REG), skr.predict(X_REG), rtol=1e-6)
    assert ours.feature_importances_.sum() == pytest.approx(1.0)
    assert "if" in ours.export_text()


def test_random_forest_and_oob():
    rf = alg.RandomForestScratch(n_estimators=30, oob_score=True, seed=0).fit(BC_X, BC_Y)
    assert rf.oob_score_ > 0.9
    assert np.mean(rf.predict(BC_X) == BC_Y) > 0.97
    rfr = alg.RandomForestScratch(n_estimators=20, task="regression", max_features=1.0).fit(X_REG, Y_REG)
    assert np.corrcoef(rfr.predict(X_REG), Y_REG)[0, 1] > 0.9


def test_boosting():
    ada = alg.AdaBoostScratch(n_estimators=40).fit(BC_XS, BC_Y)
    assert np.mean(ada.predict(BC_XS) == BC_Y) > 0.95
    gbr = alg.GradientBoostingRegressorScratch(n_estimators=50, max_depth=2).fit(X_REG, Y_REG)
    assert gbr.train_loss_[-1] < gbr.train_loss_[0] * 0.2
    staged = list(gbr.staged_predict(X_REG[:5]))
    np.testing.assert_allclose(staged[-1], gbr.predict(X_REG[:5]))
    gbc = alg.GradientBoostingClassifierScratch(n_estimators=30, max_depth=2).fit(BC_XS, BC_Y)
    assert np.mean(gbc.predict(BC_XS) == BC_Y) > 0.97


def test_svm():
    Xb, yb = make_blobs(200, centers=2, random_state=1, cluster_std=1.2)
    lin = alg.LinearSVMScratch(C=1.0, lr=0.5, n_iters=3000).fit(Xb, yb)
    assert np.mean(lin.predict(Xb) == yb) > 0.97
    rng = np.random.default_rng(0)
    Xc = rng.uniform(-1, 1, (150, 2))
    yc = (np.hypot(Xc[:, 0], Xc[:, 1]) < 0.6).astype(int)
    ksvm = alg.KernelSVMScratch(C=5, gamma=3).fit(Xc, yc)
    assert np.mean(ksvm.predict(Xc) == yc) > 0.95
    assert 0 < len(ksvm.support_) < len(Xc)


def test_kmeans_and_dbscan():
    Xb, _ = make_blobs(300, centers=3, random_state=0)
    ours = alg.KMeansScratch(3).fit(Xb)
    sk = cluster.KMeans(3, n_init=10, random_state=0).fit(Xb)
    assert ours.inertia_ == pytest.approx(sk.inertia_, rel=1e-3)
    db = alg.DBSCANScratch(eps=0.8, min_samples=5).fit_predict(Xb)
    skdb = cluster.DBSCAN(eps=0.8, min_samples=5).fit_predict(Xb)
    assert len(set(db)) == len(set(skdb))
    np.testing.assert_array_equal(db == -1, skdb == -1)


def test_pca():
    ours = alg.PCAScratch(2).fit(IRIS_X)
    sk = decomposition.PCA(2).fit(IRIS_X)
    np.testing.assert_allclose(ours.explained_variance_ratio_, sk.explained_variance_ratio_)
    np.testing.assert_allclose(np.abs(ours.transform(IRIS_X)), np.abs(sk.transform(IRIS_X)), atol=1e-8)
    assert alg.PCAScratch(0.95).fit(IRIS_X).n_components_ == 2


def test_gmm_em_increases_likelihood():
    Xb, _ = make_blobs(300, centers=3, random_state=2)
    gmm = alg.GaussianMixtureScratch(3).fit(Xb)
    ll = np.array(gmm.log_likelihood_history_)
    assert np.all(np.diff(ll) > -1e-6)
    np.testing.assert_allclose(gmm.predict_proba(Xb).sum(1), 1)


def test_mlp_learns_xor_like_data():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (400, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    mlp = alg.MLPScratch(hidden=(16,), epochs=150, lr=0.01).fit(X, y)
    assert np.mean(mlp.predict(X) == y) > 0.95
    assert mlp.history_[-1] < mlp.history_[0]
