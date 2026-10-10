import numpy as np
import pytest
from sklearn import preprocessing as skp

from ml_course import preprocessing as pp
from ml_course import validation as val

rng = np.random.default_rng(1)
X = rng.normal(5, 2, (100, 4))
X[:, 3] = 7.0  # cột hằng


def test_standard_scaler():
    ours = pp.StandardScalerScratch().fit_transform(X)
    theirs = skp.StandardScaler().fit_transform(X)
    np.testing.assert_allclose(ours, theirs, atol=1e-10)


def test_minmax_scaler():
    np.testing.assert_allclose(pp.MinMaxScalerScratch().fit_transform(X), skp.MinMaxScaler().fit_transform(X))


def test_robust_scaler():
    np.testing.assert_allclose(pp.RobustScalerScratch().fit_transform(X), skp.RobustScaler().fit_transform(X))


def test_inverse_transform():
    s = pp.StandardScalerScratch().fit(X)
    np.testing.assert_allclose(s.inverse_transform(s.transform(X)), X)


def test_one_hot_unknown_category():
    enc = pp.OneHotEncoderScratch().fit(["a", "b", "a", "c"])
    out = enc.transform(["b", "z"])
    np.testing.assert_array_equal(out, [[0, 1, 0], [0, 0, 0]])
    with pytest.raises(ValueError):
        pp.OneHotEncoderScratch(handle_unknown="error").fit(["a"]).transform(["b"])


def test_polynomial_features():
    Z = rng.normal(size=(10, 3))
    ours = pp.polynomial_features(Z, degree=3)
    theirs = skp.PolynomialFeatures(3, include_bias=False).fit_transform(Z)
    np.testing.assert_allclose(ours, theirs)


def test_train_test_split_stratified():
    y = np.array([0] * 90 + [1] * 10)
    Xtr, Xte, ytr, yte = pp.train_test_split(np.arange(100), y, test_size=0.2, stratify=y)
    assert len(Xte) == 20 and yte.sum() == 2
    assert set(Xtr) | set(Xte) == set(range(100)) and not set(Xtr) & set(Xte)


def test_kfold_partitions_everything_once():
    seen = np.concatenate([v for _, v in val.KFoldScratch(5).split(np.arange(23))])
    assert sorted(seen.tolist()) == list(range(23))


def test_stratified_kfold_keeps_ratio():
    y = np.array([0] * 80 + [1] * 20)
    for _, v in val.StratifiedKFoldScratch(5).split(np.zeros(100), y):
        assert y[v].mean() == pytest.approx(0.2)


def test_time_series_split_is_ordered():
    for tr, va in val.TimeSeriesSplitScratch(4, gap=2).split(np.arange(50)):
        assert tr.max() + 2 < va.min()


def test_group_kfold_no_group_leak():
    groups = np.repeat(np.arange(10), 5)
    for tr, va in val.GroupKFoldScratch(5).split(np.zeros(50), groups=groups):
        assert not set(groups[tr]) & set(groups[va])


def test_bootstrap_ci_contains_mean():
    est, lo, hi = val.bootstrap_ci(rng.normal(3, 1, 500))
    assert lo < est < hi and lo < 3 < hi
