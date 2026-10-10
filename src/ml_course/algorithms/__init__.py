"""Phiên bản tham chiếu (reference) của các thuật toán viết from scratch bằng NumPy.

Trong notebook, bạn sẽ *tự xây* từng thuật toán từng bước. Các class ở đây là phiên bản
hoàn chỉnh, đã được test so với scikit-learn (xem ``tests/test_algorithms.py``), để:

- chương sau dùng lại chương trước (Random Forest dùng Decision Tree, Gradient Boosting dùng Regression Tree);
- bạn đối chiếu khi code của mình chạy sai.

Quy tắc của khoá học: **from scratch trước, thư viện sau** — nhưng chỉ implement đủ để hiểu nguyên lý.
"""

from .cluster import DBSCANScratch, KMeansScratch
from .decomposition import PCAScratch
from .ensemble import (
    AdaBoostScratch,
    GradientBoostingClassifierScratch,
    GradientBoostingRegressorScratch,
    RandomForestScratch,
)
from .knn import KNNClassifierScratch, KNNRegressorScratch
from .linear_models import (
    LassoScratch,
    LinearRegressionScratch,
    LogisticRegressionScratch,
    RidgeScratch,
    SoftmaxRegressionScratch,
)
from .naive_bayes import GaussianNBScratch, MultinomialNBScratch
from .neural_net import MLPScratch
from .probabilistic import GaussianMixtureScratch
from .svm import KernelSVMScratch, LinearSVMScratch
from .tree import DecisionTreeScratch

__all__ = [
    "LinearRegressionScratch",
    "RidgeScratch",
    "LassoScratch",
    "LogisticRegressionScratch",
    "SoftmaxRegressionScratch",
    "KNNClassifierScratch",
    "KNNRegressorScratch",
    "GaussianNBScratch",
    "MultinomialNBScratch",
    "DecisionTreeScratch",
    "RandomForestScratch",
    "AdaBoostScratch",
    "GradientBoostingRegressorScratch",
    "GradientBoostingClassifierScratch",
    "LinearSVMScratch",
    "KernelSVMScratch",
    "KMeansScratch",
    "DBSCANScratch",
    "PCAScratch",
    "GaussianMixtureScratch",
    "MLPScratch",
]
