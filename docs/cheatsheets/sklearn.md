# scikit-learn — cheat sheet

```python
# API thống nhất
model.fit(X_train, y_train); model.predict(X); model.predict_proba(X)[:, 1]; model.score(X, y)
transformer.fit(X_train); transformer.transform(X); transformer.fit_transform(X_train)   # fit_transform CHỈ trên train

# Chia dữ liệu
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# Tiền xử lý + pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PolynomialFeatures, StandardScaler
pre = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
clf = make_pipeline(pre, LogisticRegression(max_iter=1000))
reg_log = TransformedTargetRegressor(make_pipeline(pre, Ridge()), func=np.log, inverse_func=np.exp)  # học trên log(y)

# Mô hình hay dùng
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# Kiểm định chéo
from sklearn.model_selection import StratifiedKFold, GroupKFold, TimeSeriesSplit, cross_val_score, cross_validate, cross_val_predict
scores = cross_val_score(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc")
oof = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]       # dự đoán out-of-fold

# Tìm siêu tham số (tên tham số trong pipeline: <bước>__<tham số>)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingRandomSearchCV  # Halving: from sklearn.experimental import enable_halving_search_cv
search = RandomizedSearchCV(clf, {"logisticregression__C": np.logspace(-3, 3, 20)}, n_iter=20, cv=5, scoring="roc_auc", random_state=0)
search.fit(X_tr, y_tr); search.best_params_; search.best_score_          # best_score_ lạc quan!

# Metric
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
                             log_loss, brier_score_loss, confusion_matrix, classification_report,
                             mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error,
                             silhouette_score, adjusted_rand_score)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

# Lưu / nạp
import joblib
joblib.dump(clf, "model.joblib"); clf = joblib.load("model.joblib")        # chỉ nạp file tin cậy (pickle chạy code)
```

## Ghi nhớ

| Quy tắc | Vì sao |
|---|---|
| mọi tiền xử lý trong `Pipeline` | tránh rò rỉ, phục vụ đúng như lúc train |
| `stratify=y` cho phân loại | giữ tỉ lệ lớp |
| `random_state` ở mọi nơi | tái lập |
| `OneHotEncoder(handle_unknown="ignore")` | không lỗi với danh mục mới — nhưng cũng **không cảnh báo**: kiểm tra đầu vào riêng |
| `class_weight="balanced"` làm lệch xác suất | hiệu chỉnh lại nếu cần xác suất |
| `predict` dùng ngưỡng 0.5 | tự đặt ngưỡng theo chi phí trên `predict_proba` |
