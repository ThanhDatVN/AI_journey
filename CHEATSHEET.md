# ⚡ Cheat sheet — Machine Learning

Tra nhanh khi làm bài và dự án. Cheat sheet thư viện (NumPy, pandas, Matplotlib, scikit-learn, PyTorch): [docs/cheatsheets/](docs/cheatsheets/README.md).

## 1. Chọn thuật toán

| Thuật toán | Bài toán | Cần chuẩn hoá? | Phi tuyến | Diễn giải | Dữ liệu ít | Điểm mạnh | Cạm bẫy | Ch. |
|---|---|:---:|:---:|:---:|:---:|---|---|:---:|
| Hồi quy tuyến tính / Ridge / Lasso | hồi quy | ✅ (khi điều chuẩn) | ❌ (trừ khi thêm đặc trưng) | ⭐⭐⭐ | ✅ | nhanh, ổn định, baseline mạnh | ngoại lai, đa cộng tuyến | 06 |
| Hồi quy logistic | phân loại | ✅ | ❌ | ⭐⭐⭐ | ✅ | xác suất hiệu chỉnh tốt, giải thích được | biên tuyến tính | 07 |
| kNN | cả hai | ✅ bắt buộc | ✅ | ⭐⭐ | ✅ | không cần huấn luyện | chậm khi dự đoán, lời nguyền chiều cao | 08 |
| Naive Bayes | phân loại (văn bản) | ❌ | — | ⭐⭐ | ✅ | cực nhanh, tốt với văn bản | xác suất quá tự tin | 09 |
| Cây quyết định | cả hai | ❌ | ✅ | ⭐⭐⭐ (nông) | ⚠️ | trực quan, xử lý tương tác | overfit, không ổn định | 10 |
| Random forest | cả hai | ❌ | ✅ | ⭐⭐ | ✅ | ít cần tinh chỉnh, OOB | mô hình lớn, kém ngoại suy | 11 |
| Gradient boosting (HGB, XGBoost, LightGBM, CatBoost) | cả hai | ❌ | ✅ | ⭐⭐ | ⚠️ | thường tốt nhất cho dữ liệu bảng | cần early stopping, dễ overfit dữ liệu nhỏ | 12 |
| SVM (kernel) | phân loại | ✅ bắt buộc | ✅ (RBF) | ⭐ | ✅ | biên lề lớn, tốt ở chiều cao | chậm với n lớn, không có xác suất sẵn | 13 |
| K-Means / GMM | phân cụm | ✅ | ❌ / elip | ⭐⭐ | ✅ | đơn giản / gán mềm, BIC | cụm cầu, cần chọn k | 20, 23 |
| DBSCAN | phân cụm | ✅ | ✅ | ⭐⭐ | ✅ | hình dạng tuỳ ý, phát hiện nhiễu | chọn eps, mật độ khác nhau | 20 |
| Isolation Forest | bất thường | ❌ | ✅ | ⭐ | ✅ | nhanh, ít giả định | ít nhãn để đánh giá | 22 |
| MLP / CNN / RNN / Transformer | mọi loại | ✅ | ✅ | ⭐ | ❌ | dữ liệu phi cấu trúc (ảnh, chữ, âm thanh) | cần nhiều dữ liệu, tính toán, tinh chỉnh | 27–33 |

**Quy tắc khởi đầu**: dữ liệu bảng → baseline đơn giản (luật nghiệp vụ, logistic/Ridge) → gradient boosting; ảnh/chữ → mô hình tiền huấn luyện + transfer learning. Mô hình phức tạp hơn phải **thắng có ý nghĩa** (khoảng tin cậy) mới đáng dùng.

## 2. Chọn metric

| Tình huống | Metric | Ghi chú |
|---|---|---|
| Hồi quy, sai số đối xứng | MAE (bền với ngoại lai), RMSE (phạt sai lớn) | báo kèm baseline trung vị/trung bình |
| Hồi quy, sai số tương đối | MAPE (giá trị thật xa 0), WAPE, sMAPE | MAPE nổ khi y ≈ 0 |
| Phân loại cân bằng | accuracy, ROC-AUC, log-loss | |
| Phân loại mất cân bằng | PR-AUC, recall@precision, F1/Fβ | accuracy vô nghĩa |
| Quyết định theo chi phí | **lợi nhuận / chi phí kỳ vọng** | ngưỡng p* = C_FP / (C_FP + C_FN) khi xác suất hiệu chỉnh |
| Dùng xác suất trực tiếp | Brier, log-loss, ECE, biểu đồ reliability | AUC không đo calibration |
| Xếp hạng / gợi ý | NDCG@k, precision@k, recall@k, coverage | loại món đã xem |
| Phân cụm (không nhãn) | silhouette, Davies–Bouldin, BIC (GMM), độ ổn định | cụm "đúng" = hữu ích |
| Dự báo | MAE/MASE theo backtest rolling-origin; độ phủ khoảng | so với seasonal naive |

## 3. Chia dữ liệu và kiểm định

| Dữ liệu | Cách chia | scikit-learn |
|---|---|---|
| i.i.d., lớp cân bằng | k-fold ngẫu nhiên | `KFold(shuffle=True)` |
| lớp mất cân bằng | phân tầng | `StratifiedKFold` |
| nhiều mẫu cùng một thực thể (bệnh nhân, khách, máy) | theo nhóm | `GroupKFold`, `StratifiedGroupKFold` |
| có thời gian | theo thời gian (có khoảng cách nếu nhãn trễ) | `TimeSeriesSplit(gap=…)` |
| chọn siêu tham số + ước lượng hiệu năng | lồng nhau | `GridSearchCV` bên trong `cross_val_score` |

**Test set chạm đúng một lần.** Mọi tiền xử lý (điền thiếu, chuẩn hoá, chọn đặc trưng, SMOTE, target encoding) nằm **trong** pipeline, fit trên phần train của từng fold.

## 4. Chẩn đoán nhanh

| Triệu chứng | Khả năng | Thử |
|---|---|---|
| train tốt, validation kém | overfitting | điều chuẩn, đơn giản hoá, thêm dữ liệu, early stopping |
| train và validation cùng kém | underfitting / đặc trưng nghèo | mô hình mạnh hơn, đặc trưng tốt hơn |
| validation đẹp bất thường | **rò rỉ** | kiểm tra đặc trưng sau thời điểm dự đoán, trùng thực thể train/test |
| offline tốt, production kém | dịch chuyển phân phối, lệch train/serving | chia theo thời gian, đóng gói pipeline, kiểm thử parity, giám sát |
| AUC tốt mà quyết định lỗ | calibration lệch, ngưỡng sai | hiệu chỉnh (Platt/isotonic), ngưỡng theo chi phí |
| loss NaN / tăng vọt | learning rate quá lớn, dữ liệu chưa chuẩn hoá, log(0) | giảm lr, chuẩn hoá, clip gradient, kiểm tra đầu vào |
| loss không giảm | lr quá nhỏ, lỗi gradient, nhãn xáo trộn | overfit thử 1 batch nhỏ, kiểm tra gradient |

## 5. Công thức hay dùng

| | Công thức |
|---|---|
| Hồi quy tuyến tính (normal equation) | $\mathbf w = (X^\top X + \lambda I)^{-1} X^\top \mathbf y$ |
| Gradient MSE | $\nabla_{\mathbf w} = \frac{2}{n} X^\top (X\mathbf w - \mathbf y)$ |
| Logistic | $p = \sigma(\mathbf w^\top \mathbf x + b)$, $\nabla = \frac1n X^\top(\mathbf p - \mathbf y)$ |
| Cross-entropy | $-\frac1n\sum_i \log p_{i, y_i}$ |
| Softmax | $\mathrm{softmax}(\mathbf z)_k = e^{z_k} / \sum_j e^{z_j}$ (trừ max trước khi exp) |
| Gini / entropy | $1 - \sum_k p_k^2$ / $-\sum_k p_k \log_2 p_k$ |
| Precision, recall, F1 | $\frac{TP}{TP+FP}$, $\frac{TP}{TP+FN}$, $\frac{2PR}{P+R}$ |
| Attention | $\mathrm{softmax}(QK^\top/\sqrt{d_k})\,V$ |
| Bellman (Q-learning) | $Q(s,a) \leftarrow Q(s,a) + \alpha\,[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$ |
| IPW (Hajek) | $\hat\tau = \frac{\sum T_iY_i/e_i}{\sum T_i/e_i} - \frac{\sum (1-T_i)Y_i/(1-e_i)}{\sum (1-T_i)/(1-e_i)}$ |
| Split conformal | $q$ = phần tử thứ $\lceil (n+1)(1-\alpha)\rceil$ nhỏ nhất của điểm không phù hợp |
| PSI | $\sum_b (c_b - r_b)\ln(c_b / r_b)$ — < 0.1 ổn, > 0.25 trôi lớn |
| Ngưỡng theo chi phí | duyệt/dự đoán dương khi $p > \frac{C_{FP}}{C_{FP} + C_{FN}}$ |

## 6. Mẫu code

```python
# Pipeline chuẩn cho dữ liệu bảng
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
])
model = make_pipeline(pre, LogisticRegression(max_iter=1000))
scores = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc")
print(f"{scores.mean():.3f} ± {scores.std():.3f}")
```

```python
# Vòng huấn luyện PyTorch tối thiểu
torch.manual_seed(0)
model = make_model().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
for epoch in range(n_epochs):
    model.train()
    for xb, yb in train_loader:
        loss = loss_fn(model(xb.to(device)), yb.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        val_loss = sum(loss_fn(model(xb.to(device)), yb.to(device)).item() for xb, yb in val_loader) / len(val_loader)
```

## 7. Checklist trước khi tin một kết quả

- [ ] Baseline đơn giản đạt bao nhiêu? Cải thiện có lớn hơn độ bất định (nhiều seed/fold, bootstrap)?
- [ ] Mọi đặc trưng có sẵn **tại thời điểm dự đoán**? Không có thực thể nào ở cả train và test?
- [ ] Tiền xử lý nằm trong pipeline? Test chỉ dùng một lần?
- [ ] Metric khớp mục tiêu kinh doanh? Đã xem theo nhóm (vùng, độ tuổi, loại khách)?
- [ ] Xác suất có hiệu chỉnh nếu dùng để ra quyết định?
- [ ] Chạy lại được bằng một lệnh, với seed, phiên bản thư viện và hash dữ liệu ghi lại?
