# Chương 02 · Mathematics for ML

> Tương ứng phần **Toán** của Tuần 1 khoá AIVN-Machine-Learning (ký hiệu toán, đại số tuyến tính, xác suất, thống kê), bổ sung giải tích, gradient descent, MLE/MAP, eigen/SVD.

**Không học toán tách rời** — mỗi khái niệm đều gắn trực tiếp với một thuật toán ML.

## 🎯 Mục tiêu

- Đọc được ký hiệu toán trong tài liệu ML và dịch sang code NumPy.
- Đại số tuyến tính: vector, ma trận, chuẩn, khoảng cách, rank, nghịch đảo, eigen, SVD.
- Giải tích: đạo hàm, gradient, chain rule, Hessian, tính lồi; tự cài đặt gradient descent.
- Xác suất: Bayes, biến ngẫu nhiên, phân phối, kỳ vọng/phương sai, LLN, CLT.
- MLE & MAP: hiểu vì sao MSE, cross-entropy, regularization "từ đâu mà có".
- Thống kê: mô tả dữ liệu, tương quan, khoảng tin cậy, bootstrap, kiểm định.

## 📋 Yêu cầu trước

[Chương 01](../01_python_for_ml/README.md) (NumPy, vectorization). Toán THPT (hàm số, đạo hàm cơ bản).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Ký hiệu toán thường dùng trong Machine Learning | 📖 Bài học | [View](01_math_notation.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/02_mathematics_for_ml/01_math_notation.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/02_mathematics_for_ml/01_math_notation.ipynb) |
| 02 | Bài 02 — Đại số tuyến tính cơ bản | 📖 Bài học | [View](02_linear_algebra.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/02_mathematics_for_ml/02_linear_algebra.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/02_mathematics_for_ml/02_linear_algebra.ipynb) |
| 03 | Bài 03 — Độc lập tuyến tính, hạng, nghịch đảo, trị riêng và SVD | 📖 Bài học | [View](03_rank_inverse_eigen_svd.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/02_mathematics_for_ml/03_rank_inverse_eigen_svd.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/02_mathematics_for_ml/03_rank_inverse_eigen_svd.ipynb) |
| 04 | Bài 04 — Giải tích cho ML: đạo hàm, gradient, chain rule | 📖 Bài học | [View](04_calculus_and_gradients.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/02_mathematics_for_ml/04_calculus_and_gradients.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/02_mathematics_for_ml/04_calculus_and_gradients.ipynb) |
| 05 | Bài 05 — Gradient Descent | 📖 Bài học | [View](05_gradient_descent.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/02_mathematics_for_ml/05_gradient_descent.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/02_mathematics_for_ml/05_gradient_descent.ipynb) |
| 06 | Bài 06 — Xác suất cho Machine Learning | 📖 Bài học | [View](06_probability.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/02_mathematics_for_ml/06_probability.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/02_mathematics_for_ml/06_probability.ipynb) |
| 07 | Bài 07 — Likelihood, MLE và MAP | 📖 Bài học | [View](07_likelihood_mle_map.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/02_mathematics_for_ml/07_likelihood_mle_map.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/02_mathematics_for_ml/07_likelihood_mle_map.ipynb) |
| 08 | Bài 08 — Thống kê cho Machine Learning | 📖 Bài học | [View](08_statistics.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/02_mathematics_for_ml/08_statistics.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/02_mathematics_for_ml/08_statistics.ipynb) |
| 90 | Bài tập — Chương 02: Toán cho ML (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/02_mathematics_for_ml/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/02_mathematics_for_ml/90_exercises.ipynb) |
| 91 | Assignment 91 — Tính hệ số tương quan giữa 2 ảnh | 💻 Assignment | [View](91_assignment_image_correlation.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/02_mathematics_for_ml/91_assignment_image_correlation.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/02_mathematics_for_ml/91_assignment_image_correlation.ipynb) |
| 99 | ✅ Checkpoint — 02 · Mathematics for ML | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/02_mathematics_for_ml/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/02_mathematics_for_ml/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## 🧭 Toán → ML

```text
dot product            → Linear Regression / Neural Network
distance / norm        → KNN / K-Means / regularization
eigenvector            → PCA
SVD                    → PCA / recommendation / nén ảnh
gradient + chain rule  → Gradient Descent → Backpropagation
Bayes theorem          → Naive Bayes, Bayesian ML
MLE                    → MSE (nhiễu Gauss), cross-entropy (Bernoulli)
MAP                    → Ridge (L2), Lasso (L1)
CLT / bootstrap        → khoảng tin cậy cho metric, so sánh model
```

## ❓ Bộ câu hỏi tự kiểm tra

[questions.md](questions.md) — trả lời không nhìn tài liệu; ôn lại sau 1 ngày, 1 tuần, 1 tháng.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, mini project PASS.

## 🔗 Tiếp theo

[03 · Data Analysis & EDA](../03_data_analysis_and_eda/README.md)

---
<sub>Bài 01, 02, 06, 08 và assignment 91 dựa trên Tuần 1 của [AIVN-Machine-Learning](https://github.com/thanhhff/AIVN-Machine-Learning) (MIT License, © Nguyen Trung Thanh / AIVIETNAM), đã được sửa lỗi, cập nhật và bổ sung.</sub>
