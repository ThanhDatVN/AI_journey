# Chương 07 · Logistic Regression

> Tương ứng **Tuần 4** (Logistic Regression, Regularization, phân loại 2 chữ số viết tay, One-vs-All) và phần **Softmax Regression** của **Tuần 5** khoá AIVN-Machine-Learning — đã sửa lỗi công thức, dùng dữ liệu offline cho các bài lý thuyết, làm lại phần CIFAR-10 thành một bài riêng (bài 08), hoàn thiện phần còn trống và mở rộng theo template "mọi model".

## 🎯 Mục tiêu

- Hiểu vì sao Linear Regression không phù hợp cho phân loại; sigmoid, xác suất, **odds / log-odds**, decision boundary.
- Suy ra **cross-entropy** từ maximum likelihood; vì sao không dùng MSE.
- Cài đặt from scratch (GD, gradient check) và dùng advanced optimization (BFGS, Newton); hiểu tham số `C` của scikit-learn.
- Regularization + feature mapping cho ranh giới phi tuyến; chọn `C` bằng CV; L1 cho chọn feature.
- Phân loại nhiều lớp: One-vs-All (from scratch) và **Softmax Regression** (from scratch, gradient check).
- Ảnh thật: ba loss cho cùng mô hình tuyến tính (One-vs-All, softmax, SVM đa lớp) trên **CIFAR-10** — gradient check, chọn siêu tham số, trọng số như ảnh mẫu, learning curve cho thấy giới hạn ~40%; Softmax Regression cho **Fashion-MNIST** (assignment 93).
- Ngưỡng quyết định theo chi phí, calibration, và 5 failure case (perfect separation, phi tuyến, mất cân bằng, outlier, đa cộng tuyến).

## 📋 Yêu cầu trước

[Chương 06](../06_linear_regression/README.md), [Chương 02](../02_mathematics_for_ml/README.md) (MLE, chain rule).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Logistic Regression | 📖 Bài học | [View](01_logistic_regression.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/07_logistic_regression/01_logistic_regression.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/07_logistic_regression/01_logistic_regression.ipynb) |
| 02 | Bài 02 — Logistic Regression from scratch: Gradient Descent, BFGS, scikit-learn | 📖 Bài học | [View](02_from_scratch_and_optimization.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/07_logistic_regression/02_from_scratch_and_optimization.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/07_logistic_regression/02_from_scratch_and_optimization.ipynb) |
| 03 | Bài 03 — Regularized Logistic Regression và Feature Mapping | 📖 Bài học | [View](03_regularized_logistic_regression.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/07_logistic_regression/03_regularized_logistic_regression.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/07_logistic_regression/03_regularized_logistic_regression.ipynb) |
| 04 | Bài 04 — Bài toán phân loại 2 chữ số viết tay | 📖 Bài học | [View](04_handwritten_digits_binary.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/07_logistic_regression/04_handwritten_digits_binary.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/07_logistic_regression/04_handwritten_digits_binary.ipynb) |
| 05 | Bài 05 — Phân loại nhiều lớp với Logistic Regression: One-vs-All | 📖 Bài học | [View](05_one_vs_all.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/07_logistic_regression/05_one_vs_all.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/07_logistic_regression/05_one_vs_all.ipynb) |
| 06 | Bài 06 — Phân loại nhiều lớp: Softmax Regression | 📖 Bài học | [View](06_softmax_regression.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/07_logistic_regression/06_softmax_regression.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/07_logistic_regression/06_softmax_regression.ipynb) |
| 07 | Bài 07 — Ngưỡng quyết định, xác suất và các trường hợp Logistic Regression thất bại | 📖 Bài học | [View](07_threshold_and_failure_cases.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/07_logistic_regression/07_threshold_and_failure_cases.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/07_logistic_regression/07_threshold_and_failure_cases.ipynb) |
| 08 | Bài 08 — Bộ phân loại tuyến tính trên ảnh thật: CIFAR-10 | 📖 Bài học | [View](08_linear_classifiers_cifar10.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/07_logistic_regression/08_linear_classifiers_cifar10.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/07_logistic_regression/08_linear_classifiers_cifar10.ipynb) |
| 90 | Bài tập — Chương 07: Logistic Regression (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/07_logistic_regression/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/07_logistic_regression/90_exercises.ipynb) |
| 91 | Assignment 91 — Logistic Regression | 💻 Assignment | [View](91_assignment_logistic_regression.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/07_logistic_regression/91_assignment_logistic_regression.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/07_logistic_regression/91_assignment_logistic_regression.ipynb) |
| 92 | Assignment 92 — Regularized Logistic Regression | 💻 Assignment | [View](92_assignment_regularized_logistic_regression.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/07_logistic_regression/92_assignment_regularized_logistic_regression.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/07_logistic_regression/92_assignment_regularized_logistic_regression.ipynb) |
| 93 | Assignment 93 — Softmax Regression: từ chữ số viết tay đến Fashion-MNIST | 💻 Assignment | [View](93_assignment_softmax_regression.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/07_logistic_regression/93_assignment_softmax_regression.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/07_logistic_regression/93_assignment_softmax_regression.ipynb) |
| 99 | ✅ Checkpoint — 07 · Logistic Regression | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/07_logistic_regression/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/07_logistic_regression/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

`src/ml_course/algorithms/linear_models.py`: `LogisticRegressionScratch`, `SoftmaxRegressionScratch`.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS.

## 🔗 Tiếp theo

[08 · K-Nearest Neighbors](../08_knn/README.md)

---
<sub>Bài 01, 03–06 và assignment 91–92 dựa trên Tuần 4–5 của [AIVN-Machine-Learning](https://github.com/thanhhff/AIVN-Machine-Learning) (MIT License, © Nguyen Trung Thanh / AIVIETNAM; bài tập gốc từ khoá Machine Learning của Andrew Ng; hình softmax từ machinelearningcoban.com). Bài 08 hoàn thiện phần CIFAR-10 và gói `algorithms/` (logistic/softmax/SVM, gradient check) của Tuần 4; assignment 93 hiện thực hoá bài tập *Softmax Regression cho Fashion-MNIST* mà khoá gốc đã dự kiến.</sub>
