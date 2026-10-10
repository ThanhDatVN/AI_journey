# Chương 06 · Linear Regression

> Tương ứng **Tuần 2–3** (hồi quy tuyến tính đơn biến, đa biến, normal equation, đa thức) và một phần **Tuần 4–5** (regularization, Batch/SGD/Mini-batch GD) của khoá AIVN-Machine-Learning, mở rộng theo template "mọi model" của khoá học.

Không chỉ `model.fit(X, y)` — ta đi qua toàn bộ chuỗi:

```text
Bài toán → Trực giác → Mô hình toán → Loss → Optimization (GD, normal equation)
→ Giả định → From scratch → scikit-learn → Regularization → Evaluation
→ Residual analysis → Failure cases → Bài tập → Checkpoint
```

## 🎯 Mục tiêu

- Cài đặt Linear Regression **from scratch**: cost function, gradient descent (vòng lặp → vector hoá), normal equation.
- Hiểu batch / stochastic / mini-batch GD; chọn learning rate; chuẩn hoá feature; học với dữ liệu lớn (theo dõi hội tụ SGD, online learning, map-reduce).
- Mở rộng phi tuyến bằng đa thức; nhận ra overfitting và nguy cơ ngoại suy.
- Góc nhìn xác suất (MLE), hình học (phép chiếu), các giả định; diễn giải hệ số đúng cách.
- Regularization: Ridge, Lasso, Elastic Net — hình học L1/L2, coefficient path, chọn λ bằng CV.
- Chẩn đoán model bằng residual; xử lý phi tuyến, heteroscedasticity, outlier, đa cộng tuyến, ngoại suy, omitted variable.

## 📋 Yêu cầu trước

[Chương 02](../02_mathematics_for_ml/README.md) (gradient, GD, MLE/MAP), [Chương 04](../04_data_preprocessing/README.md) (scaling, Pipeline), [Chương 05](../05_machine_learning_fundamentals/README.md) (train/val/test, bias–variance).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Hồi quy tuyến tính đơn biến (Linear Regression with One Variable) | 📖 Bài học | [View](01_linear_regression_one_variable.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/06_linear_regression/01_linear_regression_one_variable.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/06_linear_regression/01_linear_regression_one_variable.ipynb) |
| 02 | Bài 02 — Hồi quy tuyến tính đa biến (Linear Regression with Multiple Variables) | 📖 Bài học | [View](02_linear_regression_multiple_variables.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/06_linear_regression/02_linear_regression_multiple_variables.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/06_linear_regression/02_linear_regression_multiple_variables.ipynb) |
| 03 | Bài 03 — Gradient Descent Tips và Normal Equation | 📖 Bài học | [View](03_normal_equation_and_gd_tips.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/06_linear_regression/03_normal_equation_and_gd_tips.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/06_linear_regression/03_normal_equation_and_gd_tips.ipynb) |
| 04 | Bài 04 — Batch, Stochastic và Mini-batch Gradient Descent | 📖 Bài học | [View](04_batch_sgd_minibatch_gd.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/06_linear_regression/04_batch_sgd_minibatch_gd.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/06_linear_regression/04_batch_sgd_minibatch_gd.ipynb) |
| 05 | Bài 05 — Hồi quy đa thức (Polynomial Regression) | 📖 Bài học | [View](05_polynomial_regression.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/06_linear_regression/05_polynomial_regression.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/06_linear_regression/05_polynomial_regression.ipynb) |
| 06 | Bài 06 — Linear Regression theo template: xác suất, giả định, from scratch → thư viện, diễn giải | 📖 Bài học | [View](06_probabilistic_view_scratch_and_sklearn.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/06_linear_regression/06_probabilistic_view_scratch_and_sklearn.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/06_linear_regression/06_probabilistic_view_scratch_and_sklearn.ipynb) |
| 07 | Bài 07 — Regularization: Ridge, Lasso, Elastic Net | 📖 Bài học | [View](07_regularization_ridge_lasso.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/06_linear_regression/07_regularization_ridge_lasso.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/06_linear_regression/07_regularization_ridge_lasso.ipynb) |
| 08 | Bài 08 — Chẩn đoán model và các trường hợp Linear Regression thất bại | 📖 Bài học | [View](08_diagnostics_and_failure_cases.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/06_linear_regression/08_diagnostics_and_failure_cases.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/06_linear_regression/08_diagnostics_and_failure_cases.ipynb) |
| 90 | Bài tập — Chương 06: Linear Regression (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/06_linear_regression/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/06_linear_regression/90_exercises.ipynb) |
| 91 | Assignment 91 — Linear Regression with One Variable | 💻 Assignment | [View](91_assignment_linear_regression_one_variable.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/06_linear_regression/91_assignment_linear_regression_one_variable.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/06_linear_regression/91_assignment_linear_regression_one_variable.ipynb) |
| 92 | Assignment 92 — Linear Regression with Multiple Variables | 💻 Assignment | [View](92_assignment_linear_regression_multiple_variables.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/06_linear_regression/92_assignment_linear_regression_multiple_variables.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/06_linear_regression/92_assignment_linear_regression_multiple_variables.ipynb) |
| 93 | Assignment 93 — Stochastic, Mini-batch, Batch Gradient Descent | 💻 Assignment | [View](93_assignment_sgd_minibatch.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/06_linear_regression/93_assignment_sgd_minibatch.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/06_linear_regression/93_assignment_sgd_minibatch.ipynb) |
| 99 | ✅ Checkpoint — 06 · Linear Regression | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/06_linear_regression/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/06_linear_regression/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md) — 20 câu hỏi phải trả lời được trước khi sang chương sau.

## 🧩 Code tham chiếu

`src/ml_course/algorithms/linear_models.py`: `LinearRegressionScratch`, `RidgeScratch`, `LassoScratch` (đã test so với scikit-learn trong `tests/test_algorithms.py`).

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS.

## 🔗 Tiếp theo

[07 · Logistic Regression](../07_logistic_regression/README.md)

---
<sub>Bài 01–05, 07 và assignment 91–93 dựa trên [AIVN-Machine-Learning](https://github.com/thanhhff/AIVN-Machine-Learning), Tuần 2–5 (MIT License, © Nguyen Trung Thanh / AIVIETNAM; bài tập gốc từ khoá Machine Learning của Andrew Ng), đã được sửa lỗi, vector hoá và bổ sung. Bài 04 có thêm phần *Learning with Large Datasets* của Tuần 5 (bản gốc chưa có tài liệu).</sub>
