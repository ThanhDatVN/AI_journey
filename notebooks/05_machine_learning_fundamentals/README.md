# Chương 05 · Machine Learning Fundamentals

> Bài 01–02 dựa trên *Giới thiệu về Machine Learning* và *Các nhóm thuật toán trong Machine Learning* (Tuần 2) của khoá AIVN-Machine-Learning. Bài 03–05 hiện thực hoá nội dung **Tuần 6 — Áp dụng Machine Learning** mà khoá AIVN đã lên kế hoạch (đánh giá hàm giả thuyết, train/validation/test, bias–variance, learning curves, ML system design). Bài 06–08 bổ sung theo thiết kế khoá học.

**Đây là chương cực kỳ quan trọng trước khi học các model.**

## 🎯 Mục tiêu

- Hiểu ML là gì, khi nào nên/không nên dùng; định nghĩa T–P–E của Tom Mitchell.
- Phân biệt supervised / unsupervised / semi-supervised / self-supervised / reinforcement; regression vs classification; parameter vs hyperparameter.
- Dùng đúng train / validation / test; chọn model bằng validation.
- Hiểu underfitting/overfitting, **bias–variance tradeoff** (bằng mô phỏng), regularization.
- Chẩn đoán model bằng **learning curve**; quyết định "nên làm gì tiếp theo"; **error analysis**.
- Curse of dimensionality, No Free Lunch.
- **Data leakage** (6 dạng, có code tái hiện) và **distribution shift** (covariate / label / concept).

## 📋 Yêu cầu trước

[Chương 02](../02_mathematics_for_ml/README.md) (gradient descent, MLE/MAP), [Chương 04](../04_data_preprocessing/README.md) (Pipeline).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Machine Learning là gì? | 📖 Bài học | [View](01_what_is_machine_learning.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/05_machine_learning_fundamentals/01_what_is_machine_learning.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/05_machine_learning_fundamentals/01_what_is_machine_learning.ipynb) |
| 02 | Bài 02 — Phân nhóm các thuật toán và thuật ngữ cơ bản | 📖 Bài học | [View](02_types_of_machine_learning.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/05_machine_learning_fundamentals/02_types_of_machine_learning.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/05_machine_learning_fundamentals/02_types_of_machine_learning.ipynb) |
| 03 | Bài 03 — Train / Validation / Test và lựa chọn model | 📖 Bài học | [View](03_train_validation_test_and_model_selection.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/05_machine_learning_fundamentals/03_train_validation_test_and_model_selection.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/05_machine_learning_fundamentals/03_train_validation_test_and_model_selection.ipynb) |
| 04 | Bài 04 — Underfitting, Overfitting và Bias–Variance Tradeoff | 📖 Bài học | [View](04_overfitting_and_bias_variance.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/05_machine_learning_fundamentals/04_overfitting_and_bias_variance.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/05_machine_learning_fundamentals/04_overfitting_and_bias_variance.ipynb) |
| 05 | Bài 05 — Learning curves, chẩn đoán model và error analysis | 📖 Bài học | [View](05_learning_curves_and_error_analysis.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/05_machine_learning_fundamentals/05_learning_curves_and_error_analysis.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/05_machine_learning_fundamentals/05_learning_curves_and_error_analysis.ipynb) |
| 06 | Bài 06 — Curse of Dimensionality và No Free Lunch | 📖 Bài học | [View](06_curse_of_dimensionality_and_no_free_lunch.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/05_machine_learning_fundamentals/06_curse_of_dimensionality_and_no_free_lunch.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/05_machine_learning_fundamentals/06_curse_of_dimensionality_and_no_free_lunch.ipynb) |
| 07 | Bài 07 — Data Leakage: lỗi nguy hiểm nhất trong ML thực tế | 📖 Bài học | [View](07_data_leakage.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/05_machine_learning_fundamentals/07_data_leakage.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/05_machine_learning_fundamentals/07_data_leakage.ipynb) |
| 08 | Bài 08 — Distribution Shift: khi dữ liệu thật khác dữ liệu train | 📖 Bài học | [View](08_distribution_shift.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/05_machine_learning_fundamentals/08_distribution_shift.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/05_machine_learning_fundamentals/08_distribution_shift.ipynb) |
| 90 | Bài tập — Chương 05: ML Fundamentals (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/05_machine_learning_fundamentals/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/05_machine_learning_fundamentals/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 05 · ML Fundamentals | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/05_machine_learning_fundamentals/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/05_machine_learning_fundamentals/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## 🧭 Bản đồ khái niệm

```text
Machine Learning
├── Supervised: Regression · Classification · Ranking
├── Unsupervised: Clustering · Dim. reduction · Anomaly detection
├── Semi-/Self-supervised
└── Reinforcement Learning

Feature · Target · Sample · Dataset
Train → Validation → Test
Parameters ↔ Hyperparameters
Loss → Objective → Optimization → Generalization

Underfitting ↔ Overfitting  =  Bias ↔ Variance
Learning curves · Error analysis
Curse of dimensionality · No Free Lunch
Data leakage · Distribution shift
```

## ❓ Bộ câu hỏi tự kiểm tra

[questions.md](questions.md) — trả lời không nhìn tài liệu; ôn lại sau 1 ngày, 1 tuần, 1 tháng.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS.

## 🔗 Tiếp theo

[06 · Linear Regression](../06_linear_regression/README.md)

---
<sub>Bài 01–02 dựa trên Tuần 2 của [AIVN-Machine-Learning](https://github.com/thanhhff/AIVN-Machine-Learning) (MIT License, © Nguyen Trung Thanh / AIVIETNAM); hình minh hoạ overfitting/bias–variance từ Tuần 4 của khoá này.</sub>
