# Chương 10 · Decision Trees

> Model "giống cách con người ra quyết định": một chuỗi câu hỏi có/không. Dễ đọc khi nhỏ, nguy hiểm khi lớn — và là viên gạch của Random Forest, Gradient Boosting (chương 11–12). Không có trong khoá AIVN gốc — bổ sung mới theo template "mọi model".

## 🎯 Mục tiêu

- Trực giác cây quyết định; ranh giới song song trục; impurity (entropy, Gini, misclassification) và information gain — tính tay theo ID3.
- Cài đặt CART **from scratch**: tìm split $O(n\log n)$ bằng sắp xếp + cumsum, xây cây đệ quy, MDI feature importance.
- Regression tree: giảm phương sai, dự đoán bậc thang, MSE vs MAE, **không ngoại suy**.
- Overfitting, pre-pruning, **cost-complexity pruning**; variance cao/không ổn định.
- Failure case (ranh giới chéo, quan hệ tuyến tính mượt) và diễn giải: luật, decision path, bẫy của MDI importance.

## 📋 Yêu cầu trước

[Chương 05](../05_machine_learning_fundamentals/README.md) (bias–variance, CV), [Chương 02](../02_mathematics_for_ml/README.md) (entropy, xác suất).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Decision Tree: trực giác, entropy và Gini | 📖 Bài học | [View](01_intuition_entropy_gini.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/10_decision_trees/01_intuition_entropy_gini.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/10_decision_trees/01_intuition_entropy_gini.ipynb) |
| 02 | Bài 02 — Cài đặt Decision Tree (CART) from scratch | 📖 Bài học | [View](02_classifier_from_scratch.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/10_decision_trees/02_classifier_from_scratch.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/10_decision_trees/02_classifier_from_scratch.ipynb) |
| 03 | Bài 03 — Regression Tree | 📖 Bài học | [View](03_regression_trees.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/10_decision_trees/03_regression_trees.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/10_decision_trees/03_regression_trees.ipynb) |
| 04 | Bài 04 — Overfitting, pre-pruning và cost-complexity pruning | 📖 Bài học | [View](04_overfitting_and_pruning.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/10_decision_trees/04_overfitting_and_pruning.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/10_decision_trees/04_overfitting_and_pruning.ipynb) |
| 05 | Bài 05 — Failure case và diễn giải Decision Tree | 📖 Bài học | [View](05_failure_cases_and_interpretation.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/10_decision_trees/05_failure_cases_and_interpretation.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/10_decision_trees/05_failure_cases_and_interpretation.ipynb) |
| 90 | Bài tập — Chương 10: Decision Trees (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/10_decision_trees/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/10_decision_trees/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 10 · Decision Trees | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/10_decision_trees/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/10_decision_trees/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

`src/ml_course/algorithms/tree.py`: `DecisionTreeScratch` (classification/regression, gini/entropy/mse, `max_features` cho Random Forest).

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS.

## 🔗 Tiếp theo

[11 · Random Forest](../11_random_forest/README.md)
