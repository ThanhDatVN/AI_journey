# Chương 11 · Random Forest

> Biến một model không ổn định (cây quyết định) thành một trong những model "out-of-the-box" mạnh nhất cho dữ liệu bảng — chỉ bằng **lấy trung bình** và **ngẫu nhiên hoá có chủ đích**. Không có trong khoá AIVN gốc — bổ sung mới theo template "mọi model".

## 🎯 Mục tiêu

- Vì sao trung bình giảm variance và vì sao **tương quan** đặt giới hạn: $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$ (kiểm chứng bằng mô phỏng).
- Bootstrap, bagging, đo bias–variance bằng mô phỏng: bagging giảm variance, không giảm bias.
- Random Forest **from scratch**: bootstrap + random feature subset ở mỗi split; đánh đổi `max_features` ↔ tương quan.
- OOB score from scratch, feature importance (MDI vs permutation) và bẫy feature tương quan, tuning, Extra Trees.
- Failure case: ngoại suy, quan hệ tuyến tính cộng dồn, chi phí triển khai, xác suất chưa calibrate.

## 📋 Yêu cầu trước

[Chương 10](../10_decision_trees/README.md) (cây quyết định), [Chương 05](../05_machine_learning_fundamentals/README.md) (bias–variance).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Bagging: giảm variance bằng cách lấy trung bình | 📖 Bài học | [View](01_bagging_and_variance_reduction.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/11_random_forest/01_bagging_and_variance_reduction.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/11_random_forest/01_bagging_and_variance_reduction.ipynb) |
| 02 | Bài 02 — Random Forest from scratch | 📖 Bài học | [View](02_random_forest_from_scratch.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/11_random_forest/02_random_forest_from_scratch.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/11_random_forest/02_random_forest_from_scratch.ipynb) |
| 03 | Bài 03 — OOB score, feature importance, tuning và failure case | 📖 Bài học | [View](03_oob_importance_and_tuning.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/11_random_forest/03_oob_importance_and_tuning.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/11_random_forest/03_oob_importance_and_tuning.ipynb) |
| 90 | Bài tập — Chương 11: Random Forest (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/11_random_forest/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/11_random_forest/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 11 · Random Forest | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/11_random_forest/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/11_random_forest/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

`src/ml_course/algorithms/ensemble.py`: `RandomForestScratch` (dùng `DecisionTreeScratch` của chương 10, có OOB score).

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS.

## 🔗 Tiếp theo

[12 · Gradient Boosting](../12_gradient_boosting/README.md)
