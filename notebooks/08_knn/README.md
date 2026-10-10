# Chương 08 · K-Nearest Neighbors

> Model "không học gì" nhưng dạy rất nhiều: khoảng cách, scaling, bias–variance qua một núm vặn $k$, và curse of dimensionality. Không có trong khoá AIVN gốc — bổ sung mới theo template "mọi model".

## 🎯 Mục tiêu

- Hiểu ý tưởng "hỏi hàng xóm", lazy learning, non-parametric; decision boundary theo $k$ (Voronoi khi $k = 1$).
- Các thước đo khoảng cách (Euclid, Manhattan, Minkowski, cosine, Hamming) và khi nào dùng cái nào.
- Cài đặt KNN classifier/regressor from scratch: vòng lặp → vector hoá; hoà phiếu; trọng số khoảng cách; KD-tree vs brute force.
- Trả lời bằng thực nghiệm: **vì sao KNN gần như bắt buộc phải scaling?** Chọn $k$, `weights`, `metric` bằng CV.
- 5 failure case: nhiều chiều, feature vô ích, mất cân bằng, dữ liệu lớn, ngoại suy — và cách khắc phục.

## 📋 Yêu cầu trước

[Chương 05](../05_machine_learning_fundamentals/README.md) (bias–variance, CV, curse of dimensionality), [Chương 04](../04_data_preprocessing/README.md) (scaling, Pipeline).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — K-Nearest Neighbors: trực giác và khoảng cách | 📖 Bài học | [View](01_intuition_and_distances.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/08_knn/01_intuition_and_distances.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/08_knn/01_intuition_and_distances.ipynb) |
| 02 | Bài 02 — Cài đặt KNN from scratch | 📖 Bài học | [View](02_from_scratch.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/08_knn/02_from_scratch.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/08_knn/02_from_scratch.ipynb) |
| 03 | Bài 03 — Vì sao KNN gần như bắt buộc phải scaling? Chọn k thế nào? | 📖 Bài học | [View](03_scaling_and_choosing_k.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/08_knn/03_scaling_and_choosing_k.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/08_knn/03_scaling_and_choosing_k.ipynb) |
| 04 | Bài 04 — Khi nào KNN thất bại? Curse of dimensionality và các failure case | 📖 Bài học | [View](04_failure_cases_and_curse_of_dimensionality.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/08_knn/04_failure_cases_and_curse_of_dimensionality.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/08_knn/04_failure_cases_and_curse_of_dimensionality.ipynb) |
| 90 | Bài tập — Chương 08: K-Nearest Neighbors (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/08_knn/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/08_knn/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 08 · K-Nearest Neighbors | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/08_knn/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/08_knn/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

`src/ml_course/algorithms/knn.py`: `pairwise_distances`, `KNNClassifierScratch`, `KNNRegressorScratch`.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS.

## 🔗 Tiếp theo

[09 · Naive Bayes](../09_naive_bayes/README.md)
