# Chương 21 · Dimensionality Reduction

> Dữ liệu nhiều chiều thường "sống" trên một cấu trúc ít chiều hơn nhiều. PCA tìm cấu trúc tuyến tính đó (nén, khử nhiễu, tiền xử lý); t-SNE và UMAP vẽ bản đồ phi tuyến để khám phá — với những bẫy diễn giải cần biết. Không có trong khoá AIVN gốc — bổ sung mới theo template "mọi model".

## 🎯 Mục tiêu

- **PCA from scratch** bằng trị riêng và SVD; hai góc nhìn (tối đa phương sai ↔ tối thiểu sai số tái tạo); explained variance; power iteration.
- PCA thực tế: scaling, trực quan hoá, nén & tái tạo ảnh (eigendigits), khử nhiễu, tiền xử lý chọn bằng CV, khi PCA làm hại; Kernel PCA, TruncatedSVD, IncrementalPCA.
- **t-SNE** (perplexity, Student-t, KL) và **UMAP**; bẫy diễn giải (kích thước, khoảng cách, cụm từ nhiễu); trustworthiness.

## 📋 Yêu cầu trước

[Chương 02](../02_mathematics_for_ml/README.md) (trị riêng, SVD), [Chương 05](../05_machine_learning_fundamentals/README.md) (curse of dimensionality), [Chương 20](../20_clustering/README.md).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — PCA from scratch | 📖 Bài học | [View](01_pca_from_scratch.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/21_dimensionality_reduction/01_pca_from_scratch.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/21_dimensionality_reduction/01_pca_from_scratch.ipynb) |
| 02 | Bài 02 — PCA trong thực tế | 📖 Bài học | [View](02_pca_in_practice.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/21_dimensionality_reduction/02_pca_in_practice.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/21_dimensionality_reduction/02_pca_in_practice.ipynb) |
| 03 | Bài 03 — t-SNE và UMAP: trực quan hoá phi tuyến | 📖 Bài học | [View](03_tsne_umap.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/21_dimensionality_reduction/03_tsne_umap.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/21_dimensionality_reduction/03_tsne_umap.ipynb) |
| 90 | Bài tập — Chương 21: Dimensionality Reduction (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/21_dimensionality_reduction/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/21_dimensionality_reduction/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 21 · Dimensionality Reduction | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/21_dimensionality_reduction/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/21_dimensionality_reduction/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

`src/ml_course/algorithms/decomposition.py`: `PCAScratch` (SVD, `n_components` dạng số hoặc tỉ lệ phương sai, `inverse_transform`).

## 🧰 Thư viện tuỳ chọn

`umap-learn` — `pip install -e ".[viz]"` (notebook tự bỏ qua phần UMAP nếu chưa cài).

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS.

## 🔗 Tiếp theo

[22 · Anomaly Detection](../22_anomaly_detection/README.md)
