# Chương 20 · Clustering

> Học không giám sát đầu tiên: tìm nhóm khi không có nhãn. Ba họ thuật toán — dựa trên tâm (K-Means), phân cấp (agglomerative), dựa trên mật độ (DBSCAN/HDBSCAN) — mỗi họ mang một định nghĩa "cụm" khác nhau. Không có trong khoá AIVN gốc — bổ sung mới theo template "mọi model".

## 🎯 Mục tiêu

- **K-Means from scratch**: hàm mục tiêu, Lloyd, vì sao hội tụ, k-means++ và cực tiểu địa phương; phân khúc khách hàng (đơn vị/scaling), lượng tử hoá màu ảnh, Mini-Batch.
- Chọn K: elbow, **silhouette from scratch**, Calinski-Harabasz, Davies-Bouldin; ARI/NMI; 5 failure case của K-Means.
- **Hierarchical clustering from scratch**: linkage (single/complete/average/Ward), dendrogram, chaining.
- **DBSCAN from scratch**: core/border/noise, k-distance plot; HDBSCAN cho mật độ khác nhau; bảng so sánh thuật toán.

## 📋 Yêu cầu trước

[Chương 08](../08_knn/README.md) (khoảng cách, scaling), [Chương 05](../05_machine_learning_fundamentals/README.md) (học không giám sát).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — K-Means from scratch | 📖 Bài học | [View](01_kmeans_from_scratch.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/20_clustering/01_kmeans_from_scratch.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/20_clustering/01_kmeans_from_scratch.ipynb) |
| 02 | Bài 02 — Chọn số cụm K và khi nào K-Means thất bại | 📖 Bài học | [View](02_choosing_k_and_failure_cases.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/20_clustering/02_choosing_k_and_failure_cases.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/20_clustering/02_choosing_k_and_failure_cases.ipynb) |
| 03 | Bài 03 — Hierarchical clustering | 📖 Bài học | [View](03_hierarchical_clustering.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/20_clustering/03_hierarchical_clustering.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/20_clustering/03_hierarchical_clustering.ipynb) |
| 04 | Bài 04 — DBSCAN và HDBSCAN: phân cụm theo mật độ | 📖 Bài học | [View](04_dbscan.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/20_clustering/04_dbscan.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/20_clustering/04_dbscan.ipynb) |
| 90 | Bài tập — Chương 20: Clustering (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/20_clustering/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/20_clustering/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 20 · Clustering | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/20_clustering/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/20_clustering/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

`src/ml_course/algorithms/cluster.py`: `KMeansScratch` (k-means++, n_init), `DBSCANScratch`.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS.

## 🔗 Tiếp theo

[21 · Dimensionality Reduction](../21_dimensionality_reduction/README.md)
