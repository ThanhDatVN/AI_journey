# Chương 26 · Recommender Systems

> Hệ gợi ý trả lời câu hỏi "nên cho user này xem gì?" từ dữ liệu thưa, đuôi dài và đầy thiên lệch chọn lựa. Chương đi từ baseline popularity và content-based, qua collaborative filtering láng giềng, matrix factorization (SGD, ALS, PureSVD), tới đánh giá đúng cách — vì sao RMSE tốt không có nghĩa là gợi ý tốt. Không có trong khoá AIVN gốc — bổ sung mới theo template "mọi model".

## 🎯 Mục tiêu

- Phản hồi tường minh/ngầm; dự đoán rating vs top-N; dữ liệu thưa, đuôi dài; giao thức đánh giá theo từng user.
- Popularity (đếm, trung bình giảm chấn), content-based (hồ sơ item/user), lai.
- Baseline predictor $\mu + b_u + b_i$; KNN user/item (phần dư, shrinkage); KNN trên phản hồi ngầm.
- Matrix factorization: SGD (Funk SVD), ALS, regularization, diễn giải nhân tố; PureSVD; BPR (bài tập).
- Metric xếp hạng (precision/recall@k, MRR, MAP, NDCG); MNAR; coverage, novelty, diversity; MMR; offline vs online; cold start.

## 📋 Yêu cầu trước

[Chương 08](../08_knn/README.md) (KNN, cosine), [Chương 21](../21_dimensionality_reduction/README.md) (SVD), [Chương 06](../06_linear_regression/README.md) (Ridge), [Chương 14](../14_model_evaluation/README.md).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Bài toán gợi ý, popularity và content-based | 📖 Bài học | [View](01_popularity_and_content_based.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/26_recommender_systems/01_popularity_and_content_based.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/26_recommender_systems/01_popularity_and_content_based.ipynb) |
| 02 | Bài 02 — Collaborative filtering dựa trên láng giềng | 📖 Bài học | [View](02_collaborative_filtering.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/26_recommender_systems/02_collaborative_filtering.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/26_recommender_systems/02_collaborative_filtering.ipynb) |
| 03 | Bài 03 — Matrix factorization | 📖 Bài học | [View](03_matrix_factorization.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/26_recommender_systems/03_matrix_factorization.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/26_recommender_systems/03_matrix_factorization.ipynb) |
| 04 | Bài 04 — Đánh giá hệ gợi ý: metric xếp hạng và hơn thế nữa | 📖 Bài học | [View](04_ranking_metrics.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/26_recommender_systems/04_ranking_metrics.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/26_recommender_systems/04_ranking_metrics.ipynb) |
| 90 | Bài tập — Chương 26: Recommender Systems (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/26_recommender_systems/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/26_recommender_systems/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 26 · Recommender Systems | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/26_recommender_systems/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/26_recommender_systems/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🗂️ Dữ liệu

`data/sample/ratings.csv`, `items.csv` — sinh bởi `ml_course.datasets.make_ratings`: mô hình nhân tố ẩn + thiên lệch, độ phổ biến Zipf, và **user tự chọn phim hợp gu** (`selection_strength`) → rating quan sát được lệch cao như dữ liệu thật.

## 🧩 Code tham chiếu

`ml_course.metrics`: `precision_at_k`, `recall_at_k`, `average_precision_at_k`, `ndcg_at_k`. Bài 02: baseline predictor + KNN; bài 03: `mf_sgd`, `mf_als`, PureSVD; bài 04: `evaluate` (precision, NDCG, coverage, novelty, diversity), MMR. Bài tập: `FunkSVD`, BPR, chiến lược cold start.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS (top-10 vượt popularity).

## 🔗 Tiếp theo

[27 · Deep Learning Foundations](../27_deep_learning_foundations/README.md)
