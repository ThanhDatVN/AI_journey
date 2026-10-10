# Chương 04 · Data Preprocessing

> Bài 02 dựa trên *Feature Normalization* (Tuần 3) của khoá AIVN-Machine-Learning — đã sửa lỗi công thức/thuật ngữ và mở rộng; các bài còn lại bổ sung theo thiết kế khoá học.

## 🎯 Mục tiêu

- Điền missing value đúng cách (median, cột chỉ báo, KNN) và xử lý outlier mà không xoá dữ liệu.
- Hiểu sâu feature scaling: min-max, z-score, mean normalization, robust; log/Box-Cox/Yeo-Johnson; model nào cần scaling và vì sao.
- Mã hoá biến phân loại: one-hot, ordinal, high-cardinality (gộp hiếm, frequency, **target encoding out-of-fold**).
- Ghép mọi bước vào `Pipeline` + `ColumnTransformer` — **không rò rỉ**, dễ cross-validate và triển khai.

## 📋 Yêu cầu trước

[Chương 03](../03_data_analysis_and_eda/README.md).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Xử lý missing value và outlier đúng cách | 📖 Bài học | [View](01_missing_values_and_outliers.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/04_data_preprocessing/01_missing_values_and_outliers.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/04_data_preprocessing/01_missing_values_and_outliers.ipynb) |
| 02 | Bài 02 — Feature Scaling: chuẩn hoá dữ liệu | 📖 Bài học | [View](02_feature_scaling.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/04_data_preprocessing/02_feature_scaling.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/04_data_preprocessing/02_feature_scaling.ipynb) |
| 03 | Bài 03 — Mã hoá biến phân loại (categorical encoding) | 📖 Bài học | [View](03_encoding_categorical.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/04_data_preprocessing/03_encoding_categorical.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/04_data_preprocessing/03_encoding_categorical.ipynb) |
| 04 | Bài 04 — Pipeline và ColumnTransformer: tiền xử lý không rò rỉ | 📖 Bài học | [View](04_pipelines_and_column_transformer.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/04_data_preprocessing/04_pipelines_and_column_transformer.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/04_data_preprocessing/04_pipelines_and_column_transformer.ipynb) |
| 90 | Bài tập — Chương 04: Data Preprocessing (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/04_data_preprocessing/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/04_data_preprocessing/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 04 · Data Preprocessing | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/04_data_preprocessing/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/04_data_preprocessing/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## 🔑 Nguyên tắc vàng của chương

```text
Mọi bước HỌC TỪ DỮ LIỆU (imputer, scaler, encoder, selector, sampler...)
→ chỉ fit trên TRAIN
→ đặt trong Pipeline để cross-validation tự động làm đúng
```

## ❓ Bộ câu hỏi tự kiểm tra

[questions.md](questions.md) — trả lời không nhìn tài liệu; ôn lại sau 1 ngày, 1 tuần, 1 tháng.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, mini project (pipeline giá nhà) PASS.

## 🔗 Tiếp theo

[05 · ML Fundamentals](../05_machine_learning_fundamentals/README.md)

---
<sub>Bài 02 dựa trên [AIVN-Machine-Learning](https://github.com/thanhhff/AIVN-Machine-Learning), Tuần 3 (MIT License, © Nguyen Trung Thanh / AIVIETNAM).</sub>
