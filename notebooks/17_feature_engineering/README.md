# Chương 17 · Feature Engineering

> "Dữ liệu và feature quyết định giới hạn trên; model và thuật toán chỉ tiến gần giới hạn đó." Chương này biến hiểu biết về bài toán thành feature — và đóng gói tất cả vào một Pipeline không rò rỉ. Không có trong khoá AIVN gốc — bổ sung mới.

## 🎯 Mục tiêu

- Biến đổi số: log/Box-Cox/Yeo-Johnson/quantile, binning, **spline**, winsorize, feature miền (tỉ số); model nào cần biến đổi nào.
- Mã hoá category: one-hot (`min_frequency`, `handle_unknown`), ordinal, frequency, hashing, **target encoding** (bẫy rò rỉ, cross-fit, smoothing), native categorical.
- Tương tác, feature thời gian (mã hoá chu kỳ), feature văn bản đơn giản + TF-IDF, **aggregation theo thực thể chỉ từ quá khứ**.
- Transformer tuỳ chỉnh, Pipeline end-to-end nhận dữ liệu thô, ablation, lưu/tải pipeline.

## 📋 Yêu cầu trước

[Chương 04](../04_data_preprocessing/README.md) (Pipeline, ColumnTransformer), [Chương 15](../15_cross_validation/README.md) (CV đúng cách).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Biến đổi feature số | 📖 Bài học | [View](01_numeric_transforms.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/17_feature_engineering/01_numeric_transforms.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/17_feature_engineering/01_numeric_transforms.ipynb) |
| 02 | Bài 02 — Mã hoá feature phân loại | 📖 Bài học | [View](02_categorical_encoding.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/17_feature_engineering/02_categorical_encoding.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/17_feature_engineering/02_categorical_encoding.ipynb) |
| 03 | Bài 03 — Tương tác, thời gian, văn bản và feature tổng hợp (aggregation) | 📖 Bài học | [View](03_interaction_datetime_text_aggregation.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/17_feature_engineering/03_interaction_datetime_text_aggregation.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/17_feature_engineering/03_interaction_datetime_text_aggregation.ipynb) |
| 04 | Bài 04 — Pipeline feature engineering end-to-end | 📖 Bài học | [View](04_end_to_end_pipeline.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/17_feature_engineering/04_end_to_end_pipeline.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/17_feature_engineering/04_end_to_end_pipeline.ipynb) |
| 90 | Bài tập — Chương 17: Feature Engineering (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/17_feature_engineering/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/17_feature_engineering/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 17 · Feature Engineering | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/17_feature_engineering/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/17_feature_engineering/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## 🧭 Cheat sheet

| Tình huống | Feature engineering |
|---|---|
| Feature/target lệch phải, quan hệ nhân | log / Box-Cox (đặc biệt cho model tuyến tính) |
| Quan hệ phi tuyến, model tuyến tính | spline, binning, tương tác |
| Category ít giá trị | one-hot |
| Category nhiều giá trị | target encoding (cross-fit + smoothing), hashing, native |
| Thời gian | thành phần + mã hoá chu kỳ + khoảng cách tới sự kiện |
| Hành vi theo thực thể | aggregation từ quá khứ (count/mean/ratio trong cửa sổ) |
| Văn bản | độ dài, từ khoá, TF-IDF (→ embedding, chương 32) |

## ❓ Bộ câu hỏi tự kiểm tra

[questions.md](questions.md) — trả lời không nhìn tài liệu; ôn lại sau 1 ngày, 1 tuần, 1 tháng.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS.

## 🔗 Tiếp theo

[18 · Feature Selection](../18_feature_selection/README.md)
