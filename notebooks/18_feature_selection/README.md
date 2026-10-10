# Chương 18 · Feature Selection

> Ít feature hơn có thể nghĩa là model nhanh hơn, rẻ hơn, dễ hiểu hơn — và với dữ liệu ít, chính xác hơn. Nhưng mỗi phương pháp chọn mang giả định riêng, và chọn sai cách là một trong những nguồn rò rỉ phổ biến nhất. Không có trong khoá AIVN gốc — bổ sung mới.

## 🎯 Mục tiêu

- Khi nào chọn feature giúp (ít dữ liệu, chi phí, diễn giải, model dựa trên khoảng cách) và khi nào không cần.
- **Filter**: variance threshold, F-test/Pearson/Spearman, **mutual information** (phi tuyến), VIF cho đa cộng tuyến.
- **Wrapper**: forward selection from scratch, SFS forward/backward, RFE/RFECV, độ ổn định, nested CV cho quy trình chọn.
- **Embedded**: Lasso path, stability selection, `SelectFromModel`, shadow features (Boruta); bẫy: feature tương quan, chọn ≠ nhân quả.

## 📋 Yêu cầu trước

[Chương 15](../15_cross_validation/README.md) (rò rỉ khi chọn feature ngoài CV), [Chương 06](../06_linear_regression/README.md) (L1/L2), [Chương 10–11](../10_decision_trees/README.md) (importance).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Feature selection: vì sao và filter methods | 📖 Bài học | [View](01_filter_methods.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/18_feature_selection/01_filter_methods.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/18_feature_selection/01_filter_methods.ipynb) |
| 02 | Bài 02 — Wrapper methods: forward, backward, RFE | 📖 Bài học | [View](02_wrapper_methods.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/18_feature_selection/02_wrapper_methods.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/18_feature_selection/02_wrapper_methods.ipynb) |
| 03 | Bài 03 — Embedded methods và các bẫy khi chọn feature | 📖 Bài học | [View](03_embedded_methods_and_pitfalls.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/18_feature_selection/03_embedded_methods_and_pitfalls.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/18_feature_selection/03_embedded_methods_and_pitfalls.ipynb) |
| 90 | Bài tập — Chương 18: Feature Selection (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/18_feature_selection/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/18_feature_selection/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 18 · Feature Selection | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/18_feature_selection/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/18_feature_selection/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## 🧭 Chọn phương pháp

| Họ | Ví dụ | Ưu | Nhược |
|---|---|---|---|
| Filter | variance, F-test, MI, chi² | rất nhanh, độc lập model | bỏ sót tương tác, không thấy dư thừa |
| Wrapper | forward/backward, RFE(CV) | theo đúng model, thấy dư thừa | đắt, dễ overfit quá trình chọn |
| Embedded | Lasso, Elastic Net, SelectFromModel, shadow | chọn trong lúc train | phụ thuộc model và importance |

## ❓ Bộ câu hỏi tự kiểm tra

[questions.md](questions.md) — trả lời không nhìn tài liệu; ôn lại sau 1 ngày, 1 tuần, 1 tháng.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS.

## 🔗 Tiếp theo

[19 · Imbalanced Learning](../19_imbalanced_learning/README.md)
