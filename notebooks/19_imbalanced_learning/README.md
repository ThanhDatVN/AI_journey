# Chương 19 · Imbalanced Learning

> Gian lận, bệnh hiếm, lỗi sản xuất, khách hàng rời bỏ — những thứ quan trọng thường hiếm. Chương này phân biệt vấn đề thật (ít mẫu thiểu số, ngưỡng sai, metric sai) với "thuốc chữa" phổ biến nhưng thường bị dùng sai (SMOTE, cân bằng 50/50). Không có trong khoá AIVN gốc — bổ sung mới.

## 🎯 Mục tiêu

- Accuracy paradox; metric đúng (PR-AUC, precision/recall@k, chi phí); vì sao xác suất "thấp" (prior); số mẫu thiểu số quan trọng hơn tỉ lệ.
- Undersampling, oversampling, **SMOTE from scratch**; resampling đúng chỗ (meta-estimator trong CV); **hiệu chỉnh prior**; bẫy resample trước khi chia.
- Class weight, **chọn ngưỡng theo chi phí** (kể cả chi phí phụ thuộc từng mẫu), so sánh công bằng các kỹ thuật, quy trình khuyến nghị.

## 📋 Yêu cầu trước

[Chương 14](../14_model_evaluation/README.md) (PR curve, ngưỡng, chi phí), [Chương 15](../15_cross_validation/README.md) (rò rỉ trong CV).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Dữ liệu mất cân bằng: vì sao accuracy nói dối | 📖 Bài học | [View](01_why_accuracy_lies.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/19_imbalanced_learning/01_why_accuracy_lies.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/19_imbalanced_learning/01_why_accuracy_lies.ipynb) |
| 02 | Bài 02 — Resampling: undersampling, oversampling, SMOTE | 📖 Bài học | [View](02_resampling_and_smote.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/19_imbalanced_learning/02_resampling_and_smote.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/19_imbalanced_learning/02_resampling_and_smote.ipynb) |
| 03 | Bài 03 — Class weight, chọn ngưỡng và học theo chi phí | 📖 Bài học | [View](03_class_weights_threshold_cost.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/19_imbalanced_learning/03_class_weights_threshold_cost.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/19_imbalanced_learning/03_class_weights_threshold_cost.ipynb) |
| 90 | Bài tập — Chương 19: Imbalanced Learning (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/19_imbalanced_learning/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/19_imbalanced_learning/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 19 · Imbalanced Learning | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/19_imbalanced_learning/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/19_imbalanced_learning/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## 🧭 Quy trình khuyến nghị

```text
1. Metric đúng (PR-AUC, chi phí, @k) + chia stratified, đủ positive trong valid/test (báo cáo CI).
2. Model mạnh, chưa resample: Logistic Regression (baseline) + Gradient Boosting.
3. Chọn model theo PR-AUC / chi phí trên validation.
4. Chọn ngưỡng theo chi phí hoặc năng lực vận hành (validation).
5. class_weight / resampling chỉ khi cải thiện xếp hạng trên CV; cần xác suất đúng → hiệu chỉnh prior / calibrate.
```

## 🧰 Thư viện tuỳ chọn

`imbalanced-learn` (`imblearn.pipeline.Pipeline`, SMOTE, EasyEnsemble, BalancedRandomForest) — chương này tự cài đặt các kỹ thuật cần thiết nên không bắt buộc.

## ❓ Bộ câu hỏi tự kiểm tra

[questions.md](questions.md) — trả lời không nhìn tài liệu; ôn lại sau 1 ngày, 1 tuần, 1 tháng.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS.

## 🔗 Tiếp theo

[20 · Clustering](../20_clustering/README.md)
