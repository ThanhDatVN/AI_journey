# Chương 14 · Model Evaluation

> "Model tốt" nghĩa là gì phụ thuộc hoàn toàn vào **câu hỏi bạn đặt ra**. Chương này biến mục tiêu nghiệp vụ thành metric, đo đúng cách và báo cáo kèm độ bất định. Không có trong khoá AIVN gốc — bổ sung mới.

## 🎯 Mục tiêu

- Confusion matrix, accuracy paradox, precision/recall/specificity/F1/$F_\beta$, macro/micro/weighted, balanced accuracy, MCC — from scratch.
- ROC curve from scratch, ROC-AUC = $P(s^+ > s^-)$; PR curve, average precision; ROC vs PR khi mất cân bằng; chọn ngưỡng (Youden, F1, ràng buộc, chi phí); log-loss, Brier.
- Metric hồi quy (MAE, RMSE, $R^2$, MAPE, WAPE, RMSLE, MedAE, pinball) — metric nào tương ứng đại lượng nào; bẫy MAPE và $R^2$.
- Case study chọn metric (gian lận @k, y tế specificity@recall, gợi ý NDCG, dự báo WAPE); khoảng tin cậy bootstrap, paired bootstrap, McNemar; checklist báo cáo.

## 📋 Yêu cầu trước

[Chương 05](../05_machine_learning_fundamentals/README.md) (train/validation/test), [Chương 07](../07_logistic_regression/README.md) (xác suất, ngưỡng).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Confusion matrix và các metric phân loại | 📖 Bài học | [View](01_confusion_matrix_and_metrics.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/14_model_evaluation/01_confusion_matrix_and_metrics.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/14_model_evaluation/01_confusion_matrix_and_metrics.ipynb) |
| 02 | Bài 02 — ROC, Precision–Recall và chọn ngưỡng | 📖 Bài học | [View](02_roc_pr_and_thresholds.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/14_model_evaluation/02_roc_pr_and_thresholds.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/14_model_evaluation/02_roc_pr_and_thresholds.ipynb) |
| 03 | Bài 03 — Metric cho hồi quy | 📖 Bài học | [View](03_regression_metrics.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/14_model_evaluation/03_regression_metrics.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/14_model_evaluation/03_regression_metrics.ipynb) |
| 04 | Bài 04 — Chọn metric: case study, khoảng tin cậy và so sánh model | 📖 Bài học | [View](04_choosing_metrics_case_studies.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/14_model_evaluation/04_choosing_metrics_case_studies.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/14_model_evaluation/04_choosing_metrics_case_studies.ipynb) |
| 90 | Bài tập — Chương 14: Model Evaluation (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/14_model_evaluation/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/14_model_evaluation/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 14 · Model Evaluation | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/14_model_evaluation/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/14_model_evaluation/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## 🧭 Bản đồ metric

```text
Phân loại (nhãn)      accuracy · precision · recall · specificity · F1/Fβ · MCC · balanced acc
Phân loại (xếp hạng)  ROC-AUC · PR-AUC (AP) · precision@k · recall@k
Phân loại (xác suất)  log-loss · Brier · calibration (chương 37)
Hồi quy               MAE (trung vị) · MSE/RMSE (trung bình) · pinball (phân vị) · R² · MAPE/WAPE · RMSLE
Xếp hạng/gợi ý        precision@k · recall@k · MAP · NDCG (chương 26)
Độ bất định           bootstrap CI · paired bootstrap · McNemar
```

## ❓ Bộ câu hỏi tự kiểm tra

[questions.md](questions.md) — trả lời không nhìn tài liệu; ôn lại sau 1 ngày, 1 tuần, 1 tháng.

## 🧩 Code tham chiếu

`src/ml_course/metrics.py` — các metric viết bằng NumPy (đã test so với scikit-learn trong `tests/test_metrics.py`).

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS.

## 🔗 Tiếp theo

[15 · Cross-Validation](../15_cross_validation/README.md)
