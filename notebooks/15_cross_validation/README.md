# Chương 15 · Cross-Validation

> Một con số đánh giá chỉ có giá trị khi cách chia dữ liệu **mô phỏng đúng** cách model sẽ được dùng. Chương này dạy chọn chiến lược validation, nhận diện rò rỉ trong CV và quy trình đánh giá chuẩn. Không có trong khoá AIVN gốc — bổ sung mới.

## 🎯 Mục tiêu

- Variance của hold-out; K-Fold from scratch, chọn $k$; Stratified, Repeated K-Fold; `cross_validate`, `cross_val_predict`.
- Dữ liệu có nhóm → **GroupKFold** (bệnh nhân, khách hàng); dữ liệu thời gian → **TimeSeriesSplit** (expanding/sliding, `gap`); LOO.
- So sánh ước lượng CV với hiệu năng thật: CV sai cách lạc quan đến mức nào.
- Rò rỉ khi chọn feature ngoài CV; **selection bias** khi báo cáo `best_score_`; **nested CV**; quy trình đánh giá end-to-end.

## 📋 Yêu cầu trước

[Chương 05](../05_machine_learning_fundamentals/README.md) (train/validation/test, data leakage), [Chương 04](../04_data_preprocessing/README.md) (Pipeline), [Chương 14](../14_model_evaluation/README.md).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Hold-out, K-Fold và Stratified K-Fold | 📖 Bài học | [View](01_holdout_kfold_stratified.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/15_cross_validation/01_holdout_kfold_stratified.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/15_cross_validation/01_holdout_kfold_stratified.ipynb) |
| 02 | Bài 02 — Group K-Fold, Time Series Split và Leave-One-Out | 📖 Bài học | [View](02_group_timeseries_loo.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/15_cross_validation/02_group_timeseries_loo.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/15_cross_validation/02_group_timeseries_loo.ipynb) |
| 03 | Bài 03 — Nested CV, rò rỉ trong CV và quy trình đánh giá chuẩn | 📖 Bài học | [View](03_nested_cv_and_protocol.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/15_cross_validation/03_nested_cv_and_protocol.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/15_cross_validation/03_nested_cv_and_protocol.ipynb) |
| 90 | Bài tập — Chương 15: Cross-Validation (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/15_cross_validation/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/15_cross_validation/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 15 · Cross-Validation | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/15_cross_validation/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/15_cross_validation/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## 🧭 Chọn chiến lược validation

```text
Mẫu độc lập?                     ── có ──► (Stratified) K-Fold 5–10, Repeated nếu ít dữ liệu
   │ không
   ├─ nhiều mẫu / thực thể      ──► GroupKFold / StratifiedGroupKFold
   ├─ theo thời gian / dự báo    ──► TimeSeriesSplit (+ gap, sliding window)
   └─ theo không gian            ──► spatial / block CV
Có tune hyperparameter và cần con số trung thực? ──► nested CV hoặc test hold-out dùng MỘT lần
```

## ❓ Bộ câu hỏi tự kiểm tra

[questions.md](questions.md) — trả lời không nhìn tài liệu; ôn lại sau 1 ngày, 1 tuần, 1 tháng.

## 🧩 Code tham chiếu

`src/ml_course/validation.py`: `KFoldScratch`, `StratifiedKFoldScratch`, `TimeSeriesSplitScratch`, `GroupKFoldScratch`, `cross_val_score_scratch`, `bootstrap_ci`.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS.

## 🔗 Tiếp theo

[16 · Hyperparameter Optimization](../16_hyperparameter_optimization/README.md)
