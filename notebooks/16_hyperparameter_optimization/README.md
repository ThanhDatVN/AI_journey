# Chương 16 · Hyperparameter Optimization

> Tuning là tối ưu một hàm **đắt, nhiễu, không có gradient**. Chương này đi từ grid search đến random search, successive halving/Hyperband và Bayesian optimization (GP, TPE với Optuna) — kèm cảnh báo về overfit validation. Không có trong khoá AIVN gốc — bổ sung mới.

## 🎯 Mục tiêu

- Vì sao random search thắng grid search (low effective dimensionality); chọn phân phối tìm kiếm (log-uniform); early stopping thay vì tune số vòng.
- **Successive halving from scratch**, `HalvingRandomSearchCV`, Hyperband — đánh giá rẻ để loại sớm.
- **Bayesian optimization from scratch** (Gaussian Process + Expected Improvement); Optuna: TPE, không gian điều kiện, pruning.
- Thực hành tốt: baseline trước, không gian hợp lý, ngân sách, tránh overfit validation, tái lập.

## 📋 Yêu cầu trước

[Chương 15](../15_cross_validation/README.md) (CV, nested CV), [Chương 12](../12_gradient_boosting/README.md).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Grid Search vs Random Search | 📖 Bài học | [View](01_grid_vs_random_search.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/16_hyperparameter_optimization/01_grid_vs_random_search.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/16_hyperparameter_optimization/01_grid_vs_random_search.ipynb) |
| 02 | Bài 02 — Successive Halving và Hyperband | 📖 Bài học | [View](02_successive_halving_hyperband.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/16_hyperparameter_optimization/02_successive_halving_hyperband.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/16_hyperparameter_optimization/02_successive_halving_hyperband.ipynb) |
| 03 | Bài 03 — Bayesian Optimization và Optuna | 📖 Bài học | [View](03_bayesian_optimization_optuna.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/16_hyperparameter_optimization/03_bayesian_optimization_optuna.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/16_hyperparameter_optimization/03_bayesian_optimization_optuna.ipynb) |
| 90 | Bài tập — Chương 16: Hyperparameter Optimization (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/16_hyperparameter_optimization/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/16_hyperparameter_optimization/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 16 · Hyperparameter Optimization | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/16_hyperparameter_optimization/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/16_hyperparameter_optimization/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## 🧭 Chọn phương pháp

| Phương pháp | Khi nào |
|---|---|
| Grid search | ≤ 2 tham số rời rạc |
| Random search | mặc định tốt, dễ song song |
| Successive halving / Hyperband | mỗi lần thử đắt, có resource giảm được (mẫu, epoch) |
| Bayesian (GP) | ít chiều, liên tục, rất đắt |
| TPE (Optuna) + pruning | không gian hỗn hợp/điều kiện — lựa chọn thực tế phổ biến |

## 🧰 Thư viện tuỳ chọn

`optuna` — `pip install -e ".[tuning]"` (notebook tự bỏ qua phần Optuna nếu chưa cài).

## ❓ Bộ câu hỏi tự kiểm tra

[questions.md](questions.md) — trả lời không nhìn tài liệu; ôn lại sau 1 ngày, 1 tuần, 1 tháng.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS.

## 🔗 Tiếp theo

[17 · Feature Engineering](../17_feature_engineering/README.md)
