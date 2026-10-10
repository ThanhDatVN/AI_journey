# Chương 12 · Gradient Boosting

> "Vua" của dữ liệu bảng: nhiều cây nông, mỗi cây sửa phần lỗi còn lại của các cây trước. Từ AdaBoost (1997) đến Gradient Boosting (Friedman, 2001) và XGBoost / LightGBM / CatBoost. Không có trong khoá AIVN gốc — bổ sung mới theo template "mọi model".

## 🎯 Mục tiêu

- Bagging vs boosting; **AdaBoost from scratch**: trọng số mẫu, $\alpha_m$, lỗi train giảm theo hàm mũ, exponential loss, nhạy với nhiễu nhãn.
- **Gradient Boosting from scratch** = gradient descent trong không gian hàm: MSE (residual), Huber/MAE (robust), log-loss (+ bước Newton ở lá); learning rate × số cây; stochastic GB; early stopping.
- **XGBoost**: Taylor bậc hai + regularization → $w^* = -G/(H+\lambda)$, công thức gain (tính tay); histogram, leaf-wise (LightGBM), ordered boosting (CatBoost); so sánh 5 thư viện.
- Quy trình tuning (RandomizedSearch, Optuna), RF vs GB, monotonic constraints, template model.

## 📋 Yêu cầu trước

[Chương 10](../10_decision_trees/README.md), [Chương 11](../11_random_forest/README.md), [Chương 02](../02_mathematics_for_ml/README.md) (gradient descent, Taylor).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Boosting: học tuần tự từ sai lầm (AdaBoost) | 📖 Bài học | [View](01_boosting_intuition_adaboost.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/12_gradient_boosting/01_boosting_intuition_adaboost.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/12_gradient_boosting/01_boosting_intuition_adaboost.ipynb) |
| 02 | Bài 02 — Gradient Boosting from scratch | 📖 Bài học | [View](02_gradient_boosting_from_scratch.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/12_gradient_boosting/02_gradient_boosting_from_scratch.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/12_gradient_boosting/02_gradient_boosting_from_scratch.ipynb) |
| 03 | Bài 03 — XGBoost, LightGBM, CatBoost và HistGradientBoosting | 📖 Bài học | [View](03_xgboost_lightgbm_catboost.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/12_gradient_boosting/03_xgboost_lightgbm_catboost.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/12_gradient_boosting/03_xgboost_lightgbm_catboost.ipynb) |
| 04 | Bài 04 — Tuning Gradient Boosting, Random Forest vs Gradient Boosting | 📖 Bài học | [View](04_tuning_and_rf_vs_gb.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/12_gradient_boosting/04_tuning_and_rf_vs_gb.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/12_gradient_boosting/04_tuning_and_rf_vs_gb.ipynb) |
| 90 | Bài tập — Chương 12: Gradient Boosting (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/12_gradient_boosting/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/12_gradient_boosting/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 12 · Gradient Boosting | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/12_gradient_boosting/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/12_gradient_boosting/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

`src/ml_course/algorithms/ensemble.py`: `AdaBoostScratch`, `GradientBoostingRegressorScratch`, `GradientBoostingClassifierScratch`.

## 🧰 Thư viện tuỳ chọn

`xgboost`, `lightgbm`, `catboost`, `optuna` — cài bằng `pip install -e ".[boosting]"` (notebook tự bỏ qua phần tương ứng nếu chưa cài).

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS.

## 🔗 Tiếp theo

[13 · SVM & Kernels](../13_svm_and_kernels/README.md)
