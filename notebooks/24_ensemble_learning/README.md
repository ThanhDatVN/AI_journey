# Chương 24 · Ensemble Learning

> Bagging và boosting (chương 11–12) kết hợp nhiều model **cùng loại**. Chương này đi tiếp: kết hợp các model **khác loại** bằng voting, stacking và ensemble selection — và hiểu khi nào kết hợp thật sự có lợi (đa dạng, trọng số, rò rỉ, chi phí). Không có trong khoá AIVN gốc — bổ sung mới theo template "mọi model".

## 🎯 Mục tiêu

- Toán của ensemble: bỏ phiếu đa số (Condorcet) và vai trò tương quan; **phân rã ambiguity**; đo độ đa dạng.
- **Voting**: hard/soft, trọng số; khi nào model yếu làm hại.
- **Stacking**: dự đoán out-of-fold (vì sao bắt buộc), stacking from scratch, `StackingRegressor/Classifier`, blending.
- **Thiết kế ensemble**: nguồn đa dạng, ensemble selection (Caruana), ràng buộc chi phí, distillation, bất đồng = tín hiệu bất định.

## 📋 Yêu cầu trước

[Chương 11](../11_random_forest/README.md) (bagging), [Chương 12](../12_gradient_boosting/README.md) (boosting), [Chương 15](../15_cross_validation/README.md) (OOF, nested CV), [Chương 14](../14_model_evaluation/README.md).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Voting và sự đa dạng | 📖 Bài học | [View](01_voting_and_diversity.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/24_ensemble_learning/01_voting_and_diversity.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/24_ensemble_learning/01_voting_and_diversity.ipynb) |
| 02 | Bài 02 — Stacking và Blending | 📖 Bài học | [View](02_stacking_and_blending.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/24_ensemble_learning/02_stacking_and_blending.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/24_ensemble_learning/02_stacking_and_blending.ipynb) |
| 03 | Bài 03 — Thiết kế ensemble | 📖 Bài học | [View](03_ensemble_design.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/24_ensemble_learning/03_ensemble_design.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/24_ensemble_learning/03_ensemble_design.ipynb) |
| 90 | Bài tập — Chương 24: Ensemble Learning (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/24_ensemble_learning/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/24_ensemble_learning/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 24 · Ensemble Learning | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/24_ensemble_learning/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/24_ensemble_learning/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

Bài 03: `ensemble_selection` (Caruana et al., 2004). Bài tập: `hard_vote`/`soft_vote`, `StackingScratch` (khớp `StackingRegressor`), `budgeted_selection`, trọng số lồi bằng projected gradient lên simplex.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS (ensemble chữ số tốt hơn mọi thành viên).

## 🔗 Tiếp theo

[25 · Time Series](../25_time_series/README.md)
