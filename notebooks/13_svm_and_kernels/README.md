# Chương 13 · SVM & Kernels

> Hình học của phân loại: chọn ranh giới có **lề lớn nhất**, chỉ vài **support vectors** quyết định, và **kernel trick** để làm việc trong không gian vô hạn chiều với chi phí một phép tính "độ giống nhau". Không có trong khoá AIVN gốc — bổ sung mới theo template "mọi model".

## 🎯 Mục tiêu

- Maximum margin, khoảng cách tới siêu phẳng, hard-margin SVM là QP lồi — giải bằng `scipy.optimize`, support vectors.
- Soft margin (slack, $C$) ⇔ **hinge loss** + L2; Linear SVM **from scratch** (subgradient, Pegasos); SVM vs Logistic Regression.
- Bài toán đối ngẫu, **kernel trick** (kiểm chứng polynomial kernel), RBF ($\gamma$, $C$), SMO, ma trận Gram.
- Thực hành: scaling bắt buộc, grid search log-scale, Platt scaling, SVR, dữ liệu lớn (LinearSVC, Nystroem/RFF), mất cân bằng; template model.

## 📋 Yêu cầu trước

[Chương 07](../07_logistic_regression/README.md) (loss + regularization), [Chương 02](../02_mathematics_for_ml/README.md) (tích vô hướng, tối ưu lồi), [Chương 08](../08_knn/README.md) (khoảng cách, scaling).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — SVM: ranh giới có lề lớn nhất | 📖 Bài học | [View](01_maximum_margin_intuition.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/13_svm_and_kernels/01_maximum_margin_intuition.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/13_svm_and_kernels/01_maximum_margin_intuition.ipynb) |
| 02 | Bài 02 — Soft margin, hinge loss và Linear SVM from scratch | 📖 Bài học | [View](02_soft_margin_hinge_loss_from_scratch.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/13_svm_and_kernels/02_soft_margin_hinge_loss_from_scratch.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/13_svm_and_kernels/02_soft_margin_hinge_loss_from_scratch.ipynb) |
| 03 | Bài 03 — Kernel trick: SVM phi tuyến | 📖 Bài học | [View](03_kernel_trick.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/13_svm_and_kernels/03_kernel_trick.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/13_svm_and_kernels/03_kernel_trick.ipynb) |
| 04 | Bài 04 — SVM trong thực tế và các failure case | 📖 Bài học | [View](04_svm_in_practice_and_failure_cases.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/13_svm_and_kernels/04_svm_in_practice_and_failure_cases.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/13_svm_and_kernels/04_svm_in_practice_and_failure_cases.ipynb) |
| 90 | Bài tập — Chương 13: SVM & Kernels (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/13_svm_and_kernels/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/13_svm_and_kernels/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 13 · SVM & Kernels | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/13_svm_and_kernels/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/13_svm_and_kernels/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

`src/ml_course/algorithms/svm.py`: `LinearSVMScratch` (subgradient), `KernelSVMScratch` (simplified SMO), `rbf_kernel`, `poly_kernel`.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS.

## 🔗 Tiếp theo

[14 · Model Evaluation](../14_model_evaluation/README.md)
