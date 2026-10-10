# Chương 28 · Neural Networks

> Huấn luyện mạng nơ-ron "cho đúng": hàm kích hoạt (dựa trên bài *Activation Function* — Tuần 5 khoá AIVN, kèm bài tập lập trình gốc), khởi tạo và chuẩn hoá, các thuật toán tối ưu, regularization và quy trình huấn luyện có hệ thống. Mỗi kỹ thuật được kiểm chứng bằng thí nghiệm nhỏ chạy được trên CPU.

## 🎯 Mục tiêu

- Hàm kích hoạt: sigmoid, tanh, ReLU, PReLU, ELU, Softplus (AIVN) + GELU, SiLU; bão hoà, vanishing gradient, dying ReLU; chọn kích hoạt cho lớp ẩn/lớp ra.
- Khởi tạo: đối xứng, Xavier, He; chuẩn hoá đầu vào; Batch Norm (from scratch, train vs eval), Layer Norm.
- Optimizer from scratch: momentum, Nesterov, AdaGrad, RMSProp, Adam (khớp `torch.optim`); LR range test, lịch học, AdamW, gradient clipping.
- Regularization: weight decay, dropout, nhiễu đầu vào/augmentation, early stopping, thêm dữ liệu; quy trình huấn luyện và bảng gỡ lỗi; template MLP.

## 📋 Yêu cầu trước

[Chương 27](../27_deep_learning_foundations/README.md) (backprop, PyTorch), [Chương 06](../06_linear_regression/README.md) (gradient descent, regularization).

## 🧰 Thư viện

PyTorch — `pip install -e ".[dl]"` (Google Colab có sẵn). Mọi thí nghiệm chạy được trên CPU.

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Hàm kích hoạt (Activation Functions) | 📖 Bài học | [View](01_activation_functions.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/28_neural_networks/01_activation_functions.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/28_neural_networks/01_activation_functions.ipynb) |
| 02 | Bài 02 — Khởi tạo trọng số và chuẩn hoá | 📖 Bài học | [View](02_initialization_and_normalization.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/28_neural_networks/02_initialization_and_normalization.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/28_neural_networks/02_initialization_and_normalization.ipynb) |
| 03 | Bài 03 — Thuật toán tối ưu | 📖 Bài học | [View](03_optimizers.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/28_neural_networks/03_optimizers.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/28_neural_networks/03_optimizers.ipynb) |
| 04 | Bài 04 — Regularization và quy trình huấn luyện | 📖 Bài học | [View](04_regularization_and_training_recipes.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/28_neural_networks/04_regularization_and_training_recipes.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/28_neural_networks/04_regularization_and_training_recipes.ipynb) |
| 90 | Bài tập — Chương 28: Neural Networks (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/28_neural_networks/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/28_neural_networks/90_exercises.ipynb) |
| 91 | Assignment 91 — Programming Assignment: Activation Function | 💻 Assignment | [View](91_assignment_activation_functions.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/28_neural_networks/91_assignment_activation_functions.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/28_neural_networks/91_assignment_activation_functions.ipynb) |
| 99 | ✅ Checkpoint — 28 · Neural Networks | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/28_neural_networks/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/28_neural_networks/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

Bài 02: `BatchNorm1dScratch`; bài 03: `optimize` (SGD, momentum, Nesterov, AdaGrad, RMSProp, Adam); bài 04: `dropout_scratch`. Bài tập: optimizer dạng lớp (`MomentumSGD`, `Adam`), `init_weight`, Batch Norm backward. `91_assignment_activation_functions`: bài tập lập trình gốc của AIVN (+ đạo hàm, softmax ổn định, GELU).

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS (mạng ≥ 6 lớp ẩn, accuracy ≥ 0.95).

## 🔗 Tiếp theo

[29 · CNN](../29_cnn/README.md)
