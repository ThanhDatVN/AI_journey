# Chương 27 · Deep Learning Foundations

> Từ một nơ-ron đến mạng nhiều lớp: perceptron và giới hạn XOR, đồ thị tính toán và lan truyền ngược (tự xây một engine autograd), mạng nơ-ron NumPy theo từng lớp, rồi PyTorch. Mở đầu phần Deep Learning của khoá học; nối tiếp hồi quy logistic/softmax và gradient descent (AIVN tuần 4–5, chương 05–07 của khoá này).

## 🎯 Mục tiêu

- Nơ-ron nhân tạo; perceptron (quy tắc học, định lý hội tụ, XOR); vì sao cần lớp ẩn; trực giác xấp xỉ vạn năng.
- Đồ thị tính toán, quy tắc chuỗi, kiểm tra gradient; **autograd from scratch** (vô hướng, và cho tensor ở bài tập); backprop dạng ma trận; vì sao backprop hiệu quả.
- Mạng NumPy theo lớp (`Linear`, `ReLU`, `SoftmaxCrossEntropy`), mini-batch SGD, thử nghiệm độ rộng/sâu, overfitting.
- PyTorch: tensor, autograd, `nn.Module`, vòng lặp 5 bước, `DataLoader`, GPU, lưu/nạp, các lỗi thường gặp.
- TensorFlow/Keras: `tf.Variable`, `GradientTape`, `tf.function`, `tf.data`, Sequential/Functional API, `fit` + callback, vòng lặp tự viết, bảng tương đương PyTorch ↔ TensorFlow.

## 📋 Yêu cầu trước

[Chương 02](../02_mathematics_for_ml/README.md) (đạo hàm, gradient descent), [Chương 07](../07_logistic_regression/README.md) (logistic/softmax, cross-entropy), [Chương 05](../05_machine_learning_fundamentals/README.md) (overfitting).

## 🧰 Thư viện

PyTorch — `pip install -e ".[dl]"` (Google Colab có sẵn). Bài 01–03 chỉ cần NumPy; bài 04 và bài tập cần PyTorch; bài 05 cần TensorFlow — `pip install -e ".[tf]"` (tuỳ chọn; trên Windows TensorFlow chạy CPU). GPU không bắt buộc — mọi ví dụ chạy được trên CPU.

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Perceptron và nơ-ron nhân tạo | 📖 Bài học | [View](01_perceptron_and_neuron.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/27_deep_learning_foundations/01_perceptron_and_neuron.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/27_deep_learning_foundations/01_perceptron_and_neuron.ipynb) |
| 02 | Bài 02 — Đồ thị tính toán và lan truyền ngược | 📖 Bài học | [View](02_computational_graph_and_backprop.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/27_deep_learning_foundations/02_computational_graph_and_backprop.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/27_deep_learning_foundations/02_computational_graph_and_backprop.ipynb) |
| 03 | Bài 03 — Mạng nơ-ron from scratch với NumPy | 📖 Bài học | [View](03_neural_network_from_scratch.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/27_deep_learning_foundations/03_neural_network_from_scratch.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/27_deep_learning_foundations/03_neural_network_from_scratch.ipynb) |
| 04 | Bài 04 — PyTorch căn bản | 📖 Bài học | [View](04_pytorch_fundamentals.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/27_deep_learning_foundations/04_pytorch_fundamentals.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/27_deep_learning_foundations/04_pytorch_fundamentals.ipynb) |
| 05 | Bài 05 — TensorFlow và Keras | 📖 Bài học | [View](05_tensorflow_and_keras.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/27_deep_learning_foundations/05_tensorflow_and_keras.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/27_deep_learning_foundations/05_tensorflow_and_keras.ipynb) |
| 90 | Bài tập — Chương 27: Deep Learning Foundations (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/27_deep_learning_foundations/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/27_deep_learning_foundations/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 27 · Deep Learning Foundations | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/27_deep_learning_foundations/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/27_deep_learning_foundations/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

`src/ml_course/algorithms/neural_net.py`: `MLPScratch` (forward, backprop, mini-batch SGD/Adam). Bài 02: engine `Value` (kiểu micrograd); bài 03: `Linear`, `ReLU`, `Tanh`, `SoftmaxCrossEntropy`, `Network`; bài tập 10: autograd cho tensor.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS (MLP ≤ 10 000 tham số, accuracy ≥ 0.96).

## 🔗 Tiếp theo

[28 · Neural Networks](../28_neural_networks/README.md) — hàm kích hoạt, khởi tạo, chuẩn hoá, optimizer, regularization.

---
<sub>Chương này bao trùm Tuần 8 trong kế hoạch của [AIVN-Machine-Learning](https://github.com/thanhhff/AIVN-Machine-Learning) (*Neural Networks: Representation* — non-linear hypotheses, multiclass, cost function, backpropagation, gradient checking; khởi tạo ngẫu nhiên ở chương 28 · bài 02). Bài 05 hiện thực hoá mục *TensorFlow* (Tuần 7) — bản gốc chưa có tài liệu.</sub>
