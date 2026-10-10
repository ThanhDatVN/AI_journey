# Chương 30 · Sequence Models

> Dữ liệu tuần tự — văn bản, chuỗi sự kiện, tín hiệu — cần mô hình có "trí nhớ". Chương xây RNN và BPTT from scratch (mô hình ngôn ngữ ký tự trên review tiếng Việt), đo trực tiếp vanishing/exploding gradient, giải thích vì sao LSTM/GRU giữ được thông tin xa (adding problem), rồi tới seq2seq và attention — cầu nối sang Transformer.

## 🎯 Mục tiêu

- RNN: trạng thái ẩn, chia sẻ trọng số theo thời gian, BPTT (kiểm tra gradient), truncated BPTT, mô hình ngôn ngữ ký tự, nhiệt độ lấy mẫu.
- Vanishing/exploding gradient: tích Jacobian, bán kính phổ, bão hoà tanh; gradient tới từng bước đầu vào ở RNN vs LSTM/GRU; gradient clipping.
- LSTM, GRU (khớp `nn.LSTMCell`/`nn.GRUCell`), bias cổng quên; adding problem; bidirectional, xếp chồng, padding/packing.
- Seq2seq, teacher forcing, nút cổ chai, dot-product attention và căn chỉnh, greedy/beam search, giới hạn ngoại suy độ dài.

## 📋 Yêu cầu trước

[Chương 27](../27_deep_learning_foundations/README.md), [Chương 28](../28_neural_networks/README.md); [Chương 25](../25_time_series/README.md) (chuỗi thời gian) giúp hiểu bối cảnh.

## 🧰 Thư viện

PyTorch — `pip install -e ".[dl]"`. Bài 03–04 tự dùng GPU nếu có (huấn luyện nhiều bước); mọi bài vẫn chạy được trên CPU (chậm hơn).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Mạng nơ-ron hồi quy (RNN) from scratch | 📖 Bài học | [View](01_rnn_from_scratch.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/30_sequence_models/01_rnn_from_scratch.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/30_sequence_models/01_rnn_from_scratch.ipynb) |
| 02 | Bài 02 — Vanishing và exploding gradient trong RNN | 📖 Bài học | [View](02_vanishing_exploding_gradients.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/30_sequence_models/02_vanishing_exploding_gradients.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/30_sequence_models/02_vanishing_exploding_gradients.ipynb) |
| 03 | Bài 03 — LSTM và GRU | 📖 Bài học | [View](03_lstm_gru.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/30_sequence_models/03_lstm_gru.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/30_sequence_models/03_lstm_gru.ipynb) |
| 04 | Bài 04 — Seq2seq và attention | 📖 Bài học | [View](04_seq2seq_and_attention.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/30_sequence_models/04_seq2seq_and_attention.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/30_sequence_models/04_seq2seq_and_attention.ipynb) |
| 90 | Bài tập — Chương 30: Sequence Models (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/30_sequence_models/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/30_sequence_models/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 30 · Sequence Models | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/30_sequence_models/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/30_sequence_models/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

Bài 01: `rnn_loss_and_grads` (BPTT NumPy), `sample`. Bài 03: `lstm_step`, `gru_step`, `adding_batch`. Bài 04: `Seq2Seq` (GRU + attention). Bài tập: `lstm_forward`, `attention` (khớp `scaled_dot_product_attention`), `last_hidden`, beam search.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS (học thứ tự sự kiện, accuracy ≥ 0.95).

## 🔗 Tiếp theo

[31 · Transformers](../31_transformers/README.md)
