# Chương 34 · Representation Learning — Học biểu diễn

> Dữ liệu không nhãn thì nhiều, nhãn thì đắt. Chương này học biểu diễn **không cần nhãn**: autoencoder (và vì sao nó học "sai thứ"), variational autoencoder để sinh dữ liệu, rồi học tương phản (SimCLR) — với vài chục nhãn đã vượt mạng có giám sát train từ đầu. Câu hỏi xuyên suốt: biểu diễn học được **chứa gì**, và mục tiêu huấn luyện quyết định điều đó thế nào.

## 🎯 Mục tiêu

- Autoencoder: AE tuyến tính ≈ PCA (kiểm chứng), AE tích chập; đo latent mã hoá gì (vị trí/kích thước vs loại hình); denoising AE; phát hiện bất thường bằng sai số tái tạo và giới hạn.
- VAE: ELBO, reparameterization, KL; lấy mẫu từ prior, nội suy; β và rate–distortion, chiều latent "chết"; vì sao đánh giá mô hình sinh khó; bản đồ GAN/diffusion/autoregressive.
- Học tương phản: NT-Xent from scratch, SimCLR trên 20 000 ảnh không nhãn; linear probe với 50/500 nhãn so với pixel, encoder ngẫu nhiên, AE, có giám sát; augmentation quyết định bất biến; BYOL/DINO/MAE/CLIP.

## 📋 Yêu cầu trước

[Chương 21](../21_dimensionality_reduction/README.md) (PCA), [Chương 29](../29_cnn/README.md) (CNN), [Chương 33](../33_computer_vision/README.md) (augmentation, transfer learning); [Chương 23](../23_probabilistic_machine_learning/README.md) giúp hiểu phần xác suất của VAE.

## 🧰 Thư viện

PyTorch — `pip install -e ".[dl]"`. Mỗi bài train vài mô hình nhỏ (vài phút trên GPU; trên CPU lâu hơn đáng kể — có thể giảm số bước).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Autoencoder | 📖 Bài học | [View](01_autoencoders.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/34_representation_learning/01_autoencoders.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/34_representation_learning/01_autoencoders.ipynb) |
| 02 | Bài 02 — Variational Autoencoder (VAE) | 📖 Bài học | [View](02_variational_autoencoder.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/34_representation_learning/02_variational_autoencoder.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/34_representation_learning/02_variational_autoencoder.ipynb) |
| 03 | Bài 03 — Học tương phản và tự giám sát | 📖 Bài học | [View](03_contrastive_and_self_supervised.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/34_representation_learning/03_contrastive_and_self_supervised.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/34_representation_learning/03_contrastive_and_self_supervised.ipynb) |
| 90 | Bài tập — Chương 34: Representation Learning (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/34_representation_learning/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/34_representation_learning/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 34 · Representation Learning | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/34_representation_learning/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/34_representation_learning/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

Bài 01: `ConvAE`, `train_ae` (có denoising). Bài 02: `VAE` (encode, reparameterization, sample), `elbo_terms`. Bài 03: `augment`, `nt_xent`, `Encoder`, `train_simclr`, `linear_probe`. Bài tập: `reparameterize`, `kl_to_standard_normal`, `knn_predict`, sửa posterior collapse, ablation augmentation, tìm ảnh tương tự, phát hiện bất thường bằng kNN trên embedding.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS (SimCLR + linear probe 100 nhãn, accuracy ≥ 0.75).

## 🔗 Tiếp theo

[35 · Reinforcement Learning](../35_reinforcement_learning/README.md)
