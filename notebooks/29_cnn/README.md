# Chương 29 · CNN

> Mạng tích chập: tích chập và pooling from scratch (đối chiếu PyTorch), vì sao CNN vượt MLP trên ảnh — đo bằng thí nghiệm về **thiên kiến quy nạp** (train ở nửa trái ảnh, test ở nửa phải) — và tiến hoá kiến trúc từ LeNet tới ResNet, MobileNet. Bài 01–03 dùng ảnh sinh offline (`make_shapes`) để thấy rõ từng hiện tượng; bài 04 áp dụng trên ảnh thật Fashion-MNIST và CIFAR-10.

## 🎯 Mục tiêu

- Tích chập 2D (stride, padding, nhiều kênh), im2col, pooling, trường tiếp nhận, đồng biến vs bất biến với phép dịch.
- CNN trong PyTorch; CNN vs MLP (độ chính xác, số tham số, hiệu quả dữ liệu); global average pooling; data augmentation; trực quan hoá bộ lọc/bản đồ đặc trưng.
- Kiến trúc: LeNet, AlexNet, VGG (3×3), Inception (1×1, đa nhánh), ResNet (residual — thí nghiệm degradation), MobileNet (depthwise separable), EfficientNet; model zoo `torchvision`.
- Ảnh thật: Fashion-MNIST và CIFAR-10 — chuẩn hoá theo kênh, augmentation, BatchNorm, lịch learning rate one-cycle; so sánh tuyến tính → MLP → CNN trên cùng dữ liệu.

## 📋 Yêu cầu trước

[Chương 27](../27_deep_learning_foundations/README.md), [Chương 28](../28_neural_networks/README.md).

## 🧰 Thư viện và dữ liệu

PyTorch (+ torchvision cho model zoo) — `pip install -e ".[dl]"`. Bài 04 dùng Fashion-MNIST và CIFAR-10 thật qua `ml_course.real_datasets` (tải một lần ~200 MB vào `data/raw/`). Bộ ảnh `ml_course.datasets.make_shapes`: ảnh xám 28×28 gồm 5 loại hình (tròn, vuông, tam giác, chữ thập, vành khuyên), tham số `region` để tạo dịch chuyển phân phối theo vị trí. Bài 03 tự dùng GPU nếu có (mạng 41 lớp); mọi bài vẫn chạy được trên CPU.

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Tích chập from scratch | 📖 Bài học | [View](01_convolution_from_scratch.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/29_cnn/01_convolution_from_scratch.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/29_cnn/01_convolution_from_scratch.ipynb) |
| 02 | Bài 02 — CNN với PyTorch | 📖 Bài học | [View](02_cnn_with_pytorch.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/29_cnn/02_cnn_with_pytorch.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/29_cnn/02_cnn_with_pytorch.ipynb) |
| 03 | Bài 03 — Kiến trúc CNN: từ LeNet đến ResNet | 📖 Bài học | [View](03_architectures_lenet_to_resnet.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/29_cnn/03_architectures_lenet_to_resnet.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/29_cnn/03_architectures_lenet_to_resnet.ipynb) |
| 04 | Bài 04 — CNN trên ảnh thật: Fashion-MNIST và CIFAR-10 | 📖 Bài học | [View](04_cnn_on_real_images.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/29_cnn/04_cnn_on_real_images.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/29_cnn/04_cnn_on_real_images.ipynb) |
| 90 | Bài tập — Chương 29: CNN (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/29_cnn/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/29_cnn/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 29 · CNN | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/29_cnn/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/29_cnn/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

Bài 01: `conv2d_naive`, `im2col`, `conv2d_im2col`, `max_pool2d`, `receptive_field`. Bài 03: `LeNet5`, khối residual. Bài tập: conv batch vector hoá, max-pool forward/backward, **conv backward** (khớp autograd).

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS (train nửa trái, đạt ≥ 0.75 ở nửa phải).

## 🔗 Tiếp theo

[30 · Sequence Models](../30_sequence_models/README.md)
