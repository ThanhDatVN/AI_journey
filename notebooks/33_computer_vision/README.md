# Chương 33 · Computer Vision — Thị giác máy tính

> Từ CNN (chương 29) đến các bài toán thị giác thực tế: xử lý ảnh như tensor, data augmentation đúng cách (và khi nào nó không giúp), shortcut learning, transfer learning khi ít nhãn, rồi phát hiện vật thể (IoU, NMS, mAP, một detector dạng YOLO tự viết) và phân đoạn ảnh (U-Net). Mọi thí nghiệm chạy offline trên ảnh hình học tổng hợp; mô hình tiền huấn luyện thật có ở các ô tham khảo.

## 🎯 Mục tiêu

- Ảnh là tensor: HWC/CHW, kiểu dữ liệu, chuẩn hoá, nội suy và antialias, bộ lọc cổ điển.
- Augmentation bằng `affine_grid`/`grid_sample`; augmentation nào giúp (dịch, lật) và nào không (xoay khi dữ liệu đã xoay); phải giữ nhãn; shortcut learning do màu và cách phá vỡ.
- Transfer learning: tiền huấn luyện backbone, đặc trưng chuyển giao sang lớp mới; từ đầu vs linear probe vs fine-tune (LR phân tầng, head trước) theo số nhãn; dùng ResNet ImageNet.
- Detection: IoU, detector lưới (objectness + lớp + hộp), giải mã, NMS, precision–recall, AP/mAP; các họ detector thực tế. Segmentation: U-Net có/không skip, mIoU vs pixel accuracy.

## 📋 Yêu cầu trước

[Chương 29](../29_cnn/README.md) (CNN), [Chương 28](../28_neural_networks/README.md) (huấn luyện mạng); [Chương 19](../19_imbalanced_learning/README.md) và chương đánh giá mô hình giúp hiểu precision–recall.

## 🧰 Thư viện

PyTorch — `pip install -e ".[dl]"` (gồm torchvision, chỉ dùng để đối chiếu IoU/NMS và ở các ô tham khảo `RUN_PRETRAINED = False`). Bài 02–03 train nhiều mô hình nhỏ: tự dùng GPU nếu có; trên CPU mất vài phút mỗi bài.

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Ảnh số và data augmentation | 📖 Bài học | [View](01_images_and_augmentation.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/33_computer_vision/01_images_and_augmentation.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/33_computer_vision/01_images_and_augmentation.ipynb) |
| 02 | Bài 02 — Transfer learning và fine-tuning | 📖 Bài học | [View](02_transfer_learning_and_fine_tuning.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/33_computer_vision/02_transfer_learning_and_fine_tuning.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/33_computer_vision/02_transfer_learning_and_fine_tuning.ipynb) |
| 03 | Bài 03 — Phát hiện vật thể và phân đoạn ảnh | 📖 Bài học | [View](03_detection_and_segmentation_concepts.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/33_computer_vision/03_detection_and_segmentation_concepts.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/33_computer_vision/03_detection_and_segmentation_concepts.ipynb) |
| 90 | Bài tập — Chương 33: Computer Vision (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/33_computer_vision/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/33_computer_vision/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 33 · Computer Vision | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/33_computer_vision/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/33_computer_vision/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

Bài 01: `random_affine`, `color_augment`. Bài 02: `Net` (backbone + head), `fine_tune` (LR phân tầng, head trước). Bài 03: `box_iou`, `GridDetector`, `encode_targets`, `detect`, `nms`, `average_precision`, `UNet`, `segmentation_metrics`. Bài tập: `hflip_with_boxes`, `to_chw`, báo cáo độ bền, `iou_and_giou`. Dữ liệu: `make_shapes` (`classes`, `rgb`, `color_bias`), `SHAPE_CLASSES_EXTRA`, `make_shape_scenes` (`ml_course.datasets`).

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS (300 ảnh + augmentation, accuracy ≥ 0.88).

## 🔗 Tiếp theo

[34 · Representation Learning](../34_representation_learning/README.md)
