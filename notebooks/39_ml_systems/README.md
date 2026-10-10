# Chương 39 · ML Systems — Từ notebook tới hệ thống

> Một mô hình tốt trong notebook mới là bước đầu. Chương này chuyển trọng tâm từ "mô hình" sang **hệ thống**: thiết kế theo bài toán và ràng buộc vận hành, hai lỗi kinh điển chỉ lộ ra khi triển khai (rò rỉ thời điểm, lệch train/serving), **tái lập** thí nghiệm (seed, môi trường, hash dữ liệu, experiment tracking), và **đóng gói** mô hình an toàn kèm metadata, kiểm tra đầu vào, model registry với cổng thăng hạng — tự viết từ đầu trước, rồi dùng MLflow, skops, pydantic, ONNX.

## 🎯 Mục tiêu

- Thành phần của hệ thống ML; khung thiết kế bài toán → metric → dữ liệu/nhãn → baseline → phục vụ → đánh giá → giám sát; batch / online / streaming.
- Tái hiện và phòng tránh **rò rỉ thời điểm** (point-in-time) và **lệch train/serving**; chọn mô hình theo độ trễ, kích thước, độ chính xác.
- Nguồn bất định (seed, `PYTHONHASHSEED`, số thực, môi trường, dữ liệu); độ biến thiên theo seed và so sánh theo cặp; hash dữ liệu; tracker tự viết và MLflow; lời nguyền người thắng.
- Đóng gói pipeline đầy đủ; bẫy `__main__`; rủi ro bảo mật của pickle và skops; metadata, kiểm tra toàn vẹn, model card; kiểm tra đầu vào bằng pydantic và schema học từ dữ liệu; model registry (phiên bản, alias, cổng thăng hạng, rollback); MLflow Model Registry; xuất ONNX từ đầu.

## 📋 Yêu cầu trước

[Chương 14](../14_model_evaluation/README.md) (đánh giá), [Chương 15](../15_cross_validation/README.md) (chia dữ liệu, rò rỉ, chia theo thời gian), [Chương 16](../16_hyperparameter_optimization/README.md) (tìm siêu tham số), [Chương 17](../17_feature_engineering/README.md) (pipeline); nên học [Chương 37](../37_uncertainty_and_calibration/README.md) và [Chương 36](../36_explainable_ai/README.md) trước.

## 🧰 Thư viện

scikit-learn, joblib. Cài thêm `pip install -e ".[mlops]"`: MLflow (tracking, registry), pydantic (kiểm tra đầu vào), skops (lưu mô hình an toàn), onnx + onnxruntime. Các phần MLflow/skops/ONNX tự bỏ qua nếu chưa cài. Dữ liệu: `make_customer_churn` trong `ml_course.datasets`.

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Thiết kế hệ thống ML | 📖 Bài học | [View](01_ml_system_design.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/39_ml_systems/01_ml_system_design.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/39_ml_systems/01_ml_system_design.ipynb) |
| 02 | Bài 02 — Tái lập và theo dõi thí nghiệm | 📖 Bài học | [View](02_reproducibility_and_experiment_tracking.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/39_ml_systems/02_reproducibility_and_experiment_tracking.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/39_ml_systems/02_reproducibility_and_experiment_tracking.ipynb) |
| 03 | Bài 03 — Đóng gói mô hình và Model Registry | 📖 Bài học | [View](03_packaging_and_model_registry.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/39_ml_systems/03_packaging_and_model_registry.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/39_ml_systems/03_packaging_and_model_registry.ipynb) |
| 90 | Bài tập — Chương 39: ML Systems (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/39_ml_systems/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/39_ml_systems/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 39 · ML Systems | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/39_ml_systems/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/39_ml_systems/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

Bài 01: `simulate_activity`, `features(activity, as_of)` (đặc trưng đúng thời điểm), mô phỏng lệch train/serving, bảng độ trễ/kích thước. Bài 02: `set_seed`, `capture_environment`, `fingerprint`, `LocalTracker`/`Run`, `TrainConfig` + `train`, mô phỏng lời nguyền người thắng, MLflow tracking. Bài 03: `save_package`/`load_package` (hash + phiên bản), `infer_schema`, `check_against_schema`, `CustomerFeatures` (pydantic), `model_card`, `ModelRegistry`, `promotion_gate`, đồ thị ONNX tự dựng. Bài tập: hash dữ liệu, `JsonTracker`, kiểm tra bản ghi, kiểm thử parity online/batch, so sánh theo cặp, cổng thăng hạng, tập huấn luyện nhiều ngày chụp + kiểm thử tính chất.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS (KTC theo cặp, as-of join), phân tích (rubric) PASS, mini project PASS (gói mô hình có metadata + kiểm tra hash, nạp lại khớp).

## 🔗 Tiếp theo

[40 · MLOps](../40_mlops/README.md)
