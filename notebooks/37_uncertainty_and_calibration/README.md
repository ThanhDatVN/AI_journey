# Chương 37 · Uncertainty & Calibration — Bất định và hiệu chỉnh xác suất

> Một dự đoán chỉ hữu ích khi ta biết nó đáng tin đến đâu. Chương này: xác suất có **đúng nghĩa** không (calibration — trên dữ liệu có xác suất thật đã biết), sửa xác suất sau khi train (Platt, isotonic/PAV, temperature scaling, điều chỉnh tỉ lệ nền), tách **bất định aleatoric và epistemic** bằng deep ensemble, và **conformal prediction** — khoảng/tập dự đoán có đảm bảo độ phủ cho mọi mô hình, cùng giới hạn khi dữ liệu trôi.

## 🎯 Mục tiêu

- Calibration vs xếp hạng; reliability diagram, ECE (và sàn nhiễu), Brier, log-loss, phân rã Murphy; calibration của các họ mô hình; quyết định theo chi phí cần xác suất đúng.
- Platt (Newton) và isotonic (PAV) from scratch; cỡ tập calibration; `CalibratedClassifierCV`; temperature scaling; điều chỉnh tỉ lệ nền sau undersampling.
- Aleatoric vs epistemic; Gaussian NLL; deep ensemble và phân rã phương sai/entropy; phát hiện OOD; giới hạn của ước lượng bất định.
- Split conformal from scratch và mô phỏng đảm bảo; phủ biên vs có điều kiện; CQR, Mondrian; tập dự đoán phân loại; trôi dữ liệu.

## 📋 Yêu cầu trước

[Chương 14](../14_model_evaluation/README.md) (log-loss, Brier), [Chương 19](../19_imbalanced_learning/README.md) (undersampling), [Chương 23](../23_probabilistic_machine_learning/README.md) (mô hình xác suất); bài 03 dùng PyTorch (chương 27).

## 🧰 Thư viện

scikit-learn, SciPy; bài 03 dùng PyTorch (`pip install -e ".[dl]"`) cho mạng dự đoán phương sai (CPU, vài chục giây). Thư viện conformal thực tế: MAPIE (không bắt buộc).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Calibration: reliability diagram, ECE, Brier | 📖 Bài học | [View](01_calibration_reliability_brier.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/37_uncertainty_and_calibration/01_calibration_reliability_brier.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/37_uncertainty_and_calibration/01_calibration_reliability_brier.ipynb) |
| 02 | Bài 02 — Calibration hậu kiểm: Platt, isotonic, temperature scaling | 📖 Bài học | [View](02_platt_and_isotonic.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/37_uncertainty_and_calibration/02_platt_and_isotonic.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/37_uncertainty_and_calibration/02_platt_and_isotonic.ipynb) |
| 03 | Bài 03 — Bất định aleatoric và epistemic | 📖 Bài học | [View](03_aleatoric_and_epistemic.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/37_uncertainty_and_calibration/03_aleatoric_and_epistemic.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/37_uncertainty_and_calibration/03_aleatoric_and_epistemic.ipynb) |
| 04 | Bài 04 — Conformal prediction | 📖 Bài học | [View](04_conformal_prediction.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/37_uncertainty_and_calibration/04_conformal_prediction.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/37_uncertainty_and_calibration/04_conformal_prediction.ipynb) |
| 90 | Bài tập — Chương 37: Uncertainty & Calibration (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/37_uncertainty_and_calibration/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/37_uncertainty_and_calibration/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 37 · Uncertainty & Calibration | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/37_uncertainty_and_calibration/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/37_uncertainty_and_calibration/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

Bài 01: `reliability`, `ece`, `brier_decomposition`. Bài 02: `fit_platt` (Newton), `pav`, `fit_isotonic`, temperature scaling, điều chỉnh tỉ lệ nền. Bài 03: ensemble Gaussian NLL, phân rã phương sai/entropy. Bài 04: `conformal_quantile`, split conformal, CQR, Mondrian, tập dự đoán LAC. Bài tập: ECE thích ứng, `fit_temperature`, dự đoán có chọn lọc, Adaptive Conformal Inference.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS (calibrate Naive Bayes bằng isotonic: ECE < 0.04, AUC giữ nguyên).

## 🔗 Tiếp theo

[38 · Causal Machine Learning](../38_causal_machine_learning/README.md)
