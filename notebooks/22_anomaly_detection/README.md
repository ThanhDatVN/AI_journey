# Chương 22 · Anomaly Detection

> Khi bất thường cực hiếm, ít nhãn và luôn thay đổi, ta học "bình thường trông như thế nào" rồi cảnh báo điều lệch khỏi nó. Chương mở đầu bằng mô hình Gaussian với ε chọn bằng F1 (theo khoá Machine Learning của Andrew Ng), rồi đến Isolation Forest, LOF, One-Class SVM và thực hành đánh giá với ít nhãn. Không có trong khoá AIVN gốc — bổ sung mới theo template "mọi model".

## 🎯 Mục tiêu

- Phân loại bất thường (điểm, ngữ cảnh, tập thể); khi nào dùng anomaly detection thay vì phân loại có giám sát.
- Phương pháp thống kê: robust z-score (median/MAD), IQR; **mô hình Gaussian** $p(\mathbf x) < \varepsilon$ với ε chọn bằng F1; Gaussian nhiều chiều (Mahalanobis, $\chi^2$); ước lượng robust (MCD).
- **Isolation Forest** (cài đặt from scratch), **LOF**, **One-Class SVM** — điểm mạnh/yếu trên các cấu trúc dữ liệu khác nhau; tham số `contamination`.
- Ba chế độ (không giám sát, novelty, có giám sát) và điểm mù của từng chế độ; kiểu gian lận mới; sai số tái tạo; kết hợp detector; bất thường theo thời gian.

## 📋 Yêu cầu trước

[Chương 02](../02_mathematics_for_ml/README.md) (phân phối chuẩn, hiệp phương sai), [Chương 14](../14_model_evaluation/README.md) (precision/recall, PR-AUC), [Chương 19](../19_imbalanced_learning/README.md), [Chương 21](../21_dimensionality_reduction/README.md) (PCA).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Phát hiện bất thường: phương pháp thống kê và mô hình Gaussian | 📖 Bài học | [View](01_statistical_and_gaussian_methods.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/22_anomaly_detection/01_statistical_and_gaussian_methods.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/22_anomaly_detection/01_statistical_and_gaussian_methods.ipynb) |
| 02 | Bài 02 — Isolation Forest, LOF, One-Class SVM | 📖 Bài học | [View](02_isolation_forest_lof_ocsvm.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/22_anomaly_detection/02_isolation_forest_lof_ocsvm.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/22_anomaly_detection/02_isolation_forest_lof_ocsvm.ipynb) |
| 03 | Bài 03 — Đánh giá và thực hành phát hiện bất thường | 📖 Bài học | [View](03_evaluation_and_practice.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/22_anomaly_detection/03_evaluation_and_practice.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/22_anomaly_detection/03_evaluation_and_practice.ipynb) |
| 90 | Bài tập — Chương 22: Anomaly Detection (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/22_anomaly_detection/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/22_anomaly_detection/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 22 · Anomaly Detection | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/22_anomaly_detection/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/22_anomaly_detection/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

Bài 02 và Bài tập 10: Isolation Forest from scratch (cây cô lập, $c(n)$, điểm $2^{-\mathbb E[h]/c(\psi)}$). Bài tập 5: `estimate_gaussian`, `gaussian_pdf`, `select_threshold` theo phong cách khoá học của Andrew Ng.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS (phát hiện gian lận kiểu mới).

## 🔗 Tiếp theo

[23 · Probabilistic Machine Learning](../23_probabilistic_machine_learning/README.md)
