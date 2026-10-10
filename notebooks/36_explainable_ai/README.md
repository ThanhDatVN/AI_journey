# Chương 36 · Explainable AI — Giải thích mô hình

> Mô hình dựa vào đâu để dự đoán, và vì sao với từng trường hợp? Chương dùng dữ liệu rời bỏ khách hàng có **logit thật đã biết** để kiểm tra các phương pháp giải thích có tìm lại được sự thật không: permutation importance vs MDI, PDP/ICE/ALE, mô hình thay thế, LIME và giá trị Shapley (cài từ định nghĩa, khớp thư viện `shap`), counterfactual — và các cạm bẫy: đặc trưng tương quan, hiệu ứng Rashomon, rò rỉ, điểm tham chiếu, công bằng.

## 🎯 Mục tiêu

- Toàn cục: mô hình dễ hiểu trước (logistic khớp logit thật); MDI vs permutation (thiên lệch với nhiễu nhiều giá trị); PDP/ICE; ALE khi đặc trưng tương quan; surrogate + độ trung thực.
- Cục bộ: LIME from scratch và độ nhạy (số mẫu, kernel); giá trị Shapley chính xác from scratch, tính chất hiệu quả, đối chiếu `shap`, dạng đóng cho mô hình tuyến tính; tổng hợp toàn cục; counterfactual khả thi.
- Cạm bẫy: độ quan trọng bị thay thế/chia sẻ; permutation vs drop-column; Rashomon; phát hiện rò rỉ; giải thích phụ thuộc dữ liệu nền; biến ủy nhiệm và nhãn thiên lệch trong công bằng; deletion test; danh sách kiểm tra khi dùng XAI.

## 📋 Yêu cầu trước

[Chương 7](../07_logistic_regression/README.md), [Chương 11](../11_random_forest/README.md) (importance), [Chương 12](../12_gradient_boosting/README.md) (boosting); [Chương 5](../05_machine_learning_fundamentals/README.md) (data leakage).

## 🧰 Thư viện

scikit-learn. `shap` (`pip install -e ".[xai]"`) chỉ dùng để **đối chiếu** với cài đặt from scratch — bài học vẫn chạy nếu chưa cài.

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Giải thích toàn cục | 📖 Bài học | [View](01_global_explanations.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/36_explainable_ai/01_global_explanations.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/36_explainable_ai/01_global_explanations.ipynb) |
| 02 | Bài 02 — Giải thích cục bộ: LIME, SHAP và counterfactual | 📖 Bài học | [View](02_local_explanations_lime_shap.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/36_explainable_ai/02_local_explanations_lime_shap.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/36_explainable_ai/02_local_explanations_lime_shap.ipynb) |
| 03 | Bài 03 — Cạm bẫy của giải thích mô hình | 📖 Bài học | [View](03_pitfalls_of_explanations.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/36_explainable_ai/03_pitfalls_of_explanations.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/36_explainable_ai/03_pitfalls_of_explanations.ipynb) |
| 90 | Bài tập — Chương 36: Explainable AI (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/36_explainable_ai/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/36_explainable_ai/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 36 · Explainable AI | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/36_explainable_ai/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/36_explainable_ai/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

Bài 01: `ale_1d`, `pdp_1d`, surrogate. Bài 02: `lime_explain`, `exact_shapley`, `shapley_many`, counterfactual tìm kiếm theo chi phí. Bài 03: permutation vs drop-column, phát hiện rò rỉ, Shapley theo dữ liệu nền, deletion test. Bài tập: permutation importance và PDP tự cài, kiểm toán công bằng (parity, TPR, ngưỡng theo nhóm), KernelSHAP from scratch.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS (dùng permutation importance phát hiện cột rò rỉ).

## 🔗 Tiếp theo

[37 · Uncertainty & Calibration](../37_uncertainty_and_calibration/README.md)
