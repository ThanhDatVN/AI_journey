# Chương 38 · Causal Machine Learning — Suy luận nhân quả

> ML trả lời "điều gì sẽ xảy ra?"; doanh nghiệp, y tế, chính sách hỏi "**nếu ta làm X thì sao?**". Chương này: tương quan vs nhân quả qua DAG (Simpson, Berkson, bad controls), khung kết quả tiềm năng và thiên lệch chọn lọc, các phương pháp điểm xu hướng (IPW, matching, doubly robust AIPW), rồi hiệu ứng dị biệt với meta-learner và uplift modeling — tất cả trên dữ liệu mô phỏng có **hiệu ứng thật đã biết** để kiểm chứng từng phương pháp.

## 🎯 Mục tiêu

- Dự đoán vs can thiệp; DAG: confounder/mediator/collider; nghịch lý Simpson và Berkson; tiêu chí cửa sau; RCT và các giả định khi dùng dữ liệu quan sát.
- Kết quả tiềm năng, ITE/ATE/ATT/CATE; phân rã so sánh ngây thơ = ATT + thiên lệch chọn lọc; điều chỉnh bằng hồi quy, g-computation; bootstrap.
- Điểm xu hướng, overlap, cân bằng SMD; IPW (HT, Hajek), matching cho ATT; AIPW doubly robust với cross-fitting; vi phạm overlap và trimming.
- S/T/X-learner (PEHE); uplift trên dữ liệu A/B, đường Qini, chính sách theo lợi nhuận; dự đoán phản hồi ≠ dự đoán tác động.

## 📋 Yêu cầu trước

[Chương 6](../06_linear_regression/README.md), [Chương 7](../07_logistic_regression/README.md), [Chương 12](../12_gradient_boosting/README.md); [Chương 37](../37_uncertainty_and_calibration/README.md) (calibration của điểm xu hướng).

## 🧰 Thư viện

scikit-learn. Thư viện chuyên dụng (không bắt buộc): DoWhy, EconML, DoubleML, CausalML. Dữ liệu: `make_causal_marketing(..., randomized=...)` trong `ml_course.datasets`.

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Tương quan, nhân quả và đồ thị nhân quả (DAG) | 📖 Bài học | [View](01_correlation_vs_causation_dags.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/38_causal_machine_learning/01_correlation_vs_causation_dags.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/38_causal_machine_learning/01_correlation_vs_causation_dags.ipynb) |
| 02 | Bài 02 — Khung kết quả tiềm năng | 📖 Bài học | [View](02_potential_outcomes.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/38_causal_machine_learning/02_potential_outcomes.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/38_causal_machine_learning/02_potential_outcomes.ipynb) |
| 03 | Bài 03 — Phương pháp điểm xu hướng (propensity score) | 📖 Bài học | [View](03_propensity_score_methods.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/38_causal_machine_learning/03_propensity_score_methods.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/38_causal_machine_learning/03_propensity_score_methods.ipynb) |
| 04 | Bài 04 — Meta-learner và uplift modeling | 📖 Bài học | [View](04_meta_learners_and_uplift.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/38_causal_machine_learning/04_meta_learners_and_uplift.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/38_causal_machine_learning/04_meta_learners_and_uplift.ipynb) |
| 90 | Bài tập — Chương 38: Causal Machine Learning (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/38_causal_machine_learning/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/38_causal_machine_learning/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 38 · Causal Machine Learning | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/38_causal_machine_learning/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/38_causal_machine_learning/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

Bài 01: mô phỏng Simpson, collider, mediator, tiêu chí cửa sau. Bài 02: phân rã thiên lệch chọn lọc, `ols_effect`, `g_computation`. Bài 03: IPW, `smd`, matching, `aipw` (cross-fitting). Bài 04: `s_learner`, `t_learner`, `x_learner`, `qini_curve`, chính sách lợi nhuận. Bài tập: bad control, độ nhạy với confounder ẩn, CUPED, Double ML.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS (AIPW trên dữ liệu quan sát, sai lệch < 8 so với ATE thật).

## 🔗 Tiếp theo

[39 · ML Systems](../39_ml_systems/README.md)
