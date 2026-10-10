# Chương 23 · Probabilistic Machine Learning

> Thay vì một đáp án duy nhất, mô hình xác suất cho biết **mức độ chắc chắn**: phân phối trên tham số, trên hàm, trên trạng thái ẩn. Chương đi từ Gaussian nhiều chiều (viên gạch nền) tới GMM + EM, hồi quy Bayes, Gaussian Process và Hidden Markov Model — mỗi mô hình đều cài đặt from scratch rồi đối chiếu thư viện. Không có trong khoá AIVN gốc — bổ sung mới theo template "mọi model".

## 🎯 Mục tiêu

- **Gaussian nhiều chiều**: hình học hiệp phương sai, Cholesky, phân phối biên/có điều kiện, shrinkage; LDA/QDA.
- **GMM và EM**: biến ẩn, E-step/M-step, ELBO, khởi tạo, BIC, kiểu hiệp phương sai, suy biến.
- **Hồi quy tuyến tính Bayes**: posterior liên hợp, học tuần tự, phân phối dự đoán, MAP = Ridge, evidence (Occam's razor), kiểm tra độ phủ.
- **Gaussian Process**: kernel, GP regression bằng Cholesky, log marginal likelihood, thiết kế kernel để ngoại suy, $O(n^3)$, Bayesian optimization.
- **HMM**: forward, Viterbi, forward–backward, Baum–Welch; phát hiện chế độ.

## 📋 Yêu cầu trước

[Chương 02](../02_mathematics_for_ml/README.md) (xác suất, MLE/MAP, trị riêng), [Chương 06](../06_linear_regression/README.md) (Ridge), [Chương 09](../09_naive_bayes/README.md) (mô hình sinh), [Chương 20](../20_clustering/README.md) (K-Means), [Chương 21](../21_dimensionality_reduction/README.md).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Phân phối Gaussian nhiều chiều | 📖 Bài học | [View](01_multivariate_gaussian.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/23_probabilistic_machine_learning/01_multivariate_gaussian.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/23_probabilistic_machine_learning/01_multivariate_gaussian.ipynb) |
| 02 | Bài 02 — Gaussian Mixture Model và thuật toán EM | 📖 Bài học | [View](02_gmm_and_em.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/23_probabilistic_machine_learning/02_gmm_and_em.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/23_probabilistic_machine_learning/02_gmm_and_em.ipynb) |
| 03 | Bài 03 — Hồi quy tuyến tính Bayes | 📖 Bài học | [View](03_bayesian_linear_regression.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/23_probabilistic_machine_learning/03_bayesian_linear_regression.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/23_probabilistic_machine_learning/03_bayesian_linear_regression.ipynb) |
| 04 | Bài 04 — Gaussian Process | 📖 Bài học | [View](04_gaussian_processes.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/23_probabilistic_machine_learning/04_gaussian_processes.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/23_probabilistic_machine_learning/04_gaussian_processes.ipynb) |
| 05 | Bài 05 — Hidden Markov Model | 📖 Bài học | [View](05_hidden_markov_models.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/23_probabilistic_machine_learning/05_hidden_markov_models.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/23_probabilistic_machine_learning/05_hidden_markov_models.ipynb) |
| 90 | Bài tập — Chương 23: Probabilistic Machine Learning (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/23_probabilistic_machine_learning/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/23_probabilistic_machine_learning/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 23 · Probabilistic Machine Learning | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/23_probabilistic_machine_learning/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/23_probabilistic_machine_learning/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

`src/ml_course/algorithms/probabilistic.py`: `GaussianMixtureScratch` (EM, full covariance, `reg_covar`). Bài 04 và Bài tập 7: GP regression bằng Cholesky; Bài 05 và Bài tập 10: forward / Viterbi / Baum–Welch (rời rạc và Gaussian).

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS (GP ngoại suy xu hướng + mùa vụ).

## 🔗 Tiếp theo

[24 · Ensemble Learning](../24_ensemble_learning/README.md)
