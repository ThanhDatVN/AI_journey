# Chương 09 · Naive Bayes

> Model sinh đầu tiên của khoá học: học bằng cách **đếm**, phân loại bằng định lý Bayes. Nhanh, mạnh với văn bản và ít dữ liệu — và là bài học rõ nhất về khác biệt giữa *phân loại đúng* và *xác suất đúng*. Không có trong khoá AIVN gốc — bổ sung mới theo template "mọi model".

## 🎯 Mục tiêu

- Định lý Bayes, base rate fallacy; model **sinh** vs model **phân biệt**; quy tắc MAP.
- Giả định **độc lập có điều kiện** và vì sao nó giảm số tham số từ $2^d$ xuống $d$.
- Laplace smoothing (zero-frequency), tính trong không gian log + log-sum-exp (underflow).
- Cài đặt **Gaussian NB** và **Multinomial NB** from scratch; Bernoulli, Complement, Categorical NB.
- Phân loại văn bản bằng bag-of-words; giới hạn với phủ định.
- Vì sao "ngây thơ" vẫn hiệu quả (Domingos & Pazzani), NB vs Logistic Regression (Ng & Jordan), và các failure case: xác suất quá tự tin, XOR, phân phối sai giả định.

## 📋 Yêu cầu trước

[Chương 02](../02_mathematics_for_ml/README.md) (xác suất, MLE/MAP), [Chương 07](../07_logistic_regression/README.md) (Logistic Regression, softmax ổn định số học).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Định lý Bayes và giả định "ngây thơ" | 📖 Bài học | [View](01_bayes_and_naive_assumption.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/09_naive_bayes/01_bayes_and_naive_assumption.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/09_naive_bayes/01_bayes_and_naive_assumption.ipynb) |
| 02 | Bài 02 — Gaussian Naive Bayes from scratch | 📖 Bài học | [View](02_gaussian_nb_from_scratch.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/09_naive_bayes/02_gaussian_nb_from_scratch.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/09_naive_bayes/02_gaussian_nb_from_scratch.ipynb) |
| 03 | Bài 03 — Multinomial & Bernoulli Naive Bayes cho văn bản | 📖 Bài học | [View](03_multinomial_bernoulli_text.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/09_naive_bayes/03_multinomial_bernoulli_text.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/09_naive_bayes/03_multinomial_bernoulli_text.ipynb) |
| 04 | Bài 04 — Vì sao "ngây thơ" vẫn hiệu quả, và khi nào thất bại? | 📖 Bài học | [View](04_why_naive_works_and_fails.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/09_naive_bayes/04_why_naive_works_and_fails.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/09_naive_bayes/04_why_naive_works_and_fails.ipynb) |
| 90 | Bài tập — Chương 09: Naive Bayes (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/09_naive_bayes/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/09_naive_bayes/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 09 · Naive Bayes | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/09_naive_bayes/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/09_naive_bayes/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

`src/ml_course/algorithms/naive_bayes.py`: `GaussianNBScratch`, `MultinomialNBScratch`.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS.

## 🔗 Tiếp theo

[10 · Decision Trees](../10_decision_trees/README.md)
