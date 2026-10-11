# Machine Learning: Zero → Advanced

> Khoá học Machine Learning **tiếng Việt** dạng sách giáo khoa tương tác bằng Jupyter Notebook — từ Python và toán đến Deep Learning, ML đáng tin cậy và MLOps. Mỗi thuật toán được **tự code trước, dùng thư viện sau**; mỗi chương có bài tập 8 cấp độ, checkpoint tự chấm và tiến độ được lưu lại. Kế thừa và mở rộng khoá [AIVN-Machine-Learning](https://github.com/thanhhff/AIVN-Machine-Learning) của AIVIETNAM.

[![tests](https://github.com/ThanhDatVN/AI_journey/actions/workflows/test.yml/badge.svg)](https://github.com/ThanhDatVN/AI_journey/actions/workflows/test.yml)
[![notebooks](https://github.com/ThanhDatVN/AI_journey/actions/workflows/notebook-check.yml/badge.svg)](https://github.com/ThanhDatVN/AI_journey/actions/workflows/notebook-check.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/00_course_orientation/01_how_to_use_this_course.ipynb)

**42 chương · 178 bài học · 12 assignment · 42 bộ bài tập L0→L7 · 42 checkpoint · 8 project · capstone · 6 bài thi tổng hợp · 9 bộ luyện tập xen kẽ.** Dữ liệu được sinh offline hoặc có sẵn trong repo; riêng vài bài dùng ảnh thật (Fashion-MNIST, CIFAR-10) tự tải **một lần** vào `data/raw/` (có kiểm tra checksum). Mọi notebook chạy được trên laptop hoặc Google Colab.

## ✨ Điểm khác biệt

Học một thuật toán không phải là gọi `model.fit(X, y)`. Mỗi chương đi theo cùng một chuỗi:

```text
Lý thuyết → Trực giác → Toán → Tự code từ đầu → Dùng thư viện → Phân tích lỗi & khi nào thất bại
        → Bài tập L0–L7 → Quiz → Mini project → Checkpoint (PASS / REVIEW)
```

- **Tự code trước, thư viện sau**: hồi quy, logistic, cây, rừng, boosting, SVM, K-Means, PCA, GMM, mạng nơ-ron, CNN, RNN, attention, mini-GPT, Q-learning, SHAP, conformal, AIPW… đều được cài bằng NumPy rồi so khớp với scikit-learn/PyTorch.
- **Biết khi nào mô hình thất bại**: mỗi chương có phần "failure cases" — rò rỉ dữ liệu, lời nguyền chiều cao, học đường tắt, trôi khái niệm, vòng phản hồi.
- **Kiểm chứng bằng "sự thật"**: dữ liệu mô phỏng có quy luật thật đã biết (hệ số thật, hiệu ứng nhân quả thật, cụm thật), nên bạn biết chắc mô hình đúng hay sai.
- **Học để nhớ**: bài tập 8 cấp độ (nhớ → giải thích → tính tay → code → debug → phân tích → thực tế → nâng cao), bộ câu hỏi "biết model thật sự", checkpoint chỉ ra **chủ đề yếu** cần ôn, luyện tập xen kẽ, lặp lại ngắt quãng.
- **Như ngoài thực tế**: project có baseline nghiệp vụ và quyết định theo chi phí; capstone từ dữ liệu thô đến giám sát production; một hệ thống phát hiện gian lận có code, test, API, Docker.

## 🚀 Bắt đầu nhanh

```bash
git clone https://github.com/ThanhDatVN/AI_journey.git
cd AI_journey
```

| Cách | Lệnh | Ghi chú |
|---|---|---|
| **uv** (khuyên dùng) | `uv sync --group dev` rồi `uv run jupyter lab` | nhanh, khoá phiên bản bằng `uv.lock` |
| **pip + venv** | `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`) rồi `pip install -e .` | |
| **conda** | `conda env create -f environment.yml && conda activate ml-course` | |
| **Google Colab** | bấm badge "Open in Colab" ở đầu mỗi notebook | ô ⚙️ Setup tự clone repo và cài package |

Thư viện nâng cao chỉ cài khi học tới chương tương ứng: `pip install -e ".[boosting]"` (ch.12, 24), `".[tuning]"` (16), `".[stats]"` (25), `".[dl]"` (27–35), `".[tf]"` (27 · bài 05, tuỳ chọn), `".[xai]"` (36), `".[viz]"` (21), `".[mlops]"` (39–41, project 08) — hoặc tất cả: `pip install -e ".[all]"`.

Sau đó mở [`notebooks/00_course_orientation/01_how_to_use_this_course.ipynb`](notebooks/00_course_orientation/01_how_to_use_this_course.ipynb).

> Đổi `ThanhDatVN/AI_journey` thành repo của bạn để link Colab/nbviewer hoạt động: `python tools/set_github_repo.py <owner>/<repo>`.

## 🗺️ Bản đồ khoá học

Mười giai đoạn, 42 chương. Mỗi chương có README riêng (mục tiêu, yêu cầu trước, mục lục với link GitHub/nbviewer/Colab, điều kiện PASS).

<!-- COURSE:START -->
**Phase 1 — Foundations**

| Chương | Bài học | Bài tập / Assignment | Checkpoint |
|:---|:---:|:---:|:---:|
| [00 · Course Orientation](notebooks/00_course_orientation/README.md) | 5 | 1 | ✅ |
| [01 · Python for ML](notebooks/01_python_for_ml/README.md) | 7 | 5 | ✅ |

**Phase 2 — Mathematics**

| Chương | Bài học | Bài tập / Assignment | Checkpoint |
|:---|:---:|:---:|:---:|
| [02 · Mathematics for ML](notebooks/02_mathematics_for_ml/README.md) | 8 | 2 | ✅ |

**Phase 3 — Data & ML Fundamentals**

| Chương | Bài học | Bài tập / Assignment | Checkpoint |
|:---|:---:|:---:|:---:|
| [03 · Data Analysis & EDA](notebooks/03_data_analysis_and_eda/README.md) | 4 | 1 | ✅ |
| [04 · Data Preprocessing](notebooks/04_data_preprocessing/README.md) | 4 | 1 | ✅ |
| [05 · ML Fundamentals](notebooks/05_machine_learning_fundamentals/README.md) | 8 | 1 | ✅ |

**Phase 4 — Supervised Learning**

| Chương | Bài học | Bài tập / Assignment | Checkpoint |
|:---|:---:|:---:|:---:|
| [06 · Linear Regression](notebooks/06_linear_regression/README.md) | 8 | 4 | ✅ |
| [07 · Logistic Regression](notebooks/07_logistic_regression/README.md) | 8 | 4 | ✅ |
| [08 · K-Nearest Neighbors](notebooks/08_knn/README.md) | 4 | 1 | ✅ |
| [09 · Naive Bayes](notebooks/09_naive_bayes/README.md) | 4 | 1 | ✅ |
| [10 · Decision Trees](notebooks/10_decision_trees/README.md) | 5 | 1 | ✅ |
| [11 · Random Forest](notebooks/11_random_forest/README.md) | 3 | 1 | ✅ |
| [12 · Gradient Boosting](notebooks/12_gradient_boosting/README.md) | 4 | 1 | ✅ |
| [13 · SVM & Kernels](notebooks/13_svm_and_kernels/README.md) | 4 | 1 | ✅ |

**Phase 5 — Evaluation & Practical ML**

| Chương | Bài học | Bài tập / Assignment | Checkpoint |
|:---|:---:|:---:|:---:|
| [14 · Model Evaluation](notebooks/14_model_evaluation/README.md) | 4 | 1 | ✅ |
| [15 · Cross Validation](notebooks/15_cross_validation/README.md) | 3 | 1 | ✅ |
| [16 · Hyperparameter Optimization](notebooks/16_hyperparameter_optimization/README.md) | 3 | 1 | ✅ |
| [17 · Feature Engineering](notebooks/17_feature_engineering/README.md) | 4 | 1 | ✅ |
| [18 · Feature Selection](notebooks/18_feature_selection/README.md) | 3 | 1 | ✅ |
| [19 · Imbalanced Learning](notebooks/19_imbalanced_learning/README.md) | 3 | 1 | ✅ |

**Phase 6 — Unsupervised Learning**

| Chương | Bài học | Bài tập / Assignment | Checkpoint |
|:---|:---:|:---:|:---:|
| [20 · Clustering](notebooks/20_clustering/README.md) | 4 | 1 | ✅ |
| [21 · Dimensionality Reduction](notebooks/21_dimensionality_reduction/README.md) | 3 | 1 | ✅ |
| [22 · Anomaly Detection](notebooks/22_anomaly_detection/README.md) | 3 | 1 | ✅ |

**Phase 7 — Advanced Classical ML**

| Chương | Bài học | Bài tập / Assignment | Checkpoint |
|:---|:---:|:---:|:---:|
| [23 · Probabilistic ML](notebooks/23_probabilistic_machine_learning/README.md) | 5 | 1 | ✅ |
| [24 · Ensemble Learning](notebooks/24_ensemble_learning/README.md) | 3 | 1 | ✅ |
| [25 · Time Series](notebooks/25_time_series/README.md) | 4 | 1 | ✅ |
| [26 · Recommender Systems](notebooks/26_recommender_systems/README.md) | 4 | 1 | ✅ |

**Phase 8 — Deep Learning**

| Chương | Bài học | Bài tập / Assignment | Checkpoint |
|:---|:---:|:---:|:---:|
| [27 · Deep Learning Foundations](notebooks/27_deep_learning_foundations/README.md) | 5 | 1 | ✅ |
| [28 · Neural Networks](notebooks/28_neural_networks/README.md) | 4 | 2 | ✅ |
| [29 · Convolutional Neural Networks](notebooks/29_cnn/README.md) | 4 | 1 | ✅ |
| [30 · Sequence Models](notebooks/30_sequence_models/README.md) | 4 | 1 | ✅ |
| [31 · Transformers](notebooks/31_transformers/README.md) | 4 | 1 | ✅ |
| [32 · Natural Language Processing](notebooks/32_nlp/README.md) | 4 | 1 | ✅ |
| [33 · Computer Vision](notebooks/33_computer_vision/README.md) | 3 | 1 | ✅ |
| [34 · Representation Learning](notebooks/34_representation_learning/README.md) | 3 | 1 | ✅ |
| [35 · Reinforcement Learning](notebooks/35_reinforcement_learning/README.md) | 5 | 1 | ✅ |

**Phase 9 — Trustworthy ML**

| Chương | Bài học | Bài tập / Assignment | Checkpoint |
|:---|:---:|:---:|:---:|
| [36 · Explainable AI](notebooks/36_explainable_ai/README.md) | 3 | 1 | ✅ |
| [37 · Uncertainty & Calibration](notebooks/37_uncertainty_and_calibration/README.md) | 4 | 1 | ✅ |
| [38 · Causal ML](notebooks/38_causal_machine_learning/README.md) | 4 | 1 | ✅ |

**Phase 10 — ML Systems & MLOps**

| Chương | Bài học | Bài tập / Assignment | Checkpoint |
|:---|:---:|:---:|:---:|
| [39 · ML Systems](notebooks/39_ml_systems/README.md) | 3 | 1 | ✅ |
| [40 · MLOps](notebooks/40_mlops/README.md) | 4 | 1 | ✅ |
| [41 · Capstone Projects](notebooks/41_capstone_projects/README.md) | 2 | 1 | ✅ |
<!-- COURSE:END -->

- 📅 **Học theo tuần**: [SCHEDULE.md](SCHEDULE.md) — lịch 44 tuần (6–10 giờ/tuần), kế thừa cách tổ chức theo tuần của khoá AIVN.
- 🧭 **Lộ trình theo mục tiêu**: [ROADMAP.md](ROADMAP.md) — Data Scientist, ML Engineer, Deep Learning, lộ trình cấp tốc.

## 🧩 Cấu trúc một chương

| Tệp | Nội dung |
|---|---|
| `README.md` | mục tiêu, yêu cầu trước, thư viện, mục lục, code tham chiếu, điều kiện PASS |
| `01_…ipynb`, `02_…ipynb`, … | bài học: lý thuyết → trực giác → toán → tự code → thư viện → failure cases → kiểm tra nhanh → tóm tắt → tài liệu |
| `9N_assignment_…ipynb` | assignment kiểu AIVN/Coursera (có ở chương 01, 02, 06, 07, 28) |
| `90_exercises.ipynb` | bài tập **L0 → L7**: nhớ · giải thích · tính tay · code · debug · phân tích · thực tế · nâng cao — ô `START/END CODE HERE` và kiểm tra tự động |
| `99_checkpoint.ipynb` | quiz lý thuyết, quiz toán, bài coding, phân tích (rubric tự chấm), mini project → **PASS / REVIEW** |
| `questions.md` | (từ chương 02) bộ câu hỏi "biết model thật sự" kèm gợi ý — ôn lại sau 1 ngày, 1 tuần, 1 tháng |

## ✅ Checkpoint và tiến độ

Mỗi `99_checkpoint.ipynb` chấm tự động (đáp án được băm, không lộ trong repo) và in bảng kết quả:

```text
══════════════════════════════════════════════
Chapter: 06 · Linear Regression
──────────────────────────────────────────────
Theory Quiz:      9/10
Math Quiz:        5/5
Coding Exercise:  PASS (2/2 tasks)
Mini Project:     PASS
Model Analysis:   PASS
──────────────────────────────────────────────
STATUS: ✅ PASS
NEXT: 07 · Logistic Regression
══════════════════════════════════════════════
```

Chưa đạt thì báo **REVIEW** kèm danh sách chủ đề yếu và bài học cần xem lại. Kết quả lưu vào `progress/progress.json` (commit file này trong fork của bạn). Xem tiến độ: `python -m ml_course.progress`; cập nhật khối dưới đây: `python -m ml_course.progress --update-readme`.

<!-- PROGRESS:START -->
```text
Foundations                   ░░░░░░░░░░   0%  (0/2)
Mathematics                   ░░░░░░░░░░   0%  (0/1)
Data & ML Fundamentals        ░░░░░░░░░░   0%  (0/3)
Supervised Learning           ░░░░░░░░░░   0%  (0/8)
Evaluation & Practical ML     ░░░░░░░░░░   0%  (0/6)
Unsupervised Learning         ░░░░░░░░░░   0%  (0/3)
Advanced Classical ML         ░░░░░░░░░░   0%  (0/4)
Deep Learning                 ░░░░░░░░░░   0%  (0/9)
Trustworthy ML                ░░░░░░░░░░   0%  (0/3)
ML Systems & MLOps            ░░░░░░░░░░   0%  (0/3)

TOTAL                         ░░░░░░░░░░   0%  (0/42)
NEXT: 00 · Course Orientation
```
<!-- PROGRESS:END -->

## 🛠️ Thực hành ngoài chương

| | Nội dung |
|---|---|
| [**Projects**](projects/README.md) | 8 dự án tăng dần độ khó: định giá nhà · khách hàng rời bỏ · phân khúc · dự báo doanh số · gợi ý phim · phân loại tin tiếng Việt · CNN và học đường tắt · **hệ thống phát hiện gian lận end-to-end** (package, CLI, API, test, Docker, giám sát) |
| [**Capstone**](notebooks/41_capstone_projects/README.md) | hướng dẫn 6 tuần, design doc, rubric, kiểm tra cờ đỏ + capstone mẫu: chấm điểm tín dụng từ dữ liệu thô bẩn tới giám sát bằng nhãn sớm |
| [**Bài thi tổng hợp**](quizzes/final_exams/README.md) | 6 bài thi theo giai đoạn, cùng engine chấm như checkpoint |
| [**Luyện tập xen kẽ**](exercises/README.md) | 9 notebook trộn chủ đề — cơ bản, trung cấp, nâng cao |

## 📚 Tài liệu kèm theo

| Tài liệu | Dùng khi |
|---|---|
| [STUDY_GUIDE.md](STUDY_GUIDE.md) | cách học hiệu quả với khoá này: quy trình mỗi bài, khi bị bí, ôn tập ngắt quãng |
| [ROADMAP.md](ROADMAP.md) · [SCHEDULE.md](SCHEDULE.md) | chọn lộ trình và lịch học |
| [NOTATION.md](NOTATION.md) | ký hiệu toán thống nhất trong mọi notebook |
| [GLOSSARY.md](GLOSSARY.md) | thuật ngữ Anh–Việt |
| [CHEATSHEET.md](CHEATSHEET.md) · [docs/cheatsheets/](docs/cheatsheets/README.md) | tra nhanh thuật toán, metric, CV, NumPy, pandas, scikit-learn, PyTorch |
| [data/README.md](data/README.md) | mô tả mọi bộ dữ liệu và hàm sinh dữ liệu |
| [docs/AIVN_MAPPING.md](docs/AIVN_MAPPING.md) | đối chiếu từng bài của khoá AIVN gốc (kể cả các mục chưa có tài liệu và phần "nội dung tiếp theo") với khoá học này |
| [references/](references/books.md) | sách, paper, tài nguyên học tiếp |
| [CHANGELOG.md](CHANGELOG.md) | thay đổi giữa các phiên bản — chương nào đổi nội dung cần xem lại |
| [docs/NOTEBOOK_TEMPLATE.md](docs/NOTEBOOK_TEMPLATE.md) · [CONTRIBUTING.md](CONTRIBUTING.md) | viết thêm bài học, đóng góp |

## 📁 Cấu trúc repo

```text
.
├── notebooks/            # 42 chương: bài học, bài tập (90_), checkpoint (99_), README, questions.md
├── projects/             # 8 project (project.ipynb + README); 08 có mã nguồn dịch vụ thật
├── exercises/            # luyện tập xen kẽ: basic / intermediate / advanced
├── quizzes/              # ngân hàng câu hỏi đã băm đáp án (chapter_quizzes/, final_exams/)
├── solutions/            # lời giải: exercises/, projects/, practice/, checkpoints/, nguồn YAML của quiz & bài thi
├── src/ml_course/        # package dùng chung: datasets, real_datasets, checks, checkpoint, progress, metrics, thuật toán tự cài, envs (RL)
├── data/                 # sample/ (CSV nhỏ, sinh tự động) · raw/ (ảnh thật tải về + dữ liệu của bạn), processed/ — không commit
├── tools/                # build quiz/bài tập, sinh mục lục, chạy toàn bộ notebook, sinh dữ liệu mẫu
├── tests/                # unit test: metrics, preprocessing, thuật toán, engine checkpoint
├── docs/                 # cheat sheets, mẫu notebook, đối chiếu khoá AIVN
├── references/           # sách, paper, tài nguyên
├── progress/             # progress.json — tiến độ học của bạn
└── .github/              # CI (test, chạy lại notebook), mẫu issue và pull request
```

## 🧪 Kiểm thử và phát triển

```bash
uv run pytest                                  # unit test
uv run ruff check .                            # lint code + mọi notebook
uv run python tools/run_notebooks.py notebooks/06_linear_regression   # chạy lại notebook của một chương
uv run python tools/build_quizzes.py --check   # ngân hàng quiz đồng bộ với YAML
uv run python tools/make_exercises.py --check  # notebook bài tập đồng bộ với lời giải
uv run python tools/generate_index.py          # cập nhật mục lục, SCHEDULE.md
```

CI chạy unit test trên Python 3.10/3.12 và chạy lại **toàn bộ** notebook bài học + lời giải mỗi tuần để bảo đảm khoá học luôn chạy được. Xem [CONTRIBUTING.md](CONTRIBUTING.md).

## 🤝 Đóng góp và hỗ trợ

- **Gặp lỗi hoặc thắc mắc?** [Mở issue](https://github.com/ThanhDatVN/AI_journey/issues/new/choose) theo mẫu: notebook chạy lỗi, sai sót nội dung, đề xuất bài học, câu hỏi khi học.
- **Muốn sửa trực tiếp?** Đọc [CONTRIBUTING.md](CONTRIBUTING.md) (nguồn sự thật, công cụ sinh, kiểm tra trước khi gửi PR) và [Quy tắc ứng xử](CODE_OF_CONDUCT.md).
- **Lỗ hổng bảo mật** (vd trong `fraud_service`): báo riêng tư theo [SECURITY.md](SECURITY.md), không mở issue công khai.
- **Dùng khoá học để giảng dạy?** Trích dẫn theo [CITATION.cff](CITATION.cff) (nút *Cite this repository* trên GitHub).

## 🙏 Ghi nhận

- Khung và nhiều nội dung tuần 0–5 (giới thiệu, toán, Python/NumPy/Matplotlib, hồi quy tuyến tính, logistic, regularization, softmax, gradient descent, hàm kích hoạt, assignment kiểu "≈ N dòng") kế thừa từ khoá **[AIVN-Machine-Learning](https://github.com/thanhhff/AIVN-Machine-Learning)** — © 2019 Nguyen Trung Thanh / **AIVIETNAM**, giấy phép MIT — đã được viết lại, cập nhật và mở rộng. Toàn bộ 39 notebook của khoá gốc, các mục còn trống trong README gốc (Jupyter & Markdown, giải tích, lỗi Python thường gặp, học với dữ liệu lớn, Softmax cho Fashion-MNIST, TensorFlow) và phần "nội dung tiếp theo" (mạng nơ-ron, giải thuật di truyền, SVM, K-Means, PCA, phát hiện bất thường, hệ gợi ý) đều có bài tương ứng — xem [docs/AIVN_MAPPING.md](docs/AIVN_MAPPING.md).
- Tham khảo nhiều từ [Machine Learning cơ bản](https://machinelearningcoban.com) (Vũ Hữu Tiệp), [Dive into Deep Learning](https://d2l.ai), khoá Machine Learning của Andrew Ng, tài liệu scikit-learn và PyTorch — xem [references/](references/books.md).

## 📄 Giấy phép

[MIT](LICENSE). Phần kế thừa từ AIVN-Machine-Learning giữ nguyên thông báo bản quyền MIT của tác giả gốc — xem [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
