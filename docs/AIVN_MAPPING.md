# Đối chiếu với khoá AIVN-Machine-Learning

Khoá học này kế thừa khoá [AIVN-Machine-Learning](https://github.com/thanhhff/AIVN-Machine-Learning) của **AIVIETNAM** (© 2019 Nguyen Trung Thanh, giấy phép MIT). Tài liệu này ghi lại **từng** nội dung của khoá gốc đang nằm ở đâu trong khoá học. Bảng bao gồm cả các mục mà README gốc chỉ ghi "View" (chưa có tài liệu) và phần *"Nội dung tiếp theo"* (tuần 8–12).

**Kết quả rà soát (10/2026):**

- Cả 39 notebook của khoá gốc đã được tích hợp: viết lại, sửa lỗi và mở rộng.
- Dữ liệu, hình ảnh, cheat sheet PDF và mã `algorithms/` cũng đã được đưa vào khoá học.
- Mọi mục còn trống hoặc mới chỉ là kế hoạch của khoá gốc đều đã được soạn thành bài học.

Không còn cần bản sao cục bộ của khoá gốc để học hay để build khoá học.

## 1. Notebook của khoá gốc → khoá học này

| Khoá gốc | Khoá học này | Thay đổi chính |
|---|---|---|
| **Tuần 0** · Kỹ năng sử dụng Google Search | [00 · 05](../notebooks/00_course_orientation/05_google_search_and_learning_skills.ipynb) | + đọc thông báo lỗi, đọc tài liệu chính thức, câu hỏi tối giản (MRE), dùng trợ lý AI đúng cách |
| Tuần 0 · Hướng dẫn sử dụng PyCharm | [00 · 04](../notebooks/00_course_orientation/04_pycharm_and_vscode.ipynb) | + cài môi trường (uv / conda / pip), VS Code; bản quyền giáo dục thay cho hướng dẫn cũ |
| Tuần 0 · Tips Google Colab (GPU miễn phí) | [00 · 03](../notebooks/00_course_orientation/03_google_colab_tips.ipynb) | + chạy notebook của khoá trên Colab; công cụ theo dõi huấn luyện hiện hành (hyperdash đã ngừng) |
| **Tuần 1** · Đại số tuyến tính cơ bản | [02 · 02](../notebooks/02_mathematics_for_ml/02_linear_algebra.ipynb) | + chuẩn, khoảng cách, cosine; hai cách nhìn phép nhân ma trận; ma trận là phép biến đổi |
| Tuần 1 · Xác suất cơ bản | [02 · 06](../notebooks/02_mathematics_for_ml/06_probability.ipynb) | + biến ngẫu nhiên, phân phối, luật số lớn, định lý giới hạn trung tâm, Monte Carlo |
| Tuần 1 · Số liệu thống kê cơ bản | [02 · 08](../notebooks/02_mathematics_for_ml/08_statistics.ipynb) | + tổng thể vs mẫu, sai lệch lấy mẫu, khoảng tin cậy, kiểm định giả thuyết |
| Tuần 1 · Ký hiệu toán thường dùng | [02 · 01](../notebooks/02_mathematics_for_ml/01_math_notation.ipynb), [NOTATION.md](../NOTATION.md) | + quy ước thống nhất của khoá, từ công thức đến code |
| Tuần 1 · Python cơ bản (và `Python/Python-Basic`) | [01 · 01](../notebooks/01_python_for_ml/01_python_basics.ipynb) | + bẫy Python hay gặp khi làm ML, class theo mẫu `fit`/`predict` |
| Tuần 1 · Assignment: Cài đặt công thức toán bằng Python | [01 · 91](../notebooks/01_python_for_ml/91_assignment_math_formulas_in_python.ipynb) | khung "≈ N dòng", có lời giải và kiểm tra tự động |
| Tuần 1 · Assignment: Hệ số tương quan giữa 2 ảnh | [02 · 91](../notebooks/02_mathematics_for_ml/91_assignment_image_correlation.ipynb) | + phiên bản vectorized, phân loại chữ số bằng tương quan |
| **Tuần 2** · Giới thiệu về Machine Learning | [05 · 01](../notebooks/05_machine_learning_fundamentals/01_what_is_machine_learning.ipynb) | + khi nào **không** nên dùng ML, quy trình một dự án ML |
| Tuần 2 · Các nhóm thuật toán trong ML | [05 · 02](../notebooks/05_machine_learning_fundamentals/02_types_of_machine_learning.ipynb) | + học tự giám sát, thuật ngữ cơ bản |
| Tuần 2 · Linear Regression with One Variable | [06 · 01](../notebooks/06_linear_regression/01_linear_regression_one_variable.ipynb) | + quá trình học trên mặt $J(\theta_0, \theta_1)$, đối chiếu nghiệm đóng và scikit-learn |
| Tuần 2 · Python - Numpy (và `Python/Python-Numpy`) | [01 · 02](../notebooks/01_python_for_ml/02_numpy.ipynb), [01 · 03](../notebooks/01_python_for_ml/03_numpy_vectorization_and_broadcasting.ipynb) | + bài riêng về vectorization và broadcasting |
| Tuần 2 · Assignment (tuỳ chọn): Python - Numpy | [01 · 92](../notebooks/01_python_for_ml/92_assignment_numpy_practice.ipynb) | |
| Tuần 2 · Assignment: Python Basics With Numpy | [01 · 93](../notebooks/01_python_for_ml/93_assignment_python_basics_with_numpy.ipynb) | |
| Tuần 2 · Assignment: Linear Regression with One Variable | [06 · 91](../notebooks/06_linear_regression/91_assignment_linear_regression_one_variable.ipynb) | |
| **Tuần 3** · Linear Regression with Multiple Variables | [06 · 02](../notebooks/06_linear_regression/02_linear_regression_multiple_variables.ipynb) | + cài đặt trên dữ liệu giá nhà nhiều feature |
| Tuần 3 · Feature Normalization | [04 · 02](../notebooks/04_data_preprocessing/02_feature_scaling.ipynb) | + mô hình nào cần scaling (thực nghiệm), quy tắc fit trên train |
| Tuần 3 · Gradient Descent Tips và Normal Equation | [06 · 03](../notebooks/06_linear_regression/03_normal_equation_and_gd_tips.ipynb) | + chứng minh, ý nghĩa hình học, khi $X^\top X$ suy biến |
| Tuần 3 · Polynomial Regression | [06 · 05](../notebooks/06_linear_regression/05_polynomial_regression.ipynb) | + chọn bậc bằng validation, bùng nổ số feature |
| Tuần 3 · Python - Matplotlib (và `Python/Python-Matplotlib`) | [01 · 05](../notebooks/01_python_for_ml/05_matplotlib.ipynb) | + các loại biểu đồ dùng nhiều nhất trong ML |
| Tuần 3 · Assignment: Python - Matplotlib | [01 · 94](../notebooks/01_python_for_ml/94_assignment_matplotlib.ipynb) | |
| Tuần 3 · Assignment: Linear Regression with Multiple Variables | [06 · 92](../notebooks/06_linear_regression/92_assignment_linear_regression_multiple_variables.ipynb) | |
| **Tuần 4** · Logistic Regression | [07 · 01](../notebooks/07_logistic_regression/01_logistic_regression.ipynb) | + odds/log-odds, vì sao không dùng MSE |
| Tuần 4 · Regularization | [06 · 07](../notebooks/06_linear_regression/07_regularization_ridge_lasso.ipynb), [07 · 03](../notebooks/07_logistic_regression/03_regularized_logistic_regression.ipynb) | + Lasso, Elastic Net, coefficient path, chọn λ bằng cross-validation |
| Tuần 4 · Phân loại 2 chữ số viết tay | [07 · 04](../notebooks/07_logistic_regression/04_handwritten_digits_binary.ipynb) | + nhìn vào trọng số, xem ảnh bị đoán sai |
| Tuần 4 · Logistic Regression - One vs All (phần 1) | [07 · 05](../notebooks/07_logistic_regression/05_one_vs_all.ipynb) | One-vs-All from scratch trên `digits` |
| Tuần 4 · One vs All (phần 2: CIFAR-10) | [07 · 08](../notebooks/07_logistic_regression/08_linear_classifiers_cifar10.ipynb) | **bài mới**: OvA, softmax, SVM đa lớp trên CIFAR-10 thật; gradient check, chọn siêu tham số, learning curve |
| Tuần 4 · Assignment: Logistic Regression | [07 · 91](../notebooks/07_logistic_regression/91_assignment_logistic_regression.ipynb) | |
| Tuần 4 · Assignment: Regularized Logistic Regression | [07 · 92](../notebooks/07_logistic_regression/92_assignment_regularized_logistic_regression.ipynb) | |
| **Tuần 5** · Các hàm kích hoạt | [28 · 01](../notebooks/28_neural_networks/01_activation_functions.ipynb) | + GELU/SiLU, vanishing gradient, dying ReLU, so sánh khi huấn luyện |
| Tuần 5 · Stochastic, Mini-Batch, Batch Gradient Descent | [06 · 04](../notebooks/06_linear_regression/04_batch_sgd_minibatch_gd.ipynb) | + cài đặt đầy đủ (bản gốc là mã giả), learning rate decay, chọn batch size, học với dữ liệu lớn |
| Tuần 5 · Softmax Regression | [07 · 06](../notebooks/07_logistic_regression/06_softmax_regression.ipynb) | + hoàn thiện cross-entropy (bản gốc để trống), gradient, cài đặt from scratch |
| Tuần 5 · Assignment: Lập trình các hàm kích hoạt | [28 · 91](../notebooks/28_neural_networks/91_assignment_activation_functions.ipynb) | + đạo hàm, softmax ổn định số, GELU |
| Tuần 5 · Assignment: Stochastic, Mini-Batch GD | [06 · 93](../notebooks/06_linear_regression/93_assignment_sgd_minibatch.ipynb) | |
| **Archives** · Tutorial 3: Làm việc với dữ liệu | [03 · 01](../notebooks/03_data_analysis_and_eda/01_types_of_data_and_storage.ipynb) | + chuỗi thời gian, Parquet, cảnh báo bảo mật pickle, lấy dữ liệu từ URL/Drive/Dropbox/Kaggle |

## 2. Các mục README gốc chưa có tài liệu → đã soạn

| Mục trong README gốc | Khoá học này |
|---|---|
| Tuần 0 · Giới thiệu | [00 · 01](../notebooks/00_course_orientation/01_how_to_use_this_course.ipynb), [README](../README.md) |
| Tuần 0 · Hướng dẫn sử dụng Python trên Web AIVIETNAM | nền tảng riêng của AIVIETNAM → thay bằng Google Colab ([00 · 03](../notebooks/00_course_orientation/03_google_colab_tips.ipynb)) và cài đặt local ([00 · 04](../notebooks/00_course_orientation/04_pycharm_and_vscode.ipynb)) |
| Tuần 0 · Jupyter Notebook và kỹ thuật dùng Markdown | [00 · 02](../notebooks/00_course_orientation/02_jupyter_notebook_and_markdown.ipynb) |
| Tuần 0 · Hướng dẫn sử dụng Google Colab | [00 · 03](../notebooks/00_course_orientation/03_google_colab_tips.ipynb), mục 1 |
| Tuần 0 · Cách học hiệu quả & hướng dẫn làm bài tập | [00 · 01](../notebooks/00_course_orientation/01_how_to_use_this_course.ipynb), [STUDY_GUIDE.md](../STUDY_GUIDE.md) |
| Tuần 1 · Giải tích cơ bản | [02 · 04](../notebooks/02_mathematics_for_ml/04_calculus_and_gradients.ipynb), [02 · 05](../notebooks/02_mathematics_for_ml/05_gradient_descent.ipynb) |
| Tuần 1 · Tài liệu toán (Vũ Hữu Tiệp) | [references/useful_resources.md](../references/useful_resources.md) |
| Tuần 1 · Những lỗi thường gặp trong Python | [01 · 07](../notebooks/01_python_for_ml/07_common_python_errors.ipynb) — **bài mới** |
| Quiz các tuần (Google Forms) | quiz trong checkpoint của mọi chương ([`solutions/quizzes/`](../solutions/quizzes)) và 6 [bài thi tổng hợp](../quizzes/final_exams/README.md) |
| Tuần 5 · Learning with Large Datasets | [06 · 04](../notebooks/06_linear_regression/04_batch_sgd_minibatch_gd.ipynb), mục 7 — theo dõi hội tụ SGD, online learning, map-reduce |
| Tuần 5 · Assignment: Softmax Regression cho MNIST Fashion | [07 · 93](../notebooks/07_logistic_regression/93_assignment_softmax_regression.ipynb), mục 6 — trên Fashion-MNIST thật |
| Tuần 6 · Đánh giá hàm giả thuyết & lựa chọn mô hình | [05 · 03](../notebooks/05_machine_learning_fundamentals/03_train_validation_test_and_model_selection.ipynb), chương [14](../notebooks/14_model_evaluation/README.md)–[16](../notebooks/16_hyperparameter_optimization/README.md) |
| Tuần 6 · Train / Validation / Test | [05 · 03](../notebooks/05_machine_learning_fundamentals/03_train_validation_test_and_model_selection.ipynb), chương [15](../notebooks/15_cross_validation/README.md) |
| Tuần 6 · Chẩn đoán Bias vs. Variance & Regularization | [05 · 04](../notebooks/05_machine_learning_fundamentals/04_overfitting_and_bias_variance.ipynb), [06 · 07](../notebooks/06_linear_regression/07_regularization_ridge_lasso.ipynb) |
| Tuần 6 · Learning Curves | [05 · 05](../notebooks/05_machine_learning_fundamentals/05_learning_curves_and_error_analysis.ipynb), [07 · 08](../notebooks/07_logistic_regression/08_linear_classifiers_cifar10.ipynb) mục 8 |
| Tuần 6 · Machine Learning System Design | [05 · 05](../notebooks/05_machine_learning_fundamentals/05_learning_curves_and_error_analysis.ipynb) (error analysis), chương [14](../notebooks/14_model_evaluation/README.md), [19](../notebooks/19_imbalanced_learning/README.md), [39 · 01](../notebooks/39_ml_systems/01_ml_system_design.ipynb) |
| Tuần 7 · TensorFlow | [27 · 05](../notebooks/27_deep_learning_foundations/05_tensorflow_and_keras.ipynb) — **bài mới** (TensorFlow/Keras, đối chiếu PyTorch) |

## 3. "Nội dung tiếp theo" của khoá gốc (tuần 8–12)

| Kế hoạch | Khoá học này |
|---|---|
| Tuần 8 · Neural Networks: Representation — Non-linear Hypotheses | [27 · 01](../notebooks/27_deep_learning_foundations/01_perceptron_and_neuron.ipynb) (XOR, lớp ẩn), [06 · 05](../notebooks/06_linear_regression/05_polynomial_regression.ipynb) mục 5 (bùng nổ feature đa thức) |
| — Multiclass Classification | [27 · 03](../notebooks/27_deep_learning_foundations/03_neural_network_from_scratch.ipynb) (xoắn ốc 3 lớp, chữ số 10 lớp) |
| — Cost Function | [27 · 03](../notebooks/27_deep_learning_foundations/03_neural_network_from_scratch.ipynb) (softmax cross-entropy), [28 · 04](../notebooks/28_neural_networks/04_regularization_and_training_recipes.ipynb) (regularization) |
| — Backpropagation Algorithm | [27 · 02](../notebooks/27_deep_learning_foundations/02_computational_graph_and_backprop.ipynb) (autograd tự viết), [27 · 03](../notebooks/27_deep_learning_foundations/03_neural_network_from_scratch.ipynb) |
| — Gradient Checking | [27 · 02](../notebooks/27_deep_learning_foundations/02_computational_graph_and_backprop.ipynb), [27 · 03](../notebooks/27_deep_learning_foundations/03_neural_network_from_scratch.ipynb), [07 · 08](../notebooks/07_logistic_regression/08_linear_classifiers_cifar10.ipynb) |
| — Random Initialization | [28 · 02](../notebooks/28_neural_networks/02_initialization_and_normalization.ipynb) (khởi tạo bằng 0 không phá được đối xứng; Xavier/He) |
| Tuần 9 · Giải thuật di truyền (Neural Network + GA) | [35 · 05](../notebooks/35_reinforcement_learning/05_neuroevolution_genetic_algorithm.ipynb) (GA cho bài toán cái túi, GA tiến hoá trọng số chính sách CartPole, Evolution Strategies) |
| Tuần 10 · Support Vector Machine | chương [13](../notebooks/13_svm_and_kernels/README.md); SVM đa lớp ở [07 · 08](../notebooks/07_logistic_regression/08_linear_classifiers_cifar10.ipynb) |
| Tuần 11 · Unsupervised Learning — K-Means, PCA | chương [20](../notebooks/20_clustering/README.md) (K-Means from scratch), [21](../notebooks/21_dimensionality_reduction/README.md) (PCA from scratch) |
| Tuần 12 · Anomaly Detection & Recommender Systems | chương [22](../notebooks/22_anomaly_detection/README.md) (Gaussian, chọn ε theo F1), [26](../notebooks/26_recommender_systems/README.md) (lọc cộng tác, matrix factorization) |

## 4. Tài nguyên khác của khoá gốc

| Khoá gốc | Khoá học này |
|---|---|
| `Week 2/data/my_house_price_prediction.csv`, `Week 2/Assignment/data/ex2data.txt` | `data/sample/my_house_price_prediction.csv`, `data/sample/ex2_city_profit.txt` |
| `Week 3/data/Position_Salaries.csv`, `Week 3/Assignment/data/data.txt` | `data/sample/Position_Salaries.csv`, `data/sample/ex3_house_multivariate.txt` |
| `Week 4/Assignment/data/data1.txt`, `data2.txt` | `data/sample/ex4_exam_scores.txt`, `data/sample/ex4_microchip_tests.txt` |
| `Archives/.../data/iris.csv` | `data/sample/iris.csv` |
| `Week 4/datasets/` (CIFAR-10), `Week 4/CIFAR.py` | tải bản chính thức (có kiểm tra MD5) bằng `ml_course.real_datasets.load_cifar10` |
| `Week 4/algorithms/classifiers/` (linear classifier, loss/gradient logistic–softmax–SVM) | cài lại vectorized trong [07 · 08](../notebooks/07_logistic_regression/08_linear_classifiers_cifar10.ipynb); `LogisticRegressionScratch`, `SoftmaxRegressionScratch` trong `src/ml_course/algorithms/` |
| `Week 4/algorithms/gradient_check.py` | `ml_course.utils.numerical_gradient`, `relative_error`; `grad_check_sparse` trong [07 · 08](../notebooks/07_logistic_regression/08_linear_classifiers_cifar10.ipynb) |
| `Week 4/algorithms/regression/linear_regression.py` | `LinearRegressionScratch`, `RidgeScratch`, `LassoScratch` trong `src/ml_course/algorithms/linear_models.py` |
| Hình minh hoạ (`images/` của từng tuần) | các hình được dùng đã chép vào `notebooks/<chương>/images/` (và `solutions/exercises/<chương>/images/` cho assignment) |
| `Python/Python3_reference_cheat_sheet.pdf`, `Week 3/Matplotlib_Cheat_Sheet.pdf` | [docs/cheatsheets/](cheatsheets/README.md) |
| `LICENSE` (MIT) | [LICENSE](../LICENSE) — giữ nguyên thông báo bản quyền gốc |
| `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` | [CONTRIBUTING.md](../CONTRIBUTING.md) — mục *Quy tắc ứng xử* (Contributor Covenant) và *Ghi nhận* (cảm ơn AIVIETNAM và người đóng góp cho khoá gốc) |
| `environment.yml` | [environment.yml](../environment.yml) (cập nhật cho Python 3.11 và package `ml_course`) |
| Tổ chức học theo tuần | [SCHEDULE.md](../SCHEDULE.md) — lịch 44 tuần |
