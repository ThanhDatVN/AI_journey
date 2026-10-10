# 🏋️ Luyện tập xen kẽ (interleaved practice)

Bài tập cuối mỗi chương (`90_exercises.ipynb`) luyện **một** chủ đề. Thư mục này trộn nhiều chủ đề trong cùng một notebook — buộc bạn tự nhận ra *bài này cần công cụ nào*, kỹ năng quan trọng nhất khi làm dự án thật. Nghiên cứu về học tập (interleaving, retrieval practice) cho thấy luyện xen kẽ và **tự nhớ lại** giúp nhớ lâu hơn nhiều so với đọc lại.

| Cấp | Notebook | Nội dung | Làm sau chương |
|---|---|---|---|
| 🟢 Cơ bản | [01 · Python, NumPy, pandas](basic/01_python_numpy_pandas.ipynb) | comprehension, đếm, broadcasting, ma trận khoảng cách, groupby/transform/merge, dữ liệu thiếu | 01, 03, 04 |
| 🟢 Cơ bản | [02 · Toán và thống kê](basic/02_math_and_statistics.ipynb) | cosine, hệ phương trình, trị riêng, kiểm tra gradient, gradient descent & phân kỳ, Bayes, MLE, bootstrap | 02 |
| 🟢 Cơ bản | [03 · Quy trình ML](basic/03_ml_workflow.ipynb) | chia phân tầng, pipeline + baseline, ma trận nhầm lẫn, CV, đường cong học, lưu/nạp mô hình | 04–07, 14–15 |
| 🟡 Trung cấp | [01 · Tự cài thuật toán](intermediate/01_from_scratch_algorithms.ipynb) | logistic (GD + L2), decision stump, kNN hồi quy, K-Means, PCA, Gaussian NB — so khớp scikit-learn | 06–11, 20–21 |
| 🟡 Trung cấp | [02 · Chọn mô hình & đánh giá](intermediate/02_model_selection_and_evaluation.ipynb) | GroupKFold, TimeSeriesSplit, grid vs random search, lời nguyền người thắng & nested CV, ROC vs PR, calibration | 14–16, 19, 37 |
| 🟡 Trung cấp | [03 · Gỡ lỗi code ML](intermediate/03_debugging_ml_code.ipynb) | sai trục, `fit_transform` trên test, biên thời gian, micro vs macro F1, kiểu dữ liệu, seed, chọn đặc trưng ngoài CV | 04, 05, 14–19 |
| 🔴 Nâng cao | [01 · Kỹ thuật ML nâng cao](advanced/01_advanced_ml_techniques.ipynb) | target encoding OOF, stacking OOF, đặc trưng 7 ngày đúng thời điểm, permutation importance, bootstrap theo cặp | 17, 24, 36, 39 |
| 🔴 Nâng cao | [02 · Bên trong deep learning](advanced/02_deep_learning_internals.ipynb) | conv2d, max pooling + gradient, backprop + kiểm tra gradient, LayerNorm, multi-head attention, positional encoding | 27–31 |
| 🔴 Nâng cao | [03 · ML đáng tin cậy & nhân quả](advanced/03_trustworthy_ml_and_causality.ipynb) | Platt (Newton), ECE, conformalized quantile regression, IPW/AIPW, chỉ số công bằng, PSI phân loại | 36–38, 40 |

## Cách dùng

- Mỗi bài có ô `### START CODE HERE ###` và ô kiểm tra (✅/❌) ngay sau. Làm **không mở chương** — chỉ mở lại khi bí quá 15 phút.
- Lặp lại ngắt quãng: làm lại cùng notebook sau 1 tuần và 1 tháng (xoá code cũ). Nếu bài nào phải mở chương, đánh dấu để ôn.
- Lời giải: `solutions/practice/<cấp>/<notebook>.ipynb` — gồm cả giải thích **vì sao** cho các bài có ô ✍️.
