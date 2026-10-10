# 🎓 Bài thi tổng hợp

Sáu bài thi, mỗi bài phủ một giai đoạn của khoá học. Làm bài thi **sau khi đã PASS checkpoint của mọi chương** trong giai đoạn đó, không mở tài liệu, giới hạn khoảng 2–3 giờ. Kết quả được lưu vào `progress/progress.json` như checkpoint.

| Bài thi | Phạm vi | Lý thuyết | Toán | Coding | Mini project |
|---|---|---|---|---|---|
| [1 · Nền tảng](final_exam_1_foundations.ipynb) | 00–05: Python, toán, EDA, tiền xử lý, nền tảng ML | 20 | 8 | chuẩn hoá không rò rỉ · gradient descent | hồi quy giá nhà (MAPE < 12%) |
| [2 · Học có giám sát](final_exam_2_supervised.ipynb) | 06–13: tuyến tính, logistic, kNN, NB, cây, rừng, boosting, SVM | 20 | 8 | ngưỡng chia Gini · kNN từ đầu | chọn mô hình churn bằng CV |
| [3 · ML thực hành](final_exam_3_practical_ml.ipynb) | 14–19: metric, CV, tinh chỉnh, đặc trưng, mất cân bằng | 20 | 8 | precision/recall tại ngưỡng · ngưỡng theo chi phí | gian lận + ngưỡng chi phí |
| [4 · Không giám sát & nâng cao](final_exam_4_unsupervised_advanced.ipynb) | 20–26: phân cụm, giảm chiều, bất thường, xác suất, ensemble, chuỗi thời gian, gợi ý | 20 | 8 | một vòng K-Means · PCA bằng SVD | phân khúc khách hàng |
| [5 · Deep Learning](final_exam_5_deep_learning.ipynb) | 27–35: mạng nơ-ron, CNN, RNN, Transformer, NLP, CV, biểu diễn, RL | 20 | 8 | softmax/cross-entropy ổn định · attention | MLP 2 lớp bằng numpy |
| [6 · Đáng tin cậy & hệ thống](final_exam_6_trustworthy_systems.ipynb) | 36–41: giải thích, bất định, nhân quả, hệ thống, MLOps, capstone | 20 | 8 | khoảng conformal · PSI | hiệu chỉnh isotonic + tập conformal |

Tiêu chí PASS giống checkpoint chương: Lý thuyết ≥ 80%, Toán ≥ 80%, mọi bài coding và mini project đạt kiểm tra tự động, phần phân tích tự chấm theo rubric ≥ 80%. Bài thi chưa đạt sẽ liệt kê **chủ đề yếu** kèm chương cần ôn.

Lời giải: `solutions/exams/<bài thi>.yaml` (nguồn có đáp án) và `solutions/checkpoints/<bài thi>.ipynb` (bản điền sẵn) — chỉ xem sau khi làm.
