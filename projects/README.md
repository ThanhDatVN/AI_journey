# 🛠️ Projects — 8 dự án tăng dần độ khó

Mỗi project là một bài toán **thực tế thu nhỏ**: có bối cảnh kinh doanh, dữ liệu (thường là dữ liệu bẩn hoặc có cạm bẫy), một **baseline** của cách làm hiện tại, và tiêu chí đạt được **kiểm tra tự động** ngay trong notebook. Làm project sau khi xong các chương liên quan; làm hết trước khi bắt đầu [Capstone](../notebooks/41_capstone_projects/README.md).

| # | Project | Bài toán | Kỹ năng chính | Chương nền tảng | Độ khó | Thời gian |
|---|---|---|---|---|---|---|
| 01 | [Định giá nhà](01_regression/README.md) | Hồi quy | làm sạch dữ liệu bẩn, log-target, Ridge vs boosting, phân tích sai số | 03–06, 12, 14–15, 17 | ⭐⭐ | 4–6 giờ |
| 02 | [Khách hàng rời bỏ](02_classification/README.md) | Phân loại | so sánh mô hình bằng CV, calibration, lift, quyết định theo kinh tế chiến dịch | 07, 11–12, 14–15, 19, 37 | ⭐⭐ | 5–7 giờ |
| 03 | [Phân khúc khách hàng](03_clustering/README.md) | Phân cụm + PCA | chọn k, độ ổn định, GMM + BIC, đặt tên phân khúc | 04, 20, 21, 23 | ⭐⭐ | 4–6 giờ |
| 04 | [Dự báo doanh số](04_time_series/README.md) | Chuỗi thời gian | backtest rolling-origin, lag không rò rỉ, khoảng dự báo conformal | 15, 17, 25, 37 | ⭐⭐⭐ | 6–8 giờ |
| 05 | [Gợi ý phim](05_recommendation/README.md) | Gợi ý | chia theo thời gian, NDCG@k, item-kNN, PureSVD, cold start, thiên lệch phổ biến | 21, 26 | ⭐⭐⭐ | 6–8 giờ |
| 06 | [Phân loại tin tức tiếng Việt](06_nlp/README.md) | NLP | TF-IDF âm tiết/cặp âm tiết/ký tự, phân tích lỗi, nhãn nhiễu, embedding kiểu fastText, Transformer nhỏ | 09, 32, 31 | ⭐⭐⭐ | 5–7 giờ |
| 07 | [Nhận dạng hình khối](07_computer_vision/README.md) | Thị giác máy tính | CNN, học đường tắt (màu), dịch chuyển miền, tăng cường dữ liệu | 28, 29, 33 | ⭐⭐⭐⭐ | 6–8 giờ |
| 08 | [Hệ thống phát hiện gian lận](08_end_to_end_ml_system/README.md) | End-to-end | chính sách tổn thất kỳ vọng, package + CLI + API + test + Docker + giám sát | 14, 19, 37, 39, 40 | ⭐⭐⭐⭐⭐ | 12–20 giờ |

## Cách làm một project

1. Đọc `README.md` của project: bối cảnh, nhiệm vụ, tiêu chí đạt.
2. Mở `project.ipynb`, điền code vào các ô `### START CODE HERE ###` … `### END CODE HERE ###`; chạy ô kiểm tra ngay sau mỗi bước (✅/❌).
3. Viết các phần ✍️ (nhận xét, báo cáo) **bằng lời của bạn** — đây là phần quan trọng nhất khi đi phỏng vấn.
4. Chỉ mở lời giải (`solutions/projects/<project>/project.ipynb`) sau khi đã tự làm; so sánh cả code lẫn báo cáo.
5. Làm phần **Mở rộng** trong README nếu muốn đưa vào portfolio.

Dữ liệu đều có sẵn trong `data/sample/` hoặc sinh offline bằng `ml_course.datasets` — không cần tải gì.
