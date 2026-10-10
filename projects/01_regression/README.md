# Project 01 — Định giá nhà (Regression)

> Một sàn bất động sản muốn **gợi ý giá đăng bán** cho căn nhà ở TP.HCM. Nhân viên hiện ước lượng bằng "giá trung vị mỗi m² của quận × diện tích". Bạn có dữ liệu giao dịch **thô** và cần một mô hình tốt hơn — kèm giải thích được.

| | |
|---|---|
| Bài toán | Hồi quy (giá, tỉ đồng) |
| Dữ liệu | `data/sample/house_prices_dirty.csv` — 1.515 dòng, 8 cột; có dòng trùng, giá trị thiếu, tên quận viết 23 kiểu, diện tích gõ thừa số 0 |
| Chương nền tảng | 03, 04, 05, 06, 12, 14, 15, 17 |
| Độ khó · thời gian | ⭐⭐ · 4–6 giờ |

## 🎯 Mục tiêu học tập
- Kiểm tra và làm sạch dữ liệu thô có lý do (sửa chắc chắn vs để pipeline xử lý).
- So sánh với **baseline nghiệp vụ**, không chỉ baseline "trung vị".
- Mô hình hoá **log giá**; pipeline tiền xử lý không rò rỉ; chọn mô hình bằng cross-validation.
- Phân tích sai số theo nhóm; diễn giải hệ số theo % giá.

## 📋 Nhiệm vụ
1. Kiểm tra dữ liệu: trùng, thiếu, cách viết quận, diện tích vô lý.
2. Hàm `clean()`: bỏ trùng, chuẩn hoá tên quận, sửa lỗi gõ diện tích — **không** điền thiếu ở đây.
3. Chia 80/20; baseline trung vị và baseline giá/m² theo quận.
4. Ridge trên log giá (điền thiếu → log1p → chuẩn hoá; one-hot) và gradient boosting; so sánh 5-fold CV.
5. Đánh giá test một lần: MAE, MAPE, R².
6. Sai số theo quận và theo ngũ phân vị giá.
7. Hệ số → % chênh lệch giá; báo cáo cho trưởng phòng kinh doanh.

## ✅ Tiêu chí đạt (kiểm tra tự động)
- Dữ liệu sạch: đúng 5 quận chuẩn, không dòng trùng, không diện tích > 400 m², còn giữ giá trị thiếu cho pipeline.
- MAPE test < 16% và MAE test thấp hơn baseline nghiệp vụ ít nhất 15%.
- Sai số tuyệt đối tăng theo giá; hệ số quận/sân vườn đúng hướng.

## 📦 Sản phẩm
`project.ipynb` đã điền + báo cáo 5–8 câu (giá trị so với cách làm hiện tại, cách trình bày giá gợi ý, hạn chế, bước tiếp theo).

## 🧭 Gợi ý & cạm bẫy
- Điền giá trị thiếu **trước khi chia** train/test là rò rỉ — để `SimpleImputer` trong pipeline.
- Đừng xoá ngoại lai một cách máy móc: diện tích 900 m² cho căn 2 phòng ngủ là lỗi gõ, sửa được.
- MAPE và MAE kể hai câu chuyện khác nhau với nhà đắt/rẻ — báo cáo cả hai.

## 🚀 Mở rộng
- Khoảng dự đoán bằng quantile regression hoặc conformal prediction (chương 37) — "giá gợi ý 4.2–4.9 tỉ".
- Thêm đặc trưng tương tác (diện tích × quận); so sánh với mô hình cộng tính tổng quát.
- Giải thích từng dự đoán (chương 36) cho người bán.

## 📁 Tệp
- `project.ipynb` — notebook làm bài · lời giải: [`solutions/projects/01_regression/project.ipynb`](../../solutions/projects/01_regression/project.ipynb) (mở sau khi làm).
