# Project 04 — Dự báo doanh số theo ngày (Time Series)

> Một chuỗi cửa hàng cần dự báo doanh số **28 ngày tới** để nhập hàng và xếp ca. Lịch khuyến mãi và ngày lễ biết trước. Cách làm hiện tại: lặp lại doanh số tuần vừa rồi (seasonal naive).

| | |
|---|---|
| Bài toán | Dự báo chuỗi thời gian, đa bước (28 ngày) |
| Dữ liệu | `data/sample/sales_daily.csv` — 1.095 ngày (2022–2024), `promo`, `is_holiday` |
| Chương nền tảng | 15, 17, 25, 12, 37 |
| Độ khó · thời gian | ⭐⭐⭐ · 6–8 giờ |

## 🎯 Mục tiêu học tập
- Backtest **rolling-origin** thay vì một lần chia; baseline đúng cho chuỗi thời gian.
- Đặc trưng lag **không rò rỉ** cho dự báo trực tiếp 28 ngày (lag ≥ 28) + kiểm thử tính chất.
- Khoảng dự báo và **kiểm tra độ phủ thật**: quantile regression vs conformal theo thời gian.

## 📋 Nhiệm vụ
1. EDA: mùa vụ tuần/năm, tác động khuyến mãi. 2. `backtest()` + 3 baseline. 3. `make_features()` + gradient boosting. 4. Khoảng 80%: so độ phủ. 5. Dự báo 28 ngày tới + khoảng. 6. Báo cáo.

## ✅ Tiêu chí đạt
- Seasonal naive tốt hơn naive; mô hình tốt hơn seasonal naive ≥ 10% (MAE backtest).
- Đặc trưng không phụ thuộc doanh số 27 ngày gần nhất (kiểm thử tự động).
- Khoảng 80% conformal phủ 70–95% trên backtest.

## 🧭 Gợi ý & cạm bẫy
- Dùng lag 7 cho dự báo 28 ngày là **rò rỉ**: lúc dự báo ngày thứ 20, doanh số 7 ngày trước chưa xảy ra.
- Khoảng dự báo từ mô hình quantile học trên chính train thường quá hẹp — luôn đo độ phủ ngoài mẫu.

## 🚀 Mở rộng
- Dự báo đệ quy (recursive) với lag ngắn và so sánh; mô hình theo từng horizon.
- Thêm biến ngoại sinh (thời tiết giả lập), hiệu ứng trước/sau lễ.
- Dự báo phân cấp (nhiều cửa hàng) và hoà giải tổng.

## 📁 Tệp
- `project.ipynb` · lời giải: [`solutions/projects/04_time_series/project.ipynb`](../../solutions/projects/04_time_series/project.ipynb).
