# Project 02 — Dự đoán khách hàng rời bỏ (Classification)

> Một nhà mạng chạy **chiến dịch gọi điện giữ chân**: mỗi cuộc gọi 50.000đ; gọi đúng người sắp rời bỏ thì 30% ở lại, mỗi khách giữ được đáng giá 1.500.000đ; tổng đài chỉ gọi được 20% khách mỗi tháng. Hiện đội marketing gọi theo luật "hợp đồng tháng và dùng dưới 12 tháng".

| | |
|---|---|
| Bài toán | Phân loại nhị phân + quyết định theo chi phí |
| Dữ liệu | `data/sample/customer_churn.csv` — 5.000 khách, 9 đặc trưng, ~29% rời bỏ |
| Chương nền tảng | 07, 11, 12, 14, 15, 19, 36, 37 |
| Độ khó · thời gian | ⭐⭐ · 5–7 giờ |

## 🎯 Mục tiêu học tập
- EDA hướng câu hỏi; so sánh ba họ mô hình bằng CV với ROC-AUC, PR-AUC **và** Brier.
- Kiểm tra calibration; lift của nhóm rủi ro cao.
- Biến xác suất thành **quyết định có lợi nhuận** — có và không có giới hạn năng lực.
- Diễn giải bằng permutation importance; viết báo cáo cho người không làm kỹ thuật.

## 📋 Nhiệm vụ
1. Tỉ lệ rời bỏ theo hợp đồng, thời gian sử dụng, số cuộc gọi hỗ trợ.
2. Pipeline + logistic / random forest / gradient boosting; bảng CV.
3. Test: AUC, lift top 10%, bảng calibration theo thập phân vị.
4. `campaign_profit()`; ngưỡng kinh tế (≈ 11%); top 20% vs luật hiện tại.
5. Permutation importance; báo cáo.

## ✅ Tiêu chí đạt
- ROC-AUC test > 0.79; lift top 10% > 2; calibration lệch < 0.1 ở mọi thập phân vị.
- Ngưỡng kinh tế lãi hơn "gọi tất cả"; với năng lực 20%, mô hình lãi hơn luật hiện tại ≥ 20%.

## 🧭 Gợi ý & cạm bẫy
- Ngưỡng 0.5 không có ý nghĩa kinh doanh — ngưỡng đến từ chi phí và lợi ích.
- Mô hình dự đoán **ai sẽ rời bỏ**, không phải **ai sẽ ở lại nhờ cuộc gọi** (uplift — chương 38).

## 🚀 Mở rộng
- Thiết kế A/B test để đo tỉ lệ thành công thật của cuộc gọi; nhắm theo uplift.
- Thêm chi phí khác nhau theo giá trị khách (khách cước cao đáng giá hơn).
- Model card cho mô hình (chương 39).

## 📁 Tệp
- `project.ipynb` · lời giải: [`solutions/projects/02_classification/project.ipynb`](../../solutions/projects/02_classification/project.ipynb).
