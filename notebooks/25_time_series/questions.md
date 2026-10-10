# Chuỗi thời gian — Bạn đã thực sự "biết" dự báo?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Chuỗi thời gian khác dữ liệu bảng thông thường ở điểm nào quan trọng nhất? |
| 2 | Các thành phần của chuỗi? Cộng tính hay nhân tính — chọn thế nào? |
| 3 | Đọc biểu đồ ACF thế nào? PACF khác ACF ra sao? |
| 4 | Tính dừng là gì? Vì sao quan trọng? Làm chuỗi dừng bằng cách nào? |
| 5 | Tương quan giả là gì? Tránh thế nào? |
| 6 | Vì sao luôn cần baseline? Baseline nào? |
| 7 | MAE, RMSE, MAPE, MASE — dùng khi nào? Nhược điểm của MAPE? |
| 8 | SES, Holt, Holt–Winters khác nhau thế nào? Tham số học ra sao? |
| 9 | AR(p) và MA(q) là gì? Nhận diện bằng ACF/PACF? |
| 10 | ARIMA/SARIMA/ARIMAX gồm những gì? Khi nào ARIMAX có ích? |
| 11 | Ljung–Box kiểm tra gì? |
| 12 | Biến dự báo thành học có giám sát thế nào? Feature nào có sẵn khi dự báo h bước? |
| 13 | Kể các dạng rò rỉ trong chuỗi thời gian. |
| 14 | Vì sao KFold xáo trộn sai với chuỗi thời gian? Backtest làm thế nào? |
| 15 | Vì sao cây/boosting không ngoại suy xu hướng? Cách khắc phục? |
| 16 | Direct và recursive multi-step khác nhau thế nào? |
| 17 | Tạo khoảng dự báo thế nào? Kiểm tra ra sao? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Thứ tự thời gian: các mẫu phụ thuộc nhau, tương lai không được dùng để dự đoán quá khứ → chia/đánh giá theo thời gian, feature chỉ dùng quá khứ.
2. Xu hướng, mùa vụ (có thể nhiều chu kỳ), sự kiện, nhiễu. Nhân tính khi biên độ mùa vụ tỉ lệ với mức → log rồi cộng tính.
3. ACF: tương quan với chính nó ở độ trễ k — giảm chậm (xu hướng), đỉnh tuần hoàn (mùa vụ). PACF: tương quan ở trễ k sau khi loại ảnh hưởng các trễ nhỏ hơn — dùng nhận diện bậc AR.
4. Trung bình, phương sai, hiệp phương sai (theo độ trễ) không đổi. Nhiều model (AR, ARMA) giả định nó; chuỗi không dừng gây tương quan giả, dự báo sai. Sai phân, sai phân mùa vụ, log, khử xu hướng.
5. Hai chuỗi không dừng độc lập vẫn tương quan mạnh do cùng xu hướng/trôi. Phân tích trên sai phân/phần dư, kiểm tra đồng liên kết nếu cần.
6. Để biết mô hình phức tạp có thật sự đáng giá: naive, seasonal naive, drift, trung bình; MASE so trực tiếp với seasonal naive.
7. MAE/RMSE: theo đơn vị (RMSE phạt lỗi lớn). MAPE: theo %, bùng nổ khi y gần 0, bất đối xứng. MASE: không phụ thuộc đơn vị, so với seasonal naive, dùng được với số 0.
8. SES: chỉ mức; Holt: + xu hướng; Holt–Winters: + mùa vụ. Tham số (α, β, γ) tối thiểu tổng bình phương sai số một bước (hoặc MLE trong dạng không gian trạng thái ETS).
9. AR: hồi quy trên giá trị quá khứ (PACF cắt sau p); MA: tổ hợp các cú sốc quá khứ (ACF cắt sau q).
10. Sai phân (d, D) + ARMA (p, q) + thành phần mùa vụ (P, Q)_m + biến ngoại sinh. ARIMAX có ích khi có biến giải thích biết trước (lịch khuyến mãi, ngày lễ, giá đã lên kế hoạch).
11. Phần dư còn tự tương quan không (H₀: nhiễu trắng) — không đo độ lớn sai số.
12. Mỗi thời điểm là một dòng; target = giá trị cần dự báo; feature: lịch, Fourier, sự kiện biết trước, trễ ≥ h, thống kê trượt shift(h).
13. Rolling không shift, centered window, scaler/encoder fit trên toàn chuỗi, dùng biến không biết trước (thời tiết thực tế), KFold xáo trộn, chọn mô hình trên test.
14. Test xen giữa train → nội suy, dùng thông tin tương lai. Backtest: lặp "train đến T_k → dự báo h bước" với T_k tiến dần (expanding/sliding window), có thể thêm gap.
15. Cây trả về hằng số ngoài vùng giá trị train. Khử xu hướng (mô hình tuyến tính cho xu hướng, cây cho phần dư), dùng target sai phân/tỉ lệ, hoặc mô hình tuyến tính.
16. Direct: mỗi tầm h một mô hình (hoặc feature trễ ≥ h) — không tích luỹ sai số nhưng nhiều mô hình. Recursive: một mô hình một bước, dùng dự báo làm đầu vào — sai số tích luỹ.
17. Công thức của mô hình thống kê, mô phỏng/bootstrap phần dư, phân vị phần dư backtest (gần conformal), quantile regression. Kiểm tra độ phủ thực tế và độ rộng trên dữ liệu giữ lại.

</details>
