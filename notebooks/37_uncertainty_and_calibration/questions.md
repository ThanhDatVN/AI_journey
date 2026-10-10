# Uncertainty & Calibration — Bạn đã thực sự "biết" độ tin cậy của mô hình?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Calibration là gì? Vì sao AUC cao không đảm bảo calibration? |
| 2 | Vẽ reliability diagram và tính ECE thế nào? Giới hạn của ECE? |
| 3 | Proper scoring rule là gì? Phân rã Brier nói gì? |
| 4 | Mô hình nào thường quá tự tin / kém tự tin? Vì sao? |
| 5 | Vì sao quyết định theo chi phí cần xác suất calibrated? |
| 6 | Platt scaling và isotonic regression hoạt động thế nào? Chọn cái nào? |
| 7 | Vì sao phải calibrate trên tập riêng? |
| 8 | Temperature scaling là gì? Giới hạn? |
| 9 | Undersampling ảnh hưởng xác suất thế nào? Điều chỉnh ra sao? |
| 10 | Phân biệt bất định aleatoric và epistemic. |
| 11 | Ước lượng aleatoric và epistemic bằng cách nào? Phân rã phương sai toàn phần? |
| 12 | Vì sao softmax không đủ để phát hiện dữ liệu lạ? |
| 13 | Split conformal hoạt động thế nào? Đảm bảo gì, cần giả định gì? |
| 14 | Độ phủ biên và độ phủ có điều kiện khác nhau thế nào? CQR, Mondrian giúp gì? |
| 15 | Conformal cho phân loại tạo tập dự đoán ra sao? |
| 16 | Điều gì xảy ra khi dữ liệu trôi? Khắc phục? |
| 17 | Khi triển khai, bạn giám sát độ tin cậy của mô hình thế nào? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Dự đoán p ↔ tần suất thật p. AUC chỉ phụ thuộc thứ tự; biến đổi đơn điệu giữ AUC nhưng phá xác suất.
2. Chia bin theo p̂, so dự đoán TB với tỉ lệ thật; ECE = TB có trọng số độ lệch. Phụ thuộc số/cách chia bin; có sàn nhiễu do mẫu hữu hạn.
3. Kỳ vọng cực tiểu khi dự đoán đúng xác suất thật (Brier, log-loss). Brier ≈ reliability (calibration) − resolution (phân biệt) + uncertainty (dữ liệu).
4. Quá tự tin: Naive Bayes (đếm trùng bằng chứng), mạng nơ-ron overfit, boosting rất nhiều vòng. Kém tự tin ở hai đầu: random forest (trung bình nhiều cây).
5. Ngưỡng tối ưu (vd chi phí/lợi ích) tính từ xác suất; xác suất sai → chọn sai đối tượng dù xếp hạng tốt.
6. Platt: logistic 1 biến trên logit điểm số (2 tham số, ổn định khi ít dữ liệu). Isotonic: bậc thang đơn điệu (PAV), linh hoạt, cần nhiều dữ liệu.
7. Điểm số trên train quá khớp (gần 0/1) → ánh xạ học sai, xác suất quá tự tin trên dữ liệu mới.
8. Chia logit cho T chọn theo NLL; một tham số, giữ accuracy; không sửa được sai lệch khác nhau giữa lớp/vùng.
9. Xác suất phản ánh tỉ lệ khi train (50/50) → thổi phồng; điều chỉnh theo tỉ số tỉ lệ nền (Bayes) hoặc calibrate trên dữ liệu thật.
10. Aleatoric: nhiễu vốn có, không giảm khi thêm dữ liệu. Epistemic: thiếu hiểu biết, lớn ở vùng ít dữ liệu/OOD, giảm khi thêm dữ liệu.
11. Aleatoric: mô hình dự đoán phân phối (Gaussian NLL, hồi quy phân vị). Epistemic: ensemble, Bayes, MC dropout. Var toàn phần = TB σ²_k + Var(μ_k).
12. Mạng vẫn cho xác suất cao với dữ liệu lạ; cần tín hiệu khác (bất đồng ensemble, mật độ đặc trưng) và kiểm tra trên đúng loại OOD.
13. Điểm số không phù hợp trên tập calibration → q̂ = điểm số nhỏ thứ ⌈(n+1)(1−α)⌉ → khoảng/tập. Độ phủ biên ≥ 1−α cho mọi mô hình; cần hoán đổi được.
14. Biên: trung bình trên mọi x; có điều kiện: cho từng x/nhóm. CQR làm khoảng co giãn theo x; Mondrian calibrate riêng từng nhóm → đảm bảo theo nhóm.
15. Điểm số 1 − p̂(y|x); tập gồm các lớp có điểm ≤ q̂; tập lớn = mô hình phân vân.
16. Mất đảm bảo/calibration; calibrate lại trên dữ liệu gần đây, conformal có trọng số/thích ứng, giám sát.
17. Theo dõi ECE/reliability, độ phủ thực tế của khoảng, phân phối đầu vào và độ tự tin theo thời gian; cảnh báo khi lệch; calibrate lại định kỳ.

</details>
