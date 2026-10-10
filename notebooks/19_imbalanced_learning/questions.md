# Dữ liệu mất cân bằng — Bạn đã thực sự hiểu?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Vì sao accuracy đánh lừa khi lớp mất cân bằng? Metric nào thay thế? |
| 2 | Undersampling, oversampling, SMOTE hoạt động thế nào? Rủi ro của mỗi cách? |
| 3 | Vì sao SMOTE phải nằm trong pipeline, chỉ áp dụng trên phần train của mỗi fold? |
| 4 | Class weight làm gì với hàm mất mát? Ảnh hưởng tới xác suất dự đoán? |
| 5 | Nhiều khi chỉ cần đổi ngưỡng là đủ — vì sao? |
| 6 | Ngưỡng tối ưu theo chi phí tính thế nào? Cần điều kiện gì? |
| 7 | Khi nào mất cân bằng **không** phải là vấn đề? |
| 8 | Đánh giá mô hình phát hiện gian lận 0.1% như thế nào cho đúng? |
| 9 | Precision@k và recall ở ngưỡng cố định khác ROC-AUC thế nào về ý nghĩa kinh doanh? |
| 10 | Resampling làm sai lệch xác suất — sửa lại thế nào nếu cần xác suất thật? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Đoán lớp đa số đã đạt accuracy rất cao. Dùng PR-AUC, recall/precision ở ngưỡng hành động, F1/Fβ, chi phí kỳ vọng.
2. Undersampling bỏ bớt lớp đa số (mất thông tin); oversampling lặp lớp hiếm (overfit); SMOTE nội suy điểm mới giữa các láng giềng lớp hiếm (tạo điểm vô lý ở vùng chồng lấn).
3. Nếu tạo mẫu trên toàn bộ dữ liệu, điểm tổng hợp nội suy từ điểm validation/test → rò rỉ, điểm lạc quan.
4. Nhân trọng số loss của lớp hiếm → mô hình chú ý hơn; xác suất bị đẩy lên (tương đương đổi tỉ lệ nền) → không còn hiệu chỉnh.
5. Mô hình xếp hạng tốt nhưng ngưỡng 0.5 không phù hợp tỉ lệ nền/chi phí; dời ngưỡng thay đổi đánh đổi precision–recall mà không cần huấn luyện lại.
6. p* = C_FP/(C_FP + C_FN) với xác suất **đã hiệu chỉnh**; hoặc quét ngưỡng trên validation để tối thiểu chi phí thực tế.
7. Khi lớp tách tốt và đủ mẫu lớp hiếm, khi chỉ cần xếp hạng, hoặc mô hình xác suất được hiệu chỉnh và quyết định dựa trên ngưỡng chi phí.
8. PR-AUC, recall tại mức báo động giả chấp nhận được (theo năng lực kiểm tra), precision@k, chi phí theo số tiền; CV phân tầng, nhiều seed vì ít ca dương.
9. Precision@k/recall@ngưỡng gắn trực tiếp với số ca đội vận hành xử lý được; ROC-AUC tổng hợp mọi ngưỡng, kể cả ngưỡng không ai dùng.
10. Hiệu chỉnh lại trên dữ liệu phân phối gốc (Platt/isotonic), hoặc điều chỉnh odds theo tỉ lệ lấy mẫu: odds_thật = odds_mô_hình × (tỉ lệ lấy mẫu lớp âm / lớp dương).

</details>
