# Tối ưu siêu tham số — Bạn đã thực sự hiểu?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Tham số và siêu tham số khác nhau thế nào? Ví dụ cho ba mô hình. |
| 2 | Grid search và random search: vì sao random thường hiệu quả hơn với cùng ngân sách? |
| 3 | Vì sao learning rate, C, alpha nên tìm trên thang log? |
| 4 | Successive halving / Hyperband tiết kiệm tài nguyên thế nào? Rủi ro gì? |
| 5 | Bayesian optimization (TPE, GP) chọn cấu hình tiếp theo ra sao? |
| 6 | Khi nào Bayesian optimization đáng dùng hơn random search? |
| 7 | `best_score_` của GridSearchCV có phải ước lượng không chệch của hiệu năng? Vì sao? |
| 8 | Early stopping liên quan gì tới tìm siêu tham số? |
| 9 | Nên tinh chỉnh siêu tham số nào trước cho gradient boosting? |
| 10 | Làm sao tránh "overfit validation" khi thử hàng trăm cấu hình? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Tham số học từ dữ liệu (trọng số hồi quy, ngưỡng chia cây); siêu tham số chọn trước khi học (C của logistic, max_depth của cây, learning rate và số cây của boosting).
2. Thường chỉ vài siêu tham số thật sự quan trọng; random thử nhiều giá trị khác nhau của chúng, grid lãng phí lần thử vào chiều không quan trọng.
3. Hiệu ứng của chúng mang tính tỉ lệ (0.001 vs 0.01 khác biệt như 0.1 vs 1) — thang tuyến tính bỏ sót vùng nhỏ.
4. Cấp ít tài nguyên (ít dữ liệu/epoch/cây) cho nhiều cấu hình, loại dần cấu hình kém, dồn tài nguyên cho cấu hình tốt. Rủi ro: cấu hình "khởi động chậm" bị loại sớm.
5. Xây mô hình thay thế của hàm mục tiêu từ các lần thử trước, chọn điểm cân bằng khám phá (vùng chưa chắc chắn) và khai thác (vùng có vẻ tốt) qua hàm thu nhận.
6. Khi mỗi lần thử đắt (huấn luyện lâu), số chiều vừa phải, cần ít lần thử; với lần thử rẻ, random search song song thường đủ.
7. Không — đó là điểm của cấu hình **tốt nhất** trong nhiều cấu hình trên cùng các fold → lạc quan. Dùng nested CV hoặc tập test riêng.
8. Số epoch/số cây là siêu tham số; early stopping chọn tự động bằng validation — kết hợp với tìm learning rate, độ sâu.
9. Learning rate + số cây (early stopping) → độ phức tạp cây (max_depth/num_leaves, min_samples_leaf) → điều chuẩn (L2, subsampling).
10. Giữ tập test/holdout không dùng để chọn; nested CV; giới hạn số cấu hình; ưu tiên cấu hình đơn giản khi điểm ngang nhau; báo cáo số cấu hình đã thử.

</details>
