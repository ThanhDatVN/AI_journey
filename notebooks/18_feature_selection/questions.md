# Chọn đặc trưng — Bạn đã thực sự hiểu?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Vì sao chọn đặc trưng? Lợi ích ngoài độ chính xác? |
| 2 | Phương pháp filter, wrapper, embedded khác nhau thế nào? Ví dụ mỗi loại. |
| 3 | Tương quan, mutual information, ANOVA F-test đo gì? Hạn chế của filter đơn biến? |
| 4 | Recursive feature elimination hoạt động thế nào? Chi phí? |
| 5 | Lasso chọn đặc trưng ra sao? Vì sao không ổn định với đặc trưng tương quan? |
| 6 | Importance của cây có thiên lệch gì? |
| 7 | Vì sao chọn đặc trưng trên toàn bộ dữ liệu trước CV là rò rỉ? Minh hoạ với dữ liệu nhiễu. |
| 8 | Stability selection là gì? |
| 9 | Khi có hai đặc trưng gần trùng nhau, nên làm gì? |
| 10 | Khi nào **không** nên bỏ đặc trưng dù importance thấp? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Giảm overfit (nhất là n nhỏ, d lớn), nhanh hơn, rẻ hơn khi thu thập dữ liệu, dễ giải thích, bớt rủi ro trôi dữ liệu.
2. Filter: thống kê đơn biến độc lập mô hình (tương quan, MI). Wrapper: tìm tập con bằng cách huấn luyện mô hình (RFE, forward selection). Embedded: chọn trong lúc huấn luyện (Lasso, importance của cây).
3. Tương quan: quan hệ tuyến tính; MI: phụ thuộc bất kỳ; F-test: khác biệt trung bình giữa lớp. Bỏ qua tương tác và dư thừa giữa đặc trưng.
4. Huấn luyện, bỏ đặc trưng kém nhất, lặp lại. Tốn nhiều lần huấn luyện; phải đặt trong CV.
5. Phạt L1 đẩy hệ số kém quan trọng về 0. Với nhóm tương quan, Lasso chọn tuỳ ý một biến, thay đổi theo mẫu dữ liệu.
6. Impurity importance ưu tiên đặc trưng liên tục/nhiều giá trị, tính trên train; dùng permutation importance trên validation.
7. Bước chọn đã "nhìn" nhãn của fold validation; với hàng nghìn đặc trưng nhiễu, một số tương quan giả với nhãn → AUC CV cao giả tạo dù không có tín hiệu.
8. Chọn đặc trưng trên nhiều mẫu con (bootstrap), giữ đặc trưng được chọn thường xuyên — ổn định hơn một lần chọn.
9. Giữ một (ý nghĩa nghiệp vụ, dễ thu thập), gộp (trung bình, PCA), hoặc dùng điều chuẩn L2/elastic net.
10. Khi cần cho công bằng/giám sát, cho giải thích theo yêu cầu, khi importance thấp vì bị đặc trưng tương quan "chia" mất, hoặc đặc trưng quan trọng cho một nhóm nhỏ nhưng quan trọng.

</details>
