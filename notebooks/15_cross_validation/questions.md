# Cross-validation — Bạn đã thực sự hiểu?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Holdout và k-fold khác nhau thế nào? Ưu, nhược điểm của mỗi cách? |
| 2 | Chọn k thế nào? k lớn/nhỏ ảnh hưởng gì tới bias và phương sai của ước lượng? |
| 3 | Stratified k-fold dùng khi nào? |
| 4 | Khi nào phải dùng GroupKFold? Điều gì xảy ra nếu không dùng? |
| 5 | Vì sao không được xáo trộn dữ liệu chuỗi thời gian khi CV? TimeSeriesSplit hoạt động thế nào? |
| 6 | Khoảng cách (gap) giữa train và validation trong chia thời gian dùng để làm gì? |
| 7 | Leave-one-out CV có ưu, nhược điểm gì? |
| 8 | Nested CV giải quyết vấn đề gì? Vòng trong, vòng ngoài làm gì? |
| 9 | Vì sao mọi bước tiền xử lý phải nằm trong pipeline khi CV? |
| 10 | Mô tả một quy trình đánh giá đúng từ đầu đến cuối cho một dự án. |
| 11 | Độ lệch chuẩn của điểm CV giữa các fold cho biết gì? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Holdout: một lần chia, nhanh nhưng phụ thuộc may rủi của lần chia. K-fold: mỗi mẫu được đánh giá một lần, ước lượng ổn định hơn, tốn k lần huấn luyện.
2. Thường 5 hoặc 10. k lớn: mỗi fold train gần đủ dữ liệu (bias thấp) nhưng tốn và các ước lượng tương quan; k nhỏ: rẻ, bias bi quan hơn.
3. Phân loại, đặc biệt lớp mất cân bằng — giữ tỉ lệ lớp trong mỗi fold.
4. Khi nhiều mẫu thuộc cùng một thực thể (bệnh nhân, khách hàng, máy). Không dùng → cùng thực thể ở train và validation → mô hình "nhận ra" thực thể → điểm ảo.
5. Xáo trộn cho mô hình nhìn thấy tương lai. TimeSeriesSplit: mở rộng dần cửa sổ train, validation luôn ở sau.
6. Tránh rò rỉ khi nhãn/đặc trưng trễ (nhãn 30 ngày, đặc trưng cửa sổ trượt) — khoảng trống bằng độ trễ.
7. Dùng tối đa dữ liệu cho train, gần không chệch; nhưng phương sai cao, rất tốn, không dùng được với dữ liệu có nhóm/thời gian.
8. Ước lượng hiệu năng của **cả quy trình** có tinh chỉnh siêu tham số không bị lạc quan. Vòng trong chọn siêu tham số; vòng ngoài đánh giá mô hình đã chọn trên dữ liệu chưa dùng để chọn.
9. Nếu fit tiền xử lý (chuẩn hoá, điền thiếu, chọn đặc trưng, SMOTE) trên toàn bộ dữ liệu, fold validation đã "rò" vào huấn luyện.
10. Chia test (theo thời gian/nhóm nếu cần) trước mọi phân tích sâu → EDA trên train → pipeline → CV (đúng sơ đồ) để chọn mô hình/siêu tham số → huấn luyện lại trên toàn train → đánh giá test một lần → báo cáo kèm độ bất định.
11. Mức ổn định của mô hình và của ước lượng; chênh lệch giữa hai mô hình nhỏ hơn độ lệch này thì chưa kết luận được (so sánh theo cặp trên cùng fold).

</details>
