# Huấn luyện mạng nơ-ron — Bạn đã thực sự "biết"?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Vì sao cần hàm kích hoạt phi tuyến? |
| 2 | So sánh sigmoid, tanh, ReLU, Leaky ReLU, ELU, GELU: miền giá trị, đạo hàm, ưu nhược. |
| 3 | Vanishing/exploding gradient là gì? Nguyên nhân và cách chữa? |
| 4 | Dying ReLU là gì? Phòng tránh thế nào? |
| 5 | Vì sao không khởi tạo trọng số bằng 0? Xavier và He khác nhau thế nào? |
| 6 | Vì sao phải chuẩn hoá đầu vào? |
| 7 | Batch Norm hoạt động thế nào khi train và eval? Lợi ích và cạm bẫy? |
| 8 | Khi nào dùng LayerNorm? |
| 9 | Momentum, RMSProp, Adam cải tiến SGD thế nào? |
| 10 | Vì sao Adam cần hiệu chỉnh bias? AdamW khác Adam + L2 ra sao? |
| 11 | Chọn learning rate thế nào? Vì sao cần lịch học và warmup? |
| 12 | Gradient clipping dùng khi nào? |
| 13 | Batch size ảnh hưởng gì? Quy tắc tỉ lệ tuyến tính là gì? |
| 14 | Kể các kỹ thuật regularization cho mạng nơ-ron và cơ chế của chúng. |
| 15 | Dropout làm gì khi train và eval? Vì sao nhân 1/(1 − p)? |
| 16 | Quy trình huấn luyện có hệ thống gồm những bước nào? |
| 17 | Gặp loss không giảm / NaN / train tốt val tệ — kiểm tra gì? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Không có phi tuyến, mọi lớp tuyến tính gộp thành một → không biểu diễn được ranh giới phi tuyến.
2. Sigmoid (0,1), đạo hàm ≤ 0.25, bão hoà, không tâm 0 → lớp ra nhị phân. Tanh (−1,1), tâm 0, vẫn bão hoà. ReLU [0,∞), đạo hàm 0/1, nhanh, có thể chết. Leaky ReLU/ELU sửa phía âm. GELU/SiLU trơn, dùng trong Transformer.
3. Gradient tích của nhiều đạo hàm cục bộ → nhỏ dần (sigmoid, khởi tạo nhỏ) hoặc lớn dần (khởi tạo lớn, RNN). Chữa: ReLU/GELU, khởi tạo đúng, BN/LN, residual, gradient clipping.
4. Tiền kích hoạt âm với mọi đầu vào → gradient luôn 0. Tránh: LR hợp lý, He init, BN, Leaky ReLU/ELU/GELU.
5. Đối xứng: các nơ-ron giống hệt nhau mãi. Xavier Var = 1/n (tanh/sigmoid); He Var = 2/n (ReLU bỏ nửa phương sai).
6. Feature khác thang đo → mặt loss kéo dài, một LR không hợp mọi hướng; chuẩn hoá (fit trên train).
7. Train: chuẩn hoá bằng thống kê batch + cập nhật running stats; eval: dùng running stats; γ, β học được. Lợi: học nhanh, chịu LR lớn, ít nhạy khởi tạo. Bẫy: batch nhỏ, quên eval(), dữ liệu train/test lệch.
8. Batch nhỏ/thay đổi, chuỗi (RNN), Transformer; không phụ thuộc batch, train = eval.
9. Momentum: quán tính, triệt tiêu dao động; RMSProp: chia cho trung bình trượt của g² (bước thích nghi từng tọa độ); Adam: cả hai + hiệu chỉnh bias.
10. m, s khởi tạo 0 → lệch về 0 ở bước đầu. Với Adam, L2 trong gradient bị chia cho √ŝ (phạt không đều); AdamW trừ λθ trực tiếp.
11. LR range test; lấy ~1/10 giá trị loss bắt đầu bùng nổ. Lịch giảm dần để hội tụ (nhiễu SGD); warmup tránh bước lớn khi thống kê Adam/BN chưa ổn định.
12. Gradient thỉnh thoảng bùng nổ (RNN, mô hình lớn, LR cao) — chặn chuẩn gradient tối đa.
13. Batch lớn: ít bước/epoch, gradient ít nhiễu, tận dụng GPU; cùng LR thì underfit → tăng LR tỉ lệ với batch (kèm warmup), đến một giới hạn.
14. Weight decay (trọng số nhỏ → hàm trơn), dropout (ensemble mạng con), nhiễu/augmentation (bất biến), early stopping (giới hạn số bước), BN (nhiễu nhẹ), thêm dữ liệu.
15. Train: tắt ngẫu nhiên với xác suất p, nhân phần còn lại 1/(1 − p) để giữ kỳ vọng; eval: không làm gì.
16. Hiểu dữ liệu → baseline + đánh giá đáng tin → overfit một batch → overfit train → regularize → tune (LR trước) → ensemble/train lâu hơn.
17. Loss không giảm: LR, zero_grad, nhãn, loss sai; NaN: LR quá lớn, log(0), dữ liệu NaN; train tốt val tệ: overfitting hoặc lệch phân phối; val tốt bất thường: rò rỉ; kết quả dao động: quên eval().

</details>
