# Mô hình chuỗi — Bạn đã thực sự "biết" RNN, LSTM, attention?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Dữ liệu chuỗi có gì đặc biệt? Kể các dạng bài toán chuỗi. |
| 2 | RNN tính gì ở mỗi bước? Vì sao chia sẻ trọng số theo thời gian? |
| 3 | BPTT hoạt động thế nào? Truncated BPTT là gì? |
| 4 | Mô hình ngôn ngữ ký tự học gì? Nhiệt độ lấy mẫu ảnh hưởng ra sao? |
| 5 | Vì sao RNN bị vanishing/exploding gradient? Bán kính phổ đóng vai trò gì? |
| 6 | Gradient clipping giải quyết gì và không giải quyết gì? |
| 7 | Viết các phương trình LSTM. Mỗi cổng làm gì? |
| 8 | Vì sao LSTM giữ gradient tốt hơn? Mẹo khởi tạo nào giúp? |
| 9 | GRU khác LSTM thế nào? |
| 10 | Khi nào dùng RNN hai chiều? Khi nào không được dùng? |
| 11 | Xử lý chuỗi độ dài khác nhau trong một batch thế nào? |
| 12 | Encoder–decoder hoạt động thế nào? Teacher forcing và exposure bias là gì? |
| 13 | Nút cổ chai của seq2seq là gì? Attention giải quyết ra sao? |
| 14 | Viết công thức dot-product attention. Ma trận attention cho biết gì? |
| 15 | Greedy và beam search khác nhau thế nào? Khi nào dùng lấy mẫu? |
| 16 | Giới hạn của RNN dẫn tới Transformer như thế nào? |
| 17 | Khi nào mô hình chuỗi KHÔNG cần thiết? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Thứ tự quan trọng, độ dài thay đổi, phụ thuộc có thể xa. many-to-one (phân loại câu), one-to-many (sinh mô tả), many-to-many đồng bộ (gán nhãn từ), lệch (dịch máy).
2. $\mathbf h_t = \tanh(W_{xh}\mathbf x_t + W_{hh}\mathbf h_{t-1} + \mathbf b)$; cùng quy tắc xử lý ở mọi vị trí, số tham số không phụ thuộc độ dài, tổng quát hoá theo vị trí thời gian.
3. Trải mạng theo thời gian, forward lưu trạng thái, backward từ cuối về đầu, cộng gradient trọng số dùng chung. Truncated: cắt gradient sau k bước (trạng thái vẫn truyền tiếp).
4. Phân phối ký tự tiếp theo cho trước ngữ cảnh: chính tả, cụm từ, cấu trúc câu. Nhiệt độ thấp → an toàn, lặp; cao → đa dạng, nhiều lỗi.
5. Gradient qua thời gian là tích các Jacobian $\text{diag}(1 - h^2)W_{hh}$: $\rho < 1$ hoặc tanh bão hoà → tắt dần; $\rho$ lớn → bùng nổ.
6. Chặn độ dài gradient (giữ hướng) → tránh bước khổng lồ/NaN. Không chữa vanishing, không thay thế khởi tạo/LR hợp lý.
7. $f, i, o = \sigma(\cdot)$, $g = \tanh(\cdot)$, $c_t = f\odot c_{t-1} + i\odot g$, $h_t = o\odot\tanh c_t$; quên/ghi/xuất.
8. Đường cộng của $c_t$: $\partial c_t/\partial c_{t-1} = \text{diag}(f)$ không nhân $W$ hay tanh'; bias cổng quên = 1.
9. GRU gộp trạng thái, hai cổng (cập nhật $z$, reset $r$), $h_t = (1 - z)n + z\,h_{t-1}$; ít tham số, thường tương đương.
10. Khi có toàn bộ chuỗi lúc dự đoán (phân loại câu, gán nhãn). Không dùng cho dự báo/sinh tuần tự, chấm điểm thời gian thực.
11. Padding + mask hoặc `pack_padded_sequence`; lấy trạng thái tại vị trí thật cuối cùng; bỏ qua PAD trong loss/attention.
12. Encoder nén nguồn, decoder sinh đích từng token. Teacher forcing: train với token đúng làm đầu vào; exposure bias: lúc suy luận dùng dự đoán của chính mình → lỗi tích luỹ.
13. Một vector ngữ cảnh cố định phải chứa cả chuỗi → chuỗi dài mất thông tin. Attention: mỗi bước decoder truy cập trung bình có trọng số mọi trạng thái encoder.
14. $\alpha = \text{softmax}(\mathbf s^\top H)$, $\mathbf c = \sum\alpha_j\mathbf h_j$ (Transformer chia $\sqrt d$). Ma trận attention thể hiện căn chỉnh nguồn–đích.
15. Greedy: argmax từng bước; beam: giữ k ứng viên, tối ưu tổng log-prob (dịch, ASR). Sinh tự do: lấy mẫu với nhiệt độ/top-p để đa dạng.
16. Tuần tự (không song song), đường gradient dài, nút cổ chai → Transformer bỏ hồi quy, self-attention giữa mọi cặp vị trí, train song song.
17. Khi thứ tự không mang tín hiệu (bag-of-words đủ, như bộ review của khoá), khi đặc trưng tổng hợp + boosting đủ tốt, hoặc khi chỉ cần ngữ cảnh rất ngắn — luôn so với baseline.

</details>
