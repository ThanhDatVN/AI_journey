# Transformers — Bạn đã thực sự "biết" attention và GPT?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Attention là gì nếu nhìn như một phép tra cứu? Query, key, value đóng vai trò gì? |
| 2 | Viết công thức scaled dot-product attention. Vì sao chia $\sqrt{d_k}$? |
| 3 | Padding mask và causal mask dùng khi nào? Điều gì xảy ra nếu một hàng bị che toàn bộ? |
| 4 | Vì sao self-attention cần mã hoá vị trí? Khi nào KHÔNG nên thêm? |
| 5 | Multi-head attention khác single-head thế nào? Số tham số phụ thuộc gì? |
| 6 | So sánh mã hoá vị trí sinusoidal, learned, RoPE. |
| 7 | Một khối Transformer gồm những gì? Pre-LN và post-LN khác nhau ra sao? |
| 8 | GPT được huấn luyện thế nào? Vì sao train song song được mọi vị trí? |
| 9 | Làm sao phát hiện mô hình ngôn ngữ "nhìn trộm" tương lai? |
| 10 | Nhiệt độ, top-k, top-p ảnh hưởng thế nào tới văn bản sinh ra? |
| 11 | KV cache là gì? Nó đánh đổi điều gì? |
| 12 | Transformer vượt RNN ở đâu? Khi nào hai bên ngang nhau? |
| 13 | So sánh encoder-only, decoder-only, encoder–decoder: mask, mục tiêu, ứng dụng. |
| 14 | MLM của BERT hoạt động thế nào? Vì sao pretrain → fine-tune hiệu quả khi ít nhãn? |
| 15 | ViT biến ảnh thành token thế nào? Vì sao cần nhiều dữ liệu hơn CNN? |
| 16 | Giới hạn chính của Transformer là gì? Các hướng khắc phục? |
| 17 | Khi nào KHÔNG nên dùng Transformer? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Tra cứu mềm: so query với mọi key → softmax → trung bình có trọng số các value. Query = "đang tìm gì", key = "nhãn để so", value = "nội dung trả về"; cả ba là phép chiếu học được.
2. $\text{softmax}(QK^\top/\sqrt{d_k})V$. Tích vô hướng có phương sai $\propto d_k$ → softmax bão hoà, gradient nhỏ; chia $\sqrt{d_k}$ giữ phương sai ≈ 1.
3. Padding: bỏ vị trí đệm trong batch; causal: mô hình sinh, không nhìn tương lai. Hàng toàn $-\infty$ → softmax NaN; luôn để ít nhất một vị trí hợp lệ.
4. Self-attention hoán vị-đẳng biến — không phân biệt thứ tự. Không nên thêm khi dữ liệu là tập hợp (đám mây điểm, tập vật thể) — tính đối xứng là đúng.
5. $h$ attention song song trên các không gian con $d/h$ chiều, mỗi đầu học một kiểu quan hệ; tham số $4(d^2 + d)$, không phụ thuộc $h$.
6. Sinusoidal: không tham số, dịch chuyển = phép quay; learned: bảng học, giới hạn `max_len`; RoPE: quay Q/K, điểm chỉ phụ thuộc khoảng cách, phổ biến trong LLM. Không cách nào tự động ngoại suy độ dài tốt.
7. MHA + FFN, mỗi phần có residual và LayerNorm. Pre-LN: LN trước lớp con, đường residual sạch, train ổn định; post-LN (gốc): LN sau phép cộng, cần warmup.
8. Dự đoán token tiếp theo, cross-entropy trên mọi vị trí; causal mask đảm bảo vị trí $t$ chỉ thấy $\le t$ nên mọi vị trí tính cùng lúc trong một lượt forward.
9. Đổi token từ vị trí $k$ trở đi, logits ở các vị trí $< k$ phải không đổi. Triệu chứng rò rỉ: loss train giảm về gần 0 bất thường, sinh văn bản vô nghĩa.
10. Nhiệt độ thấp: an toàn, lặp; cao: đa dạng, nhiều lỗi. Top-k: giữ $k$ token; top-p: giữ tập nhỏ nhất có tổng xác suất ≥ $p$ — cắt phần đuôi gây lỗi.
11. Lưu key/value của các token đã xử lý ở mọi lớp; mỗi bước chỉ tính token mới. Đổi tính toán lấy bộ nhớ — cache lớn theo số lớp × độ dài × $d$ (MQA/GQA giảm).
12. Truy cập trực tiếp mọi vị trí (sao chép, tra cứu trong ngữ cảnh), song song hoá khi train, mở rộng quy mô. Ngang nhau khi ngữ cảnh cần ngắn và dữ liệu nhỏ (như bộ review).
13. Encoder-only: hai chiều, MLM — phân loại, embedding. Decoder-only: causal, token tiếp theo — sinh, đa tác vụ. Encoder–decoder: cross-attention — dịch, tóm tắt, ASR.
14. Che 15% token (80% `[MASK]`, 10% ngẫu nhiên, 10% giữ), dự đoán token gốc. Biểu diễn học từ văn bản không nhãn được tái sử dụng → cần ít nhãn hơn. Nhưng luôn so baseline: trên dữ liệu dễ, TF-IDF có thể ngang bằng.
15. Cắt patch $P\times P$, chiếu tuyến tính thành token, thêm `[CLS]` và vị trí. Không có sẵn tính cục bộ/bất biến dịch như CNN → phải học từ dữ liệu → cần nhiều dữ liệu, augmentation hoặc tiền huấn luyện.
16. Chi phí $O(n^2)$ theo độ dài; ngoại suy độ dài kém; cần nhiều dữ liệu/tính toán. Hướng: FlashAttention, attention thưa/cửa sổ, SSM, RoPE scaling, mô hình nhỏ + distillation.
17. Dữ liệu bảng (boosting tốt hơn), dữ liệu rất ít mà không có mô hình tiền huấn luyện phù hợp, bài toán mà baseline đơn giản đã đủ, ràng buộc độ trễ/chi phí chặt.

</details>
