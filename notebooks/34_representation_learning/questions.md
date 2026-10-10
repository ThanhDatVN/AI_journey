# Representation Learning — Bạn đã thực sự "biết" học biểu diễn?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | "Biểu diễn" là gì? Vì sao học biểu diễn từ dữ liệu không nhãn có giá trị? |
| 2 | Autoencoder hoạt động thế nào? Quan hệ với PCA? |
| 3 | Biểu diễn của autoencoder chứa gì? Vì sao? |
| 4 | Denoising autoencoder khác AE thường thế nào? Liên hệ với diffusion? |
| 5 | Khi nào phát hiện bất thường bằng sai số tái tạo thất bại? |
| 6 | Vì sao autoencoder thường không sinh được dữ liệu mới? |
| 7 | Viết ELBO của VAE và giải thích từng số hạng. |
| 8 | Reparameterization trick là gì, vì sao cần? |
| 9 | β trong β-VAE ảnh hưởng thế nào? Posterior collapse là gì? |
| 10 | Đánh giá mô hình sinh thế nào? Cạm bẫy? |
| 11 | So sánh VAE, GAN, diffusion, autoregressive. |
| 12 | Học tương phản hoạt động thế nào? Viết loss InfoNCE/NT-Xent. |
| 13 | Augmentation đóng vai trò gì trong học tương phản? |
| 14 | Linear probe và kNN probe là gì? Vì sao dùng để đánh giá biểu diễn? |
| 15 | Sụp đổ biểu diễn là gì? Các phương pháp tránh nó ra sao? |
| 16 | MAE và CLIP khác SimCLR thế nào? |
| 17 | Khi nào nên tự train self-supervised, khi nào dùng mô hình nền tảng? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Cách mã hoá dữ liệu thành vector sao cho thông tin hữu ích dễ truy xuất (vd bằng mô hình tuyến tính). Dữ liệu không nhãn rẻ và nhiều; biểu diễn tốt giảm mạnh số nhãn cần cho tác vụ đích.
2. Encoder nén vào latent ít chiều, decoder tái tạo, tối thiểu sai số tái tạo. AE tuyến tính + MSE ≈ không gian con PCA; AE phi tuyến nén tốt hơn.
3. Những gì giúp giảm sai số tái tạo nhiều nhất — với ảnh: vị trí, kích thước, độ sáng — không nhất thiết là thông tin tác vụ cần (loại hình).
4. Đầu vào nhiễu, đích sạch → học cấu trúc dữ liệu thay vì sao chép. Diffusion = khử nhiễu nhiều mức, sinh bằng khử nhiễu dần từ nhiễu thuần.
5. Khi AE tổng quát hoá đủ tốt để tái tạo cả bất thường, hoặc bất thường khác biệt ở những gì AE không mã hoá; dùng kNN trên embedding tốt, so sánh nhiều phương pháp.
6. Latent của dữ liệu không bị ràng buộc phân phối — điểm ngẫu nhiên rơi vào vùng decoder chưa thấy.
7. E_q[log p(x|z)] − KL(q(z|x) ‖ p(z)): tái tạo tốt + latent gần prior, liền mạch → lấy mẫu được; là cận dưới của log p(x).
8. z = μ + σ·ε với ε ~ N(0, I): đưa tính ngẫu nhiên ra ngoài để gradient qua μ, σ.
9. β lớn: KL nhỏ, latent ít thông tin, tái tạo/mẫu mờ, nhiều chiều "chết"; β nhỏ: gần AE. Posterior collapse: q(z|x) = prior, latent vô dụng — vd do loss tái tạo bị trung bình theo pixel.
10. FID, precision/recall cho mô hình sinh, kiểm tra ghi nhớ (láng giềng gần nhất), đánh giá người. Cạm bẫy: bộ phân loại tự tin với cả nhiễu; một chỉ số không đo cả chất lượng lẫn đa dạng.
11. VAE: ổn định, mờ. GAN: sắc, khó train, mode collapse. Diffusion: chất lượng + đa dạng tốt nhất, sinh chậm. Autoregressive: likelihood chính xác, sinh tuần tự (GPT).
12. Hai góc nhìn tăng cường cùng ảnh là cặp dương; cross-entropy chọn đúng cặp giữa 2N − 1 ứng viên với cosine/τ.
13. Định nghĩa thế nào là "cùng một thứ" → biểu diễn bất biến với những gì augmentation thay đổi; chọn sai thì bất biến sai (xoá thông tin cần).
14. Đóng băng encoder, train bộ phân loại tuyến tính / dùng kNN trên đặc trưng với ít nhãn — đo thông tin dễ truy xuất trong biểu diễn.
15. Mọi đầu vào cùng một vector; tránh bằng mẫu âm, bất đối xứng + dừng gradient/giáo viên EMA, ràng buộc phương sai.
16. MAE: che phần lớn patch, tái tạo phần bị che (phải suy luận). CLIP: tương phản ảnh–văn bản, zero-shot. SimCLR: tương phản ảnh–ảnh qua augmentation.
17. Tự train khi có nhiều dữ liệu miền riêng không nhãn mà mô hình nền tảng chưa thấy (y tế, công nghiệp, vệ tinh); còn lại dùng embedding của mô hình nền tảng + linear probe/fine-tune.

</details>
