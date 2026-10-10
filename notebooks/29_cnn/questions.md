# CNN — Bạn đã thực sự "biết" mạng tích chập?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Vì sao không dùng MLP trực tiếp cho ảnh lớn? |
| 2 | Phép tích chập trong CNN tính gì? Khác tích chập toán học ở đâu? |
| 3 | Stride, padding ảnh hưởng kích thước đầu ra thế nào? Viết công thức. |
| 4 | Một lớp conv có bao nhiêu tham số? Vì sao không phụ thuộc kích thước ảnh? |
| 5 | Nhiều kênh vào/ra hoạt động thế nào? |
| 6 | Pooling dùng để làm gì? Max khác average? Global average pooling? |
| 7 | Trường tiếp nhận là gì? Tăng nó bằng cách nào? |
| 8 | Đồng biến và bất biến với phép dịch khác nhau thế nào? Thành phần nào tạo ra mỗi tính chất? |
| 9 | Thiên kiến quy nạp của CNN là gì? Lợi ích đo được ra sao? |
| 10 | Data augmentation là gì? Nguyên tắc chọn phép biến đổi? |
| 11 | CNN học những đặc trưng gì ở các lớp khác nhau? |
| 12 | AlexNet, VGG, Inception, ResNet, MobileNet đóng góp ý tưởng gì? |
| 13 | Conv 1×1 và depthwise separable dùng để làm gì? |
| 14 | Degradation problem là gì? Residual giải quyết thế nào? |
| 15 | Backward của conv tính thế nào? Liên hệ transposed convolution? |
| 16 | Khi nào CNN gặp khó khăn? |
| 17 | Quy trình chuẩn khi huấn luyện CNN trên ảnh màu thật (vd CIFAR-10) gồm những gì? |
| 18 | Trên cùng dữ liệu, so sánh tuyến tính → MLP → CNN trên Fashion-MNIST và CIFAR-10. Bài học rút ra? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Số tham số khổng lồ, bỏ qua cấu trúc không gian (pixel kề nhau), phải học lại cùng một mẫu ở mọi vị trí → cần rất nhiều dữ liệu.
2. Tổng có trọng số của một vùng lân cận với kernel, trượt khắp ảnh (cross-correlation); tích chập toán học lật kernel — không quan trọng vì kernel được học.
3. $H_{out} = \lfloor(H + 2p - k)/s\rfloor + 1$; padding giữ kích thước/biên, stride giảm kích thước.
4. $C_{out}(C_{in}k^2 + 1)$ — cùng kernel dùng ở mọi vị trí.
5. Mỗi bộ lọc có độ sâu $C_{in}$, cộng qua mọi kênh vào; $C_{out}$ bộ lọc cho $C_{out}$ kênh ra.
6. Giảm kích thước, tăng trường tiếp nhận, chút bất biến cục bộ. Max giữ phản ứng mạnh nhất, average làm trơn; GAP lấy trung bình toàn bản đồ → một số/kênh, bất biến vị trí, ít tham số.
7. Vùng ảnh gốc ảnh hưởng tới một đơn vị; tăng bằng xếp chồng lớp, pooling/stride, kernel lớn hoặc dilated conv.
8. Đồng biến: đầu vào dịch → đặc trưng dịch theo (conv). Bất biến: đầu ra không đổi (pooling, đặc biệt global pooling).
9. Cục bộ + chia sẻ trọng số + đồng biến dịch. Thí nghiệm: CNN+GAP giữ nguyên độ chính xác ở vị trí chưa từng thấy (MLP về mức đoán mò); CNN với 200 ảnh vượt MLP với 2 000 ảnh.
10. Tạo dữ liệu mới bằng phép biến đổi giữ nhãn (dịch, lật, xoay, màu...) — chỉ chọn bất biến thật của bài toán.
11. Lớp đầu: cạnh, hướng, màu; giữa: kết cấu, bộ phận; sâu: đối tượng/khái niệm.
12. AlexNet: ReLU, GPU, dropout, augmentation; VGG: chồng 3×3 đều đặn; Inception: đa nhánh, 1×1 bottleneck; ResNet: kết nối tắt; MobileNet: depthwise separable cho thiết bị nhỏ.
13. 1×1: trộn kênh, đổi số kênh rẻ. Depthwise separable: conv từng kênh + 1×1, giảm chi phí ~$1/C_{out} + 1/k^2$ lần.
14. Mạng trơn rất sâu có loss train cao hơn mạng nông — khó tối ưu, không phải overfitting. Residual $y = F(x) + x$ cho phép học phần dư và gradient chảy thẳng.
15. dW = tương quan giữa đầu vào và dY; db = tổng dY; dX = tích chập đầy đủ của dY với kernel lật (transposed convolution).
16. Quan hệ xa trong ảnh (trường tiếp nhận hữu hạn), dịch chuyển miền (ảnh khác thiết bị), bất biến không có sẵn (xoay, co giãn), dữ liệu có nhãn ít khi train từ đầu (→ transfer learning).
17. Chuẩn hoá theo kênh (mean/std tính trên train); augmentation hợp lý với miền dữ liệu (lật ngang, dịch vài pixel — không lật dọc ảnh tự nhiên); khối Conv–BatchNorm–ReLU + pooling; lịch learning rate (one-cycle/cosine); theo dõi bằng validation, đánh giá test một lần ở cuối.
18. Fashion-MNIST ~83% → ~88% → ~91%; CIFAR-10 ~40% (tuyến tính) → ~73% (CNN nhỏ, 20 000 ảnh, 8 epoch). Thiên kiến quy nạp phù hợp với ảnh (đặc trưng cục bộ, dùng chung trọng số, pooling, phân cấp) quan trọng hơn số tham số — CNN chỉ nhiều hơn ~2.4 lần số tham số của mô hình tuyến tính.

</details>
