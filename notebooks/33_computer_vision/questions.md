# Computer Vision — Bạn đã thực sự "biết" thị giác máy tính?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Ảnh được biểu diễn thế nào trong máy? HWC và CHW khác nhau ra sao? |
| 2 | Vì sao cần chuẩn hoá ảnh? Chuẩn hoá theo thống kê nào? |
| 3 | Augmentation là gì và vì sao nó hoạt động? |
| 4 | Làm sao chọn phép augmentation phù hợp? Khi nào augmentation có hại? |
| 5 | Shortcut learning là gì? Cho ví dụ và cách phòng tránh. |
| 6 | Transfer learning dựa trên ý tưởng gì? |
| 7 | So sánh train từ đầu, linear probe và fine-tune. Khi nào dùng cái nào? |
| 8 | Những chi tiết gì quan trọng khi fine-tune mô hình tiền huấn luyện? |
| 9 | Phân biệt phân loại, phát hiện, phân đoạn ngữ nghĩa và thực thể. |
| 10 | IoU là gì? Vì sao khắt khe với vật nhỏ? |
| 11 | Detector dạng lưới dự đoán gì ở mỗi ô? |
| 12 | NMS hoạt động thế nào? Vì sao cần? |
| 13 | Tính AP/mAP thế nào? mAP@0.5 khác mAP@[0.5:0.95] ra sao? |
| 14 | So sánh các họ detector: hai giai đoạn, một giai đoạn, DETR. |
| 15 | U-Net có cấu trúc gì? Skip connection để làm gì? |
| 16 | Đánh giá phân đoạn bằng chỉ số nào? Vì sao không dùng pixel accuracy? |
| 17 | Làm sao biết mô hình thị giác sẽ hoạt động tốt khi triển khai? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Tensor số (uint8 0–255 hoặc float); ảnh màu có 3 kênh. HWC (cao, rộng, kênh) — PIL/OpenCV; CHW — PyTorch; đổi bằng `permute`, không phải `reshape`. OpenCV dùng BGR.
2. Đưa đầu vào về thang ổn định cho tối ưu; chuẩn hoá theo mean/std từng kênh của tập train — hoặc của bộ dữ liệu tiền huấn luyện khi dùng mô hình có sẵn.
3. Tạo biến thể giữ nhãn mỗi lần lấy batch — đưa kiến thức về bất biến vào mô hình, như có thêm dữ liệu; regularization mạnh khi ít dữ liệu.
4. Phép biến đổi phản ánh biến thiên có thật ở dữ liệu triển khai mà tập train còn thiếu, và giữ nhãn; kiểm chứng bằng validation. Có hại khi đổi nhãn (6 ↔ 9) hoặc tạo ảnh không thực tế, hay làm bài toán khó hơn mà không thêm thông tin.
5. Mô hình dựa vào đặc điểm tương quan giả trong dữ liệu train (màu, nền, ký hiệu bệnh viện). Phòng: tập test theo điều kiện triển khai, augmentation phá tương quan, kiểm tra vùng mô hình nhìn, thu dữ liệu đa dạng.
6. Đặc trưng học từ bài toán lớn (phân cấp: cạnh → bộ phận → vật thể) tái sử dụng được cho bài toán mới ít nhãn.
7. Từ đầu: chỉ khi rất nhiều dữ liệu và không có mô hình phù hợp. Linear probe: ít nhãn, miền gần. Fine-tune: nhiều nhãn hơn hoặc miền khác hơn; LR phân tầng, có thể train head trước.
8. Khớp tiền xử lý (kích thước, chuẩn hoá, số kênh), LR nhỏ cho backbone, cẩn thận BatchNorm với batch nhỏ, ít epoch, validation để dừng sớm.
9. Phân loại: 1 nhãn/ảnh. Phát hiện: nhiều hộp + nhãn + điểm. Phân đoạn ngữ nghĩa: nhãn mỗi pixel. Thực thể: mask riêng từng vật.
10. Giao/hợp diện tích hai hộp. Vật nhỏ: lệch vài pixel đã là tỉ lệ lớn của kích thước → IoU giảm mạnh.
11. Objectness (có tâm vật trong ô), xác suất lớp, vị trí tâm trong ô và kích thước hộp.
12. Sắp hộp theo điểm, giữ hộp tốt nhất, loại các hộp cùng lớp có IoU với nó vượt ngưỡng, lặp lại. Cần vì nhiều ô/anchor cùng phát hiện một vật.
13. Mỗi lớp: xếp dự đoán theo điểm, TP khi IoU ≥ ngưỡng với hộp thật chưa ghép; AP = diện tích dưới đường PR (precision nội suy); mAP = trung bình các lớp. COCO lấy trung bình trên nhiều ngưỡng IoU 0.5–0.95 → thưởng định vị chính xác.
14. Hai giai đoạn (Faster R-CNN): đề xuất vùng rồi phân loại — chính xác, chậm hơn. Một giai đoạn (YOLO, RetinaNet, FCOS): dự đoán dày đặc một lượt — nhanh. DETR: Transformer, dự đoán tập hợp, không NMS.
15. Nhánh mã hoá (giảm kích thước, tăng ngữ cảnh) + nhánh giải mã (phóng to) + skip connection nối các mức cùng độ phân giải → giữ chi tiết vị trí, biên sắc.
16. mIoU, Dice theo lớp (và chỉ số theo biên); pixel accuracy bị nền chiếm đa số làm sai lệch.
17. Tập test giống điều kiện triển khai (thiết bị, ánh sáng, địa điểm, thời gian), báo cáo độ bền dưới nhiễu loạn, phân tích lỗi theo nhóm, giám sát sau triển khai.

</details>
