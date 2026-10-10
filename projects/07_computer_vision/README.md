# Project 07 — Nhận dạng hình khối và "học đường tắt" (Computer Vision)

> Camera trên dây chuyền phải nhận dạng **hình dạng** chi tiết. Ở nhà máy cũ, mỗi loại chi tiết gần như luôn được sơn một màu riêng; nhà máy mới sơn màu ngẫu nhiên. Mô hình phải nhìn **hình**, không phải **màu**.

| | |
|---|---|
| Bài toán | Phân loại ảnh 5 lớp, dịch chuyển miền |
| Dữ liệu | `make_shapes(..., rgb=True, color_bias=...)` — train 6.000 ảnh 28×28 (95% màu theo lớp), test nhà máy cũ / nhà máy mới (màu ngẫu nhiên) |
| Chương nền tảng | 28, 29, 33, 05, 36 — cần `pip install -e ".[dl]"` |
| Độ khó · thời gian | ⭐⭐⭐⭐ · 6–8 giờ (GPU giúp nhanh hơn, CPU vẫn chạy được) |

## 🎯 Mục tiêu học tập
- Chứng minh một **đường tắt** tồn tại bằng baseline đơn giản; thấy CNN học nó.
- Đánh giá trên **dữ liệu khác phân phối** — validation cùng phân phối không phát hiện được.
- Hai cách sửa: loại thông tin gây nhiễu khỏi đầu vào; tăng cường dữ liệu phá tương quan giả.

## 📋 Nhiệm vụ
1. Xem dữ liệu. 2. Baseline "chỉ nhìn màu". 3. CNN ảnh màu. 4. Chẩn đoán lỗi. 5. Ảnh xám chuẩn hoá; xáo trộn màu. 6. Ma trận nhầm lẫn, báo cáo.

## ✅ Tiêu chí đạt
- Baseline màu > 85% nhà máy cũ, < 35% nhà máy mới.
- CNN ảnh màu > 95% nhà máy cũ và tụt > 15 điểm ở nhà máy mới; lỗi trùng dự đoán theo màu nhiều hơn mức ngẫu nhiên.
- Ít nhất một cách sửa > 88% ở nhà máy mới, chênh < 6 điểm.

## 🧭 Gợi ý & cạm bẫy
- Chuyển xám thôi chưa đủ: độ sáng của từng màu vẫn mang thông tin lớp → chuẩn hoá từng ảnh.
- Đường tắt thật ngoài đời: nền ảnh, logo bệnh viện trên phim X-quang, watermark, góc chụp.

## 🚀 Mở rộng
- Grad-CAM (chương 36) để thấy mô hình nhìn vào đâu trước và sau khi sửa.
- 9 lớp (`SHAPE_CLASSES + SHAPE_CLASSES_EXTRA`) — khó hơn nhiều; transfer learning (chương 33).
- Group DRO / reweighting khi biết thuộc tính nhiễu.

## 📁 Tệp
- `project.ipynb` · lời giải: [`solutions/projects/07_computer_vision/project.ipynb`](../../solutions/projects/07_computer_vision/project.ipynb).
