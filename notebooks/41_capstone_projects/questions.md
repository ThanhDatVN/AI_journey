# Capstone — Bạn đã sẵn sàng làm một dự án ML thật?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại trước khi bắt đầu và sau khi hoàn thành capstone.

| # | Câu hỏi |
|---|---|
| 1 | Capstone khác một bài thi Kaggle ở những điểm nào? |
| 2 | Bốn tiêu chí chọn đề tài? Vì sao cần baseline nghiệp vụ? |
| 3 | Vì sao phải có kết quả đầu–cuối ngay tuần đầu? |
| 4 | Design doc gồm những phần nào? |
| 5 | Ba loại vấn đề trong dữ liệu thô và cách xử lý mỗi loại? |
| 6 | Hợp đồng dữ liệu là gì? Vì sao hàm làm sạch là "code sản phẩm"? |
| 7 | Nhãn chưa trưởng thành là gì? Sai lầm phổ biến khi xử lý? |
| 8 | Làm sao phát hiện một cột là thông tin sau quyết định? |
| 9 | Ngưỡng quyết định được suy ra từ kinh tế khoản vay thế nào? Vì sao calibration quan trọng? |
| 10 | Mô hình có AUC test tốt nhưng lỗ hơn luật cũ — chuyện gì đã xảy ra? |
| 11 | Nhãn sớm dùng thế nào để ước lượng lại mức rủi ro? Giả định gì? |
| 12 | Lý do từ chối (reason codes) với hồi quy logistic tính ra sao? |
| 13 | Vì sao phải kiểm tra calibration theo nhóm? |
| 14 | Backtest có tính độ trễ nhãn khác chia train/test một lần thế nào? |
| 15 | Một báo cáo capstone trung thực cần có gì? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Tự định nghĩa quyết định và metric kinh doanh; dữ liệu thô; so với cách làm hiện tại; đánh giá có độ bất định, theo nhóm, calibration, chi phí; kết thúc bằng hệ thống (gói, API, giám sát) và báo cáo.
2. Có quyết định rõ ràng, có baseline không ML, có nhãn/cách đánh giá đáng tin, vừa sức. Baseline nghiệp vụ là thứ mô hình phải thay thế — vượt tỉ lệ nền chưa đủ.
3. Phát hiện sớm vấn đề tích hợp, dữ liệu, metric; mỗi tuần sau là cải thiện có đo lường trên một hệ thống chạy được.
4. Bối cảnh & quyết định, metric (kinh doanh, mô hình, guardrail), dữ liệu & nhãn (thời điểm, cách chia), cách tiếp cận, rủi ro, triển khai & giám sát.
5. Định dạng (ngày, số tiền, danh mục) — sửa bằng quy tắc; giá trị (đơn vị, tuổi vô lý) — sửa khi chắc chắn, còn lại để thiếu; ngữ nghĩa (nhãn chưa trưởng thành, cột sau quyết định) — phải hiểu nghiệp vụ.
6. Ràng buộc kiểm tra tự động trên đầu ra làm sạch (khoá duy nhất, kiểu, miền, danh mục). Hàm làm sạch được dùng lại khi phục vụ nên phải tất định, có test.
7. Hồ sơ chưa đủ thời gian quan sát kết quả. Sai lầm: điền 0 ("không vỡ nợ") — gây chệch, nhất là dữ liệu gần đây.
8. Hỏi giá trị được ghi khi nào; AUC tăng vọt; tên cột chứa kết quả; cột thiếu với hồ sơ gần đây.
9. Duyệt khi (1 − p)·g − p·ℓ > 0 ⇔ p < g/(g + ℓ); ngưỡng so trực tiếp với xác suất nên xác suất phải đúng mức, không chỉ đúng thứ tự.
10. Trôi khái niệm (cú sốc kinh tế) làm mọi người rủi ro hơn; thứ hạng giữ nguyên nhưng xác suất bị đánh giá thấp → duyệt quá tay.
11. Ước lượng TPR/FPR của nhãn sớm trên dữ liệu có cả hai nhãn; suy ngược (tỉ lệ nhãn sớm − FPR)/(TPR − FPR); hiệu chỉnh hệ số chặn. Giả định TPR/FPR không đổi và cú sốc nhân odds đều.
12. Đóng góp vào log-odds = hệ số × giá trị đã chuẩn hoá; chọn các đóng góp dương lớn nhất, diễn đạt bằng ngôn ngữ nghiệp vụ.
13. Cú sốc/trôi tác động không đều; nhóm bị đánh giá thấp rủi ro nhiều nhất chịu thiệt (hoặc gây lỗ) nhiều nhất.
14. Mỗi thời điểm chỉ huấn luyện trên hồ sơ đã có nhãn (ngày nộp + 12 tháng ≤ thời điểm) — phản ánh dữ liệu "nguội" một năm và hiệu năng thật khi triển khai.
15. Kết quả chính so với baseline kèm khoảng tin cậy, số cấu hình đã thử, phân tích theo nhóm, những gì không hiệu quả, hạn chế, kế hoạch giám sát.

</details>
