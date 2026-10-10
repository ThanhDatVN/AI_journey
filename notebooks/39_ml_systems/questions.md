# ML Systems — Bạn đã thực sự "biết" đưa mô hình vào hệ thống?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Kể các thành phần của một hệ thống ML ngoài mô hình. Thành phần nào thường tốn công nhất? |
| 2 | Trước khi huấn luyện, khung thiết kế cần trả lời những câu hỏi nào? |
| 3 | Rò rỉ thời điểm là gì? Vì sao train/test ngẫu nhiên không phát hiện được? Phòng tránh thế nào? |
| 4 | Lệch train/serving xảy ra thế nào? Kể ba nguyên nhân và ba cách phòng tránh. |
| 5 | Khi nào chọn batch, online, streaming? Ràng buộc nào ảnh hưởng tới việc chọn mô hình? |
| 6 | Phân biệt lặp lại, tái lập, tái hiện. |
| 7 | Cố định seed giải quyết được gì và không giải quyết được gì? Báo cáo hiệu năng trung thực thế nào? |
| 8 | Vì sao so sánh hai mô hình nên làm theo cặp? |
| 9 | Kể các nguồn bất định ngoài seed và cách kiểm soát. |
| 10 | Một run thí nghiệm cần lưu gì để chạy lại được? Hash dữ liệu dùng để làm gì? |
| 11 | Lời nguyền người thắng là gì? Liên quan gì tới việc "nhìn" tập test nhiều lần? |
| 12 | Vì sao phải đóng gói cả pipeline tiền xử lý? Bẫy `__main__` khi pickle là gì? |
| 13 | Vì sao pickle không an toàn? Các lựa chọn thay thế? |
| 14 | Metadata của một gói mô hình gồm những gì? Model card để làm gì? |
| 15 | Vì sao mô hình cần lớp kiểm tra đầu vào? Hai lớp kiểm tra nên có? |
| 16 | Model registry, phiên bản, alias là gì? Rollback diễn ra thế nào? |
| 17 | Cổng thăng hạng nên kiểm tra những gì? |
| 18 | ONNX là gì? Khi nào nên xuất mô hình sang ONNX? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Thu thập/kiểm tra dữ liệu, đặc trưng (feature store), huấn luyện, đánh giá, registry, phục vụ, giám sát, vòng phản hồi, cấu hình, hạ tầng. Dữ liệu và đặc trưng thường tốn công nhất.
2. Quyết định nào thay đổi nhờ mô hình; metric kinh doanh ↔ mô hình ↔ vận hành; nhãn định nghĩa và độ trễ của nhãn; đặc trưng có sẵn tại thời điểm dự đoán; baseline; kiểu phục vụ và ràng buộc; cách đánh giá (offline → shadow → A/B); giám sát và người chịu trách nhiệm.
3. Đặc trưng dùng thông tin sau thời điểm dự đoán. Train và test đều mang cùng rò rỉ nên điểm offline đều đẹp. Xây dữ liệu đúng thời điểm (as-of join, feature store), đánh giá theo thời gian, kiểm thử tính chất "đổi dữ liệu tương lai không đổi đặc trưng".
4. Logic viết lại ở dịch vụ khác, nguồn dữ liệu khác (batch vs online), cách điền thiếu/đơn vị khác. Phòng tránh: đóng gói tiền xử lý vào pipeline, dùng chung feature pipeline, kiểm thử parity, log đặc trưng lúc phục vụ và so với lúc train.
5. Batch: không cần tức thì (danh sách hằng đêm). Online: theo yêu cầu (ms–s). Streaming: sự kiện liên tục (gian lận). Ràng buộc: độ trễ p99, thông lượng, kích thước, chi phí, khả năng giải thích.
6. Lặp lại: máy mình, cùng kết quả. Tái lập: người khác từ code + dữ liệu + cấu hình + môi trường. Tái hiện: dữ liệu mới, kết luận vẫn đúng.
7. Seed cho khả năng lặp lại, không làm kết quả đúng hơn. Báo cáo trung bình ± độ lệch chuẩn qua nhiều seed/lần chia, số cấu hình đã thử.
8. Độ khó của từng tập test ảnh hưởng cả hai mô hình; hiệu theo cặp loại bỏ phần nhiễu chung → phương sai nhỏ hơn nhiều.
9. Thứ tự set/dict (`PYTHONHASHSEED`), thứ tự file, thứ tự cộng số thực khi song song/GPU, phiên bản thư viện, phần cứng, dữ liệu nguồn thay đổi. Kiểm soát: sắp xếp tường minh, thuật toán tất định, lock file/Docker, hash dữ liệu.
10. Params (kể cả seed), metric, commit code, hash/phiên bản dữ liệu, môi trường, artifact (mô hình), trạng thái. Hash phát hiện dữ liệu thay đổi lặng lẽ giữa hai run.
11. Chọn cấu hình tốt nhất trong nhiều cấu hình → điểm của nó lạc quan (một phần là may). Nhìn test để chọn biến test thành validation → cũng lạc quan; test chỉ chạm một lần.
12. Phục vụ phải dùng đúng tiền xử lý đã học (tránh lệch train/serving). Pickle lưu tham chiếu `module.tên`; hàm định nghĩa trong notebook nằm ở `__main__` nên tiến trình khác không tìm thấy → đặt vào package.
13. Pickle có thể gọi hàm tuỳ ý khi nạp (`__reduce__`). Thay thế: skops (duyệt kiểu), safetensors/`weights_only`, ONNX, kiểm tra hash/ký số.
14. Phiên bản thư viện, schema đầu vào, metric (tổng và theo nhóm), nguồn và hash dữ liệu, hash file mô hình, thời điểm, mục đích sử dụng. Model card: mục đích/không nên dùng, hiệu năng theo nhóm, dữ liệu, hạn chế.
15. Mô hình không báo lỗi với dữ liệu sai (giá trị lạ → vector 0, số âm → z-score). Lớp 1: schema nghiệp vụ (kiểu, khoảng, danh mục — pydantic). Lớp 2: so với phân phối train (ngoài khoảng đã thấy → cảnh báo, giám sát).
16. Kho mô hình có phiên bản bất biến; alias (champion/challenger) trỏ tới phiên bản; phục vụ nạp theo alias; rollback = trỏ alias về phiên bản cũ.
17. Metric tổng trên holdout cố định, metric theo nhóm, độ trễ, kích thước, schema, calibration, công bằng, so với baseline.
18. Định dạng đồ thị tính toán chuẩn, chạy bằng ONNX Runtime ở nhiều ngôn ngữ/thiết bị, không chứa code tuỳ ý. Dùng khi phục vụ ngoài Python, cần tối ưu độ trễ, hoặc triển khai trên edge/mobile.

</details>
