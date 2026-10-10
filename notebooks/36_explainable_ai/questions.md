# Explainable AI — Bạn đã thực sự "biết" giải thích mô hình?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Vì sao cần giải thích mô hình? Cho bốn mục đích khác nhau. |
| 2 | Phân biệt giải thích toàn cục/cục bộ, model-specific/model-agnostic. |
| 3 | Khi nào nên dùng mô hình vốn dễ hiểu thay vì giải thích hộp đen? |
| 4 | MDI và permutation importance khác nhau thế nào? Thiên lệch của MDI? |
| 5 | PDP, ICE, ALE là gì? Khi nào PDP sai lệch? |
| 6 | Global surrogate là gì? Vì sao phải báo cáo độ trung thực? |
| 7 | LIME hoạt động thế nào? Điểm yếu? |
| 8 | Định nghĩa giá trị Shapley. Bốn tính chất của nó? |
| 9 | Hàm giá trị interventional là gì? Vai trò của dữ liệu nền? |
| 10 | Shapley của mô hình tuyến tính có dạng gì? KernelSHAP liên hệ LIME thế nào? |
| 11 | Counterfactual explanation là gì? Giới hạn? |
| 12 | Đặc trưng tương quan ảnh hưởng thế nào đến độ quan trọng? |
| 13 | Permutation importance và drop-column importance trả lời câu hỏi nào? |
| 14 | Hiệu ứng Rashomon là gì? Hệ quả? |
| 15 | Dùng XAI để gỡ lỗi (rò rỉ, đường tắt) như thế nào? |
| 16 | Vì sao bỏ thuộc tính nhạy cảm không đảm bảo công bằng? |
| 17 | Làm sao kiểm tra một lời giải thích có trung thực? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Gỡ lỗi (rò rỉ, đường tắt), niềm tin của người dùng, tuân thủ quy định/kiểm toán, khám phá khoa học (cẩn thận với nhân quả).
2. Toàn cục: mô hình nói chung; cục bộ: một dự đoán. Model-specific dùng cấu trúc bên trong (hệ số, MDI); model-agnostic chỉ cần predict.
3. Khi hiệu năng tương đương (thường với dữ liệu bảng) và quyết định có hệ quả lớn — Rudin: đừng giải thích hộp đen khi có thể dùng mô hình dễ hiểu.
4. MDI: giảm impurity trên train, thiên lệch về đặc trưng nhiều giá trị và overfit. Permutation: hiệu năng giảm khi xáo cột trên dữ liệu giữ lại.
5. PDP: trung bình dự đoán khi cố định một giá trị cho mọi mẫu; ICE: từng mẫu; ALE: cộng dồn hiệu cục bộ trên dữ liệu thật. PDP ngoại suy khi đặc trưng tương quan.
6. Mô hình dễ hiểu bắt chước dự đoán của mô hình đen; độ trung thực thấp thì "giải thích" sai.
7. Nhiễu loạn quanh điểm, hỏi mô hình, fit tuyến tính có trọng số. Nhạy với số mẫu, kernel, cách nhiễu loạn; không có đảm bảo hiệu quả.
8. Trung bình đóng góp biên qua mọi tập con (có trọng số); hiệu quả, đối xứng, dummy, cộng tính.
9. v(S) = trung bình f khi đặc trưng trong S lấy từ x, còn lại từ dữ liệu nền; nền quyết định "so với ai".
10. β_j(x_j − E x_j). KernelSHAP = hồi quy tuyến tính có trọng số (Shapley kernel) trên các tập con — LIME với trọng số đúng.
11. Thay đổi nhỏ nhất, khả thi làm đổi dự đoán; nói về mô hình, không đảm bảo tác động thật; cần ràng buộc khả thi.
12. Bị chia sẻ hoặc thay thế: bỏ một đặc trưng, đặc trưng tương quan gánh thay; độ quan trọng thấp ≠ không liên quan.
13. Permutation: mô hình này dựa vào đặc trưng bao nhiêu. Drop-column: dữ liệu mất bao nhiêu thông tin không thay thế được.
14. Nhiều mô hình tốt ngang nhau có giải thích khác nhau → giải thích mô tả mô hình, không phải thế giới; xem qua nhiều mô hình.
15. Đặc trưng áp đảo + hiệu năng đẹp bất thường → hỏi "có sẵn lúc dự đoán không?"; xem vùng ảnh/đặc trưng mô hình dựa vào.
16. Biến ủy nhiệm (khu vực, tên...) mang thông tin nhóm; nhãn lịch sử thiên lệch; phải đo kết quả theo nhóm.
17. Deletion/insertion test, so sánh với ngẫu nhiên, kiểm tra trên dữ liệu có sự thật đã biết, kiểm tra độ ổn định (seed, nền, mô hình).

</details>
