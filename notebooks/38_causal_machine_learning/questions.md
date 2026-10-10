# Causal ML — Bạn đã thực sự "biết" suy luận nhân quả?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Câu hỏi dự đoán và câu hỏi nhân quả khác nhau thế nào? Cho ví dụ. |
| 2 | Confounder, mediator, collider là gì? Kiểm soát mỗi loại gây ra điều gì? |
| 3 | Giải thích nghịch lý Simpson và Berkson bằng DAG. |
| 4 | Tiêu chí cửa sau là gì? Vì sao "thêm mọi biến vào hồi quy" nguy hiểm? |
| 5 | Kết quả tiềm năng là gì? ITE, ATE, ATT, CATE? |
| 6 | Vì sao RCT là chuẩn vàng? |
| 7 | So sánh ngây thơ trên dữ liệu quan sát chệch thế nào? |
| 8 | Ba giả định để suy luận nhân quả từ dữ liệu quan sát? |
| 9 | Điểm xu hướng là gì? IPW hoạt động thế nào? |
| 10 | Kiểm tra overlap và cân bằng ra sao? |
| 11 | AIPW là gì? "Doubly robust" nghĩa là gì? Vì sao cần cross-fitting? |
| 12 | Vi phạm overlap gây ra điều gì? |
| 13 | S-, T-, X-learner khác nhau thế nào? |
| 14 | Uplift modeling là gì? Đánh giá bằng đường Qini thế nào? |
| 15 | Vì sao dự đoán phản hồi khác dự đoán tác động? |
| 16 | Khi báo cáo kết quả nhân quả từ dữ liệu quan sát cần kèm gì? |
| 17 | CUPED và Double ML giải quyết vấn đề gì? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Dự đoán: E[Y|X] dưới phân phối hiện có ("ai sẽ huỷ?"). Nhân quả: E[Y|do(X)] khi can thiệp ("nếu giảm giá thì có bớt huỷ?").
2. Confounder (nguyên nhân chung) — kiểm soát chặn tương quan giả. Mediator (trên đường nhân quả) — kiểm soát mất tác động gián tiếp. Collider (hệ quả chung) — kiểm soát/chọn mẫu tạo tương quan giả.
3. Simpson: confounder (mức độ bệnh) làm chiều tương quan gộp đảo ngược so với trong nhóm. Berkson: chọn mẫu theo collider (nổi tiếng) tạo tương quan âm giả.
4. Chặn mọi đường cửa sau, không chứa hậu duệ của điều trị. Thêm mediator/collider/biến hậu điều trị gây chệch.
5. Y(1), Y(0): kết quả nếu được/không được điều trị. ITE = Y(1) − Y(0); ATE trung bình; ATT trên người được điều trị; CATE theo đặc điểm x.
6. Ngẫu nhiên hoá làm điều trị độc lập với mọi confounder, kể cả chưa đo → so sánh nhóm không chệch.
7. = ATT + thiên lệch chọn lọc (người được điều trị khác người không ngay cả khi không điều trị).
8. Không có confounder chưa đo, overlap/positivity, SUTVA/consistency.
9. e(x) = P(T=1|x); IPW cân mỗi người theo 1/e hoặc 1/(1−e) để dữ liệu giống RCT.
10. Overlap: phân phối e(x) hai nhóm chồng nhau, e không quá gần 0/1. Cân bằng: SMD của confounder sau khi cân < 0.1.
11. Kết hợp mô hình kết quả và xu hướng; đúng nếu một trong hai đúng; cross-fitting tránh overfit của ML làm chệch.
12. Trọng số cực lớn, phương sai bùng nổ, ước lượng phụ thuộc vài người; dữ liệu không có thông tin ở vùng đó; trimming đổi tập đối tượng.
13. S: một mô hình với T là đặc trưng. T: hai mô hình riêng. X: gán hiệu ứng từng người bằng phản thực tế dự đoán, fit lại, trộn theo e(x).
14. Ước lượng CATE để chọn ai nên tác động; Qini: sắp theo uplift dự đoán, ước lượng kết quả tăng thêm ở top k trên dữ liệu thí nghiệm.
15. Người có kết quả cao có thể đạt kết quả đó dù không bị tác động; tác động là chênh lệch giữa hai kết quả tiềm năng.
16. DAG/giả định tường minh, chẩn đoán overlap & cân bằng, nhiều phương pháp, khoảng tin cậy, phân tích độ nhạy với confounder ẩn.
17. CUPED: giảm phương sai A/B test bằng biến trước thí nghiệm. Double ML: ước lượng tác động khi confounding phi tuyến, dùng ML + orthogonalization + cross-fitting.

</details>
