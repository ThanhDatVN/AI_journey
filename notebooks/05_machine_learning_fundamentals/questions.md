# Nền tảng ML — Bạn đã thực sự hiểu?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Học có giám sát, không giám sát, tăng cường khác nhau thế nào? Mỗi loại một ví dụ. |
| 2 | Vai trò của train, validation, test? Vì sao không dùng test để chọn mô hình? |
| 3 | Overfitting và underfitting là gì? Nhận ra mỗi loại bằng lỗi train/validation thế nào? |
| 4 | Đánh đổi bias–variance là gì? Độ phức tạp mô hình ảnh hưởng thế nào? |
| 5 | Đọc đường cong học: khi nào thêm dữ liệu giúp, khi nào không? |
| 6 | Phân tích lỗi (error analysis) gồm những bước nào? |
| 7 | Lời nguyền chiều cao là gì? Ảnh hưởng tới kNN thế nào? |
| 8 | Định lý No Free Lunch nói gì? Hệ quả thực tế? |
| 9 | Rò rỉ dữ liệu là gì? Kể bốn dạng rò rỉ. |
| 10 | Vì sao rò rỉ thường không gây ra lỗi nào mà chỉ làm kết quả "quá đẹp"? |
| 11 | Covariate shift, label shift, concept drift khác nhau thế nào? |
| 12 | Làm sao phát hiện tập test khác phân phối tập train? |
| 13 | Baseline là gì và vì sao luôn cần? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Có giám sát: học từ cặp (x, y) — dự đoán giá nhà. Không giám sát: chỉ có x — phân khúc khách hàng. Tăng cường: học hành động qua phần thưởng — robot, game, đặt giá động.
2. Train học tham số; validation chọn mô hình/siêu tham số; test ước lượng hiệu năng cuối cùng một lần. Dùng test để chọn → test thành validation → điểm lạc quan.
3. Overfit: train tốt, validation kém (học nhiễu). Underfit: cả hai kém (mô hình quá đơn giản/đặc trưng nghèo).
4. Lỗi kỳ vọng = bias² + phương sai + nhiễu. Mô hình đơn giản: bias cao, phương sai thấp; phức tạp: ngược lại. Chọn độ phức tạp tối thiểu lỗi validation.
5. Hai đường còn cách xa nhau và validation đang đi lên → thêm dữ liệu giúp (phương sai cao). Hai đường đã sát nhau ở mức lỗi cao → thêm dữ liệu không giúp; cần mô hình/đặc trưng tốt hơn.
6. Lấy mẫu các dự đoán sai, nhóm theo nguyên nhân (nhãn sai, thiếu đặc trưng, nhóm dữ liệu hiếm, trường hợp mơ hồ), đếm tỉ lệ từng nhóm, ưu tiên sửa nhóm lớn nhất.
7. Khi số chiều lớn, dữ liệu thưa thớt, mọi khoảng cách gần bằng nhau → "láng giềng gần nhất" mất ý nghĩa; cần nhiều dữ liệu hơn theo cấp số mũ, chọn/giảm đặc trưng.
8. Trung bình trên mọi bài toán, không thuật toán nào tốt hơn thuật toán khác → phải dựa vào giả định phù hợp với dữ liệu; thử nhiều họ mô hình, đánh giá trên chính bài toán của mình.
9. Đặc trưng chứa thông tin tương lai/kết quả; tiền xử lý fit trên cả test; cùng thực thể ở cả train và test; chọn đặc trưng/siêu tham số trên test; chia ngẫu nhiên dữ liệu thời gian.
10. Code chạy bình thường, mô hình "học" tín hiệu rò rỉ rất mạnh → điểm cao bất thường; chỉ lộ ra khi triển khai thật.
11. Covariate shift: P(X) đổi, P(y|X) giữ. Label shift: P(y) đổi, P(X|y) giữ. Concept drift: P(y|X) đổi.
12. So phân phối từng đặc trưng (PSI, KS), huấn luyện bộ phân loại phân biệt train và test (adversarial validation — AUC ≈ 0.5 là tốt).
13. Mốc đơn giản nhất (tỉ lệ nền, trung bình, luật nghiệp vụ) để biết mô hình có thật sự thêm giá trị; phát hiện lỗi khi mô hình phức tạp thua baseline.

</details>
