# Tiền xử lý dữ liệu — Bạn đã thực sự hiểu?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Các cách điền giá trị thiếu? Khi nào nên thêm cột "cờ thiếu"? |
| 2 | Vì sao không được điền thiếu bằng trung bình tính trên toàn bộ dữ liệu trước khi chia train/test? |
| 3 | Cách phát hiện ngoại lai bằng IQR và z-score? Hạn chế của mỗi cách? |
| 4 | Min-max scaling và standardization khác nhau thế nào? Khi nào dùng cái nào? |
| 5 | Mô hình nào **cần** chuẩn hoá, mô hình nào không? Vì sao? |
| 6 | One-hot, ordinal, target encoding — mỗi cách hợp với tình huống nào? |
| 7 | Biến danh mục có 10.000 giá trị: one-hot gặp vấn đề gì? Giải pháp? |
| 8 | Danh mục mới xuất hiện lúc dự đoán (chưa thấy khi train) — xử lý thế nào? |
| 9 | Pipeline và ColumnTransformer giải quyết vấn đề gì? |
| 10 | Vì sao `fit_transform` chỉ được gọi trên tập train? |
| 11 | Biến đổi log cho biến lệch phải giúp gì? Cẩn thận điều gì khi biến có giá trị 0? |
| 12 | Tiền xử lý viết lại thủ công ở hệ thống phục vụ gây rủi ro gì? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Trung bình/trung vị/mode, giá trị hằng, mô hình (KNN, iterative imputer), theo nhóm. Thêm cờ thiếu khi việc thiếu tự nó mang thông tin (MNAR, vd khách mới chưa có hoá đơn).
2. Thống kê chứa thông tin của tập test → rò rỉ; đánh giá lạc quan. Đặt imputer trong pipeline, fit trên train.
3. IQR: ngoài [Q1 − 1.5·IQR, Q3 + 1.5·IQR] — bền nhưng máy móc với phân phối lệch. z-score: |z| > 3 — giả định gần chuẩn, bản thân trung bình và độ lệch chuẩn bị ngoại lai kéo.
4. Min-max đưa về [0, 1], nhạy ngoại lai; standardization đưa về trung bình 0, độ lệch chuẩn 1. Mạng nơ-ron/ảnh hay dùng min-max; mô hình tuyến tính, SVM, kNN, PCA thường dùng standardization.
5. Cần: dựa trên khoảng cách hoặc gradient/điều chuẩn (kNN, SVM, K-Means, PCA, hồi quy có điều chuẩn, mạng nơ-ron). Không cần: cây và ensemble cây (chỉ dùng thứ tự giá trị).
6. One-hot: danh mục không thứ tự, ít giá trị. Ordinal: có thứ tự tự nhiên (hoặc cho mô hình cây). Target encoding: nhiều giá trị — phải tính out-of-fold và làm trơn.
7. Ma trận rất rộng, thưa, nhiều cột hiếm → overfit, chậm. Gộp danh mục hiếm thành "khác", target/frequency encoding (OOF), embedding, hashing.
8. `OneHotEncoder(handle_unknown="ignore")` (vector 0) hoặc nhóm "khác" — và **ghi log/cảnh báo** vì đây là dấu hiệu trôi dữ liệu.
9. Gói mọi bước tiền xử lý + mô hình thành một đối tượng: fit đúng trên train trong mỗi fold CV, không rò rỉ, phục vụ đúng như lúc huấn luyện.
10. Thống kê (trung bình, độ lệch chuẩn, danh mục) phải học từ train; áp dụng (`transform`) lên validation/test như dữ liệu mới.
11. Giảm độ lệch, giảm ảnh hưởng của giá trị cực lớn, biến quan hệ nhân thành cộng. log(0) = −∞ → dùng `log1p` hoặc cộng hằng số nhỏ.
12. Lệch train/serving: logic khác (điền 0 thay vì trung vị, khác đơn vị) → dự đoán sai lặng lẽ. Đóng gói pipeline, kiểm thử parity.

</details>
