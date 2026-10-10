# Phân tích dữ liệu & EDA — Bạn đã thực sự hiểu?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Phân biệt dữ liệu số liên tục, số rời rạc, danh mục (nominal), thứ bậc (ordinal). Mỗi loại xử lý khác nhau thế nào khi đưa vào mô hình? |
| 2 | CSV, Parquet, pickle khác nhau thế nào? Khi nào dùng loại nào? |
| 3 | Một quy trình EDA hướng câu hỏi gồm những bước nào? |
| 4 | Vì sao phải vẽ dữ liệu thay vì chỉ xem trung bình, độ lệch chuẩn (bộ tứ Anscombe)? |
| 5 | Trung bình và trung vị khác nhau khi nào? Phân phối lệch phải ảnh hưởng thế nào? |
| 6 | Tương quan cao giữa hai biến nói lên điều gì và không nói lên điều gì? |
| 7 | Nghịch lý Simpson là gì? Cho một ví dụ trong kinh doanh. |
| 8 | Kể năm vấn đề chất lượng dữ liệu thường gặp và cách phát hiện mỗi vấn đề. |
| 9 | Dữ liệu thiếu: MCAR, MAR, MNAR khác nhau thế nào? Vì sao quan trọng? |
| 10 | Ngoại lai: khi nào là lỗi cần sửa, khi nào là tín hiệu cần giữ? |
| 11 | Vì sao phải tách tập test **trước** khi làm EDA sâu? |
| 12 | Thiên lệch chọn mẫu (selection bias) ảnh hưởng thế nào tới kết luận từ dữ liệu? |
| 13 | Lấy dữ liệu từ Internet (URL, GitHub, Google Drive, Dropbox, Kaggle) thế nào cho tái lập được? Checksum dùng để làm gì? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Liên tục (giá, thu nhập) → có thể biến đổi log, chuẩn hoá; rời rạc (số con) → như số hoặc nhóm; nominal (thành phố) → one-hot/target encoding; ordinal (cấp độ) → mã hoá theo thứ tự.
2. CSV: văn bản, phổ biến, chậm, mất kiểu dữ liệu. Parquet: dạng cột, nén, giữ kiểu, nhanh — cho dữ liệu lớn. Pickle: đối tượng Python, nhanh nhưng **không an toàn** với file lạ và phụ thuộc phiên bản.
3. Hiểu bài toán và câu hỏi → kiểm tra cấu trúc, kiểu, thiếu, trùng → phân phối đơn biến → quan hệ với mục tiêu và giữa các biến → kiểm tra giả thuyết, phát hiện bất thường → ghi lại phát hiện và hành động.
4. Các tập có cùng thống kê tóm tắt có thể có hình dạng hoàn toàn khác (phi tuyến, ngoại lai, cụm).
5. Khác nhau khi phân phối lệch hoặc có ngoại lai; lệch phải → trung bình > trung vị (bị kéo bởi đuôi dài).
6. Có quan hệ **tuyến tính** cùng biến thiên; không nói về nhân quả, không bắt được quan hệ phi tuyến, dễ bị ngoại lai và biến gây nhiễu chi phối.
7. Xu hướng trong từng nhóm đảo ngược khi gộp nhóm do biến gây nhiễu phân bố không đều — vd tỉ lệ chuyển đổi theo kênh khi cơ cấu khách hàng mỗi kênh khác nhau.
8. Trùng lặp (`duplicated`), thiếu (`isna`), sai kiểu (số đọc thành chuỗi), danh mục viết không thống nhất (`value_counts`), giá trị vô lý/sai đơn vị (`describe`, vẽ phân phối), rò rỉ (cột sau thời điểm dự đoán).
9. MCAR: thiếu hoàn toàn ngẫu nhiên; MAR: phụ thuộc biến quan sát được; MNAR: phụ thuộc chính giá trị bị thiếu. Quyết định cách điền và liệu điền có gây chệch không; MNAR → thêm cờ "thiếu".
10. Lỗi (gõ nhầm, sai đơn vị, bất khả) → sửa hoặc đặt thiếu; giá trị hiếm nhưng thật (giao dịch lớn, gian lận) → giữ, có thể là thông tin quan trọng nhất.
11. Để quyết định dựa trên tập test không ảnh hưởng mô hình — tránh rò rỉ thông tin và kết luận lạc quan.
12. Mẫu không đại diện cho quần thể mục tiêu → kết luận/mô hình sai khi áp dụng thực tế (vd chỉ có khách đã được duyệt vay, chỉ có phim người dùng tự chọn xem).
13. Tải một lần vào `data/raw/` (không commit), không sửa tay file gốc, ghi nguồn + ngày tải + checksum (MD5/SHA-256); code xử lý ghi sang `data/processed/`. Checksum là dấu vân tay của file: đổi một byte là checksum đổi → biết chắc mọi người dùng đúng một file. GitHub dùng link raw; Dropbox đổi `?dl=0` thành `?dl=1`; Drive dùng `drive.mount` trên Colab hoặc `gdown`.

</details>
