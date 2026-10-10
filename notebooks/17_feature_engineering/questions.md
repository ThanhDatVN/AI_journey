# Feature engineering — Bạn đã thực sự hiểu?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Khi nào biến đổi log, Box-Cox, binning giúp mô hình? Mô hình nào hưởng lợi nhiều nhất? |
| 2 | Đặc trưng tỉ lệ (ratio) và tương tác (interaction) là gì? Cho ví dụ có ý nghĩa nghiệp vụ. |
| 3 | Vì sao mô hình tuyến tính cần đặc trưng tương tác còn cây quyết định thì ít cần hơn? |
| 4 | Target encoding có rủi ro gì? Làm đúng thế nào? |
| 5 | Mã hoá ngày giờ: tách những thành phần nào? Vì sao mã hoá tuần hoàn (sin/cos) cho giờ, tháng? |
| 6 | Đặc trưng tổng hợp theo nhóm/thời gian (số giao dịch 7 ngày) cần chú ý gì để không rò rỉ? |
| 7 | Biểu diễn văn bản đơn giản cho mô hình bảng: những cách nào? |
| 8 | Vì sao feature engineering phải nằm trong pipeline? |
| 9 | Đặc trưng nào là "quá tốt để là thật"? Cách kiểm tra. |
| 10 | Kiến thức nghiệp vụ đóng vai trò gì trong feature engineering? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Biến lệch, quan hệ nhân, ngoại lai lớn → log/Box-Cox làm quan hệ gần tuyến tính, giảm ảnh hưởng đuôi. Mô hình tuyến tính, kNN, mạng nơ-ron hưởng lợi; cây ít bị ảnh hưởng.
2. Ratio: nợ/thu nhập, giá/m²; interaction: diện tích × quận — hiệu ứng của một biến phụ thuộc biến khác.
3. Mô hình tuyến tính cộng các hiệu ứng độc lập; cây tự tạo tương tác qua các lần chia nối tiếp.
4. Dùng nhãn của chính mẫu → rò rỉ, overfit với danh mục hiếm. Tính out-of-fold, làm trơn về trung bình chung, áp dụng thống kê train cho test.
5. Năm, tháng, ngày trong tuần, giờ, ngày lễ, khoảng cách tới sự kiện. Sin/cos giữ tính liên tục vòng (23h gần 0h, tháng 12 gần tháng 1).
6. Chỉ dùng dữ liệu **trước** thời điểm dự đoán (as-of), loại sự kiện tại/sau thời điểm đó; tính giống hệt lúc phục vụ.
7. Độ dài, số từ, từ khoá, TF-IDF (rồi giảm chiều), embedding trung bình, cảm xúc.
8. Để fit đúng trên train trong mỗi fold, không rò rỉ, và phục vụ dùng đúng cùng logic.
9. Đặc trưng làm điểm validation tăng vọt bất thường; kiểm tra thời điểm ghi nhận, ý nghĩa nghiệp vụ, tương quan gần hoàn hảo với nhãn, hiệu năng khi bỏ đặc trưng đó.
10. Gợi ý đặc trưng có ý nghĩa (tỉ lệ nợ, mùa vụ), phát hiện rò rỉ, giải thích được mô hình — thường giá trị hơn đổi mô hình.

</details>
