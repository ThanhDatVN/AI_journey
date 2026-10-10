# Toán cho ML — Bạn đã thực sự hiểu?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Trong code, ma trận dữ liệu `X` có shape `(n, d)`. Mỗi hàng, mỗi cột là gì? `X @ w` cho ra gì? |
| 2 | Tích vô hướng đo điều gì về hình học? Cosine similarity khác khoảng cách Euclid thế nào? |
| 3 | Hạng của ma trận là gì? Vì sao đặc trưng trùng lặp làm $X^\top X$ không khả nghịch? |
| 4 | Trị riêng và vector riêng của ma trận hiệp phương sai cho biết gì (liên hệ PCA)? |
| 5 | SVD phân rã ma trận thành gì? Dùng ở đâu trong ML? |
| 6 | Gradient là gì, vì sao đi **ngược** gradient làm hàm giảm? |
| 7 | Quy tắc dây chuyền liên quan gì tới backpropagation? |
| 8 | Learning rate quá lớn / quá nhỏ gây ra hiện tượng gì? Nhận ra trên đồ thị loss thế nào? |
| 9 | Batch, stochastic, mini-batch gradient descent khác nhau thế nào? |
| 10 | Định lý Bayes là gì? Vì sao xét nghiệm "chính xác 99%" vẫn có thể cho phần lớn kết quả dương tính là sai? |
| 11 | Kỳ vọng, phương sai, hiệp phương sai, tương quan — mỗi cái đo gì? |
| 12 | MLE là gì? MLE cho phân phối Bernoulli và Gaussian ra công thức nào? |
| 13 | MAP khác MLE thế nào? Prior Gaussian tương ứng với điều chuẩn nào? |
| 14 | Khoảng tin cậy 95% nghĩa là gì (và không nghĩa là gì)? Bootstrap tạo khoảng tin cậy ra sao? |
| 15 | p-value là gì? Vì sao p nhỏ không có nghĩa là hiệu ứng lớn? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Hàng = một mẫu, cột = một đặc trưng; `X @ w` là vector `(n,)` — tổ hợp tuyến tính của đặc trưng cho từng mẫu (dự đoán tuyến tính).
2. $a \cdot b = \|a\|\|b\|\cos\theta$ — mức cùng hướng. Cosine chỉ quan tâm hướng (không phụ thuộc độ dài), Euclid phụ thuộc cả độ lớn.
3. Số cột (hàng) độc lập tuyến tính. Cột trùng/tỉ lệ → hạng thiếu → $X^\top X$ suy biến → nghiệm bình phương tối thiểu không duy nhất (Ridge khắc phục).
4. Vector riêng = hướng chính của dữ liệu; trị riêng = phương sai theo hướng đó. PCA chọn các vector riêng có trị riêng lớn nhất.
5. $X = U\Sigma V^\top$. Dùng cho PCA, giải bình phương tối thiểu ổn định, nén/khử nhiễu, phân rã ma trận trong hệ gợi ý.
6. Vector đạo hàm riêng, chỉ hướng tăng nhanh nhất; đi một bước nhỏ ngược hướng làm hàm giảm (xấp xỉ bậc nhất).
7. Backprop là quy tắc dây chuyền áp dụng có hệ thống trên đồ thị tính toán, tái sử dụng kết quả trung gian.
8. Quá lớn: loss dao động, tăng, NaN. Quá nhỏ: loss giảm rất chậm, chưa hội tụ khi hết số bước.
9. Batch: toàn bộ dữ liệu mỗi bước (ổn định, chậm); stochastic: một mẫu (nhiễu, nhanh); mini-batch: lô nhỏ (cân bằng, tận dụng vector hoá/GPU).
10. $P(A|B) = P(B|A)P(A)/P(B)$. Khi bệnh hiếm (tỉ lệ nền nhỏ), số dương tính giả từ đa số người khoẻ lấn át dương tính thật.
11. Kỳ vọng: giá trị trung bình; phương sai: độ phân tán; hiệp phương sai: mức cùng biến thiên (phụ thuộc đơn vị); tương quan: hiệp phương sai chuẩn hoá về [−1, 1], chỉ đo quan hệ tuyến tính.
12. Chọn tham số làm xác suất dữ liệu quan sát lớn nhất. Bernoulli: $\hat p$ = tỉ lệ thành công; Gaussian: $\hat\mu$ = trung bình mẫu, $\hat\sigma^2$ = phương sai chia $n$.
13. MAP cực đại hậu nghiệm = likelihood × prior. Prior Gaussian ↔ Ridge (L2); prior Laplace ↔ Lasso (L1).
14. Quy trình tạo khoảng chứa tham số thật trong 95% số lần lặp lại thí nghiệm — không phải "95% xác suất tham số nằm trong khoảng này" (theo trường phái tần suất). Bootstrap: lấy mẫu lại có hoàn lại, tính thống kê nhiều lần, lấy phân vị 2.5% và 97.5%.
15. Xác suất thấy kết quả ít nhất cực đoan như vậy **nếu giả thuyết không đúng**. Với n lớn, hiệu ứng rất nhỏ vẫn cho p nhỏ — luôn báo độ lớn hiệu ứng và khoảng tin cậy.

</details>
