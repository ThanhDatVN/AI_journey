# Nền tảng Deep Learning — Bạn đã thực sự "biết" mạng nơ-ron?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Một nơ-ron tính gì? Liên hệ với hồi quy tuyến tính/logistic? |
| 2 | Perceptron học thế nào? Khi nào hội tụ? Vì sao thất bại với XOR? |
| 3 | Lớp ẩn làm gì về mặt hình học? |
| 4 | Định lý xấp xỉ vạn năng nói gì và không nói gì? Vì sao vẫn cần mạng sâu? |
| 5 | Đồ thị tính toán là gì? Quy tắc chuỗi áp dụng thế nào? |
| 6 | Nút cộng, nhân, max chuyển gradient ra sao? |
| 7 | Vì sao gradient cộng dồn? Hệ quả trong vòng lặp huấn luyện? |
| 8 | Engine autograd hoạt động thế nào (forward ghi gì, backward làm gì)? |
| 9 | Viết các gradient của lớp tuyến tính Z = XW + b. |
| 10 | Vì sao gộp softmax với cross-entropy? Gradient là gì? |
| 11 | Kiểm tra gradient thế nào? |
| 12 | Vì sao backprop hiệu quả hơn sai phân hữu hạn? |
| 13 | Độ rộng và độ sâu của mạng ảnh hưởng gì? |
| 14 | Dấu hiệu overfitting và các hướng xử lý? |
| 15 | Vòng lặp huấn luyện PyTorch gồm những bước nào? |
| 16 | Những lỗi PyTorch phổ biến nhất và cách nhận biết? |
| 17 | Khi nào mạng nơ-ron KHÔNG phải lựa chọn tốt nhất? |
| 18 | So sánh cách tính gradient của TensorFlow (`GradientTape`) và PyTorch (`backward`). Vì sao TensorFlow không cần `zero_grad`? |
| 19 | `tf.function` làm gì? Bẫy hay gặp là gì? |
| 20 | Quy trình Keras gồm những bước nào? Khi nào phải tự viết vòng lặp huấn luyện? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. $a = g(\mathbf w^\top\mathbf x + b)$. $g$ đồng nhất → hồi quy tuyến tính; sigmoid → logistic; softmax trên nhiều nơ-ron → hồi quy softmax.
2. Đoán sai thì $\mathbf w \mathrel{+}= \eta y\mathbf x$, $b \mathrel{+}= \eta y$. Hội tụ nếu tách được tuyến tính (≤ $(R/\gamma)^2$ lỗi). XOR không tách được bằng một đường thẳng.
3. Biến đổi (phi tuyến) dữ liệu sang không gian mới nơi lớp sau tách được tuyến tính.
4. Tồn tại mạng một lớp ẩn xấp xỉ mọi hàm liên tục trên miền bị chặn. Không nói tìm được bằng GD, cần bao nhiêu nơ-ron, có tổng quát hoá không. Mạng sâu biểu diễn nhiều hàm hiệu quả hơn (tái sử dụng đặc trưng theo tầng).
5. Phép tính tách thành các nút phép toán cơ bản; gradient của một đầu vào = gradient đầu ra của nút × gradient cục bộ, nhân dọc đường đi, cộng qua các đường.
6. Cộng: chuyển nguyên gradient; nhân: gradient × giá trị đầu vào kia; max: chỉ đầu vào lớn nhất nhận gradient.
7. Biến ảnh hưởng qua nhiều nhánh → đạo hàm toàn phần là tổng. Thư viện cộng dồn vào `.grad` → phải `zero_grad()` mỗi bước.
8. Forward: mỗi phép toán tạo nút mới, nhớ cha và hàm đẩy gradient cục bộ. Backward: grad của đầu ra = 1, duyệt thứ tự tô-pô ngược, gọi hàm đẩy gradient của từng nút.
9. $\partial L/\partial W = X^\top\Delta$, $\partial L/\partial\mathbf b = \sum_{\text{hàng}}\Delta$, $\partial L/\partial X = \Delta W^\top$.
10. Gradient gộp gọn $\frac1n(\hat P - Y)$, ổn định số (log-softmax với trừ max); tính riêng thì tốn Jacobian và mất ổn định khi $\hat p \to 0$.
11. So với sai phân trung tâm $(f(\theta + h) - f(\theta - h))/2h$ trên vài tham số; sai lệch tương đối ~1e-7 trở xuống; tránh điểm không khả vi (ReLU tại 0); dùng float64.
12. Reverse-mode tính mọi gradient với chi phí cỡ một forward + một backward; sai phân hữu hạn cần 2 forward cho mỗi tham số.
13. Rộng/sâu hơn → biểu diễn mạnh hơn (fit dữ liệu phức tạp), nhưng dễ overfit, khó huấn luyện hơn (chương 28), tốn tính toán hơn.
14. Loss train giảm, loss validation ngừng giảm/tăng; khoảng cách accuracy lớn. Thêm dữ liệu/augmentation, weight decay, dropout, early stopping, mô hình nhỏ hơn/kiến trúc phù hợp.
15. `zero_grad` → forward → loss → `backward` → `step`; đánh giá với `model.eval()` + `torch.no_grad()`.
16. Quên `zero_grad`, softmax trước `CrossEntropyLoss`, nhãn sai dtype, khác thiết bị, quên `eval()`, đánh giá trên train, LR quá lớn. Mẹo: overfit một batch nhỏ trước.
17. Dữ liệu bảng nhỏ/vừa (boosting/hồi quy thường ngang hoặc hơn), cần diễn giải, ít dữ liệu, tài nguyên hạn chế. Mạng nơ-ron tỏa sáng với ảnh, văn bản, âm thanh, dữ liệu rất lớn.
18. TensorFlow ghi phép toán trong băng rồi `tape.gradient(loss, biến)` trả gradient trực tiếp, mỗi băng dùng một lần; PyTorch `loss.backward()` cộng dồn vào `.grad` của từng tensor → phải xoá mỗi bước. TensorFlow tự theo dõi `tf.Variable`; tensor thường cần `tape.watch`.
19. Truy vết hàm Python thành đồ thị ở lần gọi đầu, các lần sau chạy đồ thị (nhanh hơn, tối ưu được). Bẫy: code Python (`print`, append list) chỉ chạy lúc truy vết — dùng `tf.print`; gọi với số Python/shape thay đổi liên tục gây truy vết lại (chậm).
20. Khai báo mô hình (Sequential/Functional/kế thừa `keras.Model`) → `compile` (optimizer, loss, metric) → `fit` (validation, callback như EarlyStopping) → `evaluate`/`predict` → `save(".keras")`. Tự viết vòng lặp với `GradientTape` khi cần logic huấn luyện đặc biệt: GAN, học tăng cường, loss nhiều thành phần, cập nhật tuỳ biến.

</details>
