# Linear Regression — Bạn đã thực sự "biết" model này?

Trả lời được toàn bộ câu hỏi dưới đây (bằng lời của bạn, không nhìn tài liệu) mới thực sự **biết** Linear Regression. Ôn lại sau 1 ngày, 1 tuần, 1 tháng (spaced repetition). Mở phần gợi ý chỉ sau khi đã tự trả lời.

| # | Câu hỏi |
|---|---|
| 1 | Model này giải quyết bài toán gì? |
| 2 | Hàm giả thuyết và objective function là gì? |
| 3 | Model học tham số bằng cách nào? (ít nhất 2 cách) |
| 4 | Góc nhìn xác suất và góc nhìn hình học của model? |
| 5 | Các giả định của model? Vi phạm giả định nào nguy hiểm nhất cho dự đoán? cho diễn giải? |
| 6 | Có cần scaling không? Vì sao? |
| 7 | Có nhạy cảm với outlier không? Vì sao? Khắc phục? |
| 8 | Có xử lý quan hệ phi tuyến được không? Bằng cách nào? |
| 9 | Có xử lý missing value trực tiếp không? |
| 10 | Training complexity? Inference complexity? |
| 11 | Hyperparameter quan trọng nhất? (OLS, Ridge, Lasso) |
| 12 | Model dễ underfit hay overfit? Khi nào chuyển sang overfit? |
| 13 | Diễn giải một hệ số như thế nào? Bẫy khi diễn giải? |
| 14 | Ridge khác Lasso thế nào? Khi nào dùng cái nào? |
| 15 | Khi nào model này tốt hơn Random Forest? |
| 16 | Khi nào tuyệt đối không nên dùng? |
| 17 | Trước khi thu thập 100 triệu mẫu, kiểm tra gì để biết thêm dữ liệu có đáng không? |
| 18 | Theo dõi hội tụ của SGD trên dữ liệu rất lớn thế nào khi không thể tính loss trên toàn bộ dữ liệu? |
| 19 | Online learning là gì? Khi nào cần, và rủi ro là gì? |
| 20 | Map-reduce áp dụng cho gradient descent thế nào? Vì sao SGD từng mẫu khó song song hoá theo cách đó? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. **Regression**: dự đoán giá trị liên tục $y$ từ feature $\mathbf x$ bằng một hàm tuyến tính theo tham số.
2. $\hat y = \mathbf w^\top\mathbf x + b$; objective MSE $\frac1n\sum(y - \hat y)^2$ (Ridge/Lasso: + phạt L2/L1).
3. Nghiệm đóng (normal equation $X^\top X\mathbf w = X^\top\mathbf y$, giải bằng `lstsq`/`solve`); gradient descent (batch/mini-batch/SGD). Lasso: coordinate descent.
4. Xác suất: MLE với $y \mid \mathbf x \sim \mathcal N(\mathbf w^\top\mathbf x + b, \sigma^2)$. Hình học: $\hat{\mathbf y}$ là hình chiếu vuông góc của $\mathbf y$ lên không gian cột của $X$.
5. Tuyến tính, sai số độc lập, phương sai không đổi, sai số chuẩn, không đa cộng tuyến mạnh. Cho dự đoán: tuyến tính (sai hệ thống) và độc lập/không có leakage. Cho diễn giải: đa cộng tuyến, thiếu biến (omitted variable bias), heteroscedasticity (CI sai).
6. OLS bằng nghiệm đóng: không bắt buộc (dự đoán như nhau). Cần khi dùng GD (hội tụ), regularization (phạt công bằng), so sánh độ lớn hệ số.
7. Có — loss bình phương khuếch đại sai số lớn; điểm có leverage cao nguy hiểm nhất. Khắc phục: kiểm tra lỗi dữ liệu, Huber/MAE loss, log-transform.
8. Không trực tiếp; thêm feature phi tuyến (đa thức, log, spline, tương tác) — model vẫn tuyến tính theo tham số.
9. Không — phải impute (median + indicator...) trong Pipeline.
10. Train: $O(nd^2 + d^3)$ (nghiệm đóng) hoặc $O(k\,n\,d)$ (GD); inference $O(d)$ mỗi mẫu — rất nhanh.
11. OLS không có hyperparameter. Ridge/Lasso: $\lambda$ (`alpha` trong sklearn); Elastic Net thêm `l1_ratio`; với đa thức: bậc.
12. Thường **underfit** (high bias) vì chỉ biểu diễn quan hệ tuyến tính. Overfit khi số feature lớn so với số mẫu (vd: đa thức bậc cao, nhiều feature one-hot) → cần regularization.
13. "Khi $x_j$ tăng 1 đơn vị, **giữ các feature khác không đổi**, $\hat y$ thay đổi $w_j$". Bẫy: đơn vị khác nhau không so sánh được; ảnh hưởng có điều kiện ≠ tương quan đơn; đa cộng tuyến làm hệ số không ổn định; không phải nhân quả.
14. Ridge (L2) co đều mọi hệ số, ổn định với feature tương quan, không cho hệ số = 0. Lasso (L1) cho nghiệm thưa (chọn feature), nhưng chọn tuỳ ý trong nhóm feature tương quan. Nhiều feature hữu ích → Ridge; tin rằng ít feature quan trọng → Lasso; cả hai → Elastic Net.
15. Ít dữ liệu, quan hệ gần tuyến tính, cần **ngoại suy** (cây không ngoại suy được), cần diễn giải/suy luận thống kê, cần model cực nhẹ, nhiều feature thưa (văn bản).
16. Quan hệ phi tuyến/tương tác phức tạp mà không có feature engineering; target bị giới hạn (xác suất, số đếm — dùng logistic/Poisson); dữ liệu nhiều outlier mà vẫn dùng MSE; khi diễn giải hệ số như quan hệ nhân quả.
17. Vẽ learning curve với một tập nhỏ: train và validation đã sát nhau và cùng tệ (high bias) → thêm dữ liệu vô ích, cần mô hình/feature mạnh hơn; khoảng cách lớn (high variance) → thêm dữ liệu có ích.
18. Với mỗi mẫu, tính loss TRƯỚC khi cập nhật bằng mẫu đó (gần như miễn phí), vẽ trung bình của ~500–1 000 mẫu gần nhất. Đường tăng → learning rate quá lớn; quá nhiễu → tăng cửa sổ trung bình.
19. Cập nhật mô hình bằng từng lô dữ liệu mới rồi bỏ lô đó (`partial_fit`), không lưu và train lại toàn bộ lịch sử. Cần khi dữ liệu đến liên tục và quan hệ x → y trôi theo thời gian. Rủi ro: bị kéo lệch bởi dữ liệu bẩn/tấn công → cần giám sát và khả năng quay về phiên bản cũ.
20. Gradient batch là tổng trên các mẫu → mỗi máy tính tổng trên phần dữ liệu của mình (map), cộng lại rồi chia m (reduce) — kết quả giống hệt. SGD cập nhật tuần tự: bước sau phụ thuộc tham số của bước trước. (Huấn luyện deep learning song song dữ liệu dùng đúng ý tưởng map-reduce trên từng mini-batch: all-reduce.)

</details>
