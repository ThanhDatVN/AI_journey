# Logistic Regression — Bạn đã thực sự "biết" model này?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Model giải quyết bài toán gì? Vì sao tên có chữ "regression"? |
| 2 | Hàm giả thuyết, objective function, và chúng đến từ đâu (góc nhìn xác suất)? |
| 3 | Decision boundary có hình dạng gì? Làm sao để có ranh giới cong? |
| 4 | Diễn giải một hệ số $w_j$ như thế nào (odds ratio)? |
| 5 | Model học tham số bằng cách nào? Bài toán có lồi không? |
| 6 | Vì sao không dùng MSE làm loss? |
| 7 | Có cần scaling không? Vì sao? |
| 8 | `C` trong scikit-learn là gì? Tăng `C` thì sao? |
| 9 | Chuyện gì xảy ra khi dữ liệu tách được hoàn toàn? |
| 10 | Có nhạy cảm với outlier không? Với mất cân bằng lớp? |
| 11 | Mở rộng cho nhiều lớp thế nào? OvR khác softmax ra sao? |
| 12 | Xác suất của model có đáng tin không? Chọn ngưỡng thế nào? |
| 13 | Training / inference complexity? |
| 14 | Khi nào model này tốt hơn Random Forest? |
| 15 | Khi nào không nên dùng? |
| 16 | So sánh ba loss cho cùng mô hình tuyến tính: One-vs-All logistic, softmax, SVM đa lớp. Chúng "nghĩ" khác nhau thế nào? |
| 17 | Vì sao mô hình tuyến tính trên pixel thô chỉ đạt ~40% trên CIFAR-10? Learning curve nói gì? |
| 18 | Kiểm tra gradient (gradient check) làm thế nào? Ngưỡng nào là ổn? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Phân loại (nhị phân; mở rộng nhiều lớp). "Regression" vì nó là hồi quy tuyến tính trên **log-odds**: $\log\frac{p}{1-p} = \mathbf w^\top\mathbf x + b$.
2. $p = \sigma(\mathbf w^\top\mathbf x + b)$; objective = binary cross-entropy = negative log-likelihood của phân phối Bernoulli (MLE).
3. Siêu phẳng $\mathbf w^\top\mathbf x + b = 0$. Ranh giới cong: thêm feature phi tuyến (đa thức, tương tác), hoặc dùng kernel/model phi tuyến.
4. Tăng $x_j$ một đơn vị (giữ các feature khác cố định) → log-odds tăng $w_j$ → odds nhân $e^{w_j}$.
5. Gradient descent / SGD, L-BFGS (mặc định sklearn), Newton/IRLS. Cross-entropy với model tuyến tính là **lồi** → cực tiểu toàn cục duy nhất (nếu không tách được hoàn toàn).
6. MSE + sigmoid cho hàm không lồi, gradient gần 0 khi dự đoán sai hoàn toàn → học chậm/kẹt; cross-entropy lồi và phạt nặng dự đoán tự tin mà sai.
7. Có, nên: regularization phạt công bằng giữa các feature; tối ưu hội tụ nhanh hơn; hệ số so sánh được.
8. `C = 1/λ` — hệ số nghịch đảo của regularization. `C` lớn → regularization yếu → model phức tạp hơn (dễ overfit); `C` nhỏ → hệ số co mạnh (dễ underfit). Mặc định `C = 1` (có regularization!).
9. Không regularization: trọng số tiến ra vô cùng, GD không hội tụ, xác suất cực đoan 0/1. Regularization khắc phục.
10. Outlier ở feature có leverage nhưng sigmoid bão hoà nên ít nghiêm trọng hơn linear regression. Mất cân bằng: model thiên về lớp đa số → dùng metric phù hợp, `class_weight`, chọn ngưỡng.
11. One-vs-Rest: K classifier nhị phân độc lập, chọn điểm cao nhất (điểm không cộng thành 1). Softmax (multinomial): học K lớp cùng lúc, xác suất hợp lệ; với 2 lớp trùng logistic regression.
12. Thường khá calibrated (khi đúng giả định, regularization vừa phải); kiểm tra bằng reliability diagram. Ngưỡng chọn theo chi phí FP/FN trên validation: $t^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$ nếu xác suất calibrated.
13. Train $O(k\,n\,d)$ (k vòng lặp); inference $O(d)$ mỗi mẫu (với K lớp: $O(Kd)$).
14. Ít dữ liệu, nhiều feature thưa (văn bản), quan hệ gần tuyến tính với log-odds, cần xác suất calibrated, cần diễn giải hệ số, cần model nhẹ/nhanh, cần ngoại suy ổn định.
15. Ranh giới phi tuyến phức tạp với nhiều tương tác mà không thể feature-engineer; dữ liệu ảnh/âm thanh thô; khi coi hệ số là quan hệ nhân quả.
16. OvA: 10 bài toán nhị phân độc lập (xác suất không cộng lại bằng 1); softmax: một phân phối xác suất trên các lớp, loss luôn dương — luôn muốn tăng xác suất lớp đúng; SVM đa lớp: chỉ cần lớp đúng vượt các lớp khác một lề Δ, đạt rồi thì loss bằng 0. Trên CIFAR-10 cả ba cho ~40% — nút thắt là mô hình.
17. Mỗi lớp chỉ có một "ảnh mẫu" (trọng số) → dựa vào màu/nền, không biểu diễn được tư thế, hình dạng, vị trí khác nhau. Learning curve train ≈ validation ≈ 40% → high bias: thêm dữ liệu không giúp, cần đặc trưng học được (CNN — chương 29 đạt ~73% trên cùng dữ liệu).
18. So gradient giải tích với sai phân trung tâm (f(w + h) − f(w − h)) / 2h tại vài phần tử ngẫu nhiên, dùng float64 và tập nhỏ; sai số tương đối ~1e-7 trở xuống là đúng, > 1e-3 gần như chắc có bug. Với hàm có điểm gãy (hinge, ReLU) thỉnh thoảng sai số lớn hơn mà không phải bug.

</details>
