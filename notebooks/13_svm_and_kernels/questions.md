# SVM — Bạn đã thực sự "biết" model này?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | SVM giải quyết bài toán gì? "Lề lớn nhất" nghĩa là gì? |
| 2 | Viết bài toán tối ưu (hard và soft margin). Vì sao lồi? |
| 3 | Support vectors là gì? Vì sao nghiệm "thưa"? |
| 4 | $C$ làm gì? Tăng $C$ thì sao? |
| 5 | Hinge loss khác log-loss thế nào? |
| 6 | Kernel trick là gì? Vì sao cần bài toán đối ngẫu? |
| 7 | RBF kernel: ý nghĩa của $\gamma$? Kết hợp với $C$ ra sao? |
| 8 | Có cần scaling không? |
| 9 | SVM có cho xác suất không? |
| 10 | Training / inference complexity? Dữ liệu lớn thì sao? |
| 11 | Nhiều lớp xử lý thế nào? |
| 12 | SVR hoạt động thế nào? |
| 13 | SVM có nhạy với mất cân bằng, feature nhiễu, outlier không? |
| 14 | Khi nào dùng SVM? |
| 15 | Khi nào không nên dùng? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Phân loại (và hồi quy với SVR). Trong các siêu phẳng tách dữ liệu, chọn siêu phẳng có khoảng cách tới điểm gần nhất (margin) lớn nhất → robust với nhiễu nhỏ, tổng quát hoá tốt.
2. Hard: $\min\frac12\lVert\mathbf w\rVert^2$ s.t. $y_i(\mathbf w^\top\mathbf x_i + b) \ge 1$. Soft: thêm $C\sum\xi_i$, ràng buộc $\ge 1 - \xi_i$. Hàm mục tiêu bậc hai lồi, ràng buộc tuyến tính → QP lồi.
3. Điểm nằm trên hoặc trong lề / sai phía ($\alpha_i > 0$). Hinge loss = 0 với điểm ngoài lề → chúng không ảnh hưởng nghiệm.
4. Trọng số phạt vi phạm lề (≈ $1/\lambda$). Tăng $C$ → lề hẹp, ít vi phạm, ít support vectors, dễ overfit.
5. Hinge = 0 khi margin ≥ 1 (bỏ qua điểm dễ), tuyến tính khi vi phạm; log-loss luôn > 0, cho xác suất. Accuracy thường tương đương.
6. Dual: dữ liệu chỉ xuất hiện qua $\mathbf x_i^\top\mathbf x_j$ → thay bằng $K(\mathbf x_i, \mathbf x_j) = \phi(\mathbf x_i)^\top\phi(\mathbf x_j)$, làm việc trong không gian $\phi$ (kể cả vô hạn chiều) mà không tính $\phi$.
7. $\gamma$ = độ "hẹp" vùng ảnh hưởng của mỗi support vector: lớn → ranh giới uốn éo (overfit), nhỏ → gần tuyến tính. $C$ và $\gamma$ tương tác — vùng tốt thường là một dải chéo trên lưới log.
8. Bắt buộc (kernel dựa trên khoảng cách/tích vô hướng; regularization phạt $\lVert\mathbf w\rVert$).
9. Không trực tiếp — chỉ `decision_function`. Platt scaling (`probability=True`, CV nội bộ, chậm) hoặc `CalibratedClassifierCV`.
10. Kernel SVM: train $O(n^2)$–$O(n^3)$, bộ nhớ $O(n^2)$; dự đoán $O(n_{SV}\cdot d)$. Linear SVM: $O(nd)$. Dữ liệu lớn → LinearSVC/SGD, Nystroem/RFF, hoặc boosting.
11. One-vs-One (SVC mặc định, $K(K-1)/2$ model) hoặc One-vs-Rest (LinearSVC); Crammer–Singer multiclass.
12. $\varepsilon$-insensitive loss: bỏ qua sai số < $\varepsilon$; điểm ngoài "ống" là support vectors; $\varepsilon$ lớn → ít SV, mượt hơn.
13. Mất cân bằng: có (dùng `class_weight`). Feature nhiễu: RBF bị pha loãng như KNN. Outlier: hinge tuyến tính nên ít nhạy hơn MSE, nhưng $C$ lớn vẫn bị kéo.
14. Dữ liệu cỡ nhỏ–vừa (≤ vài chục nghìn mẫu), nhiều chiều (văn bản, gen, ảnh nhỏ), cần ranh giới phi tuyến với ít tuning, cần nghiệm toàn cục ổn định.
15. Dữ liệu rất lớn (kernel), cần xác suất calibrated hoặc diễn giải, dữ liệu bảng hỗn hợp nhiều kiểu (boosting thường tốt hơn), feature chưa chuẩn hoá được.

</details>
