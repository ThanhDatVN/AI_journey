# Gradient Boosting — Bạn đã thực sự "biết" model này?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Boosting khác bagging thế nào? Giảm bias hay variance? |
| 2 | AdaBoost cập nhật trọng số mẫu và trọng số model ra sao? |
| 3 | "Gradient" trong Gradient Boosting là gradient của cái gì, theo biến nào? |
| 4 | Pseudo-residual với MSE, MAE, log-loss? |
| 5 | Vai trò của learning rate và số cây? Vì sao cần early stopping? |
| 6 | Độ sâu cây trong GB ảnh hưởng gì? Vì sao thường dùng cây nông? |
| 7 | XGBoost khác GB cổ điển ở đâu? Công thức giá trị lá? |
| 8 | Histogram binning, leaf-wise growth, ordered boosting là gì? |
| 9 | Có cần scaling không? Missing, categorical? |
| 10 | Hyperparameter quan trọng và quy trình tuning? |
| 11 | GB có ngoại suy được không? Có nhạy với nhiễu nhãn/leakage không? |
| 12 | Xác suất của GB (log-loss) có calibrated không? |
| 13 | Training / inference complexity? |
| 14 | Khi nào dùng GB thay vì RF hay model tuyến tính? |
| 15 | Khi nào không nên dùng? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Boosting tuần tự, model sau sửa lỗi model trước, base learner yếu → giảm **bias** chủ yếu (shrinkage/subsample giảm thêm variance). Bagging song song, độc lập, giảm variance.
2. $\alpha_m = \frac12\ln\frac{1-\varepsilon_m}{\varepsilon_m}$; $w_i \leftarrow w_i e^{-\alpha_m y_i h_m(x_i)}$ rồi chuẩn hoá — mẫu sai tăng trọng số.
3. Gradient của **loss** theo **giá trị dự đoán** $F(x_i)$ tại các điểm train; cây xấp xỉ negative gradient để tổng quát hoá cho điểm mới → gradient descent trong không gian hàm.
4. MSE: $y - F$; MAE: $\operatorname{sign}(y - F)$; log-loss: $y - \sigma(F)$.
5. η nhỏ → mỗi cây đóng góp ít, cần nhiều cây hơn, tổng quát hoá tốt hơn. Thêm cây làm model phức tạp hơn → có thể overfit → early stopping trên validation.
6. Độ sâu ≈ bậc tương tác tối đa mà mỗi cây học được. Cây nông (3–8 lá/độ sâu 3–6) đủ cho hầu hết bài toán, ít overfit; boosting cộng dồn nhiều cây nông để tạo hàm phức tạp.
7. Dùng gradient **và hessian** (Taylor bậc hai), regularization trong objective ($\gamma T + \frac\lambda2\sum w^2$) → $w^* = -\frac{G}{H+\lambda}$, gain đóng; cộng thêm tối ưu hệ thống (sparsity-aware, histogram, song song, cache).
8. Histogram: gom giá trị vào ≤ 255 bin → tìm split nhanh. Leaf-wise: luôn tách lá có gain lớn nhất (LightGBM, giới hạn `num_leaves`). Ordered boosting: tính residual bằng model không thấy mẫu đó (CatBoost) để tránh prediction shift; ordered target statistics cho categorical.
9. Không cần scaling. Missing: học hướng mặc định ở mỗi split. Categorical: native (LightGBM, CatBoost, XGBoost `enable_categorical`, sklearn Hist GB) hoặc encode.
10. `learning_rate` + số cây (early stopping), độ phức tạp cây (`max_depth`/`num_leaves`), `min_child_samples`/`min_child_weight`, `subsample`, `colsample`, `reg_lambda`. Quy trình: cố định η + early stopping → độ phức tạp cây → ngẫu nhiên hoá/regularization → giảm η.
11. Không ngoại suy. Khá nhạy với nhiễu nhãn khi chạy quá nhiều vòng (AdaBoost nhạy hơn); khai thác leakage rất hiệu quả → AUC đẹp bất thường là dấu hiệu cảnh báo.
12. Với log-loss thường khá tốt, nhưng có thể quá tự tin khi overfit; kiểm tra reliability diagram, calibrate nếu cần.
13. Train $O(M \cdot n \cdot d)$ (histogram: $O(M \cdot (n + \text{bins}) \cdot d)$), tuần tự theo cây; dự đoán $O(M \cdot \text{độ sâu})$ — nhanh hơn RF thường vì cây nông.
14. Dữ liệu bảng, cần độ chính xác cao, có thời gian tune, có tương tác/phi tuyến, loss đặc thù (ranking, quantile, Tweedie...).
15. Dữ liệu phi cấu trúc (ảnh, âm thanh, văn bản thô) → deep learning; ít dữ liệu và quan hệ tuyến tính → model tuyến tính; cần ngoại suy; không có thời gian tune và cần model "khó hỏng" → RF.

</details>
