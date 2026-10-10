# Ensemble — Bạn đã thực sự "biết" voting, stacking, ensemble selection?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Vì sao kết hợp nhiều model có thể tốt hơn từng model? Điều kiện cần? |
| 2 | Định lý Condorcet nói gì? Giả định nào hiếm khi đúng trong ML? |
| 3 | Phát biểu phân rã ambiguity. Hệ quả? |
| 4 | Hard voting và soft voting khác nhau thế nào? Khi nào dùng cái nào? |
| 5 | Vì sao thêm model yếu có thể làm ensemble tệ đi? |
| 6 | Đo độ đa dạng thế nào? Bất đồng cao có luôn tốt? |
| 7 | Stacking là gì? Vì sao bắt buộc dùng dự đoán out-of-fold? |
| 8 | Meta-model nên như thế nào? Vì sao? |
| 9 | Blending khác stacking ở đâu? |
| 10 | Ensemble selection của Caruana hoạt động thế nào? Rủi ro? |
| 11 | Các nguồn tạo đa dạng? Nguồn nào hiệu quả cho dữ liệu bảng, cho mạng nơ-ron? |
| 12 | Khi nào không nên triển khai ensemble? Cách giảm chi phí? |
| 13 | Ensemble liên hệ với bagging, boosting, Random Forest thế nào? |
| 14 | Độ bất đồng giữa thành viên dùng vào việc gì? |
| 15 | Đánh giá một ensemble thế nào cho trung thực? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Các model mắc lỗi khác nhau → lỗi triệt tiêu khi kết hợp. Cần thành viên **đủ giỏi** (tốt hơn ngẫu nhiên) và **đa dạng** (lỗi ít tương quan).
2. $n$ model độc lập, mỗi model đúng với xác suất $p > 0.5$ → xác suất đa số đúng → 1 khi $n$ tăng. Giả định **độc lập**: model train trên cùng dữ liệu thường sai cùng chỗ.
3. $(\bar f - y)^2 = \frac1M\sum(f_m - y)^2 - \frac1M\sum(f_m - \bar f)^2$. Ensemble trung bình không tệ hơn trung bình thành viên; tốt hơn đúng bằng độ đa dạng; có thể thua thành viên tốt nhất.
4. Hard: đếm phiếu; soft: trung bình xác suất. Soft thường tốt hơn khi xác suất được hiệu chỉnh; hard an toàn hơn khi có model quá tự tin hoặc không có `predict_proba`.
5. Phiếu/xác suất của model yếu (hoặc quá tự tin) lật quyết định đúng của model mạnh — ví dụ 0.9 + 0.6 + 0.6 → 0.792.
6. Tỉ lệ bất đồng, tương quan của vector lỗi/phần dư (Q-statistic, kappa). Không: bất đồng cao có thể chỉ vì một model sai nhiều; muốn model đều giỏi mà sai khác chỗ.
7. Meta-model học cách kết hợp dự đoán của base model. Dự đoán in-sample lạc quan (model overfit trông hoàn hảo) → meta-model dồn trọng số cho model overfit; OOF mô phỏng dự đoán trên dữ liệu mới.
8. Đơn giản: tuyến tính có regularization hoặc trọng số không âm/tổng 1 — meta-feature ít, tương quan rất mạnh, dễ overfit (đặc biệt khi dữ liệu meta nhỏ).
9. Blending dùng một holdout cố định thay cho OOF: đơn giản, rẻ, nhưng meta-model chỉ thấy phần holdout và base model thấy ít dữ liệu hơn.
10. Tham lam, có hoàn lại: mỗi bước thêm model làm metric validation của trung bình tốt nhất; trọng số = số lần chọn. Rủi ro: overfit validation (nhỏ) → bagging quá trình chọn, khởi tạo bằng vài model tốt nhất, đánh giá trên test riêng.
11. Dữ liệu (bootstrap, mẫu con), feature (subspace), thuật toán, siêu tham số, seed, mục tiêu/loss. Bảng: khác họ thuật toán; mạng nơ-ron: khác seed khởi tạo (deep ensembles).
12. Khi lợi ích nhỏ so với chi phí độ trễ, bộ nhớ, bảo trì, diễn giải. Giảm: chọn ít thành viên (ensemble selection có ngân sách), distillation, chạy batch offline.
13. Bagging/Random Forest: ensemble cùng loại, trung bình model train trên bootstrap (giảm variance). Boosting: cộng tuần tự model yếu sửa lỗi (giảm bias). Voting/stacking: kết hợp model khác loại.
14. Tín hiệu bất định rẻ: gắn cờ dự đoán cần người kiểm tra, active learning, phát hiện dữ liệu lạ — nhưng chưa phải khoảng tin cậy đã hiệu chỉnh.
15. Mọi lựa chọn (thành viên, trọng số, meta-model, siêu tham số) chỉ dùng train/validation; báo cáo trên test chưa từng chạm hoặc nested CV; với chuỗi thời gian dùng backtest; so sánh với model đơn được tuning tốt.

</details>
