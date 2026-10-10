# Phát hiện bất thường — Bạn đã thực sự "biết" các detector?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Anomaly detection giải quyết bài toán gì? Khi nào dùng thay cho phân loại có giám sát? |
| 2 | Ba loại bất thường là gì? Ví dụ? |
| 3 | Vì sao z-score thất bại khi có nhiều outlier hoặc mẫu nhỏ? Robust z-score khắc phục thế nào? |
| 4 | Mô hình Gaussian phát hiện bất thường hoạt động thế nào? Ước lượng gì, trên dữ liệu nào? |
| 5 | Chọn ngưỡng ε thế nào? Vì sao không dùng accuracy? |
| 6 | Gaussian độc lập và Gaussian nhiều chiều khác nhau ra sao? Khi nào cần cái sau? |
| 7 | Khoảng cách Mahalanobis liên hệ với phân phối $\chi^2$ thế nào? |
| 8 | Dữ liệu train có lẫn bất thường thì sao? MCD giúp gì? |
| 9 | Isolation Forest hoạt động thế nào? Vì sao bất thường có đường đi ngắn? |
| 10 | LOF đo gì? Khi nào LOF tốt hơn Isolation Forest? |
| 11 | One-Class SVM: ý tưởng, tham số, nhược điểm? |
| 12 | `contamination` ảnh hưởng gì? Đặt ngưỡng cảnh báo thế nào trong thực tế? |
| 13 | Outlier detection khác novelty detection thế nào? |
| 14 | Vì sao model có giám sát bỏ sót kiểu bất thường mới? |
| 15 | Sai số tái tạo (PCA/autoencoder) phát hiện bất thường khi nào, thất bại khi nào? |
| 16 | Đánh giá detector khi chỉ có vài chục nhãn thế nào? |
| 17 | Phát hiện bất thường trên chuỗi thời gian cần chú ý gì? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Học mô hình của dữ liệu **bình thường** rồi cảnh báo điểm lệch khỏi nó. Dùng khi positive cực hiếm, ít/không có nhãn, kiểu bất thường đa dạng và thay đổi (cần bắt kiểu chưa từng thấy).
2. **Điểm** (một giao dịch khổng lồ), **ngữ cảnh** (35°C tháng 12), **tập thể** (chuỗi đăng nhập thất bại — từng lần bình thường).
3. Outlier kéo trung bình và thổi phồng std → z-score của chính chúng nhỏ lại (masking); với $n$ điểm, $|z| \le \sqrt{n - 1}$ (chia $n$) → $n \le 10$ thì $|z| > 3$ không bao giờ xảy ra. Median/MAD gần như không bị outlier ảnh hưởng: robust z = $0.6745(x - \text{median})/\text{MAD}$.
4. Ước lượng $\mu_j, \sigma_j^2$ trên dữ liệu (gần như) bình thường, không cần nhãn; $p(\mathbf x) = \prod_j \mathcal N(x_j; \mu_j, \sigma_j^2)$; cảnh báo khi $p(\mathbf x) < \varepsilon$. Feature nên gần chuẩn (log/căn), tạo feature mới khi kiểu bất thường bị bỏ sót.
5. Trên validation có nhãn (vài chục bất thường đã biết), chọn ε tối đa F1 (hoặc theo precision@k / năng lực xử lý). Accuracy cao nhất khi không cảnh báo gì. Với nhiều chiều, duyệt ε trên thang log.
6. Độc lập: tích các Gaussian 1 chiều, ellipse song song trục, $2d$ tham số, rẻ — bỏ sót bất thường do **tương quan**. Nhiều chiều: hiệp phương sai đầy đủ, bắt được tương quan tự động, cần $n \gg d$ và $\Sigma$ khả nghịch, $O(d^3)$.
7. $d^2 = (\mathbf x - \boldsymbol\mu)^\top\Sigma^{-1}(\mathbf x - \boldsymbol\mu)$; nếu dữ liệu Gaussian $d$ chiều thì $d^2 \sim \chi^2_d$ → ngưỡng theo phân vị (vd `chi2.ppf(0.999, d)` ≈ 0.1% báo nhầm).
8. Bất thường kéo trung bình và làm méo hiệp phương sai → chính chúng trông bình thường hơn. MCD ước lượng trên tập con có định thức hiệp phương sai nhỏ nhất (phần "sạch"), chịu được nhiễm bẩn tới gần 50%.
9. Cây chia không gian bằng cắt ngẫu nhiên (feature ngẫu nhiên, ngưỡng ngẫu nhiên); điểm ít và khác biệt bị cô lập sau ít phép cắt. Điểm số $2^{-\mathbb E[h]/c(\psi)}$; mỗi cây dùng mẫu con nhỏ (256), giới hạn chiều cao $\lceil\log_2\psi\rceil$ → nhanh.
10. Tỉ số giữa mật độ cục bộ của láng giềng và của điểm; ≈ 1 bình thường, ≫ 1 bất thường so với vùng của nó. Tốt hơn IF khi dữ liệu có vùng mật độ khác nhau hoặc hình dạng phức tạp (đa tạp cong); kém khi nhiều chiều, dữ liệu lớn, chọn $k$.
11. Tìm ranh giới (kernel RBF) bao phần lớn dữ liệu train; $\nu$ ≈ tỉ lệ điểm được phép nằm ngoài, `gamma` độ "uốn". Nhạy tham số và scaling, $O(n^2)$, khó tinh chỉnh khi không có nhãn.
12. Chỉ đặt ngưỡng cho `predict`, không đổi điểm số. Thực tế: làm việc với điểm số, ngưỡng theo năng lực xử lý (top-k mỗi ngày) hoặc validation, theo dõi tỉ lệ cảnh báo.
13. Outlier detection: train trên dữ liệu có lẫn bất thường, không nhãn. Novelty detection: train chỉ trên dữ liệu bình thường đã xác nhận, phát hiện điểm mới khác biệt (LOF `novelty=True`, Gaussian, autoencoder).
14. Nó học "bất thường trông giống các ví dụ có nhãn"; feature phân biệt kiểu mới có thể chưa từng mang tín hiệu ở train → điểm thấp. Chạy song song detector bất thường; dùng điểm bất thường làm feature.
15. Khi dữ liệu bình thường nằm gần không gian con/đa tạp ít chiều (feature tương quan, ảnh, tín hiệu): điểm lệch khỏi cấu trúc → tái tạo kém. Thất bại khi feature gần như độc lập (không có cấu trúc ít chiều) hoặc bất thường lệch dọc các thành phần chính.
16. PR-AUC, precision@k, recall theo kiểu bất thường; nhớ rằng vài chục nhãn → ước lượng rất nhiễu (bootstrap khoảng tin cậy); không có nhãn → kết hợp detector theo thứ hạng để phòng rủi ro.
17. "Bình thường" phụ thuộc ngữ cảnh (xu hướng, mùa vụ, khuyến mãi, lễ): so với quá khứ của chính chuỗi (cùng thứ, trung vị), robust (median/MAD), chỉ dùng quá khứ (`.shift(1)`), loại trừ ngữ cảnh đã biết hoặc cảnh báo trên phần dư của model dự báo.

</details>
