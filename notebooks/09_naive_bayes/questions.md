# Naive Bayes — Bạn đã thực sự "biết" model này?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Model giải quyết bài toán gì? Vì sao gọi là "Bayes"? Vì sao "naive"? |
| 2 | Model sinh khác model phân biệt thế nào? Naive Bayes thuộc loại nào? |
| 3 | Học tham số bằng cách nào? Có cần tối ưu lặp không? |
| 4 | Decision boundary có hình dạng gì (Gaussian, Multinomial)? |
| 5 | Zero-frequency problem là gì? Xử lý thế nào? |
| 6 | Vì sao phải tính trong không gian log? |
| 7 | Có cần scaling không? |
| 8 | Multinomial, Bernoulli, Gaussian, Complement, Categorical NB — dùng khi nào? |
| 9 | Vì sao giả định sai mà model vẫn phân loại tốt? |
| 10 | Xác suất của Naive Bayes có đáng tin không? |
| 11 | Khi nào Naive Bayes tốt hơn Logistic Regression và ngược lại? |
| 12 | Model học được tương tác giữa các feature không? |
| 13 | Training / inference complexity? |
| 14 | Khi nào nên dùng? |
| 15 | Khi nào không nên dùng? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Phân loại (nhị phân, nhiều lớp), nhất là văn bản. "Bayes": tính posterior $P(y \mid \mathbf x) \propto P(\mathbf x \mid y)P(y)$. "Naive": giả định các feature **độc lập có điều kiện** khi biết lớp.
2. Model sinh mô hình hoá $P(\mathbf x \mid y)$ và $P(y)$ (có thể "sinh" dữ liệu); model phân biệt mô hình hoá trực tiếp $P(y \mid \mathbf x)$ hoặc ranh giới. Naive Bayes là model sinh.
3. Maximum likelihood dạng đóng: đếm tần suất (Multinomial/Bernoulli/Categorical) hoặc tính mean/variance (Gaussian) theo lớp; smoothing ≈ MAP với Dirichlet prior. Không cần lặp — một lượt qua dữ liệu.
4. Multinomial/Bernoulli: tuyến tính theo feature (score = $\log P(y) + \sum_j x_j w_{cj}$). Gaussian: bậc hai; tuyến tính nếu các lớp dùng chung phương sai.
5. Một giá trị/từ chưa từng xuất hiện cùng lớp $c$ trong train → $P(x_j \mid c) = 0$ → cả tích bằng 0. Laplace/Lidstone smoothing (cộng $\alpha$); với Gaussian: `var_smoothing`.
6. Tích hàng nghìn xác suất nhỏ underflow về 0; tổng log ổn định; dùng log-sum-exp để đổi về posterior.
7. Không — Gaussian NB chuẩn hoá mỗi feature bằng $\sigma_{cj}$ của chính nó; Multinomial dùng đếm. Nhưng nên biến đổi feature lệch cho gần Gaussian.
8. Multinomial: đếm từ / TF-IDF. Bernoulli: feature nhị phân, văn bản ngắn (phạt từ vắng mặt). Gaussian: feature liên tục. Complement: văn bản mất cân bằng lớp. Categorical: feature phân loại.
9. Phân loại chỉ cần $\arg\max$ đúng; giả định sai làm sai **độ lớn** posterior nhưng thường không đảo **thứ tự** (Domingos & Pazzani 1997). Ngoài ra ít tham số → variance thấp.
10. Không — thường quá tự tin (đẩy về 0/1) do đếm trùng bằng chứng tương quan. Cần calibration (Platt/isotonic) nếu dùng như xác suất.
11. NB: rất ít dữ liệu, nhiều chiều thưa (văn bản), cần train cực nhanh / online. LR: đủ dữ liệu, feature tương quan — tiệm cận tốt hơn (Ng & Jordan 2002).
12. Không — mỗi feature đóng góp độc lập; XOR khiến NB đoán mò. Cần tự tạo feature tương tác hoặc dùng model khác.
13. Train $O(nd)$ một lượt; dự đoán $O(Cd)$ mỗi mẫu (với dữ liệu thưa: tỉ lệ số phần tử khác 0). Bộ nhớ $O(Cd)$.
14. Baseline văn bản (spam, sentiment, phân loại chủ đề), ít dữ liệu, cần train/predict cực nhanh, học trực tuyến, nhiều lớp, feature có missing.
15. Cần xác suất calibrated mà không calibrate; feature tương tác mạnh; feature liên tục có phân phối phức tạp; dữ liệu bảng nhiều mẫu (boosting thường tốt hơn hẳn).

</details>
