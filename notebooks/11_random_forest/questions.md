# Random Forest — Bạn đã thực sự "biết" model này?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Model giải quyết bài toán gì? Dự đoán được tính như thế nào? |
| 2 | Vì sao trung bình nhiều cây tốt hơn một cây? Giới hạn của việc lấy trung bình là gì? |
| 3 | Bootstrap là gì? OOB là gì? |
| 4 | Random Forest khác bagging ở đâu? Vì sao điều đó giúp ích? |
| 5 | Random Forest giảm bias hay variance? |
| 6 | Thêm cây có làm overfit không? |
| 7 | Hyperparameter nào quan trọng nhất? Chọn thế nào? |
| 8 | Có cần scaling không? Feature phân loại, missing? |
| 9 | Feature importance của RF có đáng tin không? |
| 10 | Training / inference complexity? Song song hoá được không? |
| 11 | Có ngoại suy được không? |
| 12 | Xác suất của RF có calibrated không? |
| 13 | RF khác Gradient Boosting thế nào? |
| 14 | Khi nào dùng? |
| 15 | Khi nào không nên dùng? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Phân loại (trung bình xác suất/bỏ phiếu) và hồi quy (trung bình) của nhiều cây quyết định train trên các mẫu bootstrap với feature ngẫu nhiên ở mỗi split.
2. Trung bình giảm variance: $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$. Giới hạn là sàn $\rho\sigma^2$ do các cây tương quan (cùng dữ liệu, cùng feature mạnh).
3. Bootstrap: rút có hoàn lại $n$ mẫu; ~63.2% điểm khác nhau. OOB: ~36.8% điểm không được chọn — dùng để đánh giá cây đó; OOB score ≈ CV miễn phí.
4. Ở mỗi split chỉ xét `max_features` feature ngẫu nhiên → cây đa dạng, tương quan $\rho$ thấp → variance của trung bình thấp hơn, dù từng cây yếu hơn chút.
5. Chủ yếu variance; bias ≈ bias của một cây sâu (thấp). Không sửa được bias của base learner.
6. Không — chỉ làm trung bình chính xác hơn; lợi ích bão hoà; chi phí tăng tuyến tính.
7. `max_features` (quan trọng nhất), `min_samples_leaf`, `max_samples`, `class_weight`; `n_estimators` chọn đủ lớn. Chọn bằng OOB hoặc CV (RandomizedSearch).
8. Không cần scaling. Feature phân loại: one-hot/ordinal (hoặc native ở một số thư viện). Missing: sklearn ≥ 1.4 hỗ trợ; hoặc impute.
9. MDI thiên vị feature liên tục/nhiều giá trị, tính trên train; permutation importance tốt hơn nhưng bị "che" bởi feature tương quan. Dùng thận trọng, gom nhóm feature tương quan, SHAP.
10. Train $O(B \cdot m \cdot n\log^2 n)$ với $m$ = `max_features`; song song hoá hoàn toàn theo cây. Dự đoán $O(B \cdot \text{độ sâu})$ — chậm hơn model tuyến tính nhiều; model có thể rất lớn.
11. Không — dự đoán nằm trong khoảng giá trị đã thấy.
12. Thường không: xác suất bị kéo vào giữa; cần calibration (Platt/isotonic) nếu dùng xác suất.
13. RF: cây sâu, độc lập, song song, giảm variance, ít cần tune. GB: cây nông, tuần tự, mỗi cây sửa lỗi cây trước, giảm bias; thường chính xác hơn khi tune kỹ nhưng dễ overfit hơn.
14. Baseline mạnh cho dữ liệu bảng, ít thời gian tuning, cần robust với outlier/nhiễu, cần OOB, dữ liệu có tương tác phi tuyến.
15. Cần ngoại suy (xu hướng), quan hệ chủ yếu tuyến tính/cộng dồn, dữ liệu thưa nhiều chiều (văn bản), yêu cầu độ trễ/bộ nhớ rất thấp, cần xác suất calibrated mà không calibrate.

</details>
