# Decision Tree — Bạn đã thực sự "biết" model này?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Model giải quyết bài toán gì? Dự đoán được tính như thế nào? |
| 2 | Decision boundary có hình dạng gì? |
| 3 | Cây chọn split bằng tiêu chí nào? Entropy và Gini khác nhau ra sao? |
| 4 | Thuật toán xây cây có tối ưu toàn cục không? |
| 5 | Độ phức tạp train và dự đoán? |
| 6 | Vì sao cây dễ overfit? Kiểm soát bằng cách nào? |
| 7 | Pre-pruning khác post-pruning thế nào? `ccp_alpha` là gì? |
| 8 | Có cần scaling không? Feature phân loại, missing values? |
| 9 | Vì sao cây không ổn định? Hệ quả? |
| 10 | Regression tree dự đoán gì ở lá? Có ngoại suy được không? |
| 11 | Feature importance của cây đáng tin đến đâu? |
| 12 | Cây có học được tương tác giữa feature không? Quan hệ tuyến tính thì sao? |
| 13 | Xác suất `predict_proba` của cây có tốt không? |
| 14 | Khi nào dùng một cây đơn lẻ? |
| 15 | Khi nào không nên dùng? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Phân loại và hồi quy. Đi từ gốc xuống lá theo các câu hỏi $x_j \le t$; lá trả về lớp đa số/tỉ lệ lớp (phân loại) hoặc trung bình (hồi quy).
2. Các hộp chữ nhật song song trục (axis-aligned); ranh giới chéo phải xấp xỉ bằng bậc thang.
3. Split tối đa hoá giảm impurity (Gini/entropy cho phân loại, phương sai cho hồi quy). Entropy = lượng thông tin (bit), Gini = xác suất gán sai ngẫu nhiên; cả hai lõm chặt, thường cho cây gần giống nhau; Gini không cần log nên nhanh hơn chút.
4. Không — greedy từng node; tìm cây tối ưu là NP-hard. Hệ quả: split tốt về sau có thể bị bỏ lỡ (XOR).
5. Train ~$O(d\,n\log^2 n)$ (sắp xếp mỗi feature ở mỗi tầng); dự đoán $O(\text{độ sâu})$ mỗi mẫu.
6. Không giới hạn, cây chia tới khi mọi lá thuần → học thuộc nhiễu. Kiểm soát: `max_depth`, `min_samples_leaf`, `min_samples_split`, `max_leaf_nodes`, `min_impurity_decrease`, `ccp_alpha` — chọn bằng CV.
7. Pre-pruning dừng sớm khi phát triển (nhanh, có thể bỏ lỡ split tốt về sau). Post-pruning trồng đầy đủ rồi cắt nhánh yếu nhất theo $R(T) + \alpha|T|$; `ccp_alpha` = $\alpha$, lớn → cây nhỏ.
8. Không cần scaling (chỉ phụ thuộc thứ tự). Feature phân loại: one-hot/ordinal (sklearn), hoặc split nhiều nhánh (ID3/C4.5, LightGBM native). Missing: sklearn ≥ 1.3 tự xử lý; hoặc impute.
9. Greedy + lỗi lan truyền: thay đổi nhỏ ở split trên → toàn bộ cây con khác. Hệ quả: variance cao, "giải thích" không ổn định → dùng bagging/Random Forest.
10. Trung bình (MSE) hoặc trung vị (MAE) của $y$ trong lá. Không ngoại suy — ngoài vùng train trả về hằng số.
11. MDI tính trên train, thiên vị feature liên tục/nhiều giá trị và feature tương quan chia nhau importance; không ổn định. Ưu tiên permutation importance trên held-out, và regularize model trước khi diễn giải.
12. Có — split con phụ thuộc split cha tạo tương tác tự nhiên. Quan hệ tuyến tính mượt lại khó: phải xấp xỉ bằng nhiều bậc → cần nhiều dữ liệu.
13. Thô (tỉ lệ lớp trong lá); cây sâu cho xác suất 0/1 → ROC-AUC kém. `min_samples_leaf` lớn hơn hoặc calibration giúp.
14. Cần luật đơn giản, minh bạch (cây nhỏ, ổn định đã kiểm định), baseline nhanh, khám phá tương tác, làm thành phần của ensemble.
15. Cần độ chính xác cao (dùng ensemble), dữ liệu có xu hướng cần ngoại suy, quan hệ chủ yếu tuyến tính/mượt với ít dữ liệu, cần xác suất tốt, cần mô hình ổn định theo thời gian.

</details>
