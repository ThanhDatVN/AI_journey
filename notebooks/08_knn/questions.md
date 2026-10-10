# K-Nearest Neighbors — Bạn đã thực sự "biết" model này?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Model giải quyết bài toán gì? Dự đoán được tính như thế nào? |
| 2 | "Train" KNN nghĩa là gì? Vì sao gọi là lazy và non-parametric? |
| 3 | Decision boundary có hình dạng gì? Thay đổi thế nào theo $k$? |
| 4 | $k$ ảnh hưởng tới bias và variance ra sao? Chọn $k$ thế nào? |
| 5 | Có cần scaling không? Vì sao? |
| 6 | Chọn thước đo khoảng cách thế nào? Cosine khác Euclid ở đâu? |
| 7 | Xử lý feature phân loại ra sao? |
| 8 | Training / inference complexity? Bộ nhớ? |
| 9 | Làm sao tăng tốc khi dữ liệu lớn? |
| 10 | Vì sao KNN tệ trong không gian nhiều chiều? |
| 11 | Có nhạy với feature vô ích, outlier, mất cân bằng lớp không? |
| 12 | KNN regressor có ngoại suy được không? |
| 13 | Xác suất `predict_proba` của KNN có đáng tin không? |
| 14 | Khi nào KNN là lựa chọn tốt? |
| 15 | Khi nào không nên dùng? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Phân loại (bỏ phiếu đa số của $k$ láng giềng) và hồi quy (trung bình $y$ của $k$ láng giềng); có thể có trọng số $1/d$. Ngoài ra: tìm kiếm tương tự, recommender, impute (KNNImputer), phát hiện bất thường (khoảng cách tới láng giềng thứ $k$).
2. "Train" = lưu dữ liệu (có thể dựng KD-tree). Lazy: mọi tính toán dồn vào lúc dự đoán. Non-parametric: không có số tham số cố định — "model" chính là dữ liệu, độ phức tạp tăng theo $n$.
3. Phi tuyến tuỳ ý; $k = 1$ là sơ đồ Voronoi (lởm chởm); $k$ tăng → ranh giới mượt hơn; $k = n$ → luôn dự đoán lớp đa số.
4. $k$ nhỏ: bias thấp, variance cao (overfit); $k$ lớn: bias cao, variance thấp. Chọn bằng CV (scaler trong Pipeline); điểm khởi đầu $k \approx \sqrt n$, lẻ với 2 lớp.
5. Gần như bắt buộc: khoảng cách cộng gộp chênh lệch mọi feature → feature thang đo lớn áp đảo. Ngoại lệ: feature cùng thang đo và cùng mức quan trọng (pixel).
6. Euclid mặc định cho feature số đã scale; Manhattan ít nhạy với chênh lệch lớn ở một chiều; cosine khi chỉ quan tâm hướng (văn bản, embedding); Hamming/Jaccard cho dữ liệu nhị phân; Gower cho dữ liệu hỗn hợp. Chọn bằng CV nếu không chắc.
7. One-hot + scale (cân nhắc trọng số), ordinal encoding cho biến thứ bậc, hoặc metric chuyên dụng (Hamming, Gower).
8. Train $O(1)$ (hoặc $O(n\log n)$ dựng cây); dự đoán $O(nd)$ mỗi điểm với brute force; bộ nhớ $O(nd)$ — phải giữ toàn bộ dữ liệu.
9. KD-tree/Ball tree (ít chiều), approximate NN (HNSW, FAISS, Annoy), giảm mẫu (condensed NN, prototype), giảm chiều (PCA), lượng tử hoá vector.
10. Khoảng cách gần nhất và xa nhất gần như bằng nhau → "láng giềng" mất ý nghĩa; cần lượng dữ liệu tăng theo hàm mũ của số chiều để phủ không gian.
11. Rất nhạy với feature vô ích (mỗi feature đóng góp đều vào khoảng cách). Outlier ảnh hưởng cục bộ ($k = 1$ nhạy nhất; $k$ lớn giảm bớt). Mất cân bằng: lớp đa số thắng phiếu → hạ ngưỡng trên `predict_proba`, `weights="distance"`, resampling.
12. Không — dự đoán luôn nằm trong khoảng $[\min y, \max y]$ của tập train; ngoài vùng dữ liệu, dự đoán phẳng.
13. Thô: chỉ nhận các giá trị $0, 1/k, 2/k, \dots$; thường không calibrated. Dùng được để xếp hạng (ROC-AUC); cần calibration (chương 37) nếu dùng như xác suất thật.
14. Ít chiều, dữ liệu vừa phải, ranh giới phức tạp/cục bộ, có metric ý nghĩa, cần baseline nhanh, cần giải thích bằng ví dụ ("giống các trường hợp này"), dữ liệu cập nhật liên tục (thêm mẫu không cần train lại).
15. Nhiều chiều/nhiều feature vô ích, dữ liệu rất lớn với yêu cầu độ trễ thấp, cần ngoại suy, feature không có thước đo khoảng cách tự nhiên, dữ liệu dạng bảng mà Gradient Boosting thường tốt hơn hẳn.

</details>
