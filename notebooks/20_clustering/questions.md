# Clustering — Bạn đã thực sự "biết" các thuật toán này?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Clustering giải quyết bài toán gì? Khác classification ở đâu? |
| 2 | K-Means tối ưu hàm gì? Thuật toán Lloyd hoạt động thế nào, vì sao hội tụ? |
| 3 | Vì sao kết quả K-Means phụ thuộc khởi tạo? k-means++ giúp gì? |
| 4 | Chọn K thế nào? Các chỉ số có luôn đồng ý không? |
| 5 | K-Means giả định gì về hình dạng cụm? Khi nào thất bại? |
| 6 | Có cần scaling không? Feature phân loại xử lý ra sao? |
| 7 | Hierarchical clustering hoạt động thế nào? Linkage ảnh hưởng gì? |
| 8 | DBSCAN định nghĩa cụm thế nào? Chọn eps, min_samples ra sao? |
| 9 | Khi nào dùng HDBSCAN thay DBSCAN? |
| 10 | Đánh giá clustering khi không có nhãn? Khi có nhãn? |
| 11 | Làm sao biết các cụm "ổn định"? |
| 12 | Diễn giải và sử dụng cụm trong thực tế thế nào? |
| 13 | Độ phức tạp của từng thuật toán? Dữ liệu lớn thì sao? |
| 14 | Clustering liên hệ thế nào với GMM, phát hiện bất thường, giảm chiều? |
| 15 | Khi nào không nên dùng clustering? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Tìm nhóm điểm "giống nhau" khi **không có nhãn** (phân khúc khách hàng, gom tài liệu, lượng tử hoá). Không có đáp án đúng để đối chiếu — kết quả phụ thuộc định nghĩa "giống nhau" và thuật toán.
2. $\min\sum_i\lVert\mathbf x_i - \boldsymbol\mu_{c_i}\rVert^2$; luân phiên gán điểm về tâm gần nhất và cập nhật tâm = trung bình; mỗi bước không tăng $J$, số cách gán hữu hạn → hội tụ (cực tiểu địa phương).
3. Bài toán không lồi → nhiều cực tiểu địa phương. k-means++ chọn tâm trải đều (xác suất ∝ $D^2$) + `n_init` lần chạy giữ nghiệm tốt nhất.
4. Elbow, silhouette, CH, DB, gap statistic, độ ổn định — thường không đồng ý; quyết định cuối dựa vào mục đích sử dụng và khả năng diễn giải/hành động.
5. Cụm lồi, đẳng hướng, kích thước/phương sai tương tự (ranh giới Voronoi). Thất bại với cụm không lồi, kéo dài, mật độ/kích thước khác nhau, outlier.
6. Cần scale (khoảng cách Euclid phụ thuộc đơn vị). Category: one-hot (cân nhắc trọng số), Gower/k-prototypes cho dữ liệu hỗn hợp.
7. Agglomerative: gộp dần hai cụm gần nhất, dendrogram. Linkage: single (bắt hình dạng dài, chaining), complete/Ward (cụm gọn), average (trung gian).
8. Cụm = vùng dày đặc (core point liên thông mật độ) + border; còn lại là nhiễu. eps từ k-distance plot; min_samples ≈ 2 × số chiều (hoặc lớn hơn khi nhiều nhiễu).
9. Khi các cụm có mật độ khác nhau, hoặc không muốn chọn eps — HDBSCAN chỉ cần `min_cluster_size`.
10. Không nhãn: silhouette, CH, DB, độ ổn định, đánh giá của chuyên gia/kinh doanh. Có nhãn tham chiếu: ARI, NMI.
11. Chạy lại trên mẫu bootstrap/dữ liệu thời điểm khác, so sánh bằng ARI; cụm không ổn định thì không nên dùng ra quyết định.
12. Profile từng cụm (trung bình/trung vị feature, kích thước), đặt tên, kiểm tra với chuyên gia; dùng làm feature, phân khúc marketing, lấy mẫu đại diện.
13. K-Means $O(nKd)$/vòng (Mini-Batch cho dữ liệu lớn); hierarchical $O(n^2)$ bộ nhớ; DBSCAN $O(n\log n)$ với index không gian (tệ hơn khi nhiều chiều); HDBSCAN tương tự.
14. K-Means = giới hạn "cứng" của GMM (chương 23); điểm xa mọi tâm / nhiễu DBSCAN dùng làm phát hiện bất thường (chương 22); giảm chiều (PCA/UMAP) trước khi phân cụm dữ liệu nhiều chiều (chương 21).
15. Khi thật ra đã có nhãn/mục tiêu rõ (dùng supervised), khi không có kế hoạch sử dụng cụm, khi dữ liệu không có cấu trúc nhóm (mọi thuật toán vẫn trả về cụm!), khi feature chưa có ý nghĩa về khoảng cách.

</details>
