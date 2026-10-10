# Giảm chiều — Bạn đã thực sự "biết" PCA, t-SNE, UMAP?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | PCA giải quyết bài toán gì? Tìm hướng nào? |
| 2 | Thành phần chính liên hệ với trị riêng/vector riêng và SVD thế nào? |
| 3 | Vì sao phải trừ trung bình? Khi nào cần chuẩn hoá? |
| 4 | Chọn số thành phần thế nào? |
| 5 | PCA dùng để làm gì trong thực tế? |
| 6 | PCA có dùng nhãn không? Khi nào PCA làm hại bài toán dự đoán? |
| 7 | Diễn giải thành phần chính (loadings) thế nào? |
| 8 | Độ phức tạp của PCA? Dữ liệu lớn/thưa thì sao? |
| 9 | PCA khác Linear Regression ra sao? |
| 10 | t-SNE hoạt động thế nào? Vì sao dùng Student-t? |
| 11 | Perplexity ảnh hưởng gì? |
| 12 | Những gì trên biểu đồ t-SNE không đáng tin? |
| 13 | UMAP khác t-SNE ở đâu? |
| 14 | Có nên dùng toạ độ t-SNE/UMAP làm feature? |
| 15 | Khi nào không nên giảm chiều? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Biểu diễn dữ liệu bằng ít chiều hơn mà giữ nhiều thông tin (phương sai) nhất; tìm các hướng trực giao có phương sai lớn nhất.
2. Thành phần chính = vector riêng của ma trận hiệp phương sai (trị riêng = phương sai dọc hướng đó); qua SVD $X_c = U\Sigma V^\top$: cột của $V$, $\lambda = \sigma^2/(n - 1)$.
3. Để mô tả biến thiên quanh trung bình (không phải vị trí). Chuẩn hoá khi feature khác đơn vị/thang đo; không cần khi cùng đơn vị có ý nghĩa (pixel).
4. Phương sai tích luỹ (90–99%), scree plot (khuỷu), hoặc CV theo hiệu năng bài toán phía sau.
5. Trực quan hoá, nén, khử nhiễu, tiền xử lý (giảm chiều cho KNN/SVM, giảm đa cộng tuyến), phát hiện bất thường (sai số tái tạo — chương 22).
6. Không. Nếu tín hiệu dự đoán nằm ở hướng phương sai nhỏ, PCA vứt bỏ nó → luôn kiểm chứng bằng CV; cân nhắc phương pháp có giám sát (PLS, LDA, feature selection).
7. Loadings = trọng số feature trong thành phần; thành phần có trọng số lớn ở một nhóm feature liên quan → đặt tên ("kích thước nhà"). Dấu tuỳ ý; nên chuẩn hoá trước khi so sánh.
8. Full SVD $O(\min(n d^2, n^2 d))$; randomized SVD khi chỉ cần vài thành phần; IncrementalPCA khi không vừa bộ nhớ; TruncatedSVD cho ma trận thưa.
9. PCA tối thiểu khoảng cách vuông góc, đối xứng giữa các biến, không có target; hồi quy tối thiểu sai số dọc y, có target.
10. Xác suất tương đồng Gaussian ở không gian gốc, Student-t ở 2D, tối thiểu KL. Student-t đuôi nặng giảm crowding problem: điểm vừa xa có thể đặt xa ở 2D.
11. ≈ số láng giềng hiệu dụng: nhỏ → cấu trúc cục bộ vụn, có thể "cụm" giả; lớn → mượt, toàn cục hơn.
12. Kích thước cụm, khoảng cách giữa cụm, vị trí tuyệt đối, hình dạng; kết quả phụ thuộc seed và perplexity. Đáng tin nhất: láng giềng cục bộ.
13. UMAP nhanh hơn, giữ cấu trúc toàn cục tốt hơn, có `transform` cho điểm mới, tham số `n_neighbors`, `min_dist`.
14. t-SNE: không (không có transform, phụ thuộc seed). UMAP: có thể (có transform), nhưng thận trọng và kiểm chứng bằng CV.
15. Khi model đã xử lý tốt nhiều chiều (boosting, regularized linear), khi cần diễn giải feature gốc, khi tín hiệu nằm ở hướng phương sai nhỏ, khi dữ liệu ít chiều sẵn.

</details>
