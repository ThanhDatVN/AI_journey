# Hệ gợi ý — Bạn đã thực sự "biết" recommender systems?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Bài toán gợi ý có những dạng nào? Phản hồi tường minh khác ngầm thế nào? |
| 2 | Vì sao popularity là baseline mạnh? Xếp hạng theo trung bình rating sai ở đâu? |
| 3 | Content-based hoạt động thế nào? Ưu và nhược? |
| 4 | Collaborative filtering là gì? User-based khác item-based ra sao? |
| 5 | Baseline predictor μ + b_u + b_i gồm gì? Vì sao quan trọng? |
| 6 | Tính độ tương đồng thế nào cho tốt (phần dư, shrinkage)? |
| 7 | Matrix factorization mô hình hoá gì? Hàm mục tiêu? |
| 8 | SGD và ALS khác nhau thế nào? Khi nào dùng ALS? |
| 9 | Vì sao không dùng thẳng SVD cho ma trận rating? PureSVD dùng được khi nào? |
| 10 | Chọn k và λ trong MF thế nào? Dấu hiệu overfitting? |
| 11 | Kể các metric xếp hạng và ý nghĩa. |
| 12 | Vì sao mô hình RMSE tốt có thể gợi ý tệ? MNAR là gì? |
| 13 | Coverage, novelty, diversity là gì? Vì sao cần? |
| 14 | Cold start là gì? Xử lý thế nào? |
| 15 | BPR tối ưu điều gì? |
| 16 | Đánh giá offline có những thiên lệch nào? Vì sao vẫn cần A/B test? |
| 17 | Kiến trúc hệ gợi ý quy mô lớn gồm những tầng nào? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Dự đoán rating, xếp hạng top-N, gợi ý theo ngữ cảnh/phiên. Tường minh: user nói rõ (rating) — ít, có tín hiệu âm; ngầm: hành vi (xem, mua) — nhiều, không có âm rõ.
2. Item phổ biến là item hay được tương tác → hay xuất hiện trong test; đúng "trung bình". Trung bình thô ưu ái item ít rating (1 rating 5★) → dùng trung bình giảm chấn.
3. Hồ sơ item (đặc trưng) → hồ sơ user (trung bình có trọng số item đã thích) → cosine. Ưu: item mới, giải thích được, không cần user khác. Nhược: phụ thuộc đặc trưng, bong bóng lọc, không xử lý user mới.
4. Dùng hành vi của nhiều user. User-based: tìm user giống; item-based: tìm item được cùng nhóm user ưa thích — ổn định hơn, tính trước được.
5. Trung bình toàn cục + thiên lệch user (chấm dễ/khó) + thiên lệch item (hay/dở), có regularization. Giải thích phần lớn biến động rating; mọi mô hình phía sau học phần dư.
6. Cosine trên phần dư (trừ thiên lệch, ≈ Pearson), shrinkage theo số user chung, giữ top-k láng giềng; với dữ liệu ngầm: cosine nhị phân, chuẩn hoá giảm thiên lệch phổ biến.
7. User và item là vector ẩn k chiều; rating ≈ μ + b_u + b_i + p_u·q_i. Tối thiểu bình phương sai số trên ô quan sát + L2.
8. SGD: cập nhật theo từng rating, linh hoạt (thêm feature, loss khác). ALS: luân phiên giải Ridge nghiệm đóng cho từng user/item — song song hoá tốt, hợp dữ liệu lớn/phản hồi ngầm.
9. Ma trận rating thiếu phần lớn ô; điền 0 coi chưa xem là ghét. PureSVD dùng cho phản hồi ngầm (0 = chưa tương tác) và mục tiêu top-N.
10. Validation (theo thời gian), early stopping. Overfitting: RMSE train giảm, test tăng lại; nhiều nhân tố với ít rating/user.
11. Precision@k, recall@k, hit rate, MRR (vị trí đúng đầu tiên), MAP (precision tại các vị trí đúng), NDCG (nhạy thứ tự, nhiều mức liên quan).
12. RMSE đo trên item user đã chọn; xếp hạng mọi item đẩy lên item user không chọn. Missing not at random: việc quan sát được phụ thuộc gu/độ phổ biến → rating quan sát lệch cao, test lệch về item phổ biến.
13. Coverage: tỉ lệ danh mục được gợi ý; novelty: mức hiếm của item gợi ý; diversity: độ khác nhau trong một danh sách. Tránh "mọi người thấy cùng thứ", hỗ trợ đuôi dài, trải nghiệm tốt hơn.
14. User/item mới chưa có tương tác. Popularity theo ngữ cảnh, onboarding (hỏi sở thích), content-based cho item mới, chuyển dần sang cá nhân hoá khi đủ tương tác.
15. Thứ hạng: item đã tương tác phải có điểm cao hơn item chưa tương tác (tối đa log σ(x_ui − x_uj)) — đúng mục tiêu top-N.
16. Dữ liệu sinh ra bởi hệ thống cũ (chỉ thấy cái đã gợi ý), MNAR, thiên lệch phổ biến, không đo tác động dài hạn. A/B test đo hành vi thật và metric kinh doanh.
17. Retrieval/ứng viên (KNN, two-tower + ANN) → ranking (boosting/mạng sâu với nhiều feature) → hậu xử lý (đa dạng, luật kinh doanh, lọc) → đánh giá online, giám sát.

</details>
