# Project 03 — Phân khúc khách hàng (Clustering + PCA)

> Một trung tâm thương mại muốn chia khách hàng thành vài **phân khúc** để thiết kế ưu đãi riêng. Không có nhãn — bạn tự tìm cấu trúc, chọn số cụm có lý do, đặt tên dễ hiểu và đề xuất hành động.

| | |
|---|---|
| Bài toán | Học không giám sát |
| Dữ liệu | `data/sample/customer_segments.csv` — 600 khách, 4 đặc trưng; cột `true_segment` chỉ dùng ở cuối để kiểm chứng |
| Chương nền tảng | 04, 20, 21, 23 |
| Độ khó · thời gian | ⭐⭐ · 4–6 giờ |

## 🎯 Mục tiêu học tập
- Chuẩn hoá; chọn k bằng elbow, silhouette, Davies–Bouldin; hiểu vì sao inertia nhỏ nhất không phải tiêu chí.
- Kiểm tra **độ ổn định** của K-Means theo khởi tạo.
- GMM, chọn k bằng BIC, gán mềm; PCA để trực quan hoá.
- Biến cụm thành **phân khúc có tên** và hành động marketing.

## 📋 Nhiệm vụ
1. Chuẩn hoá + scatter matrix. 2. Bảng chọn k (2–8). 3. K-Means, hồ sơ cụm, ổn định (10 seed, `n_init=1`). 4. GMM + BIC, khách "lưng chừng". 5. PCA 2D, ARI với nhãn thật. 6. Bảng phân khúc + hành động.

## ✅ Tiêu chí đạt
- Silhouette chọn k = 5 (silhouette > 0.4); BIC chọn k = 5.
- ARI(K-Means) > 0.85, ARI(GMM) > 0.95; 2 thành phần PCA giữ > 80% phương sai.

## 🧭 Gợi ý & cạm bẫy
- Một lần khởi tạo K-Means có thể kẹt ở nghiệm kém — luôn `n_init ≥ 10`.
- Cụm không phải "sự thật": đánh giá bằng tính hữu ích (hành động khác nhau cho từng nhóm) và độ ổn định.

## 🚀 Mở rộng
- Phân cụm phân cấp và dendrogram; DBSCAN — có tìm ra 5 cụm không?
- Theo dõi phân khúc qua thời gian (khách chuyển phân khúc).
- Đo hiệu quả hành động theo phân khúc bằng thí nghiệm.

## 📁 Tệp
- `project.ipynb` · lời giải: [`solutions/projects/03_clustering/project.ipynb`](../../solutions/projects/03_clustering/project.ipynb).
