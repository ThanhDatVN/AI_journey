# Project 05 — Gợi ý phim (Recommender System)

> Một ứng dụng xem phim muốn hiển thị **10 phim gợi ý** cá nhân hoá trên trang chủ thay cho danh sách "phổ biến nhất" chung cho mọi người.

| | |
|---|---|
| Bài toán | Gợi ý top-N từ tương tác |
| Dữ liệu | `data/sample/ratings.csv` (5.441 lượt chấm, 300 người dùng, 200 phim, có thời điểm), `items.csv` (thể loại) |
| Chương nền tảng | 21, 26, 14, 15 |
| Độ khó · thời gian | ⭐⭐⭐ · 6–8 giờ |

## 🎯 Mục tiêu học tập
- Nhận ra thiên lệch chọn lọc (MNAR) trong dữ liệu điểm số; coi bài toán là xếp hạng top-N.
- Chia **theo thời gian cho từng người dùng**; tự cài Precision@k, Recall@k, NDCG@k, độ phủ danh mục.
- Phổ biến vs item-kNN vs PureSVD (chọn hạng bằng validation); gợi ý cho người dùng mới; đo thiên lệch phổ biến.

## 📋 Nhiệm vụ
1. EDA đuôi dài. 2. `temporal_split()`. 3. `evaluate()`. 4. Ba phương pháp + chọn k. 5. Thể loại cho người mới, thiên lệch phổ biến. 6. Báo cáo.

## ✅ Tiêu chí đạt
- `evaluate()` đúng trên ví dụ nhỏ (loại phim đã xem, NDCG chuẩn hoá).
- item-kNN tốt hơn phổ biến ≥ 15% NDCG@10; PureSVD (k chọn trên validation, nhỏ) tốt hơn phổ biến và phủ danh mục ≥ 2 lần.

## 🧭 Gợi ý & cạm bẫy
- Quên loại phim đã xem khỏi danh sách gợi ý → metric đẹp giả tạo.
- Chọn hạng SVD trên test là rò rỉ — dùng validation tách từ train.
- Offline tốt chưa chắc online tốt: cần A/B test (chương 40).

## 🚀 Mở rộng
- ALS cho phản hồi ẩn (implicit ALS), BPR; đa dạng hoá (MMR); giải thích "vì bạn đã xem…".
- Đánh giá theo nhóm người dùng (ít/nhiều lịch sử).

## 📁 Tệp
- `project.ipynb` · lời giải: [`solutions/projects/05_recommendation/project.ipynb`](../../solutions/projects/05_recommendation/project.ipynb).
