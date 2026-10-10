# Project 08 — Hệ thống phát hiện gian lận end-to-end

> Ví điện tử xử lý hàng chục nghìn giao dịch thẻ mỗi ngày, ~1% là gian lận. Duyệt nhầm giao dịch gian lận → mất toàn bộ số tiền; chuyển kiểm duyệt thủ công → tốn 15.000đ. Bạn xây **toàn bộ hệ thống**: mô hình, chính sách quyết định theo chi phí, dịch vụ có kiểm thử, API, Docker, giám sát và kế hoạch triển khai.

| | |
|---|---|
| Bài toán | Phân loại cực mất cân bằng + quyết định theo chi phí + MLOps |
| Dữ liệu | `data/sample/fraud_transactions.csv` — 30.000 giao dịch, 7 đặc trưng, 1% gian lận |
| Chương nền tảng | 14, 19, 37, 39, 40 |
| Độ khó · thời gian | ⭐⭐⭐⭐⭐ · 12–20 giờ |

## 🏗️ Kiến trúc

```
fraud_transactions.csv ─► fraud.data.validate ─► fraud.train ──(cổng chất lượng)──► models/current/{model.joblib, metadata.json}
                                                                                          │
           client ──JSON──► fraud.api (FastAPI, kiểm tra hash) ──► p, tổn thất kỳ vọng ──► approve / review
                                                                                          │
           giao dịch tuần mới ──► fraud.monitoring (PSI so với mốc trong metadata) ──► ok / WARN / ALERT
```

## 🎯 Mục tiêu học tập
- Metric đúng cho dữ liệu cực lệch (PR-AUC) và vì sao **calibration** quyết định giá trị.
- Chính sách **tổn thất kỳ vọng** với ngưỡng phụ thuộc số tiền; so sánh bằng tiền.
- Tổ chức mã nguồn sản phẩm: package, CLI huấn luyện có cổng chất lượng, API, giám sát, bộ test (dữ liệu, mô hình, API, giám sát), Docker, Makefile.

## 📋 Nhiệm vụ
1. `project.ipynb` bước 1–3: EDA, so sánh mô hình (PR-AUC + calibration), chính sách chi phí.
2. Đọc mã nguồn `fraud_service/` (bản tham chiếu). **Tự viết** dịch vụ của bạn trong `my_service/` theo cùng cấu trúc — bắt đầu từ `config.py` → `features.py` → `data.py` → `model.py` → `train.py` → `api.py` → `monitoring.py`, viết test song song.
3. `project.ipynb` bước 4–6: chạy huấn luyện + test, gọi API, giám sát một tuần dữ liệu trôi.
4. Kế hoạch triển khai (shadow → canary → so sánh chi phí → toàn bộ) và báo cáo cho giám đốc rủi ro.

## ✅ Tiêu chí đạt
- Mô hình chọn: PR-AUC validation > 0.06, xác suất TB lệch < 50% so với tỉ lệ thật.
- Chính sách tổn thất kỳ vọng rẻ nhất trong các chính sách, rẻ hơn "duyệt tất cả" ≥ 40% (validation và test).
- `python -m fraud.train` qua cổng chất lượng; `pytest` của dịch vụ đạt toàn bộ; giám sát phát hiện trôi số tiền.

## ▶️ Chạy dịch vụ tham chiếu

```bash
cd projects/08_end_to_end_ml_system/fraud_service
make train          # = python -m fraud.train  → models/current/
make test           # = python -m pytest  (21 test)
make serve          # http://127.0.0.1:8000/docs
make docker-build && make docker-run      # cần Docker
```

## 🧭 Gợi ý & cạm bẫy
- `class_weight="balanced"` làm xác suất vô nghĩa cho quyết định theo chi phí — nếu dùng, phải hiệu chỉnh lại.
- Ngưỡng 0.5 gần như không bắt gì khi tỉ lệ nền 1%.
- Nhãn gian lận về trễ (chargeback vài tuần) và có vòng phản hồi: giao dịch bị chặn/kiểm duyệt có nhãn khác — giữ nhóm đối chứng.

## 🚀 Mở rộng
- PSI cho biến phân loại (`merchant_category`) trong `fraud/monitoring.py`; endpoint `/metrics` Prometheus.
- Ba quyết định (approve / review / block) với ba mức chi phí; đặc trưng tổng hợp theo thẻ (tần suất 1 giờ) bằng feature store giả lập.
- GitHub Actions cho dịch vụ (lint → train → test → build image) theo chương 40.

## 📁 Tệp
- `project.ipynb` — notebook làm bài · lời giải: [`solutions/projects/08_end_to_end_ml_system/project.ipynb`](../../solutions/projects/08_end_to_end_ml_system/project.ipynb)
- [`fraud_service/`](fraud_service/README.md) — bản tham chiếu hoàn chỉnh (đọc sau khi đã thử tự viết).
