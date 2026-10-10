# fraud_service — dịch vụ chấm điểm gian lận (bản tham chiếu)

Mã nguồn tham chiếu cho [Project 08](../README.md). Mọi lệnh chạy từ thư mục này.

## Cấu trúc

| Tệp | Vai trò |
|---|---|
| `fraud/config.py` | Chi phí kiểm duyệt, danh sách đặc trưng, đường dẫn — một nguồn sự thật |
| `fraud/features.py` | `add_features` — dùng chung cho huấn luyện và phục vụ |
| `fraud/data.py` | `load_transactions`, `validate` (hợp đồng dữ liệu), `split` (train/valid/test phân tầng) |
| `fraud/model.py` | `build_model` (logistic, giữ calibration), `decide` (p × số tiền > chi phí), `total_cost`, `baseline_policies` |
| `fraud/train.py` | CLI: kiểm tra dữ liệu → validation + **cổng chất lượng** → huấn luyện lại → test → `model.joblib` + `metadata.json` (metric, hash, phiên bản, mốc giám sát) |
| `fraud/api.py` | FastAPI: `/health`, `/v1/score` (1–1000 giao dịch), `/v1/policy`; khi khởi động kiểm tra hash mô hình và **phiên bản scikit-learn** (lệch → từ chối chạy) |
| `fraud/monitoring.py` | PSI từng đặc trưng so với mốc huấn luyện; CLI trả mã 1 khi có ALERT |
| `tests/` | dữ liệu & đặc trưng · mô hình (chất lượng tối thiểu, chính sách thắng baseline, hướng, bất biến) · API (parity, quyết định, 422) · đóng gói (hash, phiên bản) · giám sát |
| `Dockerfile`, `.dockerignore`, `requirements.txt` | đóng gói — phiên bản ghim theo môi trường huấn luyện |
| `Makefile` | `train`, `test`, `serve`, `monitor`, `docker-build`, `docker-run`, `freeze` |

## Lệnh

```bash
make train                     # hoặc: python -m fraud.train --data <csv> --out models/current --min-pr-auc 0.06
make test                      # pytest — tự huấn luyện mô hình vào thư mục tạm, không cần `make train` trước
make serve                     # uvicorn fraud.api:app --port 8000
make monitor CURRENT=new.csv   # báo cáo PSI; mã thoát 1 nếu có ALERT (dùng được trong cron/CI)
make docker-build && make docker-run    # docker-build = train → freeze (ghim phiên bản đang dùng) → build với đúng phiên bản Python
```

Vì sao `docker-build` gọi `freeze`: `model.joblib` là pickle — chỉ nạp an toàn bằng **đúng** phiên bản scikit-learn đã huấn luyện. `make freeze` ghi phiên bản của môi trường huấn luyện vào `requirements.txt`, image cài đúng các phiên bản đó, và API kiểm tra lại khi khởi động.

Biến môi trường: `FRAUD_MODEL_DIR` — thư mục mô hình cho API (mặc định `models/current`, đã nằm trong `.gitignore`).

## Ví dụ request

```bash
curl -X POST http://127.0.0.1:8000/v1/score -H "Content-Type: application/json" -d '[{"amount_vnd": 9500000, "hour": 2, "merchant_category": "electronics", "distance_from_home_km": 35, "is_foreign": true, "n_tx_last_24h": 6, "card_age_days": 40}]'
# → [{"fraud_probability": 0.97, "expected_loss_vnd": 9236812, "decision": "review", "model_version": "1.0.0"}]
```
