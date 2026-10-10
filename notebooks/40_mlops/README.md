# Chương 40 · MLOps — Phục vụ, kiểm thử, giám sát, triển khai

> Chương 39 biến mô hình thành một **gói** đáng tin cậy; chương này đưa gói đó vào **vận hành**: viết dịch vụ FastAPI và đo độ trễ qua HTTP thật, tổ chức dự án với bộ kiểm thử dành riêng cho ML, đóng gói Docker và CI/CD, giám sát trôi dữ liệu/trôi khái niệm trong 12 tuần mô phỏng, rồi triển khai an toàn bằng shadow, canary, A/B test — và nhận diện vòng phản hồi khi mô hình tác động lên chính dữ liệu của nó.

## 🎯 Mục tiêu

- Hợp đồng API có phiên bản; dịch vụ FastAPI nạp mô hình một lần (kiểm tra hash), kiểm tra đầu vào bằng pydantic, dự đoán đơn lẻ/theo lô, log dự đoán; kiểm thử bằng TestClient (parity), chạy uvicorn và đo p50/p99, thông lượng.
- Dự án triển khai được: package dùng chung cho train và serve, script huấn luyện có cổng chất lượng; kiểm thử unit, dữ liệu, hiệu năng tối thiểu, bất biến, theo hướng, API; Dockerfile chuẩn; GitHub Actions; Continuous Training; mức trưởng thành MLOps.
- Giám sát nhiều tầng; trôi dữ liệu vs trôi khái niệm; PSI, KS, chi-bình-phương, bộ phân loại miền; vì sao không dùng p-value; độ trễ của nhãn, cảnh báo, chọn cửa sổ huấn luyện lại; metric Prometheus.
- Shadow, canary tự rollback, A/B test (cỡ mẫu, power, nhìn trộm), bandit; router băm ổn định; vòng phản hồi và nhóm đối chứng.

## 📋 Yêu cầu trước

[Chương 39](../39_ml_systems/README.md) (gói mô hình, registry, kiểm tra đầu vào); [Chương 38](../38_causal_machine_learning/README.md) (thí nghiệm ngẫu nhiên, CUPED); nên biết [Chương 35](../35_reinforcement_learning/README.md) (bandit) và [Chương 37](../37_uncertainty_and_calibration/README.md) (calibration).

## 🧰 Thư viện

`pip install -e ".[mlops]"`: FastAPI, uvicorn, httpx, pydantic, prometheus-client (tuỳ chọn); pytest (nhóm `dev`). Docker **tuỳ chọn**: bài 02 chỉ build image khi có Docker và đặt `ML_COURSE_RUN_DOCKER=1`. Các bài 01–02 ghi dịch vụ/dự án mẫu vào `notebooks/40_mlops/artifacts/` (đã nằm trong `.gitignore`) — hãy mở các file đó để đọc.

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Phục vụ mô hình bằng FastAPI | 📖 Bài học | [View](01_serving_with_fastapi.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/40_mlops/01_serving_with_fastapi.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/40_mlops/01_serving_with_fastapi.ipynb) |
| 02 | Bài 02 — Docker, kiểm thử ML và CI/CD | 📖 Bài học | [View](02_docker_and_ci_cd.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/40_mlops/02_docker_and_ci_cd.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/40_mlops/02_docker_and_ci_cd.ipynb) |
| 03 | Bài 03 — Giám sát mô hình và trôi dữ liệu | 📖 Bài học | [View](03_monitoring_and_drift.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/40_mlops/03_monitoring_and_drift.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/40_mlops/03_monitoring_and_drift.ipynb) |
| 04 | Bài 04 — Chiến lược triển khai: shadow, canary, A/B test | 📖 Bài học | [View](04_deployment_strategies.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/40_mlops/04_deployment_strategies.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/40_mlops/04_deployment_strategies.ipynb) |
| 90 | Bài tập — Chương 40: MLOps (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/40_mlops/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/40_mlops/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 40 · MLOps | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/40_mlops/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/40_mlops/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

Bài 01: `artifacts/churn_service/app.py` (lifespan, pydantic, `/v1/predict`, `/v1/predict/batch`, log JSONL), TestClient, uvicorn tiến trình con. Bài 02: dự án `artifacts/churn_project/` (`churn/data.py`, `features.py`, `train.py`, `api.py`, `tests/`, `Dockerfile`, `.github/workflows/ci.yml`). Bài 03: `psi`, `psi_categorical`, `ks_statistic`, `domain_classifier`, mô phỏng 12 tuần, quy tắc cảnh báo, Prometheus. Bài 04: `bucket`/`assign`, shadow, `run_canary`, `sample_size`, mô phỏng nhìn trộm, Thompson sampling, vòng phản hồi. Bài tập: PSI, API Iris, router lỗi `hash()`, chọn cửa sổ huấn luyện lại, báo cáo A/B, trôi trong phân phối đồng thời.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS (PSI, gán biến thể ổn định), phân tích (rubric) PASS, mini project PASS (báo cáo trôi theo tuần cảnh báo đúng tuần).

## 🔗 Tiếp theo

[41 · Capstone Projects](../41_capstone_projects/README.md)
