# Chương 41 · Capstone Projects — Dự án tổng kết

> Chương cuối ghép mọi thứ thành một **dự án ML như ngoài thực tế**: bắt đầu từ một quyết định kinh doanh, dữ liệu thô bẩn, nhãn trễ, và kết thúc bằng một hệ thống có thể vận hành cùng báo cáo trung thực. Bài 01 là hướng dẫn (chọn đề tài, lộ trình 6 tuần, design doc, khung dự án, rubric, kiểm tra cờ đỏ); bài 02 là một **capstone mẫu hoàn chỉnh** — chấm điểm tín dụng từ dữ liệu thô — trong đó mô hình "tốt trên validation" gặp cú sốc kinh tế và ta học cách phát hiện, khắc phục.

## 🎯 Mục tiêu

- Phân biệt capstone với bài thi; chọn đề tài (10 gợi ý), lập kế hoạch 6 tuần với sản phẩm cụ thể; viết design doc trước khi code.
- Dựng khung dự án chuẩn bằng code; tự chấm theo rubric 8 tiêu chí; kiểm tra cờ đỏ tự động trước khi nộp.
- Walkthrough đầu–cuối: kiểm tra dữ liệu thô, làm sạch có hợp đồng dữ liệu, EDA hướng câu hỏi, chia theo thời gian, nhận diện cột rò rỉ sau quyết định, baseline nghiệp vụ, chọn mô hình có khoảng tin cậy, chính sách theo lợi nhuận kỳ vọng, đánh giá test một lần, hậu kiểm calibration, hiệu chỉnh bằng nhãn sớm, lý do từ chối, kiểm tra theo nhóm, đóng gói, kế hoạch giám sát, báo cáo.

## 📋 Yêu cầu trước

Toàn bộ khoá học, đặc biệt [Chương 14](../14_model_evaluation/README.md), [Chương 15](../15_cross_validation/README.md), [Chương 19](../19_imbalanced_learning/README.md), [Chương 36](../36_explainable_ai/README.md), [Chương 37](../37_uncertainty_and_calibration/README.md), [Chương 39](../39_ml_systems/README.md), [Chương 40](../40_mlops/README.md).

## 🧰 Thư viện

scikit-learn, pandas, matplotlib, scipy. Dữ liệu: `data/sample/credit_applications_raw.csv` (sinh bởi `make_credit_applications(messy=True)` trong `ml_course.datasets`; bản sạch dùng để kiểm chứng hàm làm sạch). Bài 01 ghi khung dự án mẫu vào `notebooks/41_capstone_projects/artifacts/` (đã nằm trong `.gitignore`).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Hướng dẫn dự án Capstone | 📖 Bài học | [View](01_capstone_guide.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/41_capstone_projects/01_capstone_guide.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/41_capstone_projects/01_capstone_guide.ipynb) |
| 02 | Bài 02 — Capstone mẫu: chấm điểm tín dụng từ dữ liệu thô | 📖 Bài học | [View](02_capstone_walkthrough.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/41_capstone_projects/02_capstone_walkthrough.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/41_capstone_projects/02_capstone_walkthrough.ipynb) |
| 90 | Bài tập — Chương 41: Capstone (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/41_capstone_projects/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/41_capstone_projects/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 41 · Capstone Projects | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/41_capstone_projects/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/41_capstone_projects/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

Bài 01: `DESIGN_DOC`, `scaffold_project`, `RUBRIC` + `grade`, `audit_project`. Bài 02: `audit_raw`, `clean_applications`, `validate_clean`, `add_features`, `make_model`, `paired_bootstrap_auc`, `loan_economics`/`policies`/`policy_table`, `odds_multiplier_for_mean` (hiệu chỉnh hệ số chặn), suy ngược tỉ lệ từ nhãn sớm, `reason_codes`, `group_report`, `score_application`, `psi`. Bài tập: đọc ngày/số tiền, chính sách lợi nhuận, sửa đánh giá rò rỉ, stress test, calibration theo nhóm, backtest có độ trễ nhãn.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini capstone PASS (làm sạch dữ liệu thô, không rò rỉ, chính sách theo mô hình lãi hơn luật hiện tại và "duyệt tất cả").

## 🎓 Sau khoá học

Chọn một đề tài ở bài 01 (hoặc dữ liệu của bạn), làm theo lộ trình 6 tuần, tự chấm bằng `grade(...)` và `audit_project(...)`. Các project ở thư mục [`projects/`](../../projects/) là bài luyện tập trước khi làm capstone.
