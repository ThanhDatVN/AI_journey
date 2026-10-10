# Chương 03 · Data Analysis & EDA

> Bài 01 dựa trên *Tutorial 3: Làm việc với dữ liệu* (thư mục Archives) của khoá AIVN-Machine-Learning; các bài 02–04 là phần mở rộng về phân tích dữ liệu khám phá.

## 🎯 Mục tiêu

- Phân biệt dữ liệu có cấu trúc / không có cấu trúc và biến chúng thành vector đặc trưng (bảng, ảnh, văn bản, chuỗi thời gian).
- Chọn định dạng lưu trữ phù hợp (CSV, Parquet, NumPy, HDF5, Pickle — và rủi ro bảo mật của pickle).
- Lấy dữ liệu từ nguồn bên ngoài (URL, GitHub, Google Drive, Dropbox, Kaggle) theo cách tái lập được: `data/raw/` bất biến, ghi nguồn và checksum.
- Thực hiện EDA theo quy trình 7 bước, kết thúc bằng danh sách insight – giả thuyết – rủi ro.
- Chọn đúng biểu đồ và thống kê cho từng cặp kiểu biến; phát hiện tương tác.
- Làm sạch dữ liệu bẩn có hệ thống; hiểu MCAR/MAR/MNAR, nghịch lý Simpson, dấu hiệu leakage.

## 📋 Yêu cầu trước

[Chương 01](../01_python_for_ml/README.md) (pandas, matplotlib), [Chương 02](../02_mathematics_for_ml/README.md) bài 08 (thống kê).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Làm việc với các loại dữ liệu và lưu trữ dữ liệu | 📖 Bài học | [View](01_types_of_data_and_storage.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/03_data_analysis_and_eda/01_types_of_data_and_storage.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/03_data_analysis_and_eda/01_types_of_data_and_storage.ipynb) |
| 02 | Bài 02 — Quy trình EDA (Exploratory Data Analysis) | 📖 Bài học | [View](02_eda_workflow.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/03_data_analysis_and_eda/02_eda_workflow.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/03_data_analysis_and_eda/02_eda_workflow.ipynb) |
| 03 | Bài 03 — Phân tích đơn biến, hai biến và đa biến | 📖 Bài học | [View](03_univariate_and_bivariate_analysis.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/03_data_analysis_and_eda/03_univariate_and_bivariate_analysis.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/03_data_analysis_and_eda/03_univariate_and_bivariate_analysis.ipynb) |
| 04 | Bài 04 — Chất lượng dữ liệu và các cạm bẫy khi phân tích | 📖 Bài học | [View](04_data_quality_and_pitfalls.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/03_data_analysis_and_eda/04_data_quality_and_pitfalls.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/03_data_analysis_and_eda/04_data_quality_and_pitfalls.ipynb) |
| 90 | Bài tập — Chương 03: Data Analysis & EDA (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/03_data_analysis_and_eda/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/03_data_analysis_and_eda/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 03 · Data Analysis & EDA | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/03_data_analysis_and_eda/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/03_data_analysis_and_eda/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi tự kiểm tra

[questions.md](questions.md) — trả lời không nhìn tài liệu; ôn lại sau 1 ngày, 1 tuần, 1 tháng.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project EDA PASS.

## 🔗 Tiếp theo

[04 · Data Preprocessing](../04_data_preprocessing/README.md)
