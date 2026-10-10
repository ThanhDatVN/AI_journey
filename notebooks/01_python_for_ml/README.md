# Chương 01 · Python for ML

> Tương ứng phần **Python** của Tuần 1–3 khoá AIVN-Machine-Learning (Python cơ bản, NumPy, Matplotlib), bổ sung Pandas, vectorization và debug.

Không biến thành khoá Python dài hàng tháng — **chỉ học những gì ML thực sự cần**.

## 🎯 Mục tiêu

- Viết Python thành thạo: kiểu dữ liệu, container, comprehension, hàm, class (theo mẫu `fit`/`predict`).
- Làm việc với **NumPy**: `ndarray`, shape, indexing, `axis`, broadcasting.
- Hiểu vì sao `X @ W` nhanh hơn vòng `for` hàng trăm lần (**vectorization**) — nền tảng của mọi thuật toán ML.
- Xử lý dữ liệu bảng bằng **Pandas**: lọc, `groupby`, `merge`, missing value.
- Vẽ đồ thị bằng **Matplotlib** theo phong cách hướng đối tượng (`fig, ax`).
- Debug lỗi shape, đo hiệu năng, cố định seed để tái lập kết quả.
- Đọc traceback, nhận diện và sửa những lỗi Python thường gặp; xử lý lỗi đúng cách (`try/except`, `raise`, `assert`).

## 📋 Yêu cầu trước

[Chương 00](../00_course_orientation/README.md). Chưa từng học Python? Hãy làm thêm một khoá Python cơ bản ngắn trước (vd: [Python Tutorial](https://docs.python.org/3/tutorial/)).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Python cơ bản cho Machine Learning | 📖 Bài học | [View](01_python_basics.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/01_python_for_ml/01_python_basics.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/01_python_for_ml/01_python_basics.ipynb) |
| 02 | Bài 02 — NumPy | 📖 Bài học | [View](02_numpy.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/01_python_for_ml/02_numpy.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/01_python_for_ml/02_numpy.ipynb) |
| 03 | Bài 03 — Vectorization & Broadcasting: vì sao `X @ W` thay cho `for` | 📖 Bài học | [View](03_numpy_vectorization_and_broadcasting.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/01_python_for_ml/03_numpy_vectorization_and_broadcasting.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/01_python_for_ml/03_numpy_vectorization_and_broadcasting.ipynb) |
| 04 | Bài 04 — Pandas cho Machine Learning | 📖 Bài học | [View](04_pandas_essentials.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/01_python_for_ml/04_pandas_essentials.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/01_python_for_ml/04_pandas_essentials.ipynb) |
| 05 | Bài 05 — Trực quan hoá dữ liệu với Matplotlib | 📖 Bài học | [View](05_matplotlib.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/01_python_for_ml/05_matplotlib.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/01_python_for_ml/05_matplotlib.ipynb) |
| 06 | Bài 06 — Debug lỗi shape, NaN và tối ưu hiệu năng | 📖 Bài học | [View](06_debugging_shapes_and_performance.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/01_python_for_ml/06_debugging_shapes_and_performance.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/01_python_for_ml/06_debugging_shapes_and_performance.ipynb) |
| 07 | Bài 07 — Những lỗi thường gặp trong Python (và cách sửa) | 📖 Bài học | [View](07_common_python_errors.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/01_python_for_ml/07_common_python_errors.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/01_python_for_ml/07_common_python_errors.ipynb) |
| 90 | Bài tập — Chương 01: Python for ML (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/01_python_for_ml/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/01_python_for_ml/90_exercises.ipynb) |
| 91 | Assignment 91 — Cài đặt công thức toán bằng Python | 💻 Assignment | [View](91_assignment_math_formulas_in_python.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/01_python_for_ml/91_assignment_math_formulas_in_python.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/01_python_for_ml/91_assignment_math_formulas_in_python.ipynb) |
| 92 | Assignment 92 — Luyện tập NumPy (10 câu) | 💻 Assignment | [View](92_assignment_numpy_practice.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/01_python_for_ml/92_assignment_numpy_practice.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/01_python_for_ml/92_assignment_numpy_practice.ipynb) |
| 93 | Assignment 93 — Python Basics With NumPy cho ML / DL | 💻 Assignment | [View](93_assignment_python_basics_with_numpy.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/01_python_for_ml/93_assignment_python_basics_with_numpy.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/01_python_for_ml/93_assignment_python_basics_with_numpy.ipynb) |
| 94 | Assignment 94 — Python Matplotlib | 💻 Assignment | [View](94_assignment_matplotlib.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/01_python_for_ml/94_assignment_matplotlib.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/01_python_for_ml/94_assignment_matplotlib.ipynb) |
| 99 | ✅ Checkpoint — 01 · Python for ML | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/01_python_for_ml/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/01_python_for_ml/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## 🧭 Bản đồ kiến thức → ML

| Kỹ năng | Dùng ở đâu trong ML |
|---|---|
| list/dict comprehension | xử lý dữ liệu, tạo feature, gom kết quả thí nghiệm |
| class với `fit` / `predict` | mọi thuật toán from scratch (chương 06+) |
| broadcasting | chuẩn hoá dữ liệu, tính khoảng cách (KNN), softmax |
| `X @ W` | Linear/Logistic Regression, Neural Network |
| `axis`, `keepdims` | mean/std theo feature, softmax theo hàng |
| boolean indexing | lọc dữ liệu, tính metric (TP/FP), chọn mẫu theo lớp |
| `groupby` | EDA, target encoding, feature tổng hợp |

## ❓ Bộ câu hỏi tự kiểm tra

[questions.md](questions.md) — trả lời không nhìn tài liệu; ôn lại sau 1 ngày, 1 tuần, 1 tháng.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math (shape/broadcasting) ≥ 80%, coding PASS, mini project PASS.

## 📎 Cheat sheet

- [Python 3 reference cheat sheet (PDF)](../../docs/cheatsheets/Python3_reference_cheat_sheet.pdf) — Laurent Pointal
- [Matplotlib cheat sheet (PDF)](../../docs/cheatsheets/Matplotlib_Cheat_Sheet.pdf) — DataCamp
- [NumPy / Pandas / Matplotlib cheat sheet của khoá học](../../docs/cheatsheets/README.md)

## 🔗 Tiếp theo

[02 · Mathematics for ML](../02_mathematics_for_ml/README.md)

---
<sub>Bài 01, 02, 05 và các assignment 91–94 dựa trên [AIVN-Machine-Learning](https://github.com/thanhhff/AIVN-Machine-Learning) (MIT License, © Nguyen Trung Thanh / AIVIETNAM; nội dung gốc tham khảo CS228/CS231n Python tutorial và Coursera Deep Learning), đã được cập nhật và bổ sung. Bài 07 hiện thực hoá mục *Những lỗi thường gặp trong Python* trong kế hoạch tuần 1 của khoá gốc (bản gốc chỉ có liên kết ngoài).</sub>
