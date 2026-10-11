# Đóng góp cho khoá học

Cảm ơn bạn muốn làm khoá học tốt hơn! Báo lỗi (sai kiến thức, notebook không chạy, câu hỏi mơ hồ), đề xuất bài học, hay sửa chính tả đều quý.

## Báo lỗi

[Mở issue](https://github.com/ThanhDatVN/AI_journey/issues/new/choose) theo mẫu phù hợp — notebook chạy lỗi, sai sót nội dung, đề xuất nội dung, câu hỏi khi học. Với lỗi chạy code, mẫu sẽ hỏi: notebook và ô bị lỗi, thông báo lỗi đầy đủ, hệ điều hành, phiên bản Python và thư viện (`pip list | grep -iE "numpy|pandas|scikit|torch"`), local hay Colab. Lỗ hổng bảo mật thì báo riêng theo [SECURITY.md](SECURITY.md).

## Môi trường phát triển

```bash
uv sync --group dev --extra all        # hoặc: pip install -e ".[all]" pytest ruff nbformat nbclient nbstripout pre-commit
pre-commit install                     # xoá output notebook, ruff, kiểm tra quiz trước mỗi commit
```

## Nguồn sự thật và công cụ sinh

Nhiều file được **sinh ra** — sửa nguồn rồi chạy công cụ, đừng sửa file sinh:

| Muốn sửa | Sửa ở | Rồi chạy |
|---|---|---|
| Bài học | `notebooks/<chương>/NN_*.ipynb` | `python tools/run_notebooks.py notebooks/<chương>` |
| Bài tập L0–L7, assignment | **lời giải** `solutions/exercises/<chương>/9N_*.ipynb` | `python tools/make_exercises.py <chương>` → sinh `notebooks/<chương>/9N_*.ipynb` |
| Project, luyện tập xen kẽ | `solutions/projects/…`, `solutions/practice/…` | `python tools/make_exercises.py` → sinh `projects/…`, `exercises/…` |
| Checkpoint chương | `solutions/quizzes/<chương>.yaml` | `python tools/build_quizzes.py <chương>` → `quizzes/chapter_quizzes/*.json`, `99_checkpoint.ipynb`, `solutions/checkpoints/*.ipynb` |
| Bài thi tổng hợp | `solutions/exams/<bài thi>.yaml` | `python tools/build_quizzes.py` → `quizzes/final_exams/*` |
| Mục lục chương, bản đồ khoá học, lịch học | `tools/schedule.yaml`, tiêu đề notebook | `python tools/generate_index.py` |
| Dữ liệu mẫu | `src/ml_course/datasets.py` | `python tools/make_sample_data.py` |
| Link Colab/nbviewer | `pyproject.toml` → `[tool.ml-course] github_repo` | `python tools/set_github_repo.py <owner>/<repo>` |

Quy tắc xoá lời giải (`make_exercises`): ô có tag `solution` bị bỏ; code giữa `### START CODE HERE ### (≈ N dòng)` và `### END CODE HERE ###` được thay bằng khung trống — dòng `# STARTER: <code>` trong khối được giữ làm khung (vd chữ ký hàm); nếu không có, mỗi phép gán `x = …` thành `x = None`.

## Quy ước viết bài

- Văn bản **tiếng Việt có dấu**; tên biến, hàm, thuật ngữ kỹ thuật phổ biến giữ tiếng Anh (lần đầu kèm giải thích tiếng Việt — xem [GLOSSARY.md](GLOSSARY.md)). Ký hiệu toán theo [NOTATION.md](NOTATION.md).
- Cấu trúc bài học theo [docs/NOTEBOOK_TEMPLATE.md](docs/NOTEBOOK_TEMPLATE.md): mục tiêu → trực giác → toán → tự code → thư viện → failure cases → kiểm tra nhanh → tóm tắt → tài liệu.
- **Mọi con số trong lời bình phải do notebook in ra** — chạy lại và đối chiếu sau mỗi lần sửa. Đặt seed cho mọi nguồn ngẫu nhiên (seed **trước** khi khởi tạo mô hình PyTorch).
- Chỉ dùng dữ liệu offline (`data/sample/`, `ml_course.datasets`, bộ dữ liệu có sẵn trong scikit-learn). Thư viện ngoài core đặt trong `try/except ImportError` với hướng dẫn cài extra.
- Kiểm tra tự động (`ml_course.checks`): ngưỡng có **biên an toàn** (đo nhiều seed nếu cần), tránh kiểm tra mong manh theo một seed.
- Notebook commit **không có output** (`nbstripout`).
- Công thức LaTeX: sửa bằng trình soạn thảo, không qua chuỗi shell/Python thường (backslash `\t`, `\f`, `\a` dễ bị biến thành ký tự điều khiển).

## Trước khi gửi pull request

```bash
ruff check .                     # code + mọi notebook (gồm cả tên chưa định nghĩa trong notebook người học)
pytest
python tools/build_quizzes.py --check && python tools/make_exercises.py --check && python tools/generate_index.py --check
python tools/run_notebooks.py <các thư mục bạn đã sửa> solutions/checkpoints/<chương>.ipynb
```

Notebook deep learning chạy lâu trên CPU? Có thể để CI chạy giúp: **Actions → notebook-check → Run workflow**, điền `targets` (vd `notebooks/29_cnn solutions/checkpoints/29_cnn.ipynb`). Lời giải checkpoint (`solutions/checkpoints/*.ipynb`) phải in `STATUS: ✅ PASS`. Sửa `projects/08_end_to_end_ml_system/fraud_service/` thì chạy thêm `pytest` trong thư mục đó. Thay đổi đáng chú ý — nhất là khi đổi nội dung bài học hoặc đáp án checkpoint — ghi vào [CHANGELOG.md](CHANGELOG.md), mục *Chưa phát hành*.

## Quy tắc ứng xử

Dự án áp dụng [Quy tắc ứng xử](CODE_OF_CONDUCT.md) điều chỉnh từ Contributor Covenant 2.1 (như khoá gốc AIVN). Tóm tắt: dùng ngôn ngữ thân thiện, tôn trọng quan điểm và kinh nghiệm khác nhau, đón nhận góp ý mang tính xây dựng, đặt lợi ích của người học lên trước. Không chấp nhận lời lẽ xúc phạm, quấy rối, công kích cá nhân hay công khai thông tin riêng tư của người khác. Người duy trì dự án có quyền chỉnh sửa hoặc từ chối đóng góp vi phạm các nguyên tắc này.

## Ghi nhận

Khoá học kế thừa khoá [AIVN-Machine-Learning](https://github.com/thanhhff/AIVN-Machine-Learning) của **AIVIETNAM**
