# Mẫu notebook bài học

Mọi bài học theo cùng một khung để người học biết trước mình sẽ gặp gì. Sao chép khung dưới đây khi viết bài mới (xem thêm [CONTRIBUTING.md](../CONTRIBUTING.md)).

## Khung bài học (`notebooks/<chương>/NN_<tên>.ipynb`)

```markdown
# Bài NN — <Tên bài>

**🎯 Mục tiêu**
- 3–5 gạch đầu dòng, động từ đo được: "tự cài…", "giải thích vì sao…", "chọn … cho …".
```

```python
# ⚙️ Setup — ô này do công cụ build chèn tự động (chạy được trên Colab)
```

```python
# Import: thư viện chuẩn → bên thứ ba → ml_course; seed cố định
import numpy as np
...
from ml_course.datasets import make_customer_churn
rng = np.random.default_rng(0)
```

Các phần (dùng `##` đánh số):

| Phần | Nội dung | Ghi chú |
|---|---|---|
| 1. Trực giác | ví dụ đời thường, hình vẽ, câu hỏi dẫn dắt | trước công thức |
| 2. Toán | công thức với ký hiệu theo [NOTATION.md](../NOTATION.md), giải thích từng ký hiệu ↔ biến trong code | đủ để cài đặt, không cần mọi chứng minh |
| 3. Tự code | cài bằng NumPy, từng bước nhỏ, kiểm tra trên ví dụ tính tay được | "from scratch before library" |
| 4. Thư viện | scikit-learn / PyTorch, **so khớp** với bản tự code (`np.allclose`) | |
| 5. Thí nghiệm | đổi tham số/dữ liệu, quan sát | mỗi lời bình đi kèm con số do notebook in ra |
| 6. Failure cases | khi nào phương pháp thất bại, vì sao, làm gì | dữ liệu mô phỏng có "sự thật" |
| 🧠 Kiểm tra nhanh | 4–7 câu hỏi tự trả lời | không có đáp án trong bài (đáp án ở `questions.md`) |
| 📌 Tóm tắt | 3–5 gạch đầu dòng | |
| 📚 Tài liệu tham khảo | sách, paper gốc, tài liệu thư viện | |

## Quy tắc

- **Số trong lời bình = số notebook in ra.** Sau mỗi lần sửa, chạy lại toàn bộ notebook và đọc lại lời bình.
- Seed cho mọi nguồn ngẫu nhiên; với PyTorch seed **trước** khi khởi tạo mô hình; CPU nhỏ dùng `torch.set_num_threads(1)` để tái lập.
- Dữ liệu: `../../data/sample/<file>` (bài học) hoặc `repo_path("data", "sample", "<file>")` (bài tập, lời giải — khác độ sâu thư mục); không tải từ internet.
- Thư viện ngoài core trong `try/except ImportError` với hướng dẫn `pip install -e ".[extra]"`.
- Mỗi notebook chạy dưới ~2 phút trên CPU (dùng dữ liệu nhỏ, ít epoch) — CI chạy lại toàn bộ khoá học mỗi tuần.
- Biểu đồ: một thông điệp mỗi hình, tiêu đề và trục có đơn vị, `plt.tight_layout()`.
- Commit không có output.

## Khung bài tập (`solutions/exercises/<chương>/90_exercises.ipynb`)

```markdown
## L0 — Nhớ khái niệm         (ô ✍️ trả lời + ô solution chứa đáp án)
## L1 — Giải thích bằng lời
## L2 — Tính tay              (ô code START/END + ô check_*)
## L3 — Code
## L4 — Debug                 (code lỗi có sẵn, sửa trong START/END, ô solution giải thích lỗi)
## L5 — Phân tích             (thí nghiệm + nhận xét)
## L6 — Thực tế
## L7 — Nâng cao
```

```python
def my_function(x):
    ### START CODE HERE ### (≈ 3 dòng)
    # STARTER: return None
    ...lời giải...
    ### END CODE HERE ###
```

```python
check_close(my_function(2), 4, message="mô tả điều đang kiểm tra")   # ngưỡng có biên an toàn
```

## Khung checkpoint (`solutions/quizzes/<chương>.yaml`)

Các phần: `theory` (mcq/tf/multi/text, mỗi câu có `topic`, `lesson`, `explanation`), `math` (numeric + `decimals`), `coding.tasks` (code + tests), `analysis` (prompt, rubric, sample_answer), `project` (code + tests). Xem ví dụ bất kỳ trong `solutions/quizzes/` và chạy `python tools/build_quizzes.py <chương>`; lời giải `solutions/checkpoints/<chương>.ipynb` phải in `STATUS: ✅ PASS`.
