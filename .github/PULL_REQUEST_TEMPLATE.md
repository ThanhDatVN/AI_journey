## Thay đổi

<!-- Sửa gì, vì sao. Liên kết issue nếu có: "Closes #12". -->

## Loại

- [ ] Sửa lỗi notebook / code
- [ ] Sửa nội dung (kiến thức, công thức, chính tả, link)
- [ ] Bài học, bài tập hoặc project mới
- [ ] Công cụ, test, CI

## Kiểm tra trước khi gửi

<!-- Xem CONTRIBUTING.md: file sinh ra (bài tập người học, quiz JSON, mục lục) phải được sinh lại từ nguồn, không sửa tay. -->

- [ ] `ruff check .` và `pytest` đều qua
- [ ] `python tools/build_quizzes.py --check`, `python tools/make_exercises.py --check`, `python tools/generate_index.py --check` đều qua
- [ ] Đã chạy lại notebook bị sửa: `python tools/run_notebooks.py <thư mục / notebook>`
- [ ] Notebook commit không có output (`nbstripout` / `pre-commit`)
- [ ] Sửa lời giải checkpoint → bản trong `solutions/checkpoints/` vẫn in `STATUS: ✅ PASS`
- [ ] Sửa `projects/08_end_to_end_ml_system/fraud_service/` → đã chạy `pytest` trong thư mục đó
