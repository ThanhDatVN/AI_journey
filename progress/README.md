# 📈 Tiến độ học

Mỗi lần bạn chạy ô cuối của một `99_checkpoint.ipynb` (hoặc một bài thi tổng hợp), kết quả được ghi vào `progress/progress.json`:

- lần làm **tốt nhất** và **gần nhất** của từng chương, kèm thời điểm;
- lịch sử 20 lần làm gần nhất (điểm từng phần: lý thuyết, toán, coding, phân tích, project);
- thời điểm PASS lần đầu.

```bash
python -m ml_course.progress                  # in bảng tiến độ theo giai đoạn + chương tiếp theo
python -m ml_course.progress --update-readme  # ghi bảng tiến độ vào khối PROGRESS trong README.md
```

`progress.json` **không** nằm trong `.gitignore` — hãy commit nó trong fork của bạn để lưu lịch sử học tập (và khoe tiến độ trên README). Muốn làm lại từ đầu: xoá file này.

Khi chạy notebook tự động (CI, `tools/run_notebooks.py`), biến môi trường `ML_COURSE_NO_RECORD=1` tắt việc ghi để không làm bẩn tiến độ thật.
