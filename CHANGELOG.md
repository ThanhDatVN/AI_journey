# Nhật ký thay đổi

Ghi lại các thay đổi đáng chú ý của khoá học. Định dạng theo [Keep a Changelog](https://keepachangelog.com/vi/1.1.0/); phiên bản theo `version` trong `pyproject.toml`.

Khi một thay đổi làm **khác nội dung bài học** (đổi công thức, đáp án checkpoint, cách chấm), hãy ghi rõ chương bị ảnh hưởng để người đang học biết cần xem lại.

## [Chưa phát hành]

### Thêm
- Mẫu issue (lỗi notebook, sai sót nội dung, đề xuất nội dung, câu hỏi), mẫu pull request, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.cff`.
- CI notebook-check báo notebook lỗi bằng annotation và job summary (xem được mà không cần mở log).

### Thay đổi
- `LICENSE` chỉ còn giấy phép MIT của khoá học; thông báo bản quyền AIVN-Machine-Learning chuyển sang `THIRD_PARTY_NOTICES.md`.
- `ROADMAP.md`: tách "Trustworthy ML" và "ML Systems & MLOps" thành hai giai đoạn, khớp với bản đồ khoá học (10 giai đoạn).

## [1.0.0] — 2026-10-10

Phiên bản đầu tiên trên `ThanhDatVN/AI_journey`, thay toàn bộ nội dung cũ của repo.

### Thêm
- 42 chương, 178 bài học trong 10 giai đoạn: Python và toán → nền tảng ML → học có giám sát → đánh giá và ML thực hành → không giám sát → ML cổ điển nâng cao → Deep Learning (PyTorch, TensorFlow/Keras) → ML đáng tin cậy → hệ thống ML và MLOps.
- Mỗi chương: README, bài tập L0–L7, `questions.md`, checkpoint tự chấm (PASS / REVIEW) và lời giải trong `solutions/`; 12 assignment kiểu "≈ N dòng".
- 8 project (project 08 là hệ thống phát hiện gian lận có package, CLI, API FastAPI, test và Docker), capstone, 6 bài thi tổng hợp, 9 notebook luyện tập xen kẽ.
- Package `ml_course`: dữ liệu mô phỏng có "sự thật" đã biết, tải Fashion-MNIST / CIFAR-10 có kiểm tra MD5, engine checkpoint, theo dõi tiến độ.
- Tài liệu: `STUDY_GUIDE.md`, `ROADMAP.md`, `SCHEDULE.md` (44 tuần), `NOTATION.md`, `GLOSSARY.md`, cheat sheet, `docs/AIVN_MAPPING.md`.
- CI: unit test (Python 3.10, 3.12), kiểm tra file sinh đồng bộ với nguồn, build Docker + smoke test cho fraud_service, chạy lại toàn bộ notebook mỗi tuần.

### Ghi nhận
- Kế thừa, viết lại và mở rộng khoá [AIVN-Machine-Learning](https://github.com/thanhhff/AIVN-Machine-Learning) (MIT, © 2019 Nguyen Trung Thanh / AIVIETNAM). Toàn bộ 39 notebook gốc và các mục "nội dung tiếp theo" trong README gốc đều có bài tương ứng.

[Chưa phát hành]: https://github.com/ThanhDatVN/AI_journey/compare/1f7ad2f...main
[1.0.0]: https://github.com/ThanhDatVN/AI_journey/tree/1f7ad2f
