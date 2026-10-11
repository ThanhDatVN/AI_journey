# Chính sách bảo mật

## Phạm vi

Đây là một khoá học: phần lớn code là notebook minh hoạ chạy trên máy người học. Báo cáo bảo mật đặc biệt hữu ích với:

- `projects/08_end_to_end_ml_system/fraud_service/`: API FastAPI, Dockerfile, cách đóng gói và nạp model (kiểm tra hash, phiên bản scikit-learn).
- `src/ml_course/`: package dùng chung, gồm `real_datasets` (tải và giải nén dữ liệu, kiểm tra MD5).
- `tools/` và `.github/workflows/`: script build và CI.
- Các bài MLOps (chương 39–41): ví dụ phục vụ model, nạp model từ file, CI/CD mẫu. Nếu ví dụ dạy một thói quen thiếu an toàn (vd `pickle` từ nguồn không tin cậy mà không cảnh báo), đó cũng là lỗi cần báo.

Chỉ nhánh `main` được hỗ trợ; bản sửa luôn được đưa vào `main`.

## Báo cáo lỗ hổng

**Đừng mở issue công khai** cho lỗ hổng bảo mật.

1. Dùng tính năng [báo cáo lỗ hổng riêng tư của GitHub](https://github.com/ThanhDatVN/AI_journey/security/advisories/new) (tab **Security → Report a vulnerability**).
2. Nếu tính năng này chưa bật, liên hệ riêng người duy trì [@ThanhDatVN](https://github.com/ThanhDatVN) qua kênh trên hồ sơ GitHub.

Báo cáo nên có: file hoặc notebook liên quan, cách tái hiện, ảnh hưởng có thể xảy ra và đề xuất sửa (nếu có).

Người duy trì sẽ phản hồi trong vòng 7 ngày và công bố bản sửa kèm ghi nhận người báo cáo, trừ khi bạn muốn ẩn danh.

## Lưu ý khi học

- Không nạp file `pickle` / `joblib` từ nguồn không tin cậy: định dạng này có thể chạy code tuỳ ý khi nạp. Chương 39 giới thiệu `skops` và ONNX làm phương án an toàn hơn.
- Không commit khoá API, mật khẩu hay dữ liệu cá nhân vào notebook. `nbstripout` chỉ xoá output, **không** xoá bí mật trong code.
