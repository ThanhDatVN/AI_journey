# 📘 Hướng dẫn học hiệu quả

Khoá học được thiết kế quanh những gì khoa học về học tập đã chứng minh hiệu quả: **tự nhớ lại** (retrieval practice), **lặp lại ngắt quãng** (spacing), **luyện xen kẽ** (interleaving), **tự giải thích** và **phản hồi ngay**. Hướng dẫn này chỉ cách tận dụng chúng.

## 1. Quy trình cho mỗi bài học (60–120 phút)

1. **Đọc mục tiêu** 🎯 ở đầu bài; tự hỏi "mình đã biết gì về chủ đề này?" (1 phút).
2. **Chạy từng ô và đoán trước** kết quả trước khi chạy. Đoán sai là lúc học nhiều nhất — dừng lại và hiểu vì sao.
3. **Gõ lại** phần "tự code" (from scratch) thay vì chỉ đọc — kể cả khi chép. Sau đó thử viết lại từ trí nhớ.
4. **Đổi một thứ** và xem chuyện gì xảy ra: tăng nhiễu, giảm dữ liệu, đổi siêu tham số, bỏ chuẩn hoá. Các phần "failure cases" là gợi ý tốt.
5. **Trả lời 🧠 Kiểm tra nhanh** cuối bài **không nhìn** lại nội dung — viết câu trả lời ra giấy.
6. **Tóm tắt** bằng 3–5 câu của bạn. Nếu không giải thích được cho người khác, bạn chưa hiểu (kỹ thuật Feynman).

## 2. Bài tập L0 → L7

| Cấp | Rèn luyện | Mẹo |
|---|---|---|
| L0 Nhớ | định nghĩa, khái niệm | trả lời trước, so đáp án sau |
| L1 Giải thích | diễn đạt bằng lời, ví dụ | viết như giải thích cho đồng nghiệp không chuyên |
| L2 Tính tay | công thức, trực giác số | dùng giấy bút, rồi mới kiểm tra bằng code |
| L3 Code | cài đặt | chạy ô kiểm tra ngay sau; đọc thông báo ❌ — có gợi ý |
| L4 Debug | đọc code người khác, tìm lỗi im lặng | hỏi: "số này có quá đẹp không?" |
| L5 Phân tích | thí nghiệm, so sánh, rút kết luận | viết nhận xét **trước** khi xem lời giải |
| L6 Thực tế | bài toán có ràng buộc kinh doanh | nghĩ về chi phí, người dùng, dữ liệu thật |
| L7 Nâng cao | mở rộng, phương pháp mới | được phép tra tài liệu |

**Quy tắc 20 phút**: bí quá 20 phút thì xem lại bài học liên quan; bí 40 phút thì mở lời giải, **đóng lại**, rồi tự viết lại từ đầu. Đánh dấu bài đó để làm lại sau một tuần.

## 3. Checkpoint: PASS / REVIEW

- Làm `99_checkpoint.ipynb` khi đã xong bài tập. Không mở tài liệu.
- **PASS** (lý thuyết ≥ 80%, toán ≥ 80%, coding + project đạt, phân tích tự chấm ≥ 80%) → sang chương tiếp theo.
- **REVIEW** → checkpoint liệt kê **chủ đề yếu** và bài cần xem lại. Ôn đúng các phần đó, đợi ít nhất một ngày, làm lại. Đừng học tiếp trên nền móng hổng — các chương sau giả định chương trước.
- Phần phân tích tự chấm theo rubric: **trung thực** với chính mình — đây là kỹ năng được hỏi nhiều nhất khi phỏng vấn.

## 4. Ôn tập ngắt quãng

Từ chương 02, mỗi chương có `questions.md` — bộ câu hỏi tự kiểm tra ("biết model thật sự" ở các chương mô hình). Lịch ôn:

| Khi nào | Làm gì |
|---|---|
| 1 ngày sau khi học | trả lời `questions.md` không nhìn tài liệu |
| 1 tuần sau | trả lời lại; làm lại các bài tập đã đánh dấu |
| 1 tháng sau | trả lời lại; làm một notebook trong [`exercises/`](exercises/README.md) (luyện xen kẽ) |
| cuối mỗi giai đoạn | bài thi tổng hợp ([`quizzes/final_exams/`](quizzes/final_exams/README.md)) |

Muốn tự động hoá: chép câu hỏi sang Anki (thẻ ghi nhớ) — chỉ chép câu bạn **đã hiểu**, đừng học thuộc câu trả lời.

## 5. Khi code không chạy hoặc kết quả lạ

1. **Đọc thông báo lỗi từ dưới lên** — dòng cuối là lỗi, các dòng trên là nơi xảy ra.
2. In **shape, dtype, vài giá trị đầu**, số NaN của mọi mảng liên quan (`X.shape`, `X.dtype`, `np.isnan(X).sum()`).
3. Thu nhỏ: chạy trên 10 dòng dữ liệu, một epoch, một đặc trưng.
4. So với một baseline đơn giản: nếu mô hình phức tạp thua baseline, lỗi thường ở dữ liệu hoặc đánh giá.
5. Kết quả **quá tốt** đáng ngờ hơn kết quả tệ: kiểm tra rò rỉ (chương 05), chia dữ liệu, đặc trưng sau thời điểm dự đoán.
6. Tìm bằng **tiếng Anh** (xem [GLOSSARY.md](GLOSSARY.md)); khi hỏi người khác, gửi ví dụ tối thiểu tái hiện được lỗi.

Chương 01 bài 06–07 (lỗi shape/NaN, những lỗi Python thường gặp) và chương 00 bài 05 có hướng dẫn chi tiết về gỡ lỗi và kỹ năng tìm kiếm.

## 6. Môi trường

- **Local**: `uv sync` hoặc `pip install -e .` (xem [README](README.md)). Cài thêm extra khi tới chương cần (README mỗi chương ghi rõ).
- **Colab**: mỗi notebook có badge "Open in Colab"; ô ⚙️ Setup đầu notebook tự clone repo và cài package. Chương deep learning: bật GPU (Runtime → Change runtime type). Lưu ý Colab xoá dữ liệu khi ngắt phiên — lưu bài làm vào Google Drive hoặc fork repo của bạn.
- Mọi notebook chạy được trên CPU; GPU chỉ giúp nhanh hơn ở chương 29–35 và project 07.

## 7. Những bẫy cần tránh

- **"Tutorial hell"**: đọc hết mà không tự làm. Tỉ lệ tốt: 30% đọc, 70% làm.
- **Bỏ qua toán**: không cần chứng minh mọi thứ, nhưng phải đọc được công thức và biết mỗi ký hiệu là gì trong code ([NOTATION.md](NOTATION.md)).
- **Đuổi theo mô hình mới nhất**: phần lớn dự án thất bại vì dữ liệu, đánh giá sai, hoặc giải sai bài toán — không vì mô hình chưa đủ mạnh.
- **Chỉ nhìn một con số**: luôn hỏi metric này đo gì, độ bất định bao nhiêu, so với baseline nào, đúng với nhóm nào.
- **Học một mình mãi**: tìm bạn học, giải thích cho nhau, review code của nhau.

## 8. Theo dõi tiến độ

`python -m ml_course.progress` in tiến độ theo giai đoạn và chương tiếp theo; `--update-readme` ghi vào README của fork. Commit `progress/progress.json` để lưu lịch sử — xem [progress/README.md](progress/README.md).
