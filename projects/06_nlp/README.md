# Project 06 — Phân loại chủ đề tin tức tiếng Việt (NLP)

> Một trang tin cần tự động gắn **chủ đề** (kinh tế, giáo dục, thể thao, sức khỏe, du lịch, công nghệ) cho bài mới. Có 3.000 tin đã gắn nhãn thủ công — kèm ~3% nhãn sai.

| | |
|---|---|
| Bài toán | Phân loại văn bản đa lớp |
| Dữ liệu | `make_news_vi(...)` trong `ml_course.datasets` — sinh offline: train 3.000, test 1.000, bộ "tin khó" 1.000 (nhiều câu gây nhiễu) |
| Chương nền tảng | 09, 07, 32, 31, 14–15 |
| Độ khó · thời gian | ⭐⭐⭐ · 5–7 giờ |

## 🎯 Mục tiêu học tập
- Baseline đúng cho văn bản (đa số, Naive Bayes); TF-IDF + logistic.
- Đặc thù tiếng Việt: từ nhiều âm tiết → **cặp âm tiết**; `token_pattern` bỏ âm tiết một chữ cái.
- Phân tích lỗi thật sự (nhãn sai vs tin đa chủ đề); đường cong học; độ bền.
- (Mở rộng) embedding kiểu fastText và **Transformer encoder nhỏ** học từ đầu với PyTorch — khi nào kiến trúc phức tạp đáng giá?

## 📋 Nhiệm vụ
1. EDA. 2. Baseline. 3. CV ba biểu diễn. 4. Test + ma trận nhầm lẫn + bảng lỗi. 5. Đường cong học + bộ tin khó. 6. Từ khoá theo chủ đề. 7. (Tuỳ chọn) fastText. 8. (Tuỳ chọn) Transformer nhỏ — tự viết mean-pooling có mask. 9. Báo cáo.

## ✅ Tiêu chí đạt
- CV chọn "âm tiết + cặp"; macro-F1 test > 0.94 và hơn âm tiết đơn; macro-F1 bộ tin khó > 0.9.
- (Mở rộng) Transformer nhỏ: macro-F1 test > 0.9, bộ tin khó > 0.88.

## 🧭 Gợi ý & cạm bẫy
- Với lớp cân bằng, macro-F1 ≈ accuracy; với lớp lệch, luôn báo macro-F1.
- Nhiều "lỗi" là nhãn sai — rà soát dữ liệu thường hiệu quả hơn đổi mô hình.

## 🚀 Mở rộng
- Tách từ (underthesea/VnCoreNLP) so với cặp âm tiết; fine-tune PhoBERT (cần GPU) — đáng không?
- Đa nhãn cho tin nhiều chủ đề; phát hiện chủ đề mới (open-set).
- Thử `reviews_vi.csv` (cảm xúc) — phủ định, mỉa mai khó hơn chủ đề thế nào?

## 📁 Tệp
- `project.ipynb` · lời giải: [`solutions/projects/06_nlp/project.ipynb`](../../solutions/projects/06_nlp/project.ipynb).
