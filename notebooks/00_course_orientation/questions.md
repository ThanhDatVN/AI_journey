# Định hướng khoá học — Bạn đã sẵn sàng học hiệu quả chưa?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Với mỗi mô hình, khoá học yêu cầu bạn trả lời được những câu hỏi nào (ngoài việc gọi `fit`/`predict`)? |
| 2 | Hệ thống bài tập L0 → L7 gồm những mức nào? Vì sao L4 (debug) và L5 (phân tích) quan trọng? |
| 3 | Checkpoint "gate" hoạt động thế nào? Khi bị REVIEW thì làm gì tiếp? |
| 4 | *Hidden state* trong Jupyter là gì? Làm sao biết notebook của bạn không bị phụ thuộc thứ tự chạy? |
| 5 | Các magic `%timeit`, `%pip`, `%debug` dùng để làm gì? |
| 6 | Viết công thức toán trong Markdown thế nào (trong dòng và thành khối riêng)? |
| 7 | Khi nào nên chuyển code từ notebook sang file `.py`? |
| 8 | Chạy notebook của khoá học trên Google Colab cần những bước nào? Lưu kết quả lâu dài ở đâu? |
| 9 | Ba cách cài môi trường của khoá học (uv, conda, pip + venv) khác nhau ra sao? File `uv.lock` để làm gì? |
| 10 | Đặt breakpoint và debug trong VS Code/PyCharm: step over, step into, watch dùng khi nào? |
| 11 | Các toán tử tìm kiếm `site:`, `intitle:`, `"..."`, `filetype:` dùng khi nào? Cho ví dụ khi tra một lỗi Python. |
| 12 | Một câu hỏi tốt khi nhờ người khác (hoặc AI) giúp gồm những gì? |
| 13 | Dùng trợ lý AI khi học thế nào để không bị "học hộ"? |
| 14 | Active recall, spaced repetition, interleaving là gì? Áp dụng vào khoá học này ra sao? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Mô hình hoạt động thế nào, giả định gì, khi nào dùng và khi nào **không** nên dùng, điểm mạnh/yếu, siêu tham số ảnh hưởng ra sao, và **vì sao nó thất bại** — kèm tự cài đặt from scratch trước khi dùng thư viện.
2. L0 nhớ khái niệm · L1 giải thích bằng lời · L2 tính tay · L3 code · L4 debug · L5 phân tích mô hình · L6 bài toán thực tế · L7 nâng cao/nghiên cứu. L4–L5 rèn đúng kỹ năng của công việc thật: tìm ra vì sao mô hình sai và hiểu siêu tham số tác động thế nào — không chỉ viết code chạy được.
3. `99_checkpoint.ipynb` chấm quiz lý thuyết, quiz toán, bài code, phân tích và mini project; đạt mọi phần → PASS và chuyển chương. REVIEW → checkpoint liệt kê chủ đề yếu và bài cần ôn; ôn đúng bài đó rồi làm lại.
4. Biến/hàm còn trong bộ nhớ từ cell đã sửa hoặc đã xoá, hoặc chạy cell không theo thứ tự → kết quả không tái lập. Kiểm tra bằng **Restart & Run All** trước khi tin kết quả.
5. `%timeit`: đo thời gian chạy (lặp nhiều lần); `%pip install`: cài thư viện vào **đúng** môi trường của kernel; `%debug`: mở trình gỡ lỗi tại chỗ vừa xảy ra exception.
6. `$...$` trong dòng, `$$...$$` cho khối riêng (cú pháp LaTeX), ví dụ `$\hat y = \mathbf w^\top \mathbf x + b$`.
7. Khi code được dùng lại ở nhiều nơi, cần test, cần chạy tự động (pipeline, API, cron) hoặc cần review/quản lý phiên bản rõ ràng — notebook để khám phá và trình bày, `.py` cho code ổn định.
8. Bấm badge "Open in Colab", chạy cell ⚙️ Setup (clone repo + `pip install -e`); bật GPU khi cần trong Runtime → Change runtime type. Phiên Colab bị xoá khi kết thúc → lưu kết quả/model vào Google Drive (`drive.mount`) hoặc tải về.
9. uv: nhanh, đọc `pyproject.toml` + `uv.lock` để cài **đúng** phiên bản đã kiểm chứng; conda: quản lý cả phiên bản Python và thư viện hệ thống (`environment.yml`); pip + venv: có sẵn trong Python. `uv.lock` ghim chính xác mọi phiên bản (kể cả phụ thuộc gián tiếp) → mọi người và CI chạy cùng một môi trường.
10. Breakpoint dừng chương trình tại một dòng; *step over* chạy hết dòng hiện tại (không đi vào hàm), *step into* đi vào bên trong hàm được gọi; *watch* theo dõi giá trị một biểu thức (vd `X.shape`) qua từng bước.
11. `site:` tìm trong một trang (`site:scikit-learn.org pipeline`), `intitle:` chỉ lấy trang có từ khoá trong tiêu đề, `"..."` tìm chính xác cụm từ (dán nguyên dòng lỗi cuối của traceback, bỏ đường dẫn riêng của máy), `filetype:pdf` tìm tài liệu PDF.
12. Mục tiêu muốn làm; đoạn code **tối giản** vẫn tái hiện được lỗi (minimal reproducible example); thông báo lỗi đầy đủ; kết quả mong đợi vs thực tế; phiên bản thư viện; những gì đã thử.
13. Tự làm trước; hỏi AI để được **giải thích** hoặc gợi ý hướng, không xin lời giải cho bài tập; luôn kiểm chứng câu trả lời (chạy code, đối chiếu tài liệu); tự viết lại lời giải bằng lời của mình.
14. Active recall: tự nhớ lại thay vì đọc lại (checkpoint, questions.md); spaced repetition: ôn lại sau khoảng thời gian tăng dần (1 ngày, 1 tuần, 1 tháng); interleaving: trộn bài của nhiều chương (thư mục `exercises/`) để luyện kỹ năng **chọn** phương pháp.

</details>
