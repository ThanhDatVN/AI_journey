# Python cho Machine Learning — Bạn đã thực sự "biết" Python, NumPy, pandas?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | `list`, `tuple`, `set`, `dict` khác nhau thế nào? Khi nào dùng loại nào? |
| 2 | `b = a` với `a` là list rồi sửa `b` — chuyện gì xảy ra với `a`? Sao chép đúng cách thế nào? |
| 3 | Vì sao `def f(x, history=[])` là một cái bẫy? |
| 4 | `ndarray` khác list Python ở những điểm nào? `dtype` và `shape` quan trọng ra sao? |
| 5 | Nêu quy tắc broadcasting. `(n,) - (n, 1)` cho ra shape gì? |
| 6 | Vì sao code vectorized nhanh hơn vòng lặp Python nhiều lần? |
| 7 | `np.mean(X, axis=0)` và `axis=1` khác nhau thế nào với ma trận dữ liệu $X$ (hàng = mẫu)? |
| 8 | `loc` và `iloc` trong pandas khác nhau ra sao? |
| 9 | `groupby` hoạt động theo mô hình nào? Cho một ví dụ trong ML. |
| 10 | `merge` và `concat` dùng khi nào? |
| 11 | Hai cách vẽ của Matplotlib (`plt.*` và `fig, ax`) — vì sao nên dùng cách hướng đối tượng? |
| 12 | Chọn loại biểu đồ cho: phân phối một biến, quan hệ hai biến số, so sánh nhóm, xu hướng theo thời gian. |
| 13 | Lỗi "âm thầm" `(n,)` vs `(n, 1)` là gì? Phòng tránh thế nào? |
| 14 | NaN/inf trong huấn luyện thường đến từ đâu? Truy tìm thế nào? |
| 15 | Đọc traceback theo thứ tự nào? Tìm dòng cần sửa ở đâu? |
| 16 | Kể 5 exception hay gặp nhất, nguyên nhân và cách sửa. |
| 17 | Vì sao `except Exception: pass` nguy hiểm? Viết `try/except` đúng cách thế nào? |
| 18 | Khi nào dùng `raise ValueError(...)`, khi nào dùng `assert`? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. `list`: dãy có thứ tự, sửa được; `tuple`: có thứ tự, bất biến (làm key của dict được); `set`: không trùng lặp, kiểm tra thuộc rất nhanh; `dict`: ánh xạ key → value (cấu hình, đếm, tra cứu).
2. Hai tên cùng trỏ một object → sửa `b` là sửa `a` (aliasing). Sao chép: `a.copy()`, `list(a)`, `copy.deepcopy(a)` với cấu trúc lồng nhau; với NumPy `arr.copy()` (lát cắt NumPy là *view*).
3. Giá trị mặc định được tạo **một lần** khi định nghĩa hàm → các lần gọi dùng chung một list, dữ liệu dồn lại. Dùng `history=None` rồi `if history is None: history = []`.
4. Một kiểu dữ liệu cho mọi phần tử, lưu liền khối trong bộ nhớ, phép toán vectorized viết bằng C; `dtype` quyết định độ chính xác/bộ nhớ (`float32` vs `float64`, tràn số với `uint8`), `shape` quyết định phép toán có hợp lệ không.
5. So chiều từ **phải sang trái**; hai chiều tương thích nếu bằng nhau hoặc một trong hai bằng 1; thiếu chiều thì coi như 1. `(n,) - (n, 1)` → `(n, n)` — thường là bug.
6. Vòng lặp chạy trong NumPy (C, dữ liệu liền khối, SIMD, BLAS đa luồng) thay vì từng phép toán Python có chi phí kiểm tra kiểu và gọi hàm.
7. `axis=0`: gộp theo hàng → một giá trị cho mỗi **cột** (trung bình từng feature); `axis=1`: gộp theo cột → một giá trị cho mỗi **mẫu**.
8. `loc` chọn theo **nhãn** (tên cột, giá trị index; lát cắt gồm cả điểm cuối); `iloc` chọn theo **vị trí** số nguyên (lát cắt không gồm điểm cuối).
9. Split → apply → combine: chia theo nhóm, tính trên từng nhóm, ghép kết quả. Ví dụ: tỉ lệ churn theo loại hợp đồng; giá trung bình theo quận (target encoding — chương 17).
10. `merge`: nối hai bảng theo **khoá** (như JOIN trong SQL — vd bảng giao dịch với bảng khách hàng); `concat`: xếp chồng bảng cùng cấu trúc (thêm hàng) hoặc ghép cột cạnh nhau.
11. `plt.*` ngầm thao tác trên "trục hiện tại" — dễ nhầm khi có nhiều biểu đồ; `fig, ax = plt.subplots(...)` chỉ rõ vẽ vào đâu, dễ làm lưới nhiều biểu đồ và viết hàm vẽ dùng lại.
12. Phân phối: histogram/KDE/boxplot; quan hệ hai biến số: scatter (thêm màu theo nhãn); so sánh nhóm: bar chart/boxplot theo nhóm; xu hướng thời gian: line plot.
13. `y_hat` shape `(n, 1)` trừ `y` shape `(n,)` broadcasting thành ma trận `(n, n)` — không báo lỗi nhưng loss sai. Thống nhất quy ước shape, `ravel()`, `assert y_hat.shape == y.shape`, so sánh với thư viện chuẩn.
14. `log(0)`, `exp` quá lớn, chia cho 0, learning rate quá lớn (phân kỳ), dữ liệu có giá trị thiếu. Truy tìm: `np.isnan(...).any()` sau từng bước, `np.errstate(all="raise")`, `clip` xác suất, chuẩn hoá đầu vào, giảm learning rate.
15. Từ **dưới lên**: dòng cuối là loại lỗi + thông điệp; đi ngược lên tới dòng đầu tiên thuộc code của bạn (không phải trong thư viện) — đó là nơi cần sửa.
16. `NameError` (gõ sai tên/chưa chạy cell định nghĩa), `TypeError` (sai kiểu, hàm quên `return` → `None`, thiếu tham số), `ValueError` (giá trị không hợp lệ: `int("3.5")`, reshape sai, unpack sai số phần tử), `KeyError`/`IndexError` (sai key/tên cột, lệch một chỉ số), `AttributeError` (list không có `.shape`; `xs = xs.sort()` → `None`), `ModuleNotFoundError` (chưa cài hoặc sai kernel), `FileNotFoundError` (đường dẫn tương đối sai thư mục làm việc).
17. Nó nuốt **mọi** lỗi kể cả lỗi lập trình → kết quả sai mà không ai biết. Đúng cách: bắt **đúng loại** lỗi dự liệu được (vd `ValueError` khi chuyển chuỗi thành số), ở phạm vi hẹp nhất, xử lý có chủ đích (ghi lại/đếm giá trị hỏng), để lỗi khác nổi lên.
18. `raise` kiểm tra **đầu vào** từ người dùng/dữ liệu (luôn chạy, thông điệp rõ ràng cách sửa); `assert` kiểm tra **giả định nội bộ** của chính code (shape, bất biến) và có thể bị tắt khi chạy Python với cờ `-O`.

</details>
