# 🗂️ Dữ liệu

Gần như toàn bộ khoá học **không cần tải dữ liệu từ internet**: dữ liệu hoặc nằm sẵn trong `data/sample/` (nhỏ, commit vào git), hoặc được sinh offline bằng `ml_course.datasets` với `seed` cố định. Ngoại lệ duy nhất là hai bộ **ảnh thật** (Fashion-MNIST, CIFAR-10) dùng ở vài bài — tự tải một lần vào `data/raw/` (xem mục cuối). Phần lớn dữ liệu là **mô phỏng có quy luật thật đã biết** (hệ số thật, hiệu ứng nhân quả thật, cụm thật, logit thật) — nhờ vậy bạn kiểm chứng được mô hình học đúng hay sai, điều không làm được với dữ liệu thật.

| Thư mục | Nội dung | Git |
|---|---|---|
| `sample/` | CSV/TXT nhỏ dùng trong bài học, bài tập, project | ✅ commit |
| `raw/` | dữ liệu tải về (Fashion-MNIST, CIFAR-10) và dữ liệu gốc của **bạn** (dự án riêng, capstone) | ❌ `.gitignore` |
| `processed/` | dữ liệu đã xử lý của bạn | ❌ `.gitignore` |

Sinh lại các CSV mô phỏng trong `sample/` (tái lập hoàn toàn bằng seed): `python tools/make_sample_data.py`.

## `data/sample/`

| Tệp | Dòng | Cột chính | Nguồn | Dùng ở |
|---|---:|---|---|---|
| `house_prices.csv` | 1.500 | diện tích, phòng, tuổi nhà, quận, khoảng cách, sân vườn → `price_billion_vnd` | `make_house_prices(seed=42)` | ch.03, 06, 08, 10–12, 14, 17, 24, bài thi 1 |
| `house_prices_dirty.csv` | 1.515 | như trên + dòng trùng, thiếu, tên quận 23 kiểu, diện tích gõ thừa số 0 | `make_house_prices(dirty=True)` | ch.01, 03–04, 17, project 01, luyện tập cơ bản |
| `customer_churn.csv` | 5.000 | hợp đồng, thời gian dùng, cước, cuộc gọi hỗ trợ… → `churn` (~29%) | `make_customer_churn(seed=42)` | ch.01, 03–04, 07–08, 10–12, 14–18, 24, 27–28, 39, project 02, bài thi 2 & 6, luyện tập |
| `fraud_transactions.csv` | 30.000 | số tiền, giờ, loại cửa hàng, khoảng cách, nước ngoài, số giao dịch 24h, tuổi thẻ → `is_fraud` (1%) | `make_fraud_transactions(seed=42)` | ch.03, 05, 07–08, 14, 17, 19, project 08, bài thi 3 |
| `customer_segments.csv` | 600 | thu nhập, điểm chi tiêu, tuổi, lượt đến → `true_segment` (5 phân khúc) | `make_customer_segments(seed=42)` | ch.05, 20–21, 23, project 03, bài thi 4 |
| `sales_daily.csv` | 1.095 | `date`, `sales`, `promo`, `is_holiday` (2022–2024) | `make_sales_series(seed=42)` | ch.01, 03, 15, 25, project 04 |
| `ratings.csv`, `items.csv` | 5.441 / 200 | lượt chấm 1–5 có thời điểm (thiên lệch chọn lọc); phim, thể loại, năm | `make_ratings(seed=42)` | ch.26, project 05, luyện tập cơ bản |
| `reviews_vi.csv` | 3.000 | review sản phẩm tiếng Việt, `sentiment` 3 lớp, `label` nhị phân | `make_text_reviews(seed=42)` | ch.09, 17, 30–31 |
| `faq_vi.csv`, `faq_queries_vi.csv` | 25 / 75 | câu hỏi thường gặp + câu hỏi người dùng gắn `faq_id` đúng | viết tay | ch.32 (truy hồi ngữ nghĩa) |
| `causal_marketing.csv` | 5.000 | tuổi, thu nhập, lượt mua → `coupon` (không ngẫu nhiên), `spend_kvnd`, `true_effect` | `make_causal_marketing(seed=42)` | bản tham chiếu — ch.38 sinh trực tiếp bằng hàm |
| `credit_applications_raw.csv` | 12.180 | hồ sơ vay **thô**: ngày 2 định dạng, số tiền dạng chuỗi, thu nhập sai đơn vị, danh mục lộn xộn, trùng; nhãn vỡ nợ 12 tháng (chưa trưởng thành với hồ sơ gần đây), nhãn sớm | `make_credit_applications(seed=42, messy=True)` | ch.41 (capstone) |
| `iris.csv` | 150 | 4 số đo hoa → `species` | Fisher (1936), phạm vi công cộng | ch.03, 09–10, 40 |
| `ex2_city_profit.txt`, `ex3_house_multivariate.txt`, `ex4_exam_scores.txt`, `ex4_microchip_tests.txt`, `Position_Salaries.csv`, `my_house_price_prediction.csv` | nhỏ | dữ liệu assignment kiểu khoá Andrew Ng | kế thừa từ khoá AIVN-Machine-Learning | ch.06, 07 (assignment) |

## `ml_course.datasets` — hàm sinh dữ liệu

```python
from ml_course.datasets import make_customer_churn
df = make_customer_churn(n=5000, seed=0)
```

| Hàm | Trả về | Điểm đặc biệt |
|---|---|---|
| `make_regression_1d(n, kind, noise, seed)` | `X (n, 1)`, `y` | `kind`: `linear`, `poly`, `sin`, `step` — minh hoạ under/overfitting |
| `make_classification_2d(kind, n, noise, seed)` | `X (n, 2)`, `y` | `linear`, `blobs`, `moons`, `circles`, `xor`, `spiral` — vẽ biên quyết định |
| `load_digits_data(as_float)` | `X`, `y`, `images` | digits 8×8 của scikit-learn (offline) |
| `make_shapes(n, size, noise, region, seed, classes, rgb, color_bias)` | `X`, `y` | ảnh hình khối; `region="left"/"right"` thử dịch chuyển; `classes=SHAPE_CLASSES_EXTRA` thêm 4 lớp; `rgb=True` + `color_bias` tạo **tương quan giả màu ↔ lớp** (học đường tắt) |
| `make_shape_scenes(n, size, max_objects, noise, seed)` | `X`, `targets` (hộp `xyxy` + nhãn), `masks` | phát hiện & phân đoạn vật thể (ch.33) |
| `make_house_prices(n, seed, dirty)` | DataFrame | giá phụ thuộc phi tuyến vào diện tích, quận, tuổi; `dirty=True` thêm lỗi dữ liệu |
| `make_customer_churn(n, seed)` | DataFrame | logit thật tuyến tính (ghi ở ch.07) |
| `make_fraud_transactions(n, fraud_rate, seed)` | DataFrame | mất cân bằng nặng, tỉ lệ gian lận điều chỉnh được |
| `make_customer_segments(n, seed)` | DataFrame | 5 phân khúc thật, hình dạng/độ phân tán khác nhau |
| `make_sales_series(n_days, start, seed)` | DataFrame | xu hướng + mùa vụ tuần/năm (Tết) + khuyến mãi + lễ |
| `make_ratings(n_users, n_items, density, k, selection_strength, seed)` | `(ratings, items)` | nhân tố ẩn + bias; **thiên lệch chọn lọc** (người dùng chọn xem phim đoán sẽ thích) |
| `make_text_reviews(n, seed, label_noise)` | DataFrame | review tiếng Việt, phủ định, nhãn nhiễu |
| `make_news_vi(n, seed, distractor, label_noise)` | DataFrame | tin tức 6 chủ đề × 2 chủ đề con, từ đồng nghĩa (`NEWS_SYNONYMS`), câu gây nhiễu |
| `make_causal_marketing(n, seed, randomized)` | DataFrame | confounding thật, `true_effect` từng người; `randomized=True` cho thí nghiệm A/B |
| `make_credit_applications(n, seed, messy)` | DataFrame | nhãn trễ 12 tháng, nhãn sớm 3 tháng, cú sốc kinh tế 01/2024; `messy=True` cho dữ liệu thô |

Hằng số: `SHAPE_CLASSES`, `SHAPE_CLASSES_EXTRA`, `NEWS_TOPICS`, `NEWS_SYNONYMS`, `NEWS_LEXICON`, `DISTRICTS`, `GENRES`, `CREDIT_OBSERVATION_DATE`, `CREDIT_MONTHLY_RATE`.

## `ml_course.envs` — môi trường học tăng cường (chương 35)

Không cần Gymnasium — các môi trường nhỏ cài bằng NumPy với giao diện `reset() → state`, `step(action) → (state, reward, done)`:

| Lớp | Mô tả |
|---|---|
| `KArmedBandit(k, seed)` | k cần gạt, phần thưởng ~ N(q*(a), 1) — khám phá vs khai thác |
| `GridWorld(layout, slip, step_reward, max_steps)` | kiểu FrozenLake (`"4x4"`, `"8x8"` hoặc bản đồ tuỳ ý), có ma trận chuyển `P[s][a] = [(p, s', r, done)]` cho quy hoạch động |
| `CliffWalking(max_steps)` | lưới 4×12, vực −100 — so sánh SARSA và Q-learning |
| `RandomWalk()` | 5 trạng thái, giá trị thật 1/6…5/6 — so sánh MC và TD |
| `CartPole(max_steps)` | động lực học giống `CartPole-v1` — DQN, policy gradient, neuroevolution |

## `ml_course.real_datasets` — ảnh thật, tải một lần

```python
from ml_course.real_datasets import load_fashion_mnist, load_cifar10
X_train, y_train, X_test, y_test = load_fashion_mnist()   # uint8 (60000, 28, 28)
X_train, y_train, X_test, y_test = load_cifar10()         # uint8 (50000, 32, 32, 3)
```

| Bộ dữ liệu | Kích thước tải | Nội dung | Nguồn · giấy phép | Dùng ở |
|---|---|---|---|---|
| Fashion-MNIST | ~30 MB | 70 000 ảnh xám 28×28, 10 loại quần áo/giày/túi (`FASHION_MNIST_CLASSES`) | Zalando Research (Xiao et al., 2017) · MIT | ch.07 assignment 93, ch.27 bài 05, ch.29 bài 04 |
| CIFAR-10 | ~170 MB | 60 000 ảnh màu 32×32, 10 lớp (`CIFAR10_CLASSES`) | Krizhevsky (2009), cs.toronto.edu | ch.07 bài 08, ch.29 bài 04 |

- Lần đầu gọi hàm sẽ tải file gốc từ nguồn chính thức, **kiểm tra MD5** với giá trị công bố (file hỏng/tải dở bị tải lại), lưu vào `data/raw/`; các lần sau đọc từ đĩa.
- Đổi thư mục cache bằng biến môi trường `ML_COURSE_DATA_DIR` (vd ổ đĩa khác hoặc Google Drive trên Colab).
- Không có mạng? Tải thủ công các file `train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`, `t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz` vào `data/raw/fashion-mnist/` và `cifar-10-python.tar.gz` vào `data/raw/`.

## Dùng dữ liệu của bạn

Đặt file vào `data/raw/` (không commit dữ liệu lớn hoặc dữ liệu cá nhân — ghi nguồn, phiên bản/hash và cách tải vào README của dự án, xem chương 39). Kiểm tra quyền sử dụng và quyền riêng tư trước khi chia sẻ notebook có output.
