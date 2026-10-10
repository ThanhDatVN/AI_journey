# Ký hiệu toán học dùng trong khoá học

Toàn bộ notebook dùng **một bộ ký hiệu thống nhất** dưới đây. Khi đọc sách/paper khác, ký hiệu có thể khác — hãy luôn kiểm tra phần "Notation" của tài liệu đó.

Xem thêm bài học [02 · 01 — Ký hiệu toán](notebooks/02_mathematics_for_ml/01_math_notation.ipynb) (dựa trên bảng ký hiệu của khoá AIVN).

## Dữ liệu

| Ký hiệu | Ý nghĩa | Trong code |
|---|---|---|
| $n$ (hoặc $m$ trong tài liệu Andrew Ng/AIVN) | số mẫu (samples) | `n = len(X)` |
| $d$ | số feature | `d = X.shape[1]` |
| $K$ | số lớp (classification) hoặc số cụm (clustering) | `K` |
| $\mathbf{x}^{(i)} \in \mathbb{R}^d$ | vector feature của mẫu thứ $i$ | `X[i]` |
| $x_j$ hoặc $x^{(i)}_j$ | feature thứ $j$ (của mẫu $i$) | `X[i, j]` |
| $X \in \mathbb{R}^{n \times d}$ | ma trận dữ liệu — **mỗi hàng là một mẫu** | `X`, shape `(n, d)` |
| $y^{(i)}$, $\mathbf{y} \in \mathbb{R}^n$ | nhãn / giá trị mục tiêu | `y`, shape `(n,)` |
| $\hat y^{(i)}$, $\hat{\mathbf y}$ | giá trị dự đoán | `y_hat`, shape `(n,)` |
| $\mathcal{D} = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^n$ | tập dữ liệu | `(X, y)` |
| $\mathcal{D}_{train}, \mathcal{D}_{val}, \mathcal{D}_{test}$ | tập train / validation / test | `X_train, X_val, X_test` |

## Model và tham số

| Ký hiệu | Ý nghĩa | Trong code |
|---|---|---|
| $\mathbf{w} \in \mathbb{R}^d$, $b$ | trọng số (weights) và bias (intercept) | `coef_`, `intercept_` |
| $\boldsymbol\theta$ | toàn bộ tham số của model (AIVN/Andrew Ng: $\theta_0 = b$, $\theta_{1..d} = \mathbf w$) | `theta` |
| $h_\theta(\mathbf x)$ hoặc $f(\mathbf x)$ | hàm giả thuyết (hypothesis) / model | `model.predict` |
| $W^{[l]}, \mathbf b^{[l]}$ | tham số của layer $l$ (neural network) | `layer.weight` |
| $\mathbf z = X\mathbf w + b$ | đầu ra tuyến tính (logit) | `z` |
| $\sigma(\cdot)$ | hàm sigmoid | `sigmoid` |
| $\lambda$ hoặc $\alpha$ | hệ số regularization (sklearn dùng `alpha`, hoặc `C = 1/λ`) | `alpha`, `C` |
| $\eta$ (hoặc $\alpha$ trong tài liệu AIVN) | learning rate | `lr` |

## Hàm mất mát và tối ưu

| Ký hiệu | Ý nghĩa |
|---|---|
| $\ell(\hat y, y)$ | loss của **một** mẫu |
| $L(\boldsymbol\theta)$ hoặc $J(\boldsymbol\theta)$ | hàm mất mát / cost trên toàn bộ dữ liệu, thường $\frac{1}{n}\sum_i \ell(\hat y^{(i)}, y^{(i)})$ |
| $\nabla_{\boldsymbol\theta} L$ | gradient của $L$ theo $\boldsymbol\theta$ (cùng shape với $\boldsymbol\theta$) |
| $\frac{\partial L}{\partial \theta_j}$ | đạo hàm riêng |
| $\boldsymbol\theta^* = \arg\min_{\boldsymbol\theta} L(\boldsymbol\theta)$ | tham số tối ưu |

## Đại số tuyến tính

| Ký hiệu | Ý nghĩa |
|---|---|
| chữ thường đậm $\mathbf{x}$ | vector (mặc định là **vector cột**) |
| chữ hoa $X, A$ | ma trận |
| $A^\top$ | chuyển vị |
| $\mathbf x^\top \mathbf y = \sum_j x_j y_j$ | tích vô hướng (dot product) |
| $A \odot B$ | tích Hadamard (nhân từng phần tử) — `A * B` |
| $AB$ | nhân ma trận — `A @ B` |
| $\lVert\mathbf x\rVert_2 = \sqrt{\sum_j x_j^2}$, $\lVert\mathbf x\rVert_1 = \sum_j \lvert x_j\rvert$ | chuẩn L2, L1 |
| $I$ | ma trận đơn vị |
| $A^{-1}$ | ma trận nghịch đảo |
| $\operatorname{tr}(A)$, $\det(A)$ | vết, định thức |

## Xác suất và thống kê

| Ký hiệu | Ý nghĩa |
|---|---|
| $P(A)$ | xác suất của sự kiện $A$ |
| $p(x)$ | hàm mật độ / hàm khối xác suất |
| $P(A \mid B)$ | xác suất có điều kiện |
| $X \sim \mathcal{N}(\mu, \sigma^2)$ | biến ngẫu nhiên có phân phối chuẩn |
| $\mathbb{E}[X]$, $\operatorname{Var}(X)$, $\operatorname{Cov}(X, Y)$ | kỳ vọng, phương sai, hiệp phương sai |
| $\bar x$, $s^2$ | trung bình mẫu, phương sai mẫu |
| $\mu$, $\sigma^2$ | trung bình, phương sai của tổng thể |
| $\Sigma$ | ma trận hiệp phương sai |
| $\mathcal{L}(\boldsymbol\theta)$ | hàm likelihood |
| $\mathbb{1}[\cdot]$ | hàm chỉ thị (bằng 1 nếu điều kiện đúng) |

## Quy ước shape trong code (rất quan trọng)

```text
X       (n, d)     mỗi hàng là một mẫu
y       (n,)       KHÔNG dùng (n, 1) để tránh lỗi broadcasting (n,) - (n,1) → (n,n)
w       (d,)
proba   (n, K)     classification nhiều lớp
images  (n, H, W) hoặc (n, C, H, W) với PyTorch
```
