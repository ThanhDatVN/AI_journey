# NumPy — cheat sheet

```python
import numpy as np
rng = np.random.default_rng(0)                 # luôn dùng Generator có seed

# Tạo mảng
np.array([[1, 2], [3, 4]]); np.zeros((3, 4)); np.ones(5); np.full((2, 2), 7)
np.arange(0, 10, 2); np.linspace(0, 1, 5); np.eye(3)
rng.normal(0, 1, (100, 3)); rng.integers(0, 10, 5); rng.choice(10, 3, replace=False); rng.permutation(10)

# Thuộc tính
a.shape, a.ndim, a.dtype, a.size

# Indexing
a[0, 1]; a[:, 0]; a[1:3, ::2]; a[-1]
a[a > 0]                     # mặt nạ bool
a[[0, 2]]                    # fancy indexing (tạo bản sao)
np.where(a > 0, a, 0)        # chọn theo điều kiện

# Hình dạng
a.reshape(2, -1); a.ravel(); a.T; a[:, None]; np.expand_dims(a, 0); a.squeeze()
np.concatenate([a, b], axis=0); np.vstack([a, b]); np.hstack([a, b]); np.column_stack([x, y])

# Tính toán theo trục (axis=0: theo cột / dọc các hàng; axis=1: theo hàng)
X.mean(axis=0); X.std(axis=0); X.sum(axis=1, keepdims=True); X.argmax(axis=1)
np.cumsum(v); np.diff(v); np.sort(v); np.argsort(-v)[:k]; np.unique(v, return_counts=True)

# Broadcasting: so shape từ phải sang trái — bằng nhau hoặc một bên là 1
X - X.mean(axis=0)                           # (n, d) - (d,)
X / np.linalg.norm(X, axis=1, keepdims=True) # (n, d) / (n, 1)
D = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)   # khoảng cách từng cặp (n, m)

# Đại số tuyến tính
A @ B; np.dot(u, v); np.linalg.norm(v); np.linalg.det(A)
np.linalg.solve(A, b)        # giải Ax = b (tốt hơn inv(A) @ b)
np.linalg.inv(A); np.linalg.pinv(A); np.linalg.lstsq(X, y, rcond=None)
vals, vecs = np.linalg.eigh(S)              # ma trận đối xứng
U, s, Vt = np.linalg.svd(X, full_matrices=False)

# Số học ổn định
np.log1p(x); np.expm1(x); np.clip(p, 1e-12, 1 - 1e-12)
z - z.max(axis=1, keepdims=True)            # trước khi exp trong softmax
np.isclose(a, b); np.allclose(A, B); np.isnan(a).sum()
```

## Bẫy thường gặp

| Bẫy | Ví dụ | Cách tránh |
|---|---|---|
| nhầm trục | `X.mean(axis=1)` khi muốn trung bình mỗi đặc trưng | đặc trưng là **cột** → `axis=0` |
| broadcasting ngoài ý muốn | `y` shape `(n, 1)` trừ `pred` shape `(n,)` → `(n, n)` | giữ vector 1D `(n,)`, kiểm tra `.shape` |
| view vs copy | sửa slice làm đổi mảng gốc | `.copy()` khi cần độc lập |
| kiểu số nguyên | `np.array([1, 2]) / 2` ok, nhưng gán float vào mảng int bị cắt | đặt `dtype=float` |
| seed toàn cục | `np.random.seed` ảnh hưởng mọi nơi | `rng = np.random.default_rng(seed)` truyền tường minh |
