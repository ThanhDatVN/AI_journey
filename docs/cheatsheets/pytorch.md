# PyTorch — cheat sheet

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)                                   # seed TRƯỚC khi khởi tạo mô hình

# Tensor
x = torch.tensor([[1.0, 2.0]]); torch.zeros(3, 4); torch.randn(8, 3); torch.arange(10)
x.shape, x.dtype, x.device; x.to(device); x.float()
x.detach().cpu().numpy()                              # về NumPy (an toàn cả khi ở GPU / có gradient)
x.view(-1, 4); x.reshape(2, -1); x.permute(0, 2, 1); x.unsqueeze(1); x.squeeze()
torch.cat([a, b], dim=0); torch.stack([a, b]); a @ b; x.sum(dim=1, keepdim=True)

# Autograd
w = torch.randn(3, requires_grad=True)
loss = ((X @ w - y) ** 2).mean(); loss.backward(); w.grad
with torch.no_grad(): ...                            # không theo dõi gradient (đánh giá, cập nhật thủ công)

# Mô hình
class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d_hidden, d_out))

    def forward(self, x):
        return self.net(x)

# Lớp hay dùng
nn.Conv2d(3, 16, kernel_size=3, padding=1); nn.MaxPool2d(2); nn.BatchNorm2d(16); nn.LayerNorm(d)
nn.Embedding(vocab, d); nn.LSTM(d, h, batch_first=True); nn.MultiheadAttention(d, n_heads, batch_first=True)

# Dữ liệu
loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y)), batch_size=64, shuffle=True)

# Huấn luyện
model = MLP(d, 64, n_classes).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=n_epochs * len(loader))
loss_fn = nn.CrossEntropyLoss()                      # nhận LOGIT (không softmax trước)
for epoch in range(n_epochs):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    model.eval()                                     # tắt dropout, BatchNorm dùng thống kê chạy
    with torch.no_grad():
        ...

# Lưu / nạp (chỉ trọng số — an toàn hơn pickle cả mô hình)
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt", weights_only=True))
```

## Bẫy thường gặp

| Bẫy | Cách tránh |
|---|---|
| quên `opt.zero_grad()` | gradient cộng dồn qua các bước |
| quên `model.eval()` khi đánh giá | dropout/BatchNorm hoạt động sai |
| softmax trước `CrossEntropyLoss` | loss nhận logit; softmax kép làm học kém |
| tensor ở hai thiết bị khác nhau | `.to(device)` cho cả mô hình và dữ liệu |
| `x.numpy()` lỗi trên GPU / khi có grad | `x.detach().cpu().numpy()` |
| không tái lập | seed trước khi tạo mô hình; `torch.backends.cudnn.deterministic = True` |
| loss NaN | giảm lr, chuẩn hoá đầu vào, clip gradient, kiểm tra log(0) |
| overfit kiểm tra nhanh | thử overfit **một batch nhỏ** — không được thì có lỗi trong code |

## PyTorch ↔ TensorFlow / Keras (chương 27 · bài 05)

| Việc cần làm | PyTorch | TensorFlow / Keras |
|---|---|---|
| tensor, tham số | `torch.tensor(x)`, `nn.Parameter` | `tf.constant(x)` (bất biến), `tf.Variable` |
| đạo hàm | `loss.backward()` → `w.grad` | `with tf.GradientTape() as tape: ...` → `tape.gradient(loss, vars)` |
| mô hình | `nn.Module` + `forward()` | `keras.Sequential([...])`, Functional API, `keras.Model` + `call()` |
| lớp tuyến tính | `nn.Linear(in, out)` | `keras.layers.Dense(out)` |
| loss phân loại (logits) | `nn.CrossEntropyLoss()` | `keras.losses.SparseCategoricalCrossentropy(from_logits=True)` |
| dữ liệu | `DataLoader(dataset, batch_size, shuffle=True)` | `tf.data.Dataset.from_tensor_slices(...).shuffle(n).batch(b).prefetch(tf.data.AUTOTUNE)` |
| huấn luyện | vòng lặp 5 bước | `model.compile(...)`; `model.fit(X, y, validation_split=0.1, callbacks=[keras.callbacks.EarlyStopping(...)])` |
| tăng tốc | `torch.compile(model)` | `@tf.function` (code Python chỉ chạy lúc truy vết — dùng `tf.print`) |
| chế độ train/eval | `model.train()` / `model.eval()` | `model(x, training=True/False)` |
| lưu / nạp | `torch.save(model.state_dict(), p)` | `model.save("m.keras")`, `keras.models.load_model("m.keras")` |
| ảnh | `(N, C, H, W)` | `(N, H, W, C)` |
