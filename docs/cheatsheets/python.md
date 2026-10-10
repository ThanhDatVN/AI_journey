# Python — cheat sheet

```python
# Kiểu dữ liệu cơ bản
x, y, s, ok, nothing = 3, 2.5, "chuỗi", True, None
a = [1, 2, 3]            # list (thay đổi được)
t = (1, 2)               # tuple (không đổi)
d = {"a": 1, "b": 2}     # dict
st = {1, 2, 3}           # set (không thứ tự — đừng dựa vào thứ tự duyệt!)

# Chuỗi
f"{x:.2f} | {y:>8,} | {0.1234:.1%}"     # định dạng: 2 chữ số, phân cách nghìn, phần trăm
"a,b,c".split(","); " x ".strip(); "-".join(["a", "b"])

# Comprehension
squares = [i * i for i in range(10) if i % 2 == 0]
lookup = {k: v for k, v in zip("abc", range(3))}

# Duyệt
for i, item in enumerate(a): ...
for k, v in d.items(): ...
for p, q in zip(a, t): ...

# Hàm: tham số mặc định, *args, **kwargs, type hints
def scale(values: list[float], factor: float = 2.0) -> list[float]:
    """Docstring mô tả hàm."""
    return [v * factor for v in values]

f = lambda v: v ** 2

# Lớp
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    lr: float = 0.1
    seed: int = 0

# Ngoại lệ
try:
    risky()
except (ValueError, KeyError) as err:
    print("lỗi:", err)
finally:
    cleanup()

# File & đường dẫn
from pathlib import Path
text = Path("data/sample/iris.csv").read_text(encoding="utf-8")
with open("out.txt", "w", encoding="utf-8") as fh:
    fh.write("xin chào\n")
```

## Lỗi thường gặp

| Lỗi | Nguyên nhân | Sửa |
|---|---|---|
| `IndentationError` | thụt lề không đều | dùng 4 dấu cách |
| `TypeError: unsupported operand` | trộn kiểu (chuỗi + số) | ép kiểu `int()`, `float()`, `str()` |
| `KeyError` | khoá không có trong dict/cột không có trong DataFrame | `d.get(k, default)`, kiểm tra `df.columns` |
| `IndexError` | chỉ số vượt độ dài | kiểm tra `len()` |
| giá trị mặc định thay đổi bí ẩn | `def f(a=[])` dùng chung list | `def f(a=None): a = a or []` |
| sao chép không như ý | `b = a` chỉ tạo tên mới | `b = a.copy()` / `copy.deepcopy(a)` |
| `float` so sánh sai | 0.1 + 0.2 != 0.3 | `math.isclose`, `np.isclose` |
