# Matplotlib — cheat sheet

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))               # một biểu đồ
fig, axes = plt.subplots(1, 3, figsize=(15, 4))      # nhiều biểu đồ: axes[0], axes[1], ...

ax.plot(x, y, marker="o", label="train")             # đường
ax.scatter(x, y, c=labels, cmap="tab10", s=10, alpha=0.6)   # điểm, màu theo nhóm
ax.bar(names, values); ax.barh(names, values)        # cột
ax.hist(values, bins=30, alpha=0.6, density=True)    # phân phối
ax.boxplot([a, b], labels=["A", "B"])                # hộp
ax.imshow(image, cmap="gray"); ax.axis("off")        # ảnh / ma trận
ax.contourf(xx, yy, Z, alpha=0.3)                    # biên quyết định
ax.fill_between(x, lower, upper, alpha=0.2)          # khoảng tin cậy / dự báo
ax.axhline(0.5, ls="--", color="gray"); ax.axvline(t, color="red")

ax.set(title="Tiêu đề", xlabel="x", ylabel="y", xlim=(0, 1), yscale="log")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
fig.savefig("hinh.png", dpi=150, bbox_inches="tight")

# pandas vẽ nhanh
df.plot(x="date", y="sales", ax=ax); df["col"].plot(kind="hist", bins=30)
```

## Biểu đồ nào cho câu hỏi nào

| Câu hỏi | Biểu đồ |
|---|---|
| phân phối một biến | histogram, boxplot, KDE |
| quan hệ hai biến số | scatter (thêm đường trung bình theo bin) |
| so sánh nhóm | bar (tỉ lệ/trung bình), boxplot theo nhóm |
| theo thời gian | line, kèm trung bình trượt |
| hiệu năng mô hình | đường cong học, ROC/PR, reliability (calibration), residual vs fitted |
| dữ liệu nhiều chiều | scatter matrix, PCA/t-SNE 2D, heatmap tương quan |

Mẹo: một biểu đồ = một thông điệp; đặt tên trục có **đơn vị**; dùng cùng màu cho cùng nhóm xuyên suốt notebook; tránh trục y cắt bớt khi so sánh độ lớn.
