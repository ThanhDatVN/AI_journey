# pandas — cheat sheet

```python
import pandas as pd

# Đọc / ghi
df = pd.read_csv("data/sample/customer_churn.csv", parse_dates=[...], dtype={...})
df.to_csv("out.csv", index=False); df.to_parquet("out.parquet")

# Xem nhanh
df.head(); df.shape; df.dtypes; df.info(); df.describe(include="all")
df["col"].value_counts(normalize=True, dropna=False); df.isna().sum(); df.duplicated().sum()

# Chọn
df["a"]; df[["a", "b"]]
df.loc[df.age > 30, ["name", "age"]]           # theo nhãn / điều kiện
df.iloc[:5, 1:3]                                # theo vị trí
df.query("age > 30 and city == 'Hà Nội'")
df[df.city.isin(["Hà Nội", "TP.HCM"])]

# Tạo / sửa cột (không sửa bản gốc ngoài ý muốn)
df = df.assign(ratio=lambda d: d.a / d.b)
df["band"] = pd.cut(df.age, [0, 30, 45, 120], labels=["<30", "30–45", ">45"])
df["q"] = pd.qcut(df.score, 5, labels=False)
df["x"] = df["x"].astype(float); df["cat"] = df["cat"].astype("category")
df["clean"] = df["raw"].str.strip().str.lower().replace({"hn": "Hà Nội"})

# Dữ liệu thiếu, trùng
df.dropna(subset=["y"]); df.fillna({"income": df.income.median()}); df.drop_duplicates()

# Gom nhóm
df.groupby("contract").churn.mean()                                   # một dòng / nhóm
df.groupby("contract").agg(n=("churn", "size"), rate=("churn", "mean"))
df["group_mean"] = df.groupby("g").x.transform("mean")                # giữ nguyên số dòng
pd.crosstab(df.a, df.b, normalize="index"); df.pivot_table(index="a", columns="b", values="y", aggfunc="mean")

# Ghép bảng
a.merge(b, on="key", how="left", validate="many_to_one")
pd.concat([df1, df2], ignore_index=True)
pd.merge_asof(left.sort_values("ts"), right.sort_values("ts"), on="ts", by="id", allow_exact_matches=False)  # đúng thời điểm

# Thời gian
df["date"] = pd.to_datetime(df.date, format="%Y-%m-%d", errors="coerce")
df.date.dt.year; df.date.dt.dayofweek; df.date.dt.to_period("M")
s.resample("W").sum(); s.rolling(7).mean(); s.shift(1); s.diff()

# Sắp xếp, đổi tên
df.sort_values(["a", "b"], ascending=[True, False]); df.rename(columns={"old": "new"}); df.reset_index(drop=True)
```

## Bẫy thường gặp

| Bẫy | Dấu hiệu | Cách tránh |
|---|---|---|
| `SettingWithCopyWarning` | gán vào lát cắt `df[df.a > 0]["b"] = …` | `df.loc[mask, "b"] = …` hoặc `.copy()` |
| merge nhân bản dòng | số dòng tăng sau merge | `validate="one_to_one"/"many_to_one"`, kiểm tra khoá trùng |
| số đọc thành chuỗi | `dtype == object` cho cột tiền | làm sạch (`str.replace`) rồi `astype(float)` |
| ngày tháng nhầm định dạng | 05/03 hiểu thành 3/5 | chỉ rõ `format=` |
| rò rỉ khi điền thiếu | `fillna(df.mean())` trên cả train + test | điền trong pipeline, fit trên train |
| `groupby` bỏ NaN | nhóm thiếu biến mất | `dropna=False` |
