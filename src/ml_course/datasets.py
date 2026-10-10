"""Bộ dữ liệu tổng hợp cho khoá học — sinh offline, tái lập được bằng ``seed``.

Vì sao dùng dữ liệu tổng hợp? Vì ta **biết quy luật thật** sinh ra dữ liệu, nên có thể kiểm
tra model học đúng hay sai (hệ số thật, hiệu ứng nhân quả thật, cụm thật, ...).
Các bộ "thực tế" (house prices, churn, fraud, ...) mô phỏng cấu trúc của dữ liệu thật:
biến phân loại, tương quan, phi tuyến, mất cân bằng, và (tuỳ chọn) dữ liệu bẩn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ============================================================================ toy datasets


def make_regression_1d(n: int = 100, kind: str = "linear", noise: float = 1.0, seed: int = 0):
    """X shape (n, 1), y shape (n,). ``kind``: linear | poly | sin | step."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-3, 3, n))
    f = {
        "linear": lambda t: 2.0 * t + 1.0,
        "poly": lambda t: 0.5 * t**3 - t**2 + 2,
        "sin": lambda t: 3 * np.sin(1.5 * t),
        "step": lambda t: np.where(t > 0, 3.0, -1.0),
    }[kind]
    y = f(x) + rng.normal(0, noise, n)
    return x.reshape(-1, 1), y


def make_classification_2d(kind: str = "blobs", n: int = 300, noise: float = 0.2, seed: int = 0):
    """Dữ liệu 2D để vẽ decision boundary. ``kind``: linear | blobs | moons | circles | xor | spiral."""
    rng = np.random.default_rng(seed)
    if kind == "linear":
        X = rng.normal(0, 1, (n, 2))
        y = (X[:, 0] + 0.7 * X[:, 1] + rng.normal(0, noise, n) > 0).astype(int)
    elif kind == "blobs":
        centers = np.array([[-1.5, -1.0], [1.5, 1.0]])
        y = rng.integers(0, 2, n)
        X = centers[y] + rng.normal(0, 0.6 + noise, (n, 2))
    elif kind == "moons":
        t = rng.uniform(0, np.pi, n)
        y = rng.integers(0, 2, n)
        X = np.where(y[:, None] == 0, np.c_[np.cos(t), np.sin(t)], np.c_[1 - np.cos(t), 0.5 - np.sin(t)])
        X = X + rng.normal(0, noise, (n, 2))
    elif kind == "circles":
        t = rng.uniform(0, 2 * np.pi, n)
        y = rng.integers(0, 2, n)
        r = np.where(y == 0, 1.0, 0.45)
        X = np.c_[r * np.cos(t), r * np.sin(t)] + rng.normal(0, noise * 0.5, (n, 2))
    elif kind == "xor":
        X = rng.uniform(-1, 1, (n, 2))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
        X = X + rng.normal(0, noise * 0.3, (n, 2))
    elif kind == "spiral":
        y = rng.integers(0, 2, n)
        t = rng.uniform(0.25, 3 * np.pi, n)
        sign = np.where(y == 0, 1, -1)
        X = np.c_[sign * t * np.cos(t), sign * t * np.sin(t)] / 3 + rng.normal(0, noise * 0.3, (n, 2))
    else:
        raise ValueError(f"kind không hợp lệ: {kind}")
    return X, y


def load_digits_data(as_float: bool = True):
    """sklearn digits (1797 ảnh 8×8, nhãn 0–9) — có sẵn offline. Trả về (X, y, images)."""
    from sklearn.datasets import load_digits

    d = load_digits()
    X = d.data / 16.0 if as_float else d.data
    return X, d.target, d.images


SHAPE_CLASSES = ("tròn", "vuông", "tam giác", "chữ thập", "vành khuyên")
SHAPE_CLASSES_EXTRA = ("ngôi sao", "lục giác", "bán nguyệt", "chữ L")       # lớp mới — cho bài transfer learning


def _polygon_mask(xx, yy, pts):
    """Điểm trong đa giác (luật chẵn–lẻ, đa giác bất kỳ)."""
    inside = np.zeros_like(xx, dtype=bool)
    for (x1, y1), (x2, y2) in zip(pts, np.roll(pts, -1, axis=0)):
        crosses = (y1 > yy) != (y2 > yy)
        x_at = (x2 - x1) * (yy - y1) / (y2 - y1 + 1e-12) + x1
        inside ^= crosses & (xx < x_at)
    return inside


def _shape_mask(kind, xx, yy, cx, cy, r, th):
    """Mặt nạ bool của một hình tâm (cx, cy), "bán kính" r, góc xoay th."""
    u = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)            # toạ độ trong hệ quay theo hình
    v = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
    d = np.hypot(u, v)
    if kind == "tròn":
        return d <= r
    if kind == "vuông":
        return (np.abs(u) <= r * 0.8) & (np.abs(v) <= r * 0.8)
    if kind == "tam giác":
        angles = th + np.array([np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3])
        pts = np.c_[cx + r * np.cos(angles), cy + r * np.sin(angles)]
        mask = np.ones_like(xx, dtype=bool)
        for i in range(3):
            (x1, y1), (x2, y2), (x3, y3) = pts[i], pts[(i + 1) % 3], pts[(i + 2) % 3]
            side = (x2 - x1) * (yy - y1) - (y2 - y1) * (xx - x1)
            ref = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            mask &= side * ref >= 0                                  # cùng phía với đỉnh còn lại
        return mask
    if kind == "chữ thập":
        arm = r * 0.3
        return ((np.abs(u) <= arm) & (np.abs(v) <= r)) | ((np.abs(v) <= arm) & (np.abs(u) <= r))
    if kind == "vành khuyên":
        return (d <= r) & (d >= r * 0.55)
    if kind == "ngôi sao":
        angles = th + np.pi / 2 + np.arange(10) * np.pi / 5
        radii = np.where(np.arange(10) % 2 == 0, r, r * 0.45)
        return _polygon_mask(xx, yy, np.c_[cx + radii * np.cos(angles), cy + radii * np.sin(angles)])
    if kind == "lục giác":
        angles = th + np.arange(6) * np.pi / 3
        return _polygon_mask(xx, yy, np.c_[cx + r * np.cos(angles), cy + r * np.sin(angles)])
    if kind == "bán nguyệt":
        return (d <= r) & (v >= -r * 0.15)
    if kind == "chữ L":
        arm = r * 0.35
        return ((np.abs(u + r * 0.5) <= arm) & (np.abs(v) <= r)) | ((np.abs(v - r * 0.65) <= arm) & (np.abs(u) <= r))
    raise ValueError(f"không biết hình {kind!r}")


def make_shapes(n: int = 2000, size: int = 28, noise: float = 0.15, region: str = "full", seed: int = 0,
                classes=None, rgb: bool = False, color_bias: float = 0.0):
    """Ảnh ``size × size`` chứa một hình (mặc định: tròn, vuông, tam giác, chữ thập, vành khuyên) ở vị trí, kích thước,
    góc xoay, độ sáng ngẫu nhiên + nhiễu Gaussian — bộ ảnh nhỏ sinh offline cho chương CNN / Computer Vision.

    ``region``: ``"full"`` (tâm hình ở bất kỳ đâu), ``"left"`` / ``"right"`` (tâm hình chỉ ở nửa trái / phải ảnh)
    — để thử khả năng tổng quát hoá theo phép dịch. ``classes``: tên các lớp (mặc định ``SHAPE_CLASSES``; có thể dùng
    ``SHAPE_CLASSES_EXTRA``). ``rgb=True``: ảnh màu (n, size, size, 3) — nền tối, hình có màu ngẫu nhiên; với xác suất
    ``color_bias`` màu hình là **màu đặc trưng của lớp** (tương quan giả để thử "học đường tắt").
    Trả về ``(X, y)`` với X float32 trong [0, 1]: (n, size, size) hoặc (n, size, size, 3).
    """
    classes = SHAPE_CLASSES if classes is None else tuple(classes)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    X = np.zeros((n, size, size, 3) if rgb else (n, size, size), dtype=np.float32)
    y = rng.integers(0, len(classes), n)
    palette = np.array([[0.95, 0.25, 0.2], [0.2, 0.8, 0.3], [0.25, 0.45, 0.95], [0.95, 0.85, 0.2], [0.85, 0.3, 0.9],
                        [0.2, 0.85, 0.9], [0.95, 0.6, 0.2], [0.6, 0.6, 0.6], [0.55, 0.35, 0.2]])
    for k in range(n):
        r = rng.uniform(0.16, 0.26) * size
        margin = r + 1
        lo_x, hi_x = margin, size - margin
        if region == "left":
            hi_x = size / 2
        elif region == "right":
            lo_x = size / 2
        cx, cy = rng.uniform(lo_x, hi_x), rng.uniform(margin, size - margin)
        th = rng.uniform(0, 2 * np.pi)
        mask = _shape_mask(classes[y[k]], xx, yy, cx, cy, r, th)
        if rgb:
            X[k] = rng.uniform(0.0, 0.25, 3)                           # màu nền
            if rng.random() < color_bias:
                color = np.clip(palette[y[k] % len(palette)] + rng.normal(0, 0.05, 3), 0, 1)
            else:
                color = rng.uniform(0.3, 1.0, 3)
            X[k][mask] = color
        else:
            X[k][mask] = rng.uniform(0.6, 1.0)
    X += rng.normal(0, noise, X.shape).astype(np.float32)
    return np.clip(X, 0, 1), y


def make_shape_scenes(n: int = 500, size: int = 64, max_objects: int = 3, noise: float = 0.1, seed: int = 0):
    """Ảnh xám ``size × size`` chứa 1–``max_objects`` hình (lớp ``SHAPE_CLASSES``) không chồng nhau — dữ liệu cho
    phát hiện vật thể và phân đoạn.

    Trả về ``(X, targets, masks)``: X (n, size, size) float32; ``targets[i]`` = dict ``boxes`` (k, 4) theo
    [x1, y1, x2, y2] (pixel, bao sát hình) và ``labels`` (k,); ``masks`` (n, size, size) int — 0 là nền, c + 1 là lớp c.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    X = np.zeros((n, size, size), dtype=np.float32)
    masks = np.zeros((n, size, size), dtype=np.int64)
    targets = []
    for k in range(n):
        boxes, labels = [], []
        for _ in range(rng.integers(1, max_objects + 1)):
            for _attempt in range(30):
                r = rng.uniform(0.09, 0.16) * size
                cx, cy = rng.uniform(r + 1, size - r - 1, 2)
                if all(abs(cx - (b[0] + b[2]) / 2) > r + (b[2] - b[0]) / 2 + 2 or abs(cy - (b[1] + b[3]) / 2) > r + (b[3] - b[1]) / 2 + 2
                       for b in boxes):
                    break
            else:
                continue
            c = rng.integers(len(SHAPE_CLASSES))
            m = _shape_mask(SHAPE_CLASSES[c], xx, yy, cx, cy, r, rng.uniform(0, 2 * np.pi))
            if m.sum() < 4:
                continue
            ys, xs = np.nonzero(m)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            labels.append(c)
            X[k][m] = rng.uniform(0.6, 1.0)
            masks[k][m] = c + 1
        targets.append({"boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4), "labels": np.array(labels, dtype=np.int64)})
    X += rng.normal(0, noise, X.shape).astype(np.float32)
    return np.clip(X, 0, 1), targets, masks


# ============================================================================ realistic tabular

DISTRICTS = {"Quận 1": 2.2, "Quận 3": 1.8, "Bình Thạnh": 1.35, "Quận 7": 1.25, "Thủ Đức": 1.0}


def make_house_prices(n: int = 1000, seed: int = 0, dirty: bool = False) -> pd.DataFrame:
    """Giá nhà (tỷ VND) phụ thuộc phi tuyến vào diện tích, quận, tuổi nhà, ...

    ``dirty=True`` thêm: missing value, outlier do gõ nhầm, category viết không thống nhất, dòng trùng.
    """
    rng = np.random.default_rng(seed)
    district = rng.choice(list(DISTRICTS), n, p=[0.12, 0.15, 0.23, 0.22, 0.28])
    area = np.round(rng.lognormal(np.log(75), 0.4, n), 1)
    bedrooms = np.clip(np.round(area / 30 + rng.normal(0, 0.7, n)), 1, 6).astype(int)
    bathrooms = np.clip(bedrooms - rng.integers(0, 2, n), 1, 5)
    age = np.clip(rng.gamma(2.0, 7.0, n), 0, 60).round().astype(int)
    distance = np.round(np.array([rng.uniform(0.5, 4) if d in ("Quận 1", "Quận 3") else rng.uniform(3, 18)
                                  for d in district]), 1)
    has_garden = rng.random(n) < np.clip(0.05 + area / 600, 0, 0.6)
    mult = np.array([DISTRICTS[d] for d in district])
    price = (0.06 * area**0.95 * mult * np.exp(-0.010 * age) * (1 + 0.08 * has_garden)
             - 0.03 * distance) * rng.lognormal(0, 0.12, n)
    df = pd.DataFrame({
        "area_m2": area, "bedrooms": bedrooms, "bathrooms": bathrooms, "age_years": age,
        "district": district, "distance_to_center_km": distance, "has_garden": has_garden,
        "price_billion_vnd": np.round(np.maximum(price, 0.5), 3),
    })
    if dirty:
        df = _make_dirty(df, rng, numeric_missing=["age_years", "bathrooms", "distance_to_center_km"],
                         typo_col="area_m2", category_col="district")
    return df


def make_customer_churn(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Khách hàng viễn thông rời bỏ (churn ≈ 25%). Logit thật được ghi trong docstring chương 07."""
    rng = np.random.default_rng(seed)
    contract = rng.choice(["month-to-month", "one_year", "two_year"], n, p=[0.55, 0.25, 0.20])
    tenure = np.where(contract == "month-to-month", rng.integers(0, 36, n), rng.integers(6, 73, n))
    internet = rng.choice(["dsl", "fiber", "none"], n, p=[0.35, 0.45, 0.20])
    monthly = np.round(20 + 25 * (internet == "dsl") + 55 * (internet == "fiber") + rng.normal(10, 8, n), 2)
    payment = rng.choice(["e-wallet", "bank_transfer", "credit_card", "cash"], n, p=[0.35, 0.25, 0.25, 0.15])
    support_calls = rng.poisson(1.2 + 0.8 * (internet == "fiber"), n)
    senior = rng.random(n) < 0.16
    n_products = rng.integers(1, 5, n)
    total = np.round(tenure * monthly * rng.uniform(0.95, 1.05, n), 2)
    logit = (-1.6 + 1.3 * (contract == "month-to-month") - 0.5 * (contract == "two_year")
             - 0.035 * tenure + 0.018 * (monthly - 70) + 0.38 * support_calls
             + 0.45 * (internet == "fiber") + 0.35 * (payment == "e-wallet") + 0.3 * senior - 0.2 * (n_products - 1))
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)], "tenure_months": tenure, "contract": contract,
        "internet_service": internet, "monthly_charges": monthly, "total_charges": total,
        "payment_method": payment, "support_calls": support_calls, "senior": senior,
        "n_products": n_products, "churn": churn,
    })


def make_fraud_transactions(n: int = 20000, fraud_rate: float = 0.01, seed: int = 0) -> pd.DataFrame:
    """Giao dịch thẻ với tỉ lệ gian lận rất thấp (mặc định 1%) — dùng cho chương 19 & project 3."""
    rng = np.random.default_rng(seed)
    amount = np.round(rng.lognormal(np.log(350_000), 1.0, n), -3)
    hour = rng.choice(24, n, p=_hour_profile())
    category = rng.choice(["grocery", "electronics", "travel", "fashion", "food", "online_games"], n,
                          p=[0.3, 0.12, 0.08, 0.2, 0.22, 0.08])
    distance = np.round(rng.exponential(8, n), 1)
    foreign = rng.random(n) < 0.04
    tx_24h = rng.poisson(2.0, n)
    card_age = rng.integers(1, 3650, n)
    score = (0.9 * np.log(amount / 350_000) + 1.2 * ((hour <= 4) | (hour >= 23)) + 1.6 * foreign
             + 0.04 * distance + 0.35 * tx_24h - 0.0006 * card_age
             + 1.0 * np.isin(category, ["electronics", "online_games"]))
    intercept = _solve_intercept(score, fraud_rate)
    is_fraud = (rng.random(n) < 1 / (1 + np.exp(-(intercept + score)))).astype(int)
    return pd.DataFrame({
        "amount_vnd": amount, "hour": hour, "merchant_category": category, "distance_from_home_km": distance,
        "is_foreign": foreign, "n_tx_last_24h": tx_24h, "card_age_days": card_age, "is_fraud": is_fraud,
    })


def make_customer_segments(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Khách hàng trung tâm thương mại với 5 phân khúc thật (cột ``true_segment``)."""
    rng = np.random.default_rng(seed)
    segments = {
        "tiết kiệm": ([25, 20, 45, 2], [6, 8, 12, 1]),
        "trung lưu": ([55, 50, 38, 4], [8, 8, 10, 1.5]),
        "chi tiêu mạnh": ([90, 85, 32, 8], [10, 7, 7, 2]),
        "giàu thận trọng": ([95, 20, 48, 3], [10, 8, 9, 1]),
        "trẻ bốc đồng": ([25, 80, 23, 7], [6, 8, 3, 2]),
    }
    names = list(segments)
    seg = rng.choice(len(names), n, p=[0.2, 0.3, 0.15, 0.15, 0.2])
    rows = np.array([rng.normal(segments[names[s]][0], segments[names[s]][1]) for s in seg])
    df = pd.DataFrame(rows, columns=["annual_income_musd", "spending_score", "age", "visits_per_month"])
    df["annual_income_musd"] = df["annual_income_musd"].clip(5).round(1)
    df["spending_score"] = df["spending_score"].clip(1, 100).round()
    df["age"] = df["age"].clip(16, 80).round()
    df["visits_per_month"] = df["visits_per_month"].clip(0).round(1)
    df["true_segment"] = [names[s] for s in seg]
    return df


def make_sales_series(n_days: int = 3 * 365, start: str = "2022-01-01", seed: int = 0) -> pd.DataFrame:
    """Doanh số bán lẻ theo ngày: trend + mùa vụ tuần + mùa vụ năm (Tết) + khuyến mãi + nhiễu."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    t = np.arange(n_days)
    trend = 200 + 0.08 * t
    weekly = np.array([0.92, 0.9, 0.93, 0.97, 1.05, 1.18, 1.12])[dates.dayofweek]
    doy = dates.dayofyear.to_numpy()
    yearly = 1 + 0.12 * np.sin(2 * np.pi * (doy - 80) / 365.25) + 0.35 * np.exp(-0.5 * ((doy - 30) / 7) ** 2)
    promo = rng.random(n_days) < 0.06
    holiday = np.isin(dates.strftime("%m-%d"), ["01-01", "04-30", "05-01", "09-02", "12-24", "12-25"])
    sales = trend * weekly * yearly * (1 + 0.3 * promo + 0.25 * holiday) * rng.lognormal(0, 0.06, n_days)
    return pd.DataFrame({"date": dates, "sales": np.round(sales, 1), "promo": promo.astype(int),
                         "is_holiday": holiday.astype(int)})


GENRES = ["hành động", "tình cảm", "hài", "kinh dị", "khoa học viễn tưởng"]


def make_ratings(n_users: int = 300, n_items: int = 200, density: float = 0.08, k: int = 5,
                 selection_strength: float = 2.0, seed: int = 0):
    """Ma trận rating 1–5 sinh từ mô hình nhân tố ẩn + bias. Trả về (ratings_df, items_df).

    Giống dữ liệu thật ở hai điểm: item phổ biến được rate nhiều hơn (phân phối Zipf), và user **tự chọn**
    item hợp gu để xem/rate (xác suất chọn ∝ độ phổ biến × exp(selection_strength · sở thích)) → rating quan sát
    được bị lệch cao so với toàn bộ cặp user–item (missing not at random). ``selection_strength=0``: chọn chỉ theo
    độ phổ biến.
    """
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 0.7, (n_users, k))
    V = np.abs(rng.normal(0, 0.7, (n_items, k)))
    b_u, b_i = rng.normal(0, 0.35, n_users), rng.normal(0, 0.45, n_items)
    popularity = 1 / np.arange(1, n_items + 1) ** 0.8
    popularity = rng.permutation(popularity / popularity.sum())
    affinity = b_i[None, :] + U @ V.T
    n_per_user = np.clip(rng.lognormal(np.log(density * n_items), 0.6, n_users).astype(int), 3, n_items // 2)
    chunks = []
    for u in range(n_users):
        w = popularity * np.exp(selection_strength * affinity[u])
        chosen = rng.choice(n_items, n_per_user[u], replace=False, p=w / w.sum())
        chunks.append(pd.DataFrame({"user_id": u, "item_id": chosen}))
    pairs = pd.concat(chunks, ignore_index=True)
    u, i = pairs["user_id"].to_numpy(), pairs["item_id"].to_numpy()
    raw = 3.4 + b_u[u] + affinity[u, i] + rng.normal(0, 0.4, len(u))
    pairs["rating"] = np.clip(np.round(raw), 1, 5).astype(int)
    pairs["timestamp"] = rng.integers(1_640_000_000, 1_700_000_000, len(u))
    pairs = pairs.sample(frac=1, random_state=seed).reset_index(drop=True)
    main_genre = np.argmax(V, axis=1)
    items_df = pd.DataFrame({
        "item_id": np.arange(n_items), "title": [f"Phim {j:03d}" for j in range(n_items)],
        "genre": [GENRES[g % len(GENRES)] for g in main_genre],
        "year": rng.integers(1995, 2025, n_items),
    })
    return pairs, items_df


_POS = ["sản phẩm rất tốt", "giao hàng nhanh", "chất lượng tuyệt vời", "đóng gói cẩn thận", "rất đáng tiền",
        "dùng rất thích", "pin trâu", "vải mát và đẹp", "shop tư vấn nhiệt tình", "sẽ ủng hộ tiếp"]
_NEG = ["hàng bị lỗi", "giao hàng quá chậm", "chất lượng kém", "không giống mô tả", "phí tiền",
        "dùng được hai ngày đã hỏng", "pin rất yếu", "vải mỏng và xấu", "shop trả lời thô lỗ", "thất vọng"]
_NEU = ["tạm được", "bình thường", "giá hợp lý", "giống ảnh", "chưa dùng nhiều nên chưa biết"]
_NEGATED_POS = ["không tốt lắm", "không đáng tiền", "không hài lòng", "không như kỳ vọng"]
_PRODUCTS = ["điện thoại", "áo thun", "tai nghe", "sách", "nồi cơm điện", "giày", "bàn phím"]


def make_text_reviews(n: int = 2000, seed: int = 0, label_noise: float = 0.03) -> pd.DataFrame:
    """Review sản phẩm tiếng Việt, nhãn ``label`` (0 = tiêu cực, 1 = tích cực) và ``sentiment`` 3 lớp.

    Có phủ định ("không tốt lắm") để thấy giới hạn của Bag-of-Words.
    """
    rng = np.random.default_rng(seed)
    texts, sentiments = [], []
    for _ in range(n):
        s = rng.choice(["pos", "neg", "neu"], p=[0.45, 0.4, 0.15])
        product = rng.choice(_PRODUCTS)
        if s == "pos":
            parts = list(rng.choice(_POS, rng.integers(1, 4), replace=False))
        elif s == "neg":
            pool = _NEG + _NEGATED_POS
            parts = list(rng.choice(pool, rng.integers(1, 4), replace=False))
        else:
            parts = list(rng.choice(_NEU, rng.integers(1, 3), replace=False))
        if rng.random() < 0.3:
            parts.insert(0, f"mua {product} cho gia đình")
        text = ", ".join(parts)
        texts.append(text[0].upper() + text[1:] + rng.choice([".", "!", "...", ""]))
        sentiments.append(s)
    sentiments = np.array(sentiments)
    label = (sentiments == "pos").astype(int)
    flip = rng.random(n) < label_noise
    label[flip] = 1 - label[flip]
    return pd.DataFrame({"text": texts, "sentiment": sentiments, "label": label})


# Mỗi chủ đề con: (chủ ngữ, động từ, tân ngữ); mỗi phần tử là một nhóm từ ĐỒNG NGHĨA (dùng thay thế nhau).
_NEWS = {
    "thể thao": {
        "bóng đá": (
            [("đội tuyển Việt Nam", "tuyển Việt Nam"), ("huấn luyện viên", "HLV"), ("tiền đạo", "chân sút"),
             ("thủ môn", "thủ thành"), ("câu lạc bộ", "CLB")],
            [("ghi bàn", "lập công"), ("giành chiến thắng", "đánh bại"), ("để thua", "thất thủ"), ("cầm hòa",)],
            [("trận đấu", "trận cầu"), ("vòng loại World Cup",), ("giải vô địch quốc gia", "V-League"),
             ("đối thủ", "đội bạn"), ("người hâm mộ", "cổ động viên")]),
        "điền kinh": (
            [("vận động viên", "VĐV"), ("tay vợt",), ("kình ngư",), ("đoàn thể thao",)],
            [("giành huy chương vàng", "đoạt HCV"), ("phá kỷ lục", "xô đổ kỷ lục"), ("lọt vào chung kết", "vào chung kết")],
            [("SEA Games",), ("giải đấu", "giải"), ("đường chạy",), ("bể bơi", "hồ bơi"), ("thành tích",)]),
    },
    "kinh tế": {
        "tài chính": (
            [("nhà đầu tư",), ("ngân hàng",), ("doanh nghiệp", "công ty"), ("Ngân hàng Nhà nước",)],
            [("tăng mạnh", "tăng vọt"), ("giảm sâu", "lao dốc"), ("điều chỉnh",), ("công bố",)],
            [("lãi suất",), ("cổ phiếu", "mã chứng khoán"), ("VN-Index", "chỉ số chứng khoán"), ("tỷ giá",),
             ("lợi nhuận", "lãi ròng")]),
        "bất động sản": (
            [("chủ đầu tư",), ("người mua nhà",), ("nhà bán lẻ", "chuỗi bán lẻ")],
            [("mở bán", "chào bán"), ("tăng giá", "đẩy giá"), ("giảm giá", "hạ giá")],
            [("căn hộ", "chung cư"), ("dự án",), ("giá nhà đất", "giá bất động sản"), ("thị trường",), ("hàng hóa",)]),
    },
    "công nghệ": {
        "thiết bị": (
            [("Apple",), ("Samsung",), ("các hãng điện thoại", "các nhà sản xuất điện thoại"), ("người dùng",)],
            [("ra mắt", "trình làng"), ("nâng cấp",), ("cập nhật",)],
            [("điện thoại thông minh", "smartphone"), ("máy tính bảng", "tablet"), ("phiên bản mới",), ("camera",),
             ("thị trường",)]),
        "phần mềm": (
            [("các công ty công nghệ", "các hãng công nghệ"), ("lập trình viên", "kỹ sư phần mềm"),
             ("công ty khởi nghiệp", "startup")],
            [("phát triển",), ("triển khai", "ứng dụng"), ("phát hiện lỗ hổng",)],
            [("trí tuệ nhân tạo", "AI"), ("ứng dụng di động", "app"), ("dữ liệu",), ("mô hình ngôn ngữ",),
             ("an ninh mạng",)]),
    },
    "sức khỏe": {
        "y tế": (
            [("bác sĩ", "y bác sĩ"), ("bệnh viện",), ("bệnh nhân", "người bệnh"), ("Bộ Y tế",)],
            [("điều trị", "chữa trị"), ("phát hiện",), ("cảnh báo",)],
            [("ca mắc", "ca bệnh"), ("sốt xuất huyết",), ("vắc xin", "vaccine"), ("ung thư",), ("ca phẫu thuật", "ca mổ")]),
        "dinh dưỡng": (
            [("chuyên gia dinh dưỡng",), ("người cao tuổi",), ("trẻ em",)],
            [("khuyến cáo", "khuyên"), ("bổ sung",), ("tập luyện", "vận động")],
            [("chế độ ăn", "thực đơn"), ("giấc ngủ",), ("rau xanh",), ("vitamin",), ("cân nặng",)]),
    },
    "giáo dục": {
        "thi cử": (
            [("thí sinh", "sĩ tử"), ("học sinh",), ("Bộ Giáo dục",), ("các trường đại học", "các trường")],
            [("công bố",), ("đăng ký",), ("đạt điểm cao",)],
            [("kỳ thi tốt nghiệp", "kỳ thi"), ("điểm chuẩn",), ("đề thi",), ("xét tuyển", "tuyển sinh"), ("học bổng",)]),
        "giảng dạy": (
            [("giáo viên", "thầy cô"), ("nhà trường",), ("sinh viên",)],
            [("giảng dạy", "dạy"), ("áp dụng",), ("đổi mới",)],
            [("chương trình học", "giáo trình"), ("sách giáo khoa",), ("học trực tuyến", "học online"), ("lớp học",),
             ("ứng dụng di động", "app")]),
    },
    "du lịch": {
        "điểm đến": (
            [("du khách", "khách du lịch"), ("khách quốc tế",), ("các công ty lữ hành",)],
            [("đổ về",), ("khám phá",), ("check-in",)],
            [("Đà Nẵng",), ("Phú Quốc",), ("Hội An",), ("bãi biển",), ("danh lam thắng cảnh",)]),
        "dịch vụ": (
            [("khách sạn", "khu nghỉ dưỡng"), ("hãng hàng không",)],
            [("giảm giá", "khuyến mãi"), ("tăng chuyến", "mở thêm đường bay"), ("kín phòng", "hết phòng")],
            [("dịp hè", "mùa hè"), ("tour", "tour du lịch"), ("vé máy bay",), ("lượng khách",), ("thị trường",)]),
    },
}
_NEWS_TIME = ["Hôm qua", "Sáng nay", "Tuần trước", "Năm nay", "Trong tháng này", "Chiều 12/5", "Đầu năm"]
_NEWS_SOURCE = ["báo cáo mới nhất", "các chuyên gia", "thông tin từ cơ quan chức năng", "số liệu thống kê"]
_NEWS_EXTRA = ["tại Hà Nội", "tại TP.HCM", "khiến nhiều người bất ngờ", "trong bối cảnh nhiều khó khăn",
               "theo kế hoạch", "nhiều hơn so với cùng kỳ năm ngoái", "", "", ""]
NEWS_TOPICS = tuple(_NEWS)
NEWS_SYNONYMS = tuple(sorted({grp for sub in _NEWS.values() for parts in sub.values() for slot in parts
                              for grp in slot if len(grp) > 1}))
NEWS_LEXICON = tuple(sorted({w for sub in _NEWS.values() for parts in sub.values() for slot in parts
                             for grp in slot for w in grp if " " in w}
                            | {"cơ quan chức năng", "thống kê", "chuyên gia", "báo cáo", "cùng kỳ", "năm ngoái",
                               "bối cảnh", "khó khăn", "kế hoạch", "bất ngờ", "Hà Nội", "trong khi đó"}))


def _news_clause(rng, parts):
    def pick(groups):
        return rng.choice(groups[rng.integers(len(groups))])

    subj, verb, obj = parts
    return f"{pick(subj)} {pick(verb)} {pick(obj)}"


def make_news_vi(n: int = 3000, seed: int = 0, distractor: float = 0.25, label_noise: float = 0.03) -> pd.DataFrame:
    """Tin tức ngắn tiếng Việt thuộc 6 chủ đề (mỗi chủ đề 2 chủ đề con) — ngữ liệu cho chương NLP.

    Mỗi khái niệm có thể xuất hiện dưới nhiều **từ đồng nghĩa** (``NEWS_SYNONYMS``: "smartphone" / "điện thoại thông
    minh", "HLV" / "huấn luyện viên"...) dùng thay thế nhau trong cùng ngữ cảnh; một số từ dùng chung giữa các chủ đề
    ("thị trường", "ứng dụng di động", "giảm giá", "công bố"); với xác suất ``distractor`` câu có thêm một mệnh đề về
    chủ đề **khác**; ``label_noise`` nhãn bị gán sai. ``NEWS_LEXICON``: danh sách từ nhiều âm tiết để tách từ.
    Trả về DataFrame ``text``, ``topic``, ``subtopic``.
    """
    rng = np.random.default_rng(seed)
    pairs = [(t, s) for t, subs in _NEWS.items() for s in subs]
    rows = []
    for _ in range(n):
        topic, sub = pairs[rng.integers(len(pairs))]
        parts = _NEWS[topic][sub]
        style = rng.integers(4)
        main = _news_clause(rng, parts)
        extra = rng.choice(_NEWS_EXTRA)
        if style == 0:
            text = f"{rng.choice(_NEWS_TIME)}, {main} {extra}".strip()
        elif style == 1:
            text = f"Theo {rng.choice(_NEWS_SOURCE)}, {main} {extra}".strip()
        elif style == 2:
            text = main
        else:
            text = f"{main}, {_news_clause(rng, parts)}"
        text = text[0].upper() + text[1:] + "."
        if rng.random() < 0.5:
            second = _news_clause(rng, parts)
            text += f" {second[0].upper() + second[1:]} {rng.choice(_NEWS_EXTRA)}".rstrip() + "."
        if rng.random() < distractor:
            other_t, other_s = pairs[rng.integers(len(pairs))]
            while other_t == topic:
                other_t, other_s = pairs[rng.integers(len(pairs))]
            text += f" Trong khi đó, {_news_clause(rng, _NEWS[other_t][other_s])}."
        rows.append((text, topic, sub))
    df = pd.DataFrame(rows, columns=["text", "topic", "subtopic"])
    noisy = rng.random(n) < label_noise
    df.loc[noisy, "topic"] = rng.choice(NEWS_TOPICS, noisy.sum())
    return df


def make_causal_marketing(n: int = 5000, seed: int = 0, randomized: bool = False) -> pd.DataFrame:
    """Dữ liệu quan sát: ai nhận coupon (treatment) KHÔNG ngẫu nhiên (bị confounding).

    Cột ``true_effect`` = hiệu ứng nhân quả thật của coupon lên chi tiêu từng người (ITE),
    ATE thật = ``true_effect.mean()``. ``randomized=True``: coupon được phát **ngẫu nhiên** 50/50 (thí nghiệm A/B —
    không confounding). Dùng ở chương 38.
    """
    rng = np.random.default_rng(seed)
    age = rng.uniform(18, 70, n)
    income = rng.lognormal(np.log(15), 0.4, n)  # triệu VND / tháng
    past = rng.poisson(3, n)
    logit_t = -2.0 + 0.35 * past + 0.04 * (income - 15)
    p_treat = np.full(n, 0.5) if randomized else 1 / (1 + np.exp(-logit_t))
    treated = (rng.random(n) < p_treat).astype(int)
    effect = 120 + 4.0 * (45 - age)  # coupon hiệu quả hơn với người trẻ (nghìn VND)
    spend = 300 + 25 * income + 60 * past + effect * treated + rng.normal(0, 120, n)
    return pd.DataFrame({"age": age.round(1), "income_mvnd": income.round(2), "past_purchases": past,
                         "coupon": treated, "spend_kvnd": spend.round(1), "true_effect": effect.round(1)})


CREDIT_OBSERVATION_DATE = "2025-07-01"
CREDIT_MONTHLY_RATE = 0.015


def make_credit_applications(n: int = 12000, seed: int = 0, messy: bool = False) -> pd.DataFrame:
    """Hồ sơ vay tiêu dùng 01/2022 → 06/2025, nhãn ``defaulted`` = vỡ nợ (quá hạn ≥ 90 ngày) trong 12 tháng đầu (chương 41).

    - Nhãn **chưa trưởng thành**: hồ sơ chưa đủ 12 tháng tính tới ngày quan sát ``CREDIT_OBSERVATION_DATE`` (2025-07-01)
      có ``defaulted`` = NaN.
    - ``early_dpd30_3m`` = quá hạn ≥ 30 ngày trong 3 tháng đầu — **nhãn sớm** (proxy) có sau 3 tháng thay vì 12; là thông tin
      **sau** quyết định cho vay nên **không** được dùng làm đặc trưng (bẫy rò rỉ), chỉ dùng để giám sát.
    - Từ 01/2024 có **cú sốc kinh tế**: tỉ lệ vỡ nợ tăng ở mọi nhóm, mạnh hơn với lao động tự do/hợp đồng (trôi khái niệm).
    - Khoản trả hằng tháng theo lãi suất ``CREDIT_MONTHLY_RATE`` (1.5%/tháng, trả góp đều); rủi ro tăng theo tỉ lệ
      nợ/thu nhập (DTI), số lần trễ hạn 12 tháng gần nhất, loại việc làm; giảm theo lịch sử tín dụng, tài sản đảm bảo.
    - ``messy=True``: dữ liệu **thô** như xuất từ hệ thống nghiệp vụ — dòng trùng, thứ tự lộn xộn, ngày nhiều định dạng,
      số tiền dạng chuỗi "120.000.000", thu nhập nhập nhầm đơn vị (triệu thay vì đồng), tuổi không hợp lệ, danh mục
      viết không thống nhất, giá trị thiếu.
    """
    rng = np.random.default_rng(seed)
    start, observed = pd.Timestamp("2022-01-01"), pd.Timestamp(CREDIT_OBSERVATION_DATE)
    n_days = (observed - start).days
    date = start + pd.to_timedelta(np.sort((rng.random(n) ** 0.85 * n_days).astype(int)), unit="D")  # số hồ sơ tăng dần
    customer = rng.integers(0, int(n * 0.85), n)
    age = rng.integers(21, 66, n)
    employment = rng.choice(["salaried", "self_employed", "contract", "unemployed"], n, p=[0.55, 0.25, 0.15, 0.05]).astype(object)
    base = pd.Series({"salaried": 16e6, "self_employed": 18e6, "contract": 11e6, "unemployed": 5e6})
    income = np.round(base[employment].values * rng.lognormal(0, 0.45, n) * (1 + 0.012 * (age - 21)), -5)
    term = rng.choice([12, 24, 36], n, p=[0.3, 0.4, 0.3])
    purpose = rng.choice(["consumer", "vehicle", "education", "business", "home_improvement"], n, p=[0.4, 0.2, 0.1, 0.15, 0.15])
    target_dti = np.clip(rng.beta(2.2, 5, n) * 0.9 * np.where(purpose == "vehicle", 1.2, 1.0), 0.03, 0.85)
    annuity = CREDIT_MONTHLY_RATE / (1 - (1 + CREDIT_MONTHLY_RATE) ** -term)
    loan = np.maximum(np.round(target_dti * income / annuity, -6), 5e6)
    dti = loan * annuity / income
    history = np.round((age - 20) * 12 * rng.uniform(0.05, 1.0, n)).astype(int)
    late = rng.poisson(0.25 + 0.45 * (employment == "unemployed") + 0.15 * (employment == "contract"), n)
    collateral = rng.random(n) < np.select([purpose == "vehicle", purpose == "home_improvement"], [0.85, 0.5], 0.1)
    region = rng.choice(["Hà Nội", "TP.HCM", "Đà Nẵng", "Khác"], n, p=[0.3, 0.35, 0.1, 0.25])
    dependents = rng.poisson(1.0, n)
    shock = np.asarray(date >= pd.Timestamp("2024-01-01"))
    score = (3.0 * dti + 0.55 * late + 0.9 * (employment == "unemployed") + 0.4 * (employment == "contract")
             + 0.25 * (employment == "self_employed") - 0.025 * history / 12 - 0.6 * collateral
             - 0.35 * np.log(income / 15e6) + 0.35 * (purpose == "business") + 0.08 * dependents)
    intercept = _solve_intercept(score, 0.08)
    score = score + shock * (0.45 + 0.4 * np.isin(employment, ["self_employed", "contract"]))
    latent = rng.random(n) < 1 / (1 + np.exp(-(intercept + score)))
    early = np.where(latent, rng.random(n) < 0.55, rng.random(n) < 0.04).astype(float)
    early[(date + pd.DateOffset(months=3)) > observed] = np.nan
    defaulted = latent.astype(float)
    defaulted[(date + pd.DateOffset(months=12)) > observed] = np.nan
    df = pd.DataFrame({
        "application_id": [f"A{i:06d}" for i in range(n)], "customer_id": [f"KH{c:06d}" for c in customer],
        "application_date": date, "age": age, "employment_type": employment, "monthly_income_vnd": income,
        "loan_amount_vnd": loan, "term_months": term, "purpose": purpose, "region": region, "n_dependents": dependents,
        "credit_history_months": history, "late_payments_12m": late, "has_collateral": collateral,
        "early_dpd30_3m": early, "defaulted": defaulted,
    })
    if not messy:
        return df

    out = df.copy()
    alt = rng.random(n) < 0.12
    out["application_date"] = np.where(alt, out.application_date.dt.strftime("%d/%m/%Y"), out.application_date.dt.strftime("%Y-%m-%d"))
    dotted = rng.random(n) < 0.35
    out["loan_amount_vnd"] = [f"{int(v):,}".replace(",", ".") if d else str(int(v)) for v, d in zip(out.loan_amount_vnd, dotted)]
    unit = rng.random(n) < 0.02
    out.loc[unit, "monthly_income_vnd"] = (out.loc[unit, "monthly_income_vnd"] / 1e6).round(1)   # nhập theo triệu
    out.loc[rng.random(n) < 0.03, "monthly_income_vnd"] = np.nan
    bad_age = rng.random(n) < 0.005
    out.loc[bad_age, "age"] = rng.choice([0, -1, 150, 999], bad_age.sum())
    variants = {"salaried": ["salaried", "Salaried", "SALARIED", " salaried"],
                "self_employed": ["self_employed", "Self-employed", "self employed"],
                "contract": ["contract", "Contract", "hop_dong"], "unemployed": ["unemployed", "Unemployed"]}
    out["employment_type"] = [str(rng.choice(variants[e])) if rng.random() < 0.15 else e for e in out.employment_type]
    out.loc[rng.random(n) < 0.02, "employment_type"] = None
    region_variants = {"TP.HCM": ["TP.HCM", "Ho Chi Minh", "HCM", "tp hcm"], "Hà Nội": ["Hà Nội", "Ha Noi", "HN"],
                       "Đà Nẵng": ["Đà Nẵng", "Da Nang"], "Khác": ["Khác"]}
    out["region"] = [str(rng.choice(region_variants[r])) if rng.random() < 0.2 else r for r in out.region]
    duplicates = out.iloc[rng.choice(n, int(0.015 * n), replace=False)]
    out = pd.concat([out, duplicates], ignore_index=True)
    return out.iloc[rng.permutation(len(out))].reset_index(drop=True)


# ============================================================================ helpers


def _hour_profile() -> np.ndarray:
    h = np.arange(24)
    p = 0.2 + np.exp(-0.5 * ((h - 12) / 4) ** 2) + 0.8 * np.exp(-0.5 * ((h - 20) / 2) ** 2)
    return p / p.sum()


def _solve_intercept(score: np.ndarray, target_rate: float) -> float:
    lo, hi = -30.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = np.mean(1 / (1 + np.exp(-(mid + score))))
        lo, hi = (mid, hi) if rate < target_rate else (lo, mid)
    return (lo + hi) / 2


def _make_dirty(df: pd.DataFrame, rng, numeric_missing, typo_col, category_col) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    for col in numeric_missing:
        df[col] = df[col].astype(float)
        df.loc[rng.random(n) < 0.05, col] = np.nan
    typo = rng.random(n) < 0.01
    df.loc[typo, typo_col] = df.loc[typo, typo_col] * 10  # gõ thừa số 0
    messy = rng.random(n) < 0.08
    df.loc[messy, category_col] = df.loc[messy, category_col].map(
        lambda s: rng.choice([s.lower(), s.upper(), f" {s} ", s.replace("Quận", "Q.")])
    )
    dup = df.sample(frac=0.01, random_state=int(rng.integers(1e9)))
    return pd.concat([df, dup], ignore_index=True)


__all__ = [
    "make_regression_1d",
    "make_classification_2d",
    "load_digits_data",
    "make_shapes",
    "make_shape_scenes",
    "SHAPE_CLASSES",
    "SHAPE_CLASSES_EXTRA",
    "make_house_prices",
    "make_customer_churn",
    "make_credit_applications",
    "CREDIT_OBSERVATION_DATE",
    "CREDIT_MONTHLY_RATE",
    "make_fraud_transactions",
    "make_customer_segments",
    "make_sales_series",
    "make_ratings",
    "make_text_reviews",
    "make_news_vi",
    "NEWS_TOPICS",
    "NEWS_SYNONYMS",
    "NEWS_LEXICON",
    "make_causal_marketing",
    "DISTRICTS",
    "GENRES",
]
