"""Sinh các file CSV mẫu trong data/sample/ từ ml_course.datasets (tái lập bằng seed cố định).

    python tools/make_sample_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ml_course import datasets as ds  # noqa: E402

OUT = ROOT / "data" / "sample"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ratings, items = ds.make_ratings(seed=42)
    tables = {
        "house_prices.csv": ds.make_house_prices(n=1500, seed=42),
        "house_prices_dirty.csv": ds.make_house_prices(n=1500, seed=42, dirty=True),
        "customer_churn.csv": ds.make_customer_churn(n=5000, seed=42),
        "fraud_transactions.csv": ds.make_fraud_transactions(n=30000, seed=42),
        "customer_segments.csv": ds.make_customer_segments(n=600, seed=42),
        "sales_daily.csv": ds.make_sales_series(seed=42),
        "ratings.csv": ratings,
        "items.csv": items,
        "reviews_vi.csv": ds.make_text_reviews(n=3000, seed=42),
        "causal_marketing.csv": ds.make_causal_marketing(n=5000, seed=42),
        "credit_applications_raw.csv": ds.make_credit_applications(n=12000, seed=42, messy=True),
    }
    for name, df in tables.items():
        df.to_csv(OUT / name, index=False, encoding="utf-8")
        print(f"OK {name:28s} {df.shape}")


if __name__ == "__main__":
    main()
