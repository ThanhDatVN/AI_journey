"""Bộ sinh dữ liệu: tái lập theo seed, đúng cấu trúc, đúng các tính chất mà bài học dựa vào."""

import numpy as np
import pandas as pd
import pytest

from ml_course import datasets as ds

TABLE_GENERATORS = {
    "house_prices": lambda seed: ds.make_house_prices(200, seed=seed),
    "house_prices_dirty": lambda seed: ds.make_house_prices(200, seed=seed, dirty=True),
    "customer_churn": lambda seed: ds.make_customer_churn(300, seed=seed),
    "fraud_transactions": lambda seed: ds.make_fraud_transactions(2000, seed=seed),
    "customer_segments": lambda seed: ds.make_customer_segments(200, seed=seed),
    "sales_series": lambda seed: ds.make_sales_series(120, seed=seed),
    "ratings": lambda seed: ds.make_ratings(30, 20, seed=seed)[0],
    "text_reviews": lambda seed: ds.make_text_reviews(100, seed=seed),
    "news_vi": lambda seed: ds.make_news_vi(100, seed=seed),
    "causal_marketing": lambda seed: ds.make_causal_marketing(300, seed=seed),
    "credit": lambda seed: ds.make_credit_applications(500, seed=seed),
    "credit_messy": lambda seed: ds.make_credit_applications(500, seed=seed, messy=True),
}


@pytest.mark.parametrize("name", sorted(TABLE_GENERATORS))
def test_tables_reproducible_and_seed_dependent(name):
    make = TABLE_GENERATORS[name]
    pd.testing.assert_frame_equal(make(0), make(0))
    assert not make(0).equals(make(1))


def test_toy_generators_shapes():
    X, y = ds.make_regression_1d(50, kind="sin", seed=0)
    assert X.shape == (50, 1) and y.shape == (50,)
    for kind in ["linear", "blobs", "moons", "circles", "xor", "spiral"]:
        X2, y2 = ds.make_classification_2d(kind, n=120, seed=0)
        assert X2.shape == (120, 2) and set(np.unique(y2)) <= {0, 1, 2, 3}


def test_shapes_options():
    X, y = ds.make_shapes(60, 28, noise=0.0, region="left", seed=0)
    assert X.shape == (60, 28, 28) and 0 <= X.min() and X.max() <= 1
    centres = np.array([np.nonzero(img > 0.3)[1].mean() for img in X])
    assert centres.max() < 15, "region='left' đặt hình ở nửa trái"
    _, y_extra = ds.make_shapes(80, 28, classes=ds.SHAPE_CLASSES_EXTRA, seed=0)
    assert set(y_extra) <= set(range(len(ds.SHAPE_CLASSES_EXTRA)))
    Xc, yc = ds.make_shapes(50, 28, rgb=True, color_bias=1.0, seed=0)
    assert Xc.shape == (50, 28, 28, 3)


def test_shape_scenes_targets():
    X, targets, masks = ds.make_shape_scenes(10, size=64, max_objects=3, seed=0)
    assert X.shape == masks.shape == (10, 64, 64) and len(targets) == 10
    for t in targets:
        boxes = t["boxes"]
        assert 1 <= len(boxes) <= 3 and len(t["labels"]) == len(boxes)
        assert (boxes[:, 0] < boxes[:, 2]).all() and (boxes[:, 1] < boxes[:, 3]).all()
        assert boxes.min() >= 0 and boxes.max() <= 64


def test_house_prices_dirty_has_documented_problems():
    dirty = ds.make_house_prices(1500, seed=42, dirty=True)
    assert dirty.duplicated().sum() > 0
    assert dirty[["age_years", "bathrooms", "distance_to_center_km"]].isna().any().all()
    assert dirty.district.nunique() > len(ds.DISTRICTS)
    assert dirty.area_m2.max() > 400                                   # diện tích gõ thừa số 0


def test_fraud_rate_and_segments():
    fraud = ds.make_fraud_transactions(20000, fraud_rate=0.02, seed=0)
    assert abs(fraud.is_fraud.mean() - 0.02) < 0.005
    seg = ds.make_customer_segments(600, seed=0)
    assert seg.true_segment.nunique() == 5


def test_causal_marketing_confounding_vs_randomized():
    obs = ds.make_causal_marketing(20000, seed=0)
    rct = ds.make_causal_marketing(20000, seed=0, randomized=True)
    assert abs(np.corrcoef(obs.coupon, obs.past_purchases)[0, 1]) > 0.1
    assert abs(np.corrcoef(rct.coupon, rct.past_purchases)[0, 1]) < 0.03
    naive_rct = rct.spend_kvnd[rct.coupon == 1].mean() - rct.spend_kvnd[rct.coupon == 0].mean()
    assert abs(naive_rct - rct.true_effect.mean()) < 15                # thí nghiệm ngẫu nhiên: so sánh trực tiếp không chệch


def test_news_topics():
    news = ds.make_news_vi(300, seed=0)
    assert set(news.topic) <= set(ds.NEWS_TOPICS)
    assert {"text", "topic", "subtopic"} <= set(news.columns)


def test_credit_label_maturity_proxy_and_shock():
    df = ds.make_credit_applications(6000, seed=0)
    observed = pd.Timestamp(ds.CREDIT_OBSERVATION_DATE)
    immature = (df.application_date + pd.DateOffset(months=12)) > observed
    assert df.defaulted[immature].isna().all() and df.defaulted[~immature].notna().all()
    assert set(df.defaulted.dropna().unique()) <= {0.0, 1.0}
    early_immature = (df.application_date + pd.DateOffset(months=3)) > observed
    assert df.early_dpd30_3m[early_immature].isna().all()
    before = df[df.application_date < "2024-01-01"].defaulted.mean()
    shock = df[(df.application_date >= "2024-01-01") & df.defaulted.notna()].defaulted.mean()
    assert 0.05 < before < 0.11 and shock > 1.3 * before
    labelled = df.dropna(subset=["defaulted", "early_dpd30_3m"])
    assert np.corrcoef(labelled.defaulted, labelled.early_dpd30_3m)[0, 1] > 0.3   # nhãn sớm là proxy hữu ích


def test_credit_messy_is_recoverable():
    clean = ds.make_credit_applications(2000, seed=1).set_index("application_id")
    messy = ds.make_credit_applications(2000, seed=1, messy=True)
    assert messy.duplicated().sum() > 0
    assert messy.application_date.astype(str).str.contains("/").any()
    assert messy.loan_amount_vnd.astype(str).str.contains(".", regex=False).any()
    dedup = messy.drop_duplicates("application_id").set_index("application_id")
    amounts = dedup.loan_amount_vnd.astype(str).str.replace(".", "", regex=False).astype(float)
    assert (amounts == clean.loc[dedup.index, "loan_amount_vnd"]).all()
    dates = dedup.application_date.astype(str)
    parsed = pd.to_datetime(dates, format="%Y-%m-%d", errors="coerce").fillna(pd.to_datetime(dates, format="%d/%m/%Y", errors="coerce"))
    assert (parsed.values == clean.loc[dedup.index, "application_date"].values).all()
