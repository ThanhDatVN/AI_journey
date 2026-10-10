import pandas as pd

from fraud.data import validate
from fraud.features import add_features


def test_sample_data_is_valid(data):
    assert validate(data) == []


def test_validate_catches_problems(data):
    broken = data.head(100).copy()
    broken.loc[0, "hour"] = 25
    broken.loc[1, "merchant_category"] = "casino"
    broken.loc[2, "amount_vnd"] = -1
    issues = validate(broken)
    assert any("hour" in i for i in issues)
    assert any("casino" in i for i in issues)
    assert any("amount_vnd" in i for i in issues)


def test_validate_without_target_for_serving(data):
    assert validate(data.drop(columns="is_fraud"), require_target=False) == []


def test_night_flag_boundaries():
    df = pd.DataFrame({"amount_vnd": [1e5] * 4, "hour": [4, 5, 22, 23], "is_foreign": [False] * 4})
    assert add_features(df).is_night.tolist() == [1, 0, 0, 1]


def test_features_do_not_mutate_input():
    df = pd.DataFrame({"amount_vnd": [1e5], "hour": [3], "is_foreign": [False]})
    add_features(df)
    assert list(df.columns) == ["amount_vnd", "hour", "is_foreign"]
