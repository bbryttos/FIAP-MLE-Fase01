"""Schema validation tests using pandera."""

import os

import pandera as pa
import pytest

from src.data.preprocessing import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    load_data,
)

DATA_PATH = os.getenv("DATA_PATH", "data/Telco_customer_churn.xlsx")

RAW_SCHEMA = pa.DataFrameSchema(
    {
        "Tenure Months": pa.Column(int, checks=pa.Check.ge(0)),
        "Monthly Charges": pa.Column(float, checks=pa.Check.ge(0)),
    }
)


@pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason="Dataset not present — copy Telco_customer_churn.xlsx to data/",
)
def test_data_loads():
    X, y = load_data(DATA_PATH)
    assert len(X) >= 5000
    assert len(y) == len(X)


@pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason="Dataset not present",
)
def test_churn_value_binary():
    _, y = load_data(DATA_PATH)
    assert set(y.unique()).issubset({0, 1}), f"Unexpected values: {y.unique()}"


@pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason="Dataset not present",
)
def test_required_features_present():
    X, _ = load_data(DATA_PATH)
    for col in NUMERICAL_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES:
        assert col in X.columns, f"Missing column: {col}"


@pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason="Dataset not present",
)
def test_no_negative_tenure():
    X, _ = load_data(DATA_PATH)
    assert (X["Tenure Months"] >= 0).all()


@pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason="Dataset not present",
)
def test_pandera_schema():
    X, _ = load_data(DATA_PATH)
    RAW_SCHEMA.validate(X)
