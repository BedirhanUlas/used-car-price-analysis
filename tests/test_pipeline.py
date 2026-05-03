import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from src.train import build_models, evaluate_model, preprocess


@pytest.fixture
def sample_df():
    """Minimal vehicles DataFrame for testing (prices within filter range)."""
    return pd.DataFrame({
        "price": [8000, 15000, 22000, 5500, 31000],
        "year": [2015, 2018, 2020, 2012, 2019],
        "odometer": [60000, 30000, 15000, 90000, 25000],
        "manufacturer": ["ford", "toyota", "honda", "chevrolet", "ford"],
        "condition": ["good", "excellent", "good", "fair", "like new"],
        "cylinders": ["6 cylinders", "4 cylinders", "4 cylinders", "6 cylinders", "4 cylinders"],
        "fuel": ["gas", "gas", "gas", "gas", "hybrid"],
        "title_status": ["clean", "clean", "clean", "clean", "clean"],
        "transmission": ["automatic", "automatic", "automatic", "manual", "automatic"],
        "drive": ["4wd", "fwd", "fwd", "rwd", "fwd"],
        "type": ["sedan", "suv", "sedan", "pickup", "sedan"],
        "paint_color": ["white", "silver", "black", "red", "blue"],
        "state": ["ca", "tx", "fl", "oh", "ny"],
    })


def test_preprocess_shape(sample_df):
    X, y, feature_names = preprocess(sample_df)
    assert X.shape[0] == len(sample_df)
    assert len(y) == len(sample_df)
    assert len(feature_names) == X.shape[1]


def test_preprocess_no_nan(sample_df):
    X, y, _ = preprocess(sample_df)
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()


def test_preprocess_target_positive(sample_df):
    _, y, _ = preprocess(sample_df)
    assert (y > 0).all()


def test_build_models_returns_three():
    models = build_models()
    assert len(models) == 3


def test_evaluate_model_returns_expected_keys(sample_df):
    X, y, _ = preprocess(sample_df)
    model = LinearRegression()
    metrics = evaluate_model("LR", model, X, y, X, y)
    assert "MAE ($)" in metrics
    assert "RMSE ($)" in metrics
    assert "R²" in metrics
    assert "Train Time (s)" in metrics


def test_mae_is_non_negative(sample_df):
    X, y, _ = preprocess(sample_df)
    model = Ridge(alpha=1.0)
    metrics = evaluate_model("Ridge", model, X, y, X, y)
    assert metrics["MAE ($)"] >= 0
    assert metrics["RMSE ($)"] >= 0
