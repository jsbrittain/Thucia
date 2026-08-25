import numpy as np
import pandas as pd
import pytest
from thucia.core.models.utils.adapter import HGBQuantileAdapter
from thucia.core.models.utils.adapter import QuantileAdapter
from thucia.core.models.utils.adapter import residual_regression
from thucia.core.models.utils.adapter import RidgeAdapter


def _embedding_df(n=2):
    return pd.DataFrame(
        {
            "GID_2": [f"G{i}" for i in range(n)],
            "feature0": [float(i) for i in range(n)],
            "feature1": [float(n - 1 - i) for i in range(n)],
        }
    )


def _predictions_df(n_dates=6, n=2, seed=0):
    dates = pd.period_range("2020-01", periods=n_dates, freq="M")
    rows = []
    for d in dates:
        for i in range(n):
            rows.append(
                {
                    "Date": d,
                    "GID_2": f"G{i}",
                    "prediction": 10.0 + i,
                    "Cases": 12.0 + i,  # constant +2 residual for every GID
                    "horizon": 1,
                    "quantile": 0.5,
                }
            )
    return pd.DataFrame(rows)


def test_ridge_adapter_learns_bias():
    ad = RidgeAdapter(_embedding_df().set_index("GID_2"), alpha=1.0)
    df = _predictions_df()
    ad.fit(df)
    # A per-GID bias is applied to predictions
    out = ad.apply(df.copy())
    # residuals (Cases - prediction) = +2 for all rows; bias should shrink them
    assert np.abs((out["prediction"] - df["Cases"]).mean()) < 2.0


def test_quantile_adapter_fit_and_apply():
    ad = QuantileAdapter(_embedding_df().set_index("GID_2"), quantile=0.5)
    ad.fit(_predictions_df())
    out = ad.apply(_predictions_df().copy())
    assert out["prediction"].notna().all()


def test_hgb_adapter_raises_without_data():
    ad = HGBQuantileAdapter(_embedding_df().set_index("GID_2"), quantile=0.5)
    with pytest.raises(ValueError):
        ad.fit(_predictions_df()[0:0])


def test_residual_regression_ridge_reduces_error():
    preds = _embedding_df()
    out = residual_regression(_predictions_df(), preds, method="ridge", geo_col="GID_2")
    assert len(out) == len(_predictions_df())
    rmse_before = np.sqrt(
        np.mean((_predictions_df()["prediction"] - _predictions_df()["Cases"]) ** 2)
    )
    rmse_after = np.sqrt(np.mean((out["prediction"] - out["Cases"]) ** 2))
    assert rmse_after < rmse_before


def test_residual_regression_unknown_method():
    with pytest.raises(Exception):
        residual_regression(
            _predictions_df(), _embedding_df(), method="nope", geo_col="GID_2"
        )


def test_residual_regression_requires_feature_columns():
    bad = _embedding_df().rename(columns={"feature0": "x0", "feature1": "x1"})
    # No feature* columns -> nothing to regress on, but must not crash badly
    out = residual_regression(_predictions_df(), bad, method="ridge", geo_col="GID_2")
    assert len(out) == len(_predictions_df())
