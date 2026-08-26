# Probing tests for the covariate feature-engineering helpers
# (thucia.core.models.utils.covariates). These assert real numeric properties
# of each pipeline op rather than just executing them.
import numpy as np
import pandas as pd
import pytest
from thucia.core.models.utils.covariates import _apply_pipeline_to_group
from thucia.core.models.utils.covariates import build_features
from thucia.core.models.utils.covariates import prepare_covariates


def _series(values):
    return pd.Series(values, dtype=float)


def test_shift_op():
    out = _apply_pipeline_to_group(
        _series([1.0, 2.0, 3.0, 4.0]), [{"op": "shift", "periods": 1}]
    )
    assert np.isnan(out.iloc[0])
    assert out.iloc[1:].tolist() == [1.0, 2.0, 3.0]


def test_shift_op_default_periods_is_one():
    out = _apply_pipeline_to_group(_series([1.0, 2.0, 3.0]), [{"op": "shift"}])
    assert np.isnan(out.iloc[0])
    assert out.iloc[1:].tolist() == [1.0, 2.0]


def test_rolling_mean_default_min_periods():
    out = _apply_pipeline_to_group(
        _series([1.0, 2.0, 3.0, 4.0]), [{"op": "rolling", "window": 2}]
    )
    # min_periods defaults to window -> first value NaN
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(1.5)
    assert out.iloc[2] == pytest.approx(2.5)


def test_rolling_with_min_periods_1():
    out = _apply_pipeline_to_group(
        _series([1.0, 2.0, 3.0]), [{"op": "rolling", "window": 3, "min_periods": 1}]
    )
    assert out.tolist() == [1.0, 1.5, 2.0]


def test_rolling_custom_agg():
    out = _apply_pipeline_to_group(
        _series([1.0, 2.0, 3.0, 4.0]),
        [{"op": "rolling", "window": 2, "agg": "max"}],
    )
    assert out.iloc[1] == 2.0
    assert out.iloc[3] == 4.0


def test_diff_op():
    out = _apply_pipeline_to_group(_series([1.0, 2.0, 4.0, 7.0]), [{"op": "diff"}])
    assert out.iloc[0] == pytest.approx(np.nan) or np.isnan(out.iloc[0])
    assert out.iloc[1:].tolist() == [1.0, 2.0, 3.0]


def test_pct_change_op():
    out = _apply_pipeline_to_group(
        _series([100.0, 110.0, 110.0]), [{"op": "pct_change"}]
    )
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(0.1)
    assert out.iloc[2] == pytest.approx(0.0)


def test_ema_op():
    out = _apply_pipeline_to_group(_series([1.0, 2.0, 3.0]), [{"op": "ema", "span": 2}])
    # ewm(span=2, adjust=False): alpha = 2/(span+1) = 2/3, y_t = alpha*x_t + (1-alpha)*y_{t-1}
    assert out.iloc[0] == pytest.approx(1.0)
    assert out.iloc[1] == pytest.approx(2.0 / 3 * 2 + 1.0 / 3 * 1.0)  # 5/3
    assert out.iloc[2] == pytest.approx(2.0 / 3 * 3 + 1.0 / 3 * (5.0 / 3))  # 23/9


def test_fillna_op():
    out = _apply_pipeline_to_group(
        _series([np.nan, 2.0, np.nan]), [{"op": "fillna", "value": 0}]
    )
    assert out.tolist() == [0.0, 2.0, 0.0]


def test_fillna_default_zero():
    out = _apply_pipeline_to_group(_series([np.nan, 5.0]), [{"op": "fillna"}])
    assert out.tolist() == [0.0, 5.0]


def test_clip_op():
    out = _apply_pipeline_to_group(
        _series([-5.0, 3.0, 10.0]), [{"op": "clip", "lower": 0.0, "upper": 5.0}]
    )
    assert out.tolist() == [0.0, 3.0, 5.0]


def test_lambda_op():
    out = _apply_pipeline_to_group(
        _series([0.0, 1.0, 2.0]), [{"op": "lambda", "func": np.log1p}]
    )
    assert out.tolist() == [0.0, np.log(2), np.log(3)]


def test_unknown_op_raises():
    with pytest.raises(ValueError, match="Unknown op"):
        _apply_pipeline_to_group(_series([1.0]), [{"op": "bogus"}])


def test_build_features_no_groupby():
    df = pd.DataFrame({"Date": range(4), "Cases": [1.0, 2.0, 3.0, 4.0]})
    out = build_features(
        df,
        [
            {
                "name": "lag1",
                "column": "Cases",
                "pipeline": [{"op": "shift", "periods": 1}],
            }
        ],
    )
    assert np.isnan(out.loc[0, "lag1"])
    assert out["lag1"].iloc[1:].tolist() == [1.0, 2.0, 3.0]


def test_build_features_groupby_isolates_rolling():
    # Two GID_2 with disjoint windows: a rolling window must never bleed
    # across regions.
    df = pd.DataFrame(
        {
            "Date": [0, 1, 2, 0, 1, 2],
            "GID_2": ["A", "A", "A", "B", "B", "B"],
            "x": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }
    )
    out = build_features(
        df,
        [
            {
                "name": "x_roll",
                "groupby": ["GID_2"],
                "column": "x",
                "pipeline": [{"op": "rolling", "window": 2}],
            }
        ],
    )
    a = out[out["GID_2"] == "A"]["x_roll"]
    b = out[out["GID_2"] == "B"]["x_roll"]
    assert np.isnan(a.iloc[0]) and a.iloc[1] == pytest.approx(1.5)
    # B's first non-NaN is its own first window, not contaminated by A
    assert np.isnan(b.iloc[0]) and b.iloc[1] == pytest.approx(15.0)


def test_build_features_reindexes_to_input():
    # Non-contiguous index must not break alignment.
    df = pd.DataFrame(
        {"Date": [0, 1, 2, 3], "Cases": [1.0, 2.0, 3.0, 4.0]}, index=[7, 8, 9, 10]
    )
    out = build_features(
        df,
        [
            {
                "name": "lag1",
                "column": "Cases",
                "pipeline": [{"op": "shift", "periods": 1}],
            }
        ],
    )
    assert out.index.tolist() == [7, 8, 9, 10]
    assert out.loc[8, "lag1"] == 1.0


def test_prepare_covariates_recipe():
    df = pd.DataFrame(
        {
            "Date": pd.period_range("2020-01", periods=5, freq="M").astype(str).tolist()
            + pd.period_range("2020-01", periods=5, freq="M").astype(str).tolist(),
            "GID_2": ["A"] * 5 + ["B"] * 5,
            "Cases": [10.0, 20.0, 30.0, 40.0, 50.0] * 2,
            "tmin": [18.0, 19.0, 20.0, 21.0, 22.0] * 2,
            "prec": [100.0, 90.0, 80.0, 70.0, 60.0] * 2,
        }
    )
    out, case_col, cov_cols = prepare_covariates(df)
    assert case_col == "Log_Cases"
    assert cov_cols == ["lag_1_log_cases", "lag_1_tmin_roll_2", "lag_1_prec_roll_2"]
    # Log_Cases is log1p of Cases
    assert (out["Log_Cases"] == np.log1p(out["Cases"])).all()
    # MONTH derived from Date
    assert out["MONTH"].iloc[0] == 1
    # lag_1_log_cases is the per-GID shifted Log_Cases
    a = out[out["GID_2"] == "A"].reset_index(drop=True)
    assert a.loc[1, "lag_1_log_cases"] == pytest.approx(a.loc[0, "Log_Cases"])
    assert np.isnan(a.loc[0, "lag_1_log_cases"])
    # tmin_roll_2 is a 2-month rolling mean (first value NaN); lag_1 shifts it,
    # so the first non-NaN lag_1_tmin_roll_2 lands on row 2 and equals
    # tmin_roll_2 at row 1.
    assert a.loc[2, "lag_1_tmin_roll_2"] == pytest.approx((18.0 + 19.0) / 2)
    assert a.loc[2, "lag_1_prec_roll_2"] == pytest.approx((100.0 + 90.0) / 2)
    assert np.isnan(a.loc[0, "lag_1_tmin_roll_2"])
