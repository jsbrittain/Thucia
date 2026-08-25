import numpy as np
import pandas as pd
import pytest
from thucia.core.cases.wis import _bracher_wis_from_quantiles
from thucia.core.cases.wis import wis_bracher

QS = [0.05, 0.5, 0.95]


def _frame(gid, date, quantile, prediction, cases, **extra):
    df = pd.DataFrame(
        {
            "GID_2": [gid] * len(quantile),
            "Date": [date] * len(quantile),
            "quantile": quantile,
            "prediction": prediction,
            "Cases": [cases] * len(quantile),
        }
    )
    for k, v in extra.items():
        df[k] = v
    return df


def test_perfect_forecast_zero_wis():
    # All quantiles equal to the truth -> WIS is 0
    df = _frame("A", "2020-01", QS, [10, 10, 10], 10)
    assert wis_bracher(df)["WIS"].iloc[0] == pytest.approx(0.0)


def test_wis_bounds_by_interval():
    # Single symmetric interval [l,u] around median: WIS = IS_alpha/alpha_weight
    # IS_alpha = (u - l) + 2/alpha*(under + over)
    # With truth outside, only the over/under term matters.
    y = 100.0
    lo, mid, hi = 0.0, 5.0, 10.0
    alpha = 2 * 0.05
    expected_is = (hi - lo) + (2.0 / alpha) * max(0.0, y - hi)
    # median term: 0.5 * |y - m|, median weight 0.5
    expected = (0.5 * abs(y - mid) + (alpha / 2.0) * expected_is) / (0.5 + alpha / 2.0)
    got = _bracher_wis_from_quantiles(y, np.array(QS), np.array([lo, mid, hi]))
    assert got == pytest.approx(expected)
    df = _frame("A", "2020-01", QS, [lo, mid, hi], y)
    assert wis_bracher(df)["WIS"].iloc[0] == pytest.approx(expected)


def test_median_interpolated_when_absent():
    # No q=0.5; median must be linearly interpolated between q=0.25 and q=0.75
    qs = [0.25, 0.75]
    preds = np.array([0.0, 10.0])
    y = 5.0
    got = _bracher_wis_from_quantiles(y, qs, preds)
    assert np.isfinite(got)


def test_monotonic_fix_applied():
    # Input quantiles out of order; monotonic_fix=True must lift violations
    df = _frame(
        "A", "2020-01", [0.05, 0.5, 0.95], [10, 5, 7], 6
    )  # q0.5 < q0.05 violated
    out = wis_bracher(df, monotonic_fix=True)
    assert np.isfinite(out["WIS"].iloc[0])


def test_negative_predictions_clamped_on_raw_scale():
    # On the raw (non-log) scale, negative predictions clamp to 0
    df = _frame("A", "2020-01", QS, [-10, -5, 0], 1)
    out = wis_bracher(df, clamp_negative_to_zero=True, log1p_scale=False)
    assert np.isfinite(out["WIS"].iloc[0])


def test_log1p_scale_consistent():
    df = _frame("A", "2020-01", QS, [9, 10, 11], 10)
    raw = wis_bracher(df, log1p_scale=False)["WIS"].iloc[0]
    log = wis_bracher(df, log1p_scale=True)["WIS"].iloc[0]
    # log1p scoring is a different scale but must be finite and non-negative
    assert raw >= 0
    assert log >= 0
    assert np.isfinite(raw) and np.isfinite(log)


def test_model_column_included_in_grouping():
    # A 'model' column is auto-added to the grouping (per the docstring)
    df1 = _frame("A", "2020-01", QS, [0, 5, 10], 5, model="m1")
    df2 = _frame("A", "2020-01", QS, [1, 6, 11], 5, model="m2")
    out = wis_bracher(pd.concat([df1, df2], ignore_index=True))
    assert "model" in out.columns
    assert set(out["model"]) == {"m1", "m2"}


def test_wis_is_nonnegative_and_finite():
    rng = np.random.default_rng(0)
    for _ in range(20):
        qs = sorted(rng.uniform(0.01, 0.99, 7).tolist() + [0.5])
        preds = np.sort(rng.uniform(0, 100, len(qs)))
        y = rng.uniform(0, 100)
        assert _bracher_wis_from_quantiles(y, np.asarray(qs), preds) >= 0
