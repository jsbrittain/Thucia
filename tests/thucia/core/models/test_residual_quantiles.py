import numpy as np
import pandas as pd
from thucia.core.models.utils.residual_quantiles import add_residual_quantiles


def _frame(n_dates=30, n_gid=2, seed=1, horizon=1, preds_fn=None):
    rng = np.random.default_rng(seed)
    dates = pd.period_range("2020-01", periods=n_dates, freq="M")
    rows = []
    for d in dates:
        for g in [f"G{i}" for i in range(n_gid)]:
            pred = preds_fn(rng) if preds_fn else float(rng.normal(5, 1))
            rows.append(
                {
                    "Date": d,
                    "GID_2": g,
                    "horizon": horizon,
                    "prediction": pred,
                    "Cases": float(rng.normal(5, 1)),
                }
            )
    return pd.DataFrame(rows)


def test_output_schema_and_quantile_count():
    out = add_residual_quantiles(_frame(), min_history=5)
    df = out.df
    assert {"Date", "GID_2", "horizon", "quantile", "prediction", "Cases"} <= set(
        df.columns
    )
    # one row per (date, gid, quantile)
    assert len(df) == 30 * 2 * 15


def test_quantiles_monotone_per_group():
    out = add_residual_quantiles(_frame(), min_history=5).df
    dates = sorted(out["Date"].unique())
    q = out[(out["GID_2"] == "G0") & (out["Date"] == dates[-1])].sort_values("quantile")
    assert (q["prediction"].diff().dropna() >= -1e-9).all()


def test_early_dates_use_point_prediction_fallback():
    # With min_history > number of available dates, all quantiles collapse to the base
    out = add_residual_quantiles(_frame(n_dates=4), min_history=100).df
    d = out["Date"].min()
    g = out[(out["Date"] == d) & (out["GID_2"] == "G0")].sort_values("quantile")
    # no history -> all quantiles identical
    assert g["prediction"].nunique() == 1


def test_window_respects_sliding_window():
    out = add_residual_quantiles(_frame(), min_history=5, window=5).df
    assert out["prediction"].notna().all()


def test_na_truth_excluded_from_residuals():
    df = _frame()
    df.loc[0, "Cases"] = np.nan  # first day truth missing
    out = add_residual_quantiles(df, min_history=1).df
    assert out["prediction"].notna().all()
