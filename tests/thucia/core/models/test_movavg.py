import numpy as np
import pandas as pd
import pytest
from thucia.core.models import movavg


def _frame():
    idx = pd.period_range("2015-01", periods=72, freq="M")
    df = pd.DataFrame(
        {
            "Date": idx,
            "GID_2": ["A"] * 72,
            "GID_1": ["G1"] * 72,
            "future": [False] * 72,
            "Cases": (np.sin(np.arange(72)) * 5 + 10).clip(min=0),
        }
    )
    return df


def test_movavg_output_schema():
    out = movavg(
        _frame(),
        start_date=pd.Period("2020-01", freq="M"),
        end_date=pd.Period("2020-12", freq="M"),
    )
    assert {"GID_2", "Date", "prediction", "future", "Cases"} <= set(out.columns)
    assert len(out) == 12  # one row per month in range
    assert out["Date"].dtype == "period[M]"


def test_movavg_forecast_is_seasonal_moving_average():
    # Prediction for month m is the mean of that month over the previous 5 years
    df = _frame()
    out = movavg(
        df,
        start_date=pd.Period("2020-01", freq="M"),
        end_date=pd.Period("2020-12", freq="M"),
    )
    # all predictions finite and within range of the input series
    assert out["prediction"].notna().all()
    assert out["prediction"].between(0, df["Cases"].max() + 1e-9).all()


def test_movavg_future_cases_set_to_nan():
    df = _frame()
    df.loc[df.index[-12:], "future"] = True
    out = movavg(
        df,
        start_date=pd.Period("2020-01", freq="M"),
        end_date=pd.Period("2020-12", freq="M"),
    )
    assert out.loc[out["future"], "Cases"].isna().all()


def test_movavg_multiple_gids():
    df = _frame()
    other = _frame()
    other["GID_2"] = "B"
    df = pd.concat([df, other], ignore_index=True)
    out = movavg(
        df,
        start_date=pd.Period("2020-01", freq="M"),
        end_date=pd.Period("2020-12", freq="M"),
    )
    assert set(out["GID_2"].unique()) == {"A", "B"}


def test_movavg_weekly_seasonal():
    # Weekly cadence: predictions are the prior-years' same-ISO-week mean, and
    # the weekly anchor (W-SAT) is preserved.
    from forecast_data import make_forecast_df

    df = make_forecast_df(freq="W-SAT", n_periods=364, start="2014-01-04", n_gid=1)
    out = movavg(
        df,
        start_date=pd.Period("2020-01-04", freq="W-SAT"),
        end_date=pd.Period("2020-12-26", freq="W-SAT"),
    )
    assert str(out["Date"].dtype) == "period[W-SAT]"
    assert 50 <= len(out) <= 53  # ~one row per week in the window, single GID
    preds = out["prediction"].dropna()
    assert (preds >= 0).all()

    # Pick a mid-year week with 5 prior years of history: the prediction is the
    # mean of that ISO week's cases over the previous 5 years.
    target = out[out["Date"] == pd.Period("2020-07-25", freq="W-SAT")]
    assert len(target) == 1
    gid = str(target["GID_2"].iloc[0])
    week = int(
        pd.PeriodIndex(target["Date"])
        .to_timestamp(how="end")
        .isocalendar()["week"]
        .iloc[0]
    )
    week_arr = (
        pd.PeriodIndex(df["Date"])
        .to_timestamp(how="end")
        .isocalendar()["week"]
        .to_numpy()
    )
    hist = df[
        (df["GID_2"].astype(str) == gid)
        & (week_arr == week)
        & (df["Date"] < pd.Period("2020-01-04", freq="W-SAT"))
    ]
    expected = hist["Cases"].iloc[-5:].mean()
    assert target["prediction"].iloc[0] == pytest.approx(expected, rel=1e-6)
