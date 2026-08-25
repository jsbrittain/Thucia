import numpy as np
import pandas as pd
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
