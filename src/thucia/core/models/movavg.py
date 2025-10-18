import logging

import numpy as np
import pandas as pd
from thucia.core.cases import align_date_types


def movavg(
    df: pd.DataFrame,
    start_date: pd.Timestamp | pd.Period = None,
    end_date: pd.Timestamp | pd.Period = None,
    gid_1: list[str] | None = None,
    method: str = "historical",  # historical / predict
    *args,
    **kwargs,
) -> pd.DataFrame:
    logging.info("Starting Seasonal Moving Average model...")

    # Admin-1 filter
    if gid_1 is not None:
        df = df[df["GID_1"].isin(gid_1)]

    # Determine start and end dates
    if start_date is None:
        start_date = pd.Timestamp.min
    if end_date is None:
        end_date = pd.Timestamp.max
    start_date = max(
        align_date_types(start_date, df["Date"]),
        df["Date"].min(),
    ).to_timestamp(how="end")
    end_date = min(
        align_date_types(end_date, df["Date"]),
        df["Date"].max(),
    ).to_timestamp(how="end")

    # Interpolate date range, ensuring we don't skip any gaps in the data
    freq = df["Date"].dtype.freq.name
    date_range = pd.date_range(
        start=start_date - pd.DateOffset(years=5),  # need 5 years of history
        end=end_date,
        freq=freq,
    )

    # Combine Cases over Status=Confirmed, Probable
    df = (
        df.groupby(["Date", "GID_2"], observed=True)
        .agg({"Cases": "sum", "future": "first"})
        .reset_index()
    )
    df["Date"] = df["Date"].dt.to_timestamp(how="end")

    # Interpolate missing dates
    multi_index = pd.MultiIndex.from_product(  # <-- implicit conversion to Timestamp
        [df["GID_2"].unique(), date_range], names=["GID_2", "Date"]
    )
    df = df.set_index(["GID_2", "Date"]).reindex(multi_index).reset_index()

    df["Cases"] = df["Cases"].fillna(0)

    df_forecast = df.copy()
    df_forecast["Year"] = df_forecast["Date"].dt.year
    df_forecast["Month"] = df_forecast["Date"].dt.month

    df_forecast = df_forecast.groupby(
        ["GID_2", "Year", "Month"], observed=True, as_index=False
    ).agg({"Cases": "mean", "future": "first"})

    df_forecast["prediction"] = (
        df_forecast.groupby(["GID_2", "Month"], observed=True)["Cases"]
        .apply(lambda s: s.shift(1).rolling(window=5, min_periods=5).mean())
        .reset_index(level=[0, 1], drop=True)
    )
    df_forecast["sample"] = 0

    df_forecast["Date"] = pd.to_datetime(
        df_forecast[["Year", "Month"]].assign(DAY=1)
    ).dt.to_period(freq[0])
    df_forecast.drop(columns=["Year", "Month"], inplace=True)

    df_forecast = df_forecast[df_forecast["Date"] >= start_date.to_period(freq[0])]
    df_forecast.loc[df_forecast["future"], "Cases"] = np.nan

    logging.info("Seasonal Moving Average model complete.")
    return df_forecast
