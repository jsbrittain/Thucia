import logging

import pandas as pd


def movavg(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp = pd.Timestamp.min,
    end_date: str | pd.Timestamp = pd.Timestamp.max,
    gid_1: list[str] | None = None,
    method: str = "historical",  # historical / predict
) -> pd.DataFrame:
    logging.info("Starting Seasonal Moving Average model...")

    # Admin-1 filter
    if gid_1 is not None:
        df = df[df["GID_1"].isin(gid_1)]

    # Determine start and end dates
    start_date = max(pd.to_datetime(start_date), df["Date"].min())
    end_date = min(pd.to_datetime(end_date), df["Date"].max())

    # Interpolate date range, ensuring we don't skip any gaps in the data
    date_range = pd.date_range(start=start_date, end=end_date, freq="ME")

    # Ensure dates are aligned to the end of the month
    date_range += pd.offsets.MonthEnd(0)  # Ensure we reset to the end of the month
    df.loc[:, "Date"] = pd.to_datetime(df["Date"]) + pd.offsets.MonthEnd(0)

    # Combine Cases over Status=Confirmed, Probable
    df = df.groupby(["Date", "GID_2"]).agg({"Cases": "sum"}).reset_index()

    # Interpolate missing dates
    multi_index = pd.MultiIndex.from_product(
        [df["GID_2"].unique(), date_range], names=["GID_2", "Date"]
    )
    df = df.set_index(["GID_2", "Date"]).reindex(multi_index).reset_index()

    df["Cases"] = df["Cases"].fillna(0)

    df_forecast = df.copy()
    df_forecast["Year"] = df_forecast["Date"].dt.year
    df_forecast["Month"] = df_forecast["Date"].dt.month

    df_forecast = df_forecast.groupby(["GID_2", "Year", "Month"], as_index=False)[
        "Cases"
    ].mean()

    df_forecast["prediction"] = (
        df_forecast.groupby(["GID_2", "Month"])["Cases"]
        .apply(lambda s: s.shift(1).rolling(window=5, min_periods=5).mean())
        .reset_index(level=[0, 1], drop=True)
    )
    df_forecast["sample"] = 0

    df_forecast["Date"] = pd.to_datetime(
        df_forecast[["Year", "Month"]].assign(DAY=1)
    ) + pd.offsets.MonthEnd(0)
    df_forecast.drop(columns=["Year", "Month"], inplace=True)

    logging.info("Seasonal Moving Average model complete.")
    return df_forecast
