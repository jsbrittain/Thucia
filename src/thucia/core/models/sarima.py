import logging

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import AutoARIMA
from darts.utils.timeseries_generation import datetime_attribute_timeseries


class SarimaSamples:
    def __init__(
        self,
        timeseries_df,
        case_col,
        withheld_months,
        num_samples=1000,
        season_length=12,
    ):
        self.df = timeseries_df
        self.case_col = case_col
        self.withheld_months = withheld_months
        self.num_samples = num_samples
        self.season_length = season_length

    def historical_forecast(self, df_filter: dict) -> pd.DataFrame:
        """
        Parameters
        ----------
        province: str
            The province identifier (GID_2) to process.
        df_filter: dict
            Dictionary of key-value pairs to filter on, e.g. {'GID_2': 'some_region'}
        """
        logging.info(f"[Historical] Processing province with filter: {df_filter}")

        province_df = self.df.copy()
        for k, v in df_filter.items():
            province_df = province_df[province_df[k] == v]
        province_df = province_df.sort_values("Date")
        province_df = province_df.iloc[: -self.withheld_months]

        series = TimeSeries.from_dataframe(
            province_df,
            time_col="Date",
            value_cols=[self.case_col],
        )

        model = AutoARIMA(season_length=self.season_length)
        model.fit(series)

        backtest = model.historical_forecasts(
            series=series,
            forecast_horizon=1,
            num_samples=self.num_samples,
            verbose=True,
        )

        # Convert predictions to DataFrame
        pred_df = backtest.to_dataframe()
        pred_df = pred_df.reset_index().melt(
            id_vars="time", var_name="sample", value_name="prediction"
        )
        pred_df["sample"] = pred_df["sample"].str.extract(r"(\d+)").astype(int)
        pred_df["prediction"] = np.expm1(pred_df["prediction"])  # Inverse log transform

        # Merge predictions with original DataFrame
        merged = pd.merge(
            province_df,
            pred_df,
            how="left",
            left_on="Date",
            right_on="time",
        ).drop(columns=["time"])

        logging.info("Finished historical forecast with reshaped output.")
        return merged

    def rolling_forecast(self, df_filter: dict) -> pd.DataFrame:
        """
        Run rolling one-step-ahead forecasts over the last `withheld_months` period,
        and return a DataFrame with 'sample', 'prediction', and metadata aligned with
        original data.

        Parameters
        ----------
        df_filter: dict
            Dictionary of key-value pairs to filter on, e.g. {'GID_2': 'some_region'}

        Returns
        -------
        pd.DataFrame
            DataFrame with predictions for each time step and sample.
        """
        logging.info(
            f"[Rolling] Processing province {df_filter} "
            f"(withheld_months: {self.withheld_months}, samples: {self.num_samples})"
        )

        # Filter and prepare the province data
        province_df = self.df.copy()
        for k, v in df_filter.items():
            province_df = province_df[province_df[k] == v]
        province_df = province_df.sort_values("Date")

        # Remove holdout period from the end
        trimmed_df = province_df.iloc[: -self.withheld_months]
        series = TimeSeries.from_dataframe(
            trimmed_df,
            time_col="Date",
            value_cols=[self.case_col],
        )
        start_idx = len(series) - self.withheld_months

        predictions = []

        # Rolling forecast for each withheld time step
        logging.info(f"Starting rolling forecast for {self.withheld_months} months.")
        for j in range(self.withheld_months):
            train_series = series[: start_idx + j]

            future_cov = datetime_attribute_timeseries(
                train_series, attribute="month", cyclic=True, add_length=6
            )
            model = AutoARIMA(season_length=self.season_length)
            model.fit(train_series, future_covariates=future_cov)

            pred = model.predict(
                n=1,
                future_covariates=future_cov,
                num_samples=self.num_samples,
            )

            pred_values = TimeSeries.all_values(pred[0]).reshape(self.num_samples)
            pred_df = pd.DataFrame(
                {
                    "sample": list(range(self.num_samples)),
                    "prediction": pred_values,
                    "forecast_date": [pred[0].time_index[0]] * self.num_samples,
                }
            )
            predictions.append(pred_df)

        # Concatenate all prediction rows
        pred_all = pd.concat(predictions, ignore_index=True)

        # Merge with original province_df on Date
        merged = pd.merge(
            province_df,
            pred_all,
            how="right",
            left_on="Date",
            right_on="forecast_date",
        ).drop(columns=["forecast_date"])
        merged["prediction"] = np.expm1(merged["prediction"])  # Inverse log transform

        logging.info("Finished rolling forecast with merged output.")
        return merged


def sarima(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp = pd.Timestamp.min,
    end_date: str | pd.Timestamp = pd.Timestamp.max,
    gid_1: list[str] | None = None,
    method: str = "historical",  # historical / predict
) -> pd.DataFrame:
    logging.info("Starting SARIMA model...")

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

    # Apply log transform for SARIMA modeling
    df["Log_Cases"] = np.log1p(df["Cases"])

    provinces = df["GID_2"].unique()
    sarima = SarimaSamples(
        timeseries_df=df,
        case_col="Log_Cases",
        withheld_months=12,
        num_samples=1000,
    )

    all_forecasts = []
    for ix, province in enumerate(provinces):
        logging.info(
            f"Processing province {province} ({ix + 1}/{len(provinces)}) "
            "for SARIMA model"
        )
        region_filter = {"GID_2": province}

        if method == "historical":
            # Historical forecast
            df_forecast = sarima.historical_forecast(df_filter=region_filter)
        elif method == "predict":
            # Rolling forecast
            df_forecast = sarima.rolling_forecast(df_filter=region_filter)
        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'historical' or 'predict'."
            )

        # Append to master list
        all_forecasts.append(df_forecast)

    # Concatenate all provinces' forecasts into a single DataFrame
    forecast_results = pd.concat(all_forecasts, ignore_index=True)

    logging.info("SARIMA model complete.")
    return forecast_results
