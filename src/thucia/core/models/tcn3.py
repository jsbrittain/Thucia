import logging

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TCNModel
from darts.utils.likelihood_models import QuantileRegression

torch.set_default_dtype(torch.float32)


class TcnSamples:
    def __init__(
        self,
        timeseries_df,
        target_col,
        covariate_cols,
        withheld_months,
        num_samples=1000,
        input_chunk_length=48,
        retrain=True,
        verbose=True,
    ):
        self.df = timeseries_df.copy()
        self.target_col = target_col
        self.covariate_cols = covariate_cols
        self.withheld_months = withheld_months
        self.num_samples = num_samples
        self.input_chunk_length = input_chunk_length
        self.retrain = retrain
        self.scaler = Scaler()
        self.verbose = verbose

    def historical_forecast(self, df_filter: dict) -> pd.DataFrame:
        logging.info(f"[Historical] Processing TCN with filter: {df_filter}")

        province_df = self.df.copy()
        for k, v in df_filter.items():
            province_df = province_df[province_df[k] == v]
        province_df = province_df.sort_values("Date")
        province_df = province_df.iloc[: -self.withheld_months]

        target_series = TimeSeries.from_dataframe(
            province_df,
            time_col="Date",
            value_cols=[self.target_col],
            fill_missing_dates=True,
            freq="ME",
        ).astype(np.float32)
        covariate_series = TimeSeries.from_dataframe(
            province_df,
            time_col="Date",
            value_cols=self.covariate_cols,
            fill_missing_dates=True,
            freq="ME",
        ).astype(np.float32)
        covariates_scaled = self.scaler.fit_transform([covariate_series])[0].astype(
            np.float32
        )

        model = TCNModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=1,
            kernel_size=3,
            num_filters=4,
            random_state=42,
            likelihood=QuantileRegression(),
            save_checkpoints=True,
            force_reset=True,
            n_epochs=150,
        )
        model.fit(
            series=target_series,
            past_covariates=covariates_scaled,
            verbose=self.verbose,
        )

        backtest = model.historical_forecasts(
            series=target_series,
            past_covariates=covariates_scaled,
            forecast_horizon=1,
            num_samples=self.num_samples,
            retrain=self.retrain,
            verbose=self.verbose,
        )

        pred_array = TimeSeries.all_values(backtest)
        num_time = pred_array.shape[0]
        pred_array = pred_array.reshape(num_time, self.num_samples)

        pred_df = pd.DataFrame(pred_array)
        pred_df["Date"] = target_series.time_index[-num_time:]
        pred_df = pred_df.melt(
            id_vars="Date", var_name="sample", value_name="prediction"
        )
        pred_df["sample"] = pred_df["sample"].astype(int)
        pred_df["prediction"] = np.expm1(pred_df["prediction"])

        merged = pd.merge(
            province_df,
            pred_df,
            how="left",
            on="Date",
        )
        return merged


def tcn3(
    df: pd.DataFrame,
    # covariates: list[str],
    start_date: str | pd.Timestamp = pd.Timestamp.min,
    end_date: str | pd.Timestamp = pd.Timestamp.max,
    gid_1: list[str] | None = None,
    retrain: bool = True,
) -> pd.DataFrame:
    logging.info("Starting TCN forecasting pipeline...")

    if gid_1 is not None:
        df = df[df["GID_1"].isin(gid_1)]

    start_date = max(pd.to_datetime(start_date), df["Date"].min())
    end_date = min(pd.to_datetime(end_date), df["Date"].max())

    date_range = pd.date_range(start=start_date, end=end_date, freq="ME")
    date_range += pd.offsets.MonthEnd(0)
    df.loc[:, "Date"] = pd.to_datetime(df["Date"]) + pd.offsets.MonthEnd(0)

    df = (
        df.groupby(["Date", "GID_2"])
        .agg(
            {
                "Cases": "sum",
                "tmin": "mean",
                "prec": "mean",
            }
        )
        .reset_index()
    )
    multi_index = pd.MultiIndex.from_product(
        [df["GID_2"].unique(), date_range], names=["GID_2", "Date"]
    )
    df = df.set_index(["GID_2", "Date"]).reindex(multi_index).reset_index()

    # Set-up target column
    df["Cases"] = df["Cases"].fillna(0)
    df["Log_Cases"] = np.log1p(df["Cases"])

    # Load and merge covariates
    covariates = ["LAG_1_LOG_CASES", "LAG_1_tmin_roll_2", "LAG_1_prec_roll_2"]
    df["LAG_1_LOG_CASES"] = df.groupby("GID_2")["Log_Cases"].shift(1).fillna(0)
    df["LAG_1_tmin_roll_2"] = (
        df.groupby("GID_2")["tmin"].shift(1).rolling(window=2).mean().fillna(0)
    )
    df["LAG_1_prec_roll_2"] = (
        df.groupby("GID_2")["prec"].shift(1).rolling(window=2).mean().fillna(0)
    )

    # Convert all float channels to float32
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].astype(np.float32)

    provinces = df["GID_2"].unique()
    tcn = TcnSamples(
        timeseries_df=df,
        target_col="Log_Cases",
        covariate_cols=covariates,
        withheld_months=12,
        num_samples=1000,
        retrain=False,  # retrain,
    )

    all_forecasts = []
    for ix, province in enumerate(provinces):
        logging.info(f"Processing province {province} ({ix + 1}/{len(provinces)})")
        region_filter = {"GID_2": province}
        df_forecast = tcn.historical_forecast(df_filter=region_filter)
        all_forecasts.append(df_forecast)

    forecast_results = pd.concat(all_forecasts, ignore_index=True)
    logging.info("TCN forecasting complete.")
    return forecast_results
