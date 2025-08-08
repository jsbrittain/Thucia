import logging

import numpy as np
import pandas as pd
from timesfm import TimesFm
from timesfm import TimesFmCheckpoint
from timesfm import TimesFmHparams


class TimeSFMSamples:
    def __init__(
        self,
        timeseries_df: pd.DataFrame,
        case_col: str,
        withheld_months: int,
        dynamic_numerical_cols: list,
        dynamic_categorical_cols: list,
        static_categorical_col: str,
        context_len: int = 24,
        horizon_len: int = 1,
        num_samples: int = 1000,
    ):
        self.df = timeseries_df.copy()
        self.case_col = case_col
        self.withheld_months = withheld_months
        self.dynamic_numerical_cols = dynamic_numerical_cols
        self.dynamic_categorical_cols = dynamic_categorical_cols
        self.static_categorical_col = static_categorical_col
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.num_samples = num_samples

        self.model = TimesFm(
            hparams=TimesFmHparams(
                backend="cpu",
                per_core_batch_size=32,
                horizon_len=12,
                num_layers=50,
                context_len=2048,
                use_positional_embedding=False,
                # quantiles cant seem to from their defaults
                # quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),  # default
                # quantiles=(0.01, 0.025, *list(np.arange(0.05, 1.0, 0.05)), 0.975, 0.99),
            ),
            checkpoint=TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
            ),
        )

    def _filter_df(self, df_filter: dict) -> pd.DataFrame:
        df = self.df.copy()
        for k, v in df_filter.items():
            df = df[df[k] == v]
        return df.sort_values("Date").reset_index(drop=True)

    def sample_from_quantiles(
        self,
        quantiles: np.ndarray,
        quantile_levels: np.ndarray,
        num_samples: int = 1000,
    ):
        """
        Simulate samples from a discrete quantile forecast using inverse transform sampling.
        Args:
            quantiles: np.array of shape [num_quantiles], e.g., forecasted values at 10%-90%
            quantile_levels: np.array of quantile probabilities, e.g., [0.1, 0.2, ..., 0.9]
            num_samples: Number of samples to generate
        Returns:
            np.array of samples
        """
        uniform_samples = np.random.uniform(0, 1, size=num_samples)
        return np.interp(uniform_samples, quantile_levels, quantiles)

    def historical_forecast(self, df_filter: dict) -> pd.DataFrame:
        logging.info(f"[Historical] Processing filter: {df_filter}")
        df = self._filter_df(df_filter).iloc[: -self.withheld_months].copy()

        predictions = []
        for i in range(self.context_len, len(df) - self.horizon_len + 1):
            logging.info(f"Processing index {i} for historical forecast.")
            context = df.iloc[i - self.context_len : i]
            future = df.iloc[i : i + self.horizon_len]

            inputs = context[self.case_col].tolist()

            dynamic_numerical = {
                col: [df[col].iloc[-(self.context_len + self.horizon_len) :].tolist()]
                for col in self.dynamic_numerical_cols
            }
            dynamic_categorical = {
                col: [df[col].iloc[-(self.context_len + self.horizon_len) :].tolist()]
                for col in self.dynamic_categorical_cols
            }
            static_categorical = {
                self.static_categorical_col: [df[self.static_categorical_col].iloc[i]]
            }

            use_covar = dynamic_numerical or dynamic_categorical or static_categorical

            if use_covar:
                # Covariate support, but no quantiles (simulate from residuals)
                forecast, ols_forecast = self.model.forecast_with_covariates(
                    inputs=[inputs],
                    dynamic_numerical_covariates=dynamic_numerical,
                    dynamic_categorical_covariates=dynamic_categorical,
                    static_numerical_covariates={},
                    static_categorical_covariates=static_categorical,
                    freq=[0],
                    xreg_mode="xreg + timesfm",
                )
                # forecast is a list of array() with a single element

                # Simulate samples from normal distribution
                # residuals = 0  # requires historical forecasting #######################
                sigma = 0.1  # np.std(residuals)
                mu = forecast[0][0]
                samples = np.random.normal(loc=mu, scale=sigma, size=self.num_samples)

                predictions.extend(
                    [
                        {
                            "Date": future["Date"].values[0],
                            "sample": ix,
                            "prediction": np.expm1(sample),
                        }
                        for ix, sample in enumerate(samples)
                    ]
                )
            else:
                # No covariates, but (limited) quantiles (mean, 0.1, 0.2, ..., 0.9)

                # ### This routine forecasts multiple steps ahead - need to control ###
                forecast, quantile_forecast = self.model.forecast(
                    inputs=[inputs],
                    freq=[0],
                )

                # Simulate samples from residuals
                samples = self.sample_from_quantiles(
                    # [first(only) series][one month ahead prediction][quantiles]
                    quantiles=quantile_forecast[0][0][1:],
                    quantile_levels=self.model.hparams.quantiles,
                    num_samples=1000,
                )
                predictions.extend(
                    [
                        {
                            "Date": future["Date"].values[0],
                            "sample": ix,
                            "prediction": np.expm1(sample),
                        }
                        for ix, sample in enumerate(samples)
                    ]
                )

        pred_df = pd.DataFrame(predictions)
        merged = pd.merge(df, pred_df, on="Date", how="left")
        logging.info("Finished historical forecast.")
        return merged

    def rolling_forecast(self, df_filter: dict) -> pd.DataFrame:
        logging.info(f"[Rolling] Processing filter: {df_filter}")
        raise NotImplementedError(
            "Rolling forecast is not implemented yet. "
            "Please use historical_forecast method."
        )


def timesfm(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp = pd.Timestamp.min,
    end_date: str | pd.Timestamp = pd.Timestamp.max,
    gid_1: list[str] | None = None,
    method: str = "historical",  # historical / predict
) -> pd.DataFrame:
    logging.info("Starting TimeSFM model...")

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
    # df = df.groupby(["Date", "GID_2"]).agg({"Cases": "sum"}).reset_index()

    # Interpolate missing dates
    multi_index = pd.MultiIndex.from_product(
        [df["GID_2"].unique(), date_range], names=["GID_2", "Date"]
    )
    df = df.set_index(["GID_2", "Date"]).reindex(multi_index).reset_index()

    df["Cases"] = df["Cases"].fillna(0)

    # Apply log transform for TimeSFM modeling
    df["Log_Cases"] = np.log1p(df["Cases"])

    # Pre-compute covariates
    df["MONTH"] = pd.to_datetime(df["Date"]).dt.month
    df["LAG_1_LOG_CASES"] = df.groupby("GID_2")["Log_Cases"].shift(1)
    df["tmin_roll_2"] = df.groupby("GID_2")["tmin"].transform(
        lambda x: x.rolling(2).mean()
    )
    df["LAG_1_tmin_roll_2"] = df.groupby("GID_2")["tmin_roll_2"].shift(1)
    df["prec_roll_2"] = df.groupby("GID_2")["prec"].transform(
        lambda x: x.rolling(2).mean()
    )
    df["LAG_1_prec_roll_2"] = df.groupby("GID_2")["prec_roll_2"].shift(1)

    provinces = df["GID_2"].unique()
    timesfm = TimeSFMSamples(
        timeseries_df=df,
        case_col="Log_Cases",
        withheld_months=12,
        dynamic_numerical_cols=[
            "LAG_1_LOG_CASES",
            "LAG_1_tmin_roll_2",
            "LAG_1_prec_roll_2",
        ],
        dynamic_categorical_cols=["MONTH"],
        static_categorical_col="GID_2",
        context_len=24,
        horizon_len=1,
    )

    all_forecasts = []
    for ix, province in enumerate(provinces):
        logging.info(
            f"Processing province {province} ({ix + 1}/{len(provinces)}) "
            "for TimeSFM model"
        )
        region_filter = {"GID_2": province}

        if method == "historical":
            df_forecast = timesfm.historical_forecast(df_filter=region_filter)
        elif method == "predict":
            df_forecast = timesfm.rolling_forecast(df_filter=region_filter)
        else:
            raise ValueError(f"Invalid method: {method}.")

        # Append to master list
        all_forecasts.append(df_forecast)

    # Concatenate all provinces' forecasts into a single DataFrame
    forecast_results = pd.concat(all_forecasts, ignore_index=True)

    logging.info("TimeSFM model complete.")
    return forecast_results
