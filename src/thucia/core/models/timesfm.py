import logging

import numpy as np
import pandas as pd
from timesfm import TimesFm
from timesfm import TimesFmCheckpoint
from timesfm import TimesFmHparams

from .utils import filter_admin1
from .utils import interpolate_missing_dates
from .utils import set_historical_na_to_zero


class TimeSFMSamples:
    def __init__(
        self,
        timeseries_df: pd.DataFrame,
        case_col: str,
        dynamic_numerical_cols: list[str] = [],
        dynamic_categorical_cols: list[str] = [],
        static_numerical_covariates: list[str] = [],
        static_categorical_cols: list[str] = [],
        per_core_batch_size: int = 32,
        num_layers: int = 50,
        context_len: int = 32,
        horizon_len: int = 1,
        num_samples: int = 1000,
    ):
        self.df = timeseries_df.copy()
        self.case_col = case_col
        self.dynamic_numerical_cols = dynamic_numerical_cols
        self.dynamic_categorical_cols = dynamic_categorical_cols
        self.static_numerical_covariates = static_numerical_covariates
        self.static_categorical_cols = static_categorical_cols
        self.per_core_batch_size = per_core_batch_size
        self.num_layers = num_layers
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.num_samples = num_samples

        self.model = TimesFm(
            hparams=TimesFmHparams(
                backend="cpu",
                context_len=self.context_len,
                horizon_len=self.horizon_len,
                num_layers=self.num_layers,
                per_core_batch_size=self.per_core_batch_size,
                use_positional_embedding=True,
                # cant seem to alter quantiles from their defaults
                # quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),  # default
                # quantiles=(0.01, 0.025, *list(np.arange(0.05, 1.0, 0.05)), 0.975, 0.99),
            ),
            checkpoint=TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
            ),
        )

    def sample_from_quantiles(
        self,
        quantiles: np.ndarray,
        quantile_levels: np.ndarray,
        num_samples: int = 1000,
    ):
        """
        Simulate samples from a discrete quantile forecast using inverse transform
        sampling.

        Args:
            quantiles: np.array of shape [num_quantiles], e.g., at 10%-90%
            quantile_levels: np.array of quantile probabilities, e.g., [0.1, ..., 0.9]
            num_samples: Number of samples to generate
        Returns:
            np.array of samples
        """
        uniform_samples = np.random.uniform(0, 1, size=num_samples)
        return np.interp(uniform_samples, quantile_levels, quantiles)

    @property
    def use_covariates(self) -> bool:
        return bool(
            self.dynamic_numerical_cols
            or self.dynamic_categorical_cols
            or self.static_numerical_covariates
            or self.static_categorical_cols
        )

    def predict(self, sigma: float = 0.0) -> pd.DataFrame:
        predictions = []
        df = self.df

        if len(df) < 2:  # (self.horizon_len + self.context_len):
            logging.warning(f"Insufficient data (n={len(df)}) for TimeSFM model. ")
            df["sample"] = np.nan
            df["prediction"] = np.nan
            return df

        # Time-series target series should not cover the horizon period
        inputs = df[self.case_col][: -self.horizon_len].tolist()

        if self.use_covariates:
            logging.debug("TimeSFM with covariates forecasting...")

            # Prepare covariates
            dynamic_numerical = {
                col: [df[col].tolist()] for col in self.dynamic_numerical_cols
            }
            dynamic_categorical = {
                col: [df[col].tolist()] for col in self.dynamic_categorical_cols
            }
            static_numerical = {
                col: df[col].iloc[-1] for col in self.static_numerical_covariates
            }
            static_categorical = {
                col: df[col].iloc[-1] for col in self.static_categorical_cols
            }

            # Covariate support, but no quantiles (simulate from residuals)
            forecasts, ols_forecasts = self.model.forecast_with_covariates(
                inputs=[inputs],
                dynamic_numerical_covariates=dynamic_numerical,
                dynamic_categorical_covariates=dynamic_categorical,
                static_numerical_covariates=static_numerical,
                static_categorical_covariates=static_categorical,
                freq=[1],  # use 1 for monthly data
                xreg_mode="xreg + timesfm",
            )
            # forecast is a list of array() with horizon_len elements
            for forecast, date in zip(forecasts[0], df["Date"][-self.horizon_len :]):
                # Simulate samples from normal distribution
                mu = forecast
                samples = np.random.normal(loc=mu, scale=sigma, size=self.num_samples)

                predictions.extend(
                    [
                        {
                            "Date": date,
                            "sample": ix,
                            "prediction": sample,
                        }
                        for ix, sample in enumerate(samples)
                    ]
                )
        else:
            # No covariates, but (limited) quantiles (mean, 0.1, 0.2, ..., 0.9)
            logging.debug("TimeSFM without covariates forecasting...")

            # Forecast to horizon_len
            forecasts, quantile_forecasts = self.model.forecast(
                inputs=[inputs],
                freq=[1],  # use 1 for monthly data
            )

            # forecast is a list of array() with horizon_len elements
            for forecast, nstep in zip(forecasts[0], range(self.horizon_len)):
                date = df["Date"].iloc[-self.horizon_len + nstep]
                # Simulate samples from residuals
                samples = self.sample_from_quantiles(
                    # [first(only) series][n-step][quantiles]
                    quantiles=quantile_forecasts[0][nstep][1:],
                    quantile_levels=self.model.hparams.quantiles,
                    num_samples=1000,
                )

                predictions.extend(
                    [
                        {
                            "Date": date,
                            "sample": ix,
                            "prediction": sample,
                        }
                        for ix, sample in enumerate(samples)
                    ]
                )

        pred_df = pd.DataFrame(predictions)
        merged = pd.merge(df, pred_df, on="Date", how="left")
        logging.debug("Finished historical forecast.")
        return merged


def timesfm(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp = pd.Timestamp.min,
    end_date: str | pd.Timestamp = pd.Timestamp.max,
    gid_1: list[str] | None = None,
    method: str = "historical",  # historical / predict
) -> pd.DataFrame:
    logging.info("Starting TimeSFM model...")

    df = df.copy()
    df = filter_admin1(df, gid_1=gid_1)
    df = interpolate_missing_dates(df, start_date, end_date)
    df = set_historical_na_to_zero(df)

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

    # Instantiate TimeSFM model
    timesfm = TimeSFMSamples(
        timeseries_df=df,
        case_col="Log_Cases",
        dynamic_numerical_cols=[
            "LAG_1_LOG_CASES",
            "LAG_1_tmin_roll_2",
            "LAG_1_prec_roll_2",
        ],
        dynamic_categorical_cols=["MONTH"],
        # static_categorical_cols=["GID_2"],
        horizon_len=1,
    )

    # Loop over provinces
    all_forecasts = []
    provinces = df["GID_2"].unique()
    dates = sorted(df["Date"].unique())
    for ix, province in enumerate(provinces):
        logging.info(
            f"Processing province {province} ({ix + 1}/{len(provinces)}) "
            "for TimeSFM model"
        )

        # Initialize the first forecast with NaN
        all_forecasts.append(
            pd.DataFrame(
                {
                    "Date": [dates[0]],
                    "GID_2": [province],
                    "sample": [0],
                    "prediction": [np.nan],  # initial prediction is NaN
                    "Cases": [
                        df.loc[(df["Date"] == dates[0]) & (df["GID_2"] == province)][
                            "Cases"
                        ].values[0]
                    ],
                }
            )
        )

        # Generate successive historical forecasts
        for date in dates[1:]:  # TimeSFM requires at least 2 data points
            logging.info(
                f"Processing date {date.strftime('%Y-%m-%d')} for province {province}"
            )
            # Subset data
            subset_df = df.copy()
            subset_df = subset_df[subset_df["GID_2"] == province]
            subset_df = subset_df[subset_df["Date"] <= date]
            subset_df = subset_df.fillna(
                0
            )  # TimeSFM cannot cope with NaN covariates #################
            timesfm.df = subset_df
            # Fit model
            df_forecast = timesfm.predict(
                sigma=0.1
            )  # ### Estimate sigma from past residuals
            # Overwrite Log_Cases with median prediction if marked 'future'
            if subset_df["future"].iloc[-1]:
                df.loc[
                    (df["Date"] == date) & (df["GID_2"] == province), "Log_Cases"
                ] = np.median(
                    df_forecast.loc[
                        (df_forecast["Date"] == date)
                        & (df_forecast["GID_2"] == province),
                        "prediction",
                    ]
                )
                logging.debug(
                    f"Backfilling Log_Cases for {date} in {province} with "
                    f"{df.loc[(df['Date'] == date) & (df['GID_2'] == province), 'Log_Cases']}"
                )
            # Append to master list
            df_forecast.loc[df_forecast["future"], "Cases"] = np.nan
            all_forecasts.append(
                df_forecast[df_forecast["Date"] == date][
                    ["Date", "GID_2", "sample", "prediction", "Cases"]
                ]
            )

    # Concatenate all province forecasts into a single DataFrame
    forecast_results = pd.concat(all_forecasts, ignore_index=True)

    # Inverse log transform
    forecast_results["prediction"] = np.clip(
        np.expm1(forecast_results["prediction"]),
        a_min=0,
        a_max=None,
    )

    logging.info("TimeSFM model complete.")
    return forecast_results
