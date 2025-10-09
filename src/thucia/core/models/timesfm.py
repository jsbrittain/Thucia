import logging
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from darts import TimeSeries
from timesfm import TimesFm
from timesfm import TimesFmCheckpoint
from timesfm import TimesFmHparams

from .darts import DartsBase


# -------- TimesFM --------
class TimesFMSamples(DartsBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_method = "samples"

    def build_model(self):
        return TimesFm(
            hparams=TimesFmHparams(
                backend="cpu",
                context_len=32,
                horizon_len=self.horizon,
                num_layers=50,
                per_core_batch_size=32,
                use_positional_embedding=True,
            ),
            checkpoint=TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
            ),
        )

    def pre_fit(self, target_gids=None, **kwargs):
        pass

    def historical_forecasts(self, ts, cov, start_date=None, retrain=True, **kwargs):
        dates = ts.time_index
        output = []
        for tix, t in enumerate(dates):  # forecast target date at horizon
            logging.info(f"Forecasting for date: {t}...")
            if tix < self.horizon:
                logging.info("Not enough data to forecast, skipping...")
                continue

            # Temporal masks
            input_mask = ts.time_index <= dates[tix - self.horizon]
            covariate_mask = ts.time_index <= t

            # Inputs
            tsv = ts.values().squeeze()
            inputs = [tsv[input_mask].tolist()]

            # Covariates
            dynamic_numerical_covariates = {}
            dcov = cov.values().squeeze()
            for i in range(dcov.shape[1]):
                dynamic_numerical_covariates[f"covariate_{i}"] = [
                    dcov[covariate_mask, i].tolist()
                ]

            # Forecast
            forecasts, _ols = self.model.forecast_with_covariates(
                inputs=inputs,
                dynamic_numerical_covariates=dynamic_numerical_covariates,
                dynamic_categorical_covariates=None,
                static_numerical_covariates=None,  # No grouping
                static_categorical_covariates=None,
                freq=[1],  # 1 = monthly
                xreg_mode="xreg + timesfm",
            )

            times = ts.time_index[covariate_mask][-self.horizon :]
            output.append(TimeSeries.from_times_and_values(times, forecasts[0]))

        # output = [time]series[horizon][1][samples]
        return output


# -------- pipeline helper --------
def timesfm(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp = pd.Timestamp.min,
    end_date: str | pd.Timestamp = pd.Timestamp.max,
    gid_1: Optional[List[str]] = None,
    horizon: int = 1,
    case_col: str = "Log_Cases",
    covariate_cols: Optional[List[str]] = None,
    retrain: bool = True,  # Only use False for a quick test
) -> pd.DataFrame:
    logging.info("Starting TimesFM forecasting pipeline...")

    # float32
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].astype(np.float32)

    preds_hist = []
    gid_list = df["GID_2"].unique().tolist()
    for ix, gid in enumerate(gid_list):
        logging.info(f"Processing GID_2: {gid}...")
        tic = pd.Timestamp.now()

        df_gid = df[df["GID_2"] == gid].copy()

        model = TimesFMSamples(
            df=df_gid,
            case_col=case_col,
            covariate_cols=covariate_cols,
            horizon=horizon,
            num_samples=1000,
            start_date=start_date,
        )

        # Historical predictions
        preds_hist.append(
            model.historical_predictions(
                start_date=start_date,
                retrain=retrain,
            )
        )

        toc = pd.Timestamp.now()
        logging.info(f"Completed GID_2: {gid} in {toc - tic}.")

        estimated_time_remaining = (toc - tic) * (len(gid_list) - ix - 1)
        logging.info(f"Estimated time remaining: {estimated_time_remaining}.")

    preds = pd.concat(preds_hist).reset_index(drop=True)
    return preds
