import logging
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from darts.models.forecasting.tcn_model import TCNModel
from darts.utils.likelihood_models import GaussianLikelihood
from darts.utils.likelihood_models import QuantileRegression

from .darts import BaseSamples


# -------- TCN --------
#
# TCN is a non-iterative, block forecaster, therefore self.horizon values are output
# at once. Iterative forecasters would step through the forecast horizon one step at a
# time.
class TcnSamples(BaseSamples):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_method = "samples"

    def build_model(self):
        return TCNModel(
            input_chunk_length=48,  # how many past steps the model can see
            output_chunk_length=self.horizon,  # how many future steps the model predicts at once
            kernel_size=3,
            num_filters=4,
            # dropout=0.2,  # enables MC dropout
            random_state=42,
            likelihood=GaussianLikelihood(),  # sampling supported
            save_checkpoints=False,
            force_reset=True,
            n_epochs=150,
            # batch_size=64,
            # optimizer_kwargs={
            #     "lr": 1e-4,
            # },
        )


# -------- pipeline helper --------
def tcn2(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp = pd.Timestamp.min,
    end_date: str | pd.Timestamp = pd.Timestamp.max,
    gid_1: Optional[List[str]] = None,
    horizon: int = 1,
    case_col: str = "Cases",
    covariate_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    logging.info("Starting TCN forecasting pipeline...")

    # float32
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].astype(np.float32)
    df = df[df["GID_2"] == df["GID_2"].iloc[0]]  # TEMP: only one region

    model = TcnSamples(
        df=df,
        case_col=case_col,
        covariate_cols=covariate_cols,
        horizon=horizon,
        num_samples=1000,
        start_date=start_date,
    )

    # Historical predictions
    preds_hist = model.historical_predictions(
        start_date=start_date,
        retrain=False,  # only use False for a quick test
    )

    # Forecasting
    preds_future = model.forecast_using_past_covariances(
        horizon=horizon,
        train=False,  # can be False if historical_predictions was called first
    )
    preds = pd.concat([preds_hist, preds_future], ignore_index=True)

    # Add Cases column to predictions
    all_gids = preds[model.geo_col].unique()
    all_dates = preds[model.date_col].unique()
    preds[case_col] = np.nan
    for gid in all_gids:
        for date in all_dates:
            mask = (df[model.geo_col] == gid) & (df[model.date_col] == date)
            if mask.any():
                preds.loc[
                    (preds[model.geo_col] == gid) & (preds[model.date_col] == date),
                    case_col,
                ] = df.loc[mask, case_col].values[0]
    df.drop(columns=["index"], inplace=True, errors="ignore")

    logging.info("TCN forecasting complete.")
    return preds
