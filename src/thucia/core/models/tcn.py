import logging
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from darts.models import TCNModel
from darts.utils.likelihood_models import QuantileRegression

from .darts import DartsBase


# -------- TCN --------
class TcnSamples(DartsBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_method = "samples"

    def build_model(self):
        return TCNModel(
            input_chunk_length=48,  # how many past steps the model can see
            output_chunk_length=1,  # how many future steps the model predicts at once
            kernel_size=3,
            num_filters=4,
            dropout=0.2,  # enables MC dropout
            random_state=42,
            likelihood=QuantileRegression(),  # sampling supported
            save_checkpoints=False,
            force_reset=True,
            n_epochs=150,  # default 100
            batch_size=64,
            optimizer_kwargs={
                "lr": 1e-4,
            },
        )

    def pre_fit(self, target_gids=None, **kwargs):
        target_list, covar_list, _ = self.get_cases(
            future=False,
            target_gids=target_gids,
        )  # historical data only
        self.model.fit(
            series=target_list,
            past_covariates=covar_list,
            verbose=True,
        )

    def historical_forecasts(self, ts, cov, start_date=None, retrain=True, **kwargs):
        bt = self.model.historical_forecasts(
            series=ts,
            past_covariates=cov,
            forecast_horizon=self.horizon,
            start=start_date,
            stride=1,
            retrain=retrain,
            last_points_only=False,  # this changes the output format
            verbose=False,
            num_samples=1000,
        )
        return bt


# -------- pipeline helper --------
def tcn(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp = pd.Timestamp.min,
    end_date: str | pd.Timestamp = pd.Timestamp.max,
    gid_1: Optional[List[str]] = None,
    horizon: int = 1,
    case_col: str = "Log_Cases",
    covariate_cols: Optional[List[str]] = None,
    retrain: bool = True,  # Only use False for a quick test
) -> pd.DataFrame:
    logging.info("Starting TCN forecasting pipeline...")

    # float32
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].astype(np.float32)

    preds_hist = []
    gid_list = df["GID_2"].unique().tolist()
    for ix, gid in enumerate(gid_list):
        logging.info(f"Processing GID_2: {gid}...")
        tic = pd.Timestamp.now()

        df_gid = df[df["GID_2"] == gid].copy()

        model = TcnSamples(
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
