import logging
from pathlib import Path
from typing import List
from typing import Optional

import pandas as pd
from darts.models import NBEATSModel
from darts.utils.likelihood_models import QuantileRegression
from thucia.core.fs import DataFrame

from .darts import DartsBase


quantiles = [
    0.01,
    0.025,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.975,
    0.99,
]


class NBEATSSamples(DartsBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_method = "quantiles"

    def build_model(self, horizon):
        return NBEATSModel(
            input_chunk_length=48,
            output_chunk_length=horizon,
            generic_architecture=True,
            dropout=0.2,
            likelihood=QuantileRegression(quantiles),
            random_state=42,
            n_epochs=150,
            # batch_size=64,
            # force_reset=False,
            # loss_fn=nn.L1Loss(),   # MAE
            # optimizer_kwargs={
            #     "lr": 1e-4,
            #     "weight_decay": 1e-5,
            # },
        )

    def pre_fit(self, target_gids=None, **kwargs):
        logging.info(
            "Fitting NBEATS model on historical data "
            f"({self.train_start_date} to {self.train_end_date})..."
        )
        target_list, covar_list, _ = self.get_cases(
            future=False,
            target_gids=target_gids,
            start_date=self.train_start_date,
            end_date=self.train_end_date,
        )  # historical data only
        self.model.fit(
            series=target_list,
            past_covariates=covar_list,
            verbose=True,
        )

    def historical_forecasts(
        self, ts, cov, horizon, start_date=None, retrain=True, **kwargs
    ):
        logging.info(
            "Generating NBEATS historical forecasts "
            f"from {start_date} with retrain={retrain}..."
        )
        bt = self.model.historical_forecasts(
            series=ts,
            past_covariates=cov,
            forecast_horizon=horizon,
            start=start_date,
            stride=1,
            retrain=retrain,
            last_points_only=True,
            verbose=False,
            num_samples=1,
            predict_likelihood_parameters=True,
        )
        return bt


def nbeats(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp = pd.Timestamp.min,
    end_date: str | pd.Timestamp = pd.Timestamp.max,
    train_start_date: str | pd.Timestamp = pd.Timestamp.min,
    train_end_date: str | pd.Timestamp = pd.Timestamp.max,
    gid_1: Optional[List[str]] = None,
    horizons: int = 1,
    case_col: str = "Log_Cases",
    covariate_cols: Optional[List[str]] = None,
    retrain: bool = True,  # Only use False for a quick test
    db_file: str | Path | None = None,
    model_admin_level: int | None = None,
    num_samples: int | None = None,
    multivariate: bool = True,
) -> DataFrame | pd.DataFrame:
    """NBEATS forecasting pipeline.

    Returns a Thucia DataFrame if db_file is specified, otherwise a pandas DataFrame.
    """
    logging.info("Starting NBEATS forecasting pipeline...")

    import numpy as np

    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].astype(np.float32)

    # Instantiate model
    model = NBEATSSamples(
        df=df,
        case_col=case_col,
        covariate_cols=covariate_cols,
        horizons=horizons,
        num_samples=num_samples,
        db_file=db_file,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        multivariate=multivariate,
    )

    # Historical predictions
    tdf = model.historical_predictions(
        start_date=start_date,
        retrain=retrain,
        model_admin_level=model_admin_level,
    )
    logging.info("Completed NBEATS forecasting pipeline.")

    return tdf
