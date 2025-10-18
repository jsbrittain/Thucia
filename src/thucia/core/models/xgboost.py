import logging
from pathlib import Path
from typing import List
from typing import Optional

import pandas as pd
from darts.models import XGBModel
from thucia.core.fs import DataFrame

from .darts import DartsBase


# -------- XGBoost --------
class XGBoostSamples(DartsBase):
    def __init__(self, *args, **kwargs):
        self.input_chunk_length = 48
        super().__init__(*args, **kwargs)

    def build_model(self):
        return XGBModel(
            lags=self.input_chunk_length,
            lags_past_covariates=self.input_chunk_length,
            lags_future_covariates=None,
            output_chunk_length=1,
            # probabilistic:
            likelihood="quantile",  # or "poisson" for count data
            quantiles=self.quantiles,
            # a few sane XGBoost defaults (tunable):
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )

    def pre_fit(self, target_gids=None, **kwargs):
        logging.info(
            "Fitting XGBoost model on historical data "
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

    def historical_forecasts(self, ts, cov, start_date=None, retrain=True, **kwargs):
        logging.info(
            "Generating XGBoost historical forecasts "
            f"from {start_date} with retrain={retrain}..."
        )
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
def xgboost(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp = pd.Timestamp.min,
    end_date: str | pd.Timestamp = pd.Timestamp.max,
    train_start_date: str | pd.Timestamp = pd.Timestamp.min,
    train_end_date: str | pd.Timestamp = pd.Timestamp.max,
    gid_1: Optional[List[str]] = None,
    horizon: int = 1,
    case_col: str = "Log_Cases",
    covariate_cols: Optional[List[str]] = None,
    retrain: bool = True,  # Only use False for a quick test
    db_file: str | Path | None = None,
    train_per_region: bool = True,  # Train a separate model for each region
) -> DataFrame | pd.DataFrame:
    logging.info("Starting XGBoost forecasting pipeline...")

    # Instantiate model
    model = XGBoostSamples(
        df=df,
        case_col=case_col,
        covariate_cols=covariate_cols,
        horizon=horizon,
        num_samples=1000,
        db_file=db_file,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
    )

    # Historical predictions
    tdf = model.historical_predictions(
        start_date=start_date,
        retrain=retrain,
        train_per_region=train_per_region,
    )
    logging.info("Completed XGBoost forecasting pipeline.")

    return tdf
