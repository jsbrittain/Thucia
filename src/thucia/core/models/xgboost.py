import logging
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from darts.models import XGBModel

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
def xgboost(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp = pd.Timestamp.min,
    end_date: str | pd.Timestamp = pd.Timestamp.max,
    gid_1: Optional[List[str]] = None,
    horizon: int = 1,
    case_col: str = "Log_Cases",
    covariate_cols: Optional[List[str]] = None,
    retrain: bool = True,  # Only use False for a quick test
) -> pd.DataFrame:
    logging.info("Starting XGBoost forecasting pipeline...")

    # float32
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].astype(np.float32)

    model = XGBoostSamples(
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
        retrain=retrain,
    )
    preds = preds_hist

    return preds
