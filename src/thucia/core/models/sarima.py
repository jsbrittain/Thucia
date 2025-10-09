import logging
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from darts.models import AutoARIMA

from .darts import DartsBase


# -------- SARIMA --------
class SarimaQuantiles(DartsBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_method = "samples"

    def build_model(self):
        # No global model
        return None

    def pre_fit(self, **kwargs):
        # No pre-fitting needed for SARIMA
        pass

    def historical_forecasts(self, ts, cov, start_date=None, **kwargs):
        model = AutoARIMA(season_length=12)
        bt = model.historical_forecasts(
            series=ts,
            future_covariates=cov,
            forecast_horizon=self.horizon,
            start=start_date,
            stride=1,
            retrain=True,
            last_points_only=False,  # this changes the output format
            verbose=False,
            num_samples=1000,
        )
        return bt


# -------- pipeline helper --------
def sarima(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp = pd.Timestamp.min,
    end_date: str | pd.Timestamp = pd.Timestamp.max,
    gid_1: Optional[List[str]] = None,
    horizon: int = 1,
    case_col: str = "Log_Cases",
    covariate_cols: Optional[List[str]] = None,
    retrain: bool = True,  # no effect
) -> pd.DataFrame:
    logging.info("Starting SARIMA forecasting pipeline...")

    # float32
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].astype(np.float32)

    model = SarimaQuantiles(
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
    )
    preds = preds_hist
    logging.info("Completed SARIMA forecasting pipeline.")

    return preds
