import logging
from pathlib import Path
from typing import List
from typing import Optional

import pandas as pd
from darts.models import AutoARIMA
from thucia.core.fs import DataFrame

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
            num_samples=self.num_samples,
        )
        return bt


# -------- pipeline helper --------
def sarima(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp | pd.Period = pd.Timestamp.min,
    end_date: str | pd.Timestamp | pd.Period = pd.Timestamp.max,
    gid_1: Optional[List[str]] = None,
    horizon: int = 1,
    case_col: str = "Log_Cases",
    covariate_cols: Optional[List[str]] = None,
    db_file: str | Path | None = None,
    num_samples: int | None = None,
    multivariate: bool = False,
    *args,
    **kwargs,
) -> DataFrame | pd.DataFrame:
    """SARIMA forecasting pipeline.

    Returns a Thucia DataFrame if db_file is specified, otherwise a pandas DataFrame.
    """
    logging.info("Starting SARIMA forecasting pipeline...")

    if args:
        logging.warning(f"Positional arguments {args} are ignored in sarima().")
    if kwargs:
        logging.warning(f"Keyword arguments {kwargs} are ignored in sarima().")

    if multivariate:
        raise ValueError("SARIMA does not support multivariate forecasting.")

    # Instantiate model
    model = SarimaQuantiles(
        df=df,
        case_col=case_col,
        covariate_cols=covariate_cols,
        horizon=horizon,
        num_samples=num_samples,
        db_file=db_file,
        multivariate=False,
    )

    # Historical predictions
    tdf = model.historical_predictions(
        start_date=start_date,
    )
    logging.info("Completed SARIMA forecasting pipeline.")

    return tdf
