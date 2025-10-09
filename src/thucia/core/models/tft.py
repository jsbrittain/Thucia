import logging
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from darts.models import TFTModel
from darts.utils.likelihood_models import GaussianLikelihood

from .darts import DartsBase


# -------- TFT --------
class TftSamples(DartsBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_method = "samples"

    def build_model(self):
        return TFTModel(
            input_chunk_length=48,
            output_chunk_length=1,
            hidden_size=16,
            lstm_layers=1,
            num_attention_heads=4,
            dropout=0.2,  # MC dropout adds stochasticity as well
            likelihood=GaussianLikelihood(),
            random_state=42,
            n_epochs=150,  # default 100
            batch_size=64,
            force_reset=True,
            add_relative_index=True,  # gives a simple future encoder even without explicit future covs
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
def tft(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp = pd.Timestamp.min,
    end_date: str | pd.Timestamp = pd.Timestamp.max,
    gid_1: Optional[List[str]] = None,
    horizon: int = 1,
    case_col: str = "Log_Cases",
    covariate_cols: Optional[List[str]] = None,
    retrain: bool = True,  # Only use False for a quick test
) -> pd.DataFrame:
    logging.info("Starting TFT forecasting pipeline...")

    # float32
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].astype(np.float32)

    model = TftSamples(
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
