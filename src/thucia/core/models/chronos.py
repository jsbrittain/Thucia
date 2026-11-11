import logging
from pathlib import Path
from typing import List
from typing import Optional

import pandas as pd
from darts.models import Chronos2Model
from darts.utils.likelihood_models import QuantileRegression
from thucia.core.fs import DataFrame

from ._meta import ModelSpec
from .darts import DartsBase

SPEC = ModelSpec(
    name="chronos",
    family="darts",
    supports=frozenset({"train_end_date", "retrain", "multivariate"}),
    sampling="quantiles",
    extras=("chronos",),
)

# Chronos-2 is pre-trained on a fixed quantile grid; this is the intersection
# with the canonical grid in thucia.core.quantiles (it lacks 0.025 / 0.975).
CHRONOS_QUANTILES = [
    0.01,
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
    0.99,
]


# -------- Chronos (DARTS implementation) --------
class ChronosQuantilesDarts(DartsBase):
    def __init__(self, *args, **kwargs):
        # Model parameters
        self.input_chunk_length = 48  # how many past steps the model can see
        super().__init__(*args, **kwargs)
        # Chronos-2 predicts quantiles directly (zero-shot, no sampling);
        # DartsBase parses them from the ``*_q<level>`` components returned by
        # historical_forecasts.
        self.sampling_method = "quantiles"

    def build_model(self, horizon):
        return Chronos2Model(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=horizon,
            likelihood=QuantileRegression(quantiles=self.quantiles),
        )

    def pre_fit(self, target_gids=None, **kwargs):
        logging.info(
            "Preparing Chronos-2 model on historical data "
            f"({self.train_start_date} to {self.train_end_date})..."
        )
        # ``fit`` with darts' default ``epochs=0`` only registers the training
        # series/covariates; Chronos-2 stays a pre-trained (zero-shot) model.
        target_list, covar_list, _ = self.get_cases(
            future=False,
            target_gids=target_gids,
            start_date=self.train_start_date,
            end_date=self.train_end_date,
        )
        self.model.fit(
            series=target_list,
            past_covariates=covar_list,
            verbose=True,
        )

    def historical_forecasts(
        self, ts, cov, start_date=None, retrain=True, horizon=1, **kwargs
    ):
        logging.info(
            "Generating Chronos-2 historical forecasts "
            f"from {start_date} with retrain={retrain}..."
        )
        # ``last_points_only=True`` returns the target step of each horizon
        # vector (Date = origin + horizon), matching the other darts models.
        # ``predict_likelihood_parameters=True`` yields the requested quantile
        # levels directly, rather than drawing (and then collapsing) samples.
        return self.model.historical_forecasts(
            series=ts,
            past_covariates=cov,
            forecast_horizon=horizon,
            start=start_date,
            stride=1,
            retrain=retrain,
            last_points_only=True,
            predict_likelihood_parameters=True,
            verbose=False,
        )


# -------- pipeline helper --------
def chronos(
    df: DataFrame | pd.DataFrame,
    start_date: str | pd.Timestamp = pd.Timestamp.min,
    end_date: str | pd.Timestamp = pd.Timestamp.max,
    train_start_date: str | pd.Timestamp = pd.Timestamp.min,
    train_end_date: str | pd.Timestamp = pd.Timestamp.max,
    geo_col: str = "GID_2",
    geo_parent: Optional[str] = "GID_1",
    geo_parent_filter: Optional[List[str]] = None,
    horizons: List[int] = [1],
    case_col: str = "Log_Cases",
    covariate_cols: Optional[List[str]] = None,
    retrain: bool = True,  # Only use False for a quick test
    db_file: str | Path | None = None,
    multivariate: bool = False,
    train_col: Optional[str] = None,
) -> DataFrame | pd.DataFrame:
    """Chronos forecasting pipeline.

    Returns a Thucia DataFrame if db_file is specified, otherwise a pandas DataFrame.
    """
    logging.info("Starting Chronos forecasting pipeline...")

    # Chronos-2 is a pre-trained foundation model; the darts wrapper runs it
    # zero-shot and returns quantiles directly (no sampling), while still
    # supporting per-region and whole-dataset training via ``train_col``.
    model = ChronosQuantilesDarts(
        df=df,
        case_col=case_col,
        geo_col=geo_col,
        geo_parent=geo_parent,
        covariate_cols=covariate_cols,
        horizons=horizons,
        db_file=db_file,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        multivariate=multivariate,
        quantiles=CHRONOS_QUANTILES,
    )

    # Historical predictions
    tdf = model.historical_predictions(
        start_date=start_date,
        retrain=retrain,
        train_col=train_col,
    )
    logging.info("Completed Chronos forecasting pipeline.")

    return tdf
