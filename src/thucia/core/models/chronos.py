import logging
from pathlib import Path
from typing import List
from typing import Optional

import pandas as pd
from darts.models import Chronos2Model
from thucia.core.fs import DataFrame

from ._meta import ModelSpec
from .darts.foundation import CHRONOS_QUANTILES
from .darts.foundation import FoundationQuantileBase

SPEC = ModelSpec(
    name="chronos",
    family="darts",
    supports=frozenset(
        {
            "train_end_date",
            "retrain",
            "multivariate",
            "quantiles",
            "median_only",
            "past_covariate_cols",
            "future_covariate_cols",
        }
    ),
    sampling="quantiles",
    extras=("torch",),
)

__all__ = ["ChronosQuantiles", "ChronosQuantilesDarts", "CHRONOS_QUANTILES"]


# -------- Chronos (DARTS implementation) --------
class ChronosQuantiles(FoundationQuantileBase):
    """Chronos-2 via darts: zero-shot, direct quantile parameters.

    The pre-trained quantile grid is :data:`CHRONOS_QUANTILES`; request any
    subset of it (or ``median_only=True`` for a deterministic median).
    """

    MODEL_QUANTILES = CHRONOS_QUANTILES

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
            likelihood=self.likelihood(),
        )


#: Backwards-compatible alias for the pre-rename class name.
ChronosQuantilesDarts = ChronosQuantiles


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
    quantiles: Optional[List[float]] = None,
    median_only: bool = False,
    past_covariate_cols: Optional[List[str]] = None,
    future_covariate_cols: Optional[List[str]] = None,
) -> DataFrame | pd.DataFrame:
    """Chronos forecasting pipeline.

    Returns a Thucia DataFrame if db_file is specified, otherwise a pandas DataFrame.
    """
    logging.info("Starting Chronos forecasting pipeline...")

    # Chronos-2 is a pre-trained foundation model; the darts wrapper runs it
    # zero-shot and returns quantiles directly (no sampling), while still
    # supporting per-region and whole-dataset training via ``train_col``.
    model = ChronosQuantiles(
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
        quantiles=quantiles,
        median_only=median_only,
        past_covariate_cols=past_covariate_cols,
        future_covariate_cols=future_covariate_cols,
    )

    # Historical predictions
    tdf = model.historical_predictions(
        start_date=start_date,
        retrain=retrain,
        train_col=train_col,
    )
    logging.info("Completed Chronos forecasting pipeline.")

    return tdf
