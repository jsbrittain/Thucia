import logging
from pathlib import Path
from typing import List
from typing import Optional

import pandas as pd
from darts.models import TimesFM3Model
from thucia.core.fs import DataFrame

from ._meta import ModelSpec
from .darts.foundation import TIMESFM_QUANTILES
from .darts.foundation import FoundationQuantileBase

SPEC = ModelSpec(
    name="timesfm3",
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

#: Human-readable licence note used in the error raised by ``accept_license``.
LICENCE_MESSAGE = (
    "TimesFM 3.0 pre-trained weights are distributed under the TimesFM "
    "Non-Commercial License v1.0 and are restricted to non-commercial, "
    "non-production use. Pass accept_license=True to acknowledge the licence "
    "(https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)."
)

__all__ = ["TimesFMQuantiles3", "TIMESFM_QUANTILES"]


# -------- TimesFM 3.0 (DARTS implementation) --------
class TimesFMQuantiles3(FoundationQuantileBase):
    """TimesFM 3.0 via darts: zero-shot, native past/future covariates.

    The pre-trained quantile grid is :data:`TIMESFM_QUANTILES` (0.1-0.9); request
    any subset of it, or ``median_only=True`` for a deterministic median.
    """

    MODEL_QUANTILES = TIMESFM_QUANTILES

    def __init__(
        self,
        *args,
        accept_license: bool = False,
        input_chunk_length: int | tuple[int, int] = (32, 512),
        enable_finetuning: bool | dict | None = None,
        **kwargs,
    ):
        if not accept_license:
            raise ValueError(LICENCE_MESSAGE)
        self.accept_license = True
        # (min, max) lets short per-region histories use whatever they have
        # instead of being NaN-padded up to a fixed window.
        self.input_chunk_length = input_chunk_length
        self.enable_finetuning = enable_finetuning
        super().__init__(*args, **kwargs)

    def build_model(self, horizon):
        kwargs = {
            "input_chunk_length": self.input_chunk_length,
            "output_chunk_length": horizon,
            "likelihood": self.likelihood(),
            "accept_license": True,
        }
        if self.enable_finetuning is not None:
            kwargs["enable_finetuning"] = self.enable_finetuning
        return TimesFM3Model(**kwargs)


# -------- pipeline helper --------
def timesfm3(
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
    accept_license: bool = False,
    input_chunk_length: int | tuple[int, int] = (32, 512),
    enable_finetuning: bool | dict | None = None,
) -> DataFrame | pd.DataFrame:
    """TimesFM 3.0 forecasting pipeline.

    Requires ``accept_license=True`` (the weights are non-commercial).

    Returns a Thucia DataFrame if db_file is specified, otherwise a pandas DataFrame.
    """
    logging.info("Starting TimesFM 3.0 forecasting pipeline...")

    model = TimesFMQuantiles3(
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
        accept_license=accept_license,
        input_chunk_length=input_chunk_length,
        enable_finetuning=enable_finetuning,
    )

    # Historical predictions
    tdf = model.historical_predictions(
        start_date=start_date,
        retrain=retrain,
        train_col=train_col,
    )
    logging.info("Completed TimesFM 3.0 forecasting pipeline.")

    return tdf
