import logging
import warnings
from pathlib import Path
from typing import List
from typing import Optional

import pandas as pd
from darts.models import TimesFM2p5Model
from thucia.core.fs import DataFrame

from ._meta import ModelSpec
from .darts.foundation import FoundationQuantileBase
from .darts.foundation import TIMESFM_QUANTILES

SPEC = ModelSpec(
    name="timesfm25",
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

__all__ = ["TimesFMQuantiles25", "TIMESFM_QUANTILES"]


# -------- TimesFM 2.5 (DARTS implementation) --------
class TimesFMQuantiles25(FoundationQuantileBase):
    """TimesFM 2.5 via darts: zero-shot, quantiles, no covariates.

    The checkpoint supports neither past nor future covariates; covariate
    columns supplied by the pipeline are ignored with a warning. The
    pre-trained quantile grid is :data:`TIMESFM_QUANTILES` (0.1-0.9).
    """

    MODEL_QUANTILES = TIMESFM_QUANTILES
    SUPPORTS_COVARIATES = False

    def __init__(
        self,
        *args,
        input_chunk_length: int | tuple[int, int] = (32, 512),
        **kwargs,
    ):
        covariate_cols = kwargs.get("covariate_cols") or []
        if covariate_cols:
            warnings.warn(
                "TimesFM 2.5 does not support covariates; ignoring "
                f"{len(covariate_cols)} covariate column(s): {list(covariate_cols)}",
                UserWarning,
                stacklevel=2,
            )
            kwargs["covariate_cols"] = []
        self.input_chunk_length = input_chunk_length
        super().__init__(*args, **kwargs)

    def build_model(self, horizon):
        # TimesFM 2.5's darts class caps output_chunk_length (+ shift) at 128.
        if horizon > 128:
            raise ValueError(f"TimesFM 2.5 supports horizons up to 128, got {horizon}.")
        return TimesFM2p5Model(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=horizon,
            likelihood=self.likelihood(),
        )


# -------- pipeline helper --------
def timesfm25(
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
    input_chunk_length: int | tuple[int, int] = (32, 512),
) -> DataFrame | pd.DataFrame:
    """TimesFM 2.5 forecasting pipeline.

    This checkpoint takes no covariates: any ``covariate_cols`` are ignored with
    a warning, and passing ``past_covariate_cols``/``future_covariate_cols``
    raises.

    Returns a Thucia DataFrame if db_file is specified, otherwise a pandas DataFrame.
    """
    logging.info("Starting TimesFM 2.5 forecasting pipeline...")

    model = TimesFMQuantiles25(
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
        input_chunk_length=input_chunk_length,
    )

    # Historical predictions
    tdf = model.historical_predictions(
        start_date=start_date,
        retrain=retrain,
        train_col=train_col,
    )
    logging.info("Completed TimesFM 2.5 forecasting pipeline.")

    return tdf
