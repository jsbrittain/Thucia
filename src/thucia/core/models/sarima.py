import logging
from pathlib import Path
from typing import List
from typing import Optional

import pandas as pd
from darts.models import ARIMA
from darts.models import AutoARIMA
from thucia.core.fs import DataFrame

from .darts import DartsBase


# -------- SARIMA --------
class SarimaQuantiles(DartsBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_method = "samples"
        self.sarima_retrain = False
        self.method = "powell"

    def set_retrain(self, retrain: bool):
        self.sarima_retrain = retrain

    def set_season_length(self, season_length: int):
        self.season_length = season_length

    def set_method(self, method: str):
        self.method = method

    def build_model(self):
        # No global model
        return None

    def remove_gid_covariate(self, cov):
        try:
            cov = cov.drop_columns("GID_2_codes")
        except Exception:
            pass

    def pre_fit(self, target_gids=None, **kwargs):
        # Only determine model order if not retraining
        if self.sarima_retrain:
            return
        # Determine model structure from historical data
        self.fixed_order = {}
        logging.info("SARIMA pre-fitting: determining model orders for each GID...")
        for ix, target_gid in enumerate(target_gids):
            tic = pd.Timestamp.now()
            target_list, covar_list, _ = self.get_cases(
                future=False,
                target_gids=[target_gid],
                start_date=self.train_start_date,
                end_date=self.train_end_date,
            )
            cov = self.remove_gid_covariate(covar_list[0])
            model = AutoARIMA(season_length=self.season_length)
            model.fit(
                target_list[0],
                future_covariates=cov,
            )
            p, q, P, Q, s, d, D = model.model.model_["arma"]
            s_info = f"{s}"
            if s != self.season_length:
                s = self.season_length
                s_info = f"{s_info} -> 12 [adjusted]"
            self.fixed_order[target_gid] = {
                "p": p,
                "q": q,
                "P": P,
                "Q": Q,
                "s": s,
                "d": d,
                "D": D,
            }
            toc = pd.Timestamp.now()
            logging.info(
                f"Determined SARIMA order for gid {target_gid}: "
                f"(p,d,q)=({p},{d},{q}), (P,D,Q,s)=({P},{D},{Q},{s_info}) "
                f"in {toc - tic}"
            )

    def historical_forecasts(self, ts, cov, gid=None, start_date=None, **kwargs):
        if self.sarima_retrain:
            model = AutoARIMA(
                season_length=self.season_length,
            )
        else:
            if not gid:
                raise ValueError("gid must be provided when not retraining SARIMA.")
            model = ARIMA(
                p=self.fixed_order[gid]["p"],
                d=self.fixed_order[gid]["d"],
                q=self.fixed_order[gid]["q"],
                seasonal_order=(
                    self.fixed_order[gid]["P"],
                    self.fixed_order[gid]["D"],
                    self.fixed_order[gid]["Q"],
                    12,  # season length (AutoArima can yield 1)
                ),
            )
        # Remove GID covariate since SARIMA is univariate
        cov = self.remove_gid_covariate(cov)
        # Historical forecasts
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
    retrain: bool = False,  # AutoARIMA at every step
    db_file: str | Path | None = None,
    model_admin_level: int = 2,  # GID 2 level
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
        logging.warning("SARIMA does not support multivariate forecasting.")

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
    model.set_season_length(season_length=12)
    model.set_retrain(True)

    # Historical predictions
    tdf = model.historical_predictions(
        start_date=start_date,
        model_admin_level=model_admin_level,
    )
    logging.info("Completed SARIMA forecasting pipeline.")

    return tdf
