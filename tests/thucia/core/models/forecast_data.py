# Synthetic seasonal case data for forecast tests.
#
# The generator produces seasonal + trend + noise case counts for a small
# number of GIDs with NaN-free covariates and a categorical geo column.
# History is long enough (>= 48 periods before any forecast origin) for
# lags-based models such as xgboost (input_chunk_length=48).
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def season_length_for_freq(freq: str) -> int:
    """Number of periods per seasonal cycle (monthly 12, weekly 52, daily 365)."""
    if freq.startswith("W"):
        return 52
    if freq.startswith("D"):
        return 365
    return 12


def make_forecast_df(
    n_periods: int = 60,
    n_gid: int = 2,
    start: str = "2016-01",
    seed: int = 0,
    with_covariates: bool = True,
    gid_prefix: str = "X",
    freq: str = "M",
    season_length: Optional[int] = None,
) -> pd.DataFrame:
    """Return a synthetic case DataFrame with the columns models expect.

    Columns: Date (period[<freq>]), GID_1, GID_2 (categorical), future, Cases,
    Log_Cases, and covariates (tmin, prec) that are NaN-free.

    `freq` may be any pandas Period frequency ("M", "W-SAT", "W-SUN", "D", ...).
    Fits are kept fast by forecasting only a short trailing window against a
    modest history.
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start, periods=n_periods, freq=freq)
    season = (
        season_length if season_length is not None else season_length_for_freq(freq)
    )
    bases = np.linspace(20.0, 60.0, n_gid)
    rows = []
    for gi, base in enumerate(bases):
        gid_1 = f"{gid_prefix}.{gi + 1}_1"
        gid_2 = f"{gid_prefix}.{gi + 1}.1_2"
        for i, d in enumerate(idx):
            seasonal = 1 + 0.5 * np.sin(2 * np.pi * i / season)
            trend = 1 + 0.005 * i
            cases = base * seasonal * trend + rng.uniform(-2, 2)
            rows.append(
                {
                    "Date": d,
                    "GID_1": gid_1,
                    "GID_2": gid_2,
                    "future": False,
                    "Cases": max(cases, 0.0),
                }
            )
    df = pd.DataFrame(rows)
    df["GID_1"] = df["GID_1"].astype("category")
    df["GID_2"] = df["GID_2"].astype("category")
    df["Log_Cases"] = np.log1p(df["Cases"])

    if with_covariates:
        cov_rng = np.random.default_rng(seed + 1)
        df["tmin"] = cov_rng.uniform(15, 25, len(df))
        df["prec"] = cov_rng.uniform(50, 200, len(df))
    return df


COVARIATE_COLS = ["tmin", "prec"]
