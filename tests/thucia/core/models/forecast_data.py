# Synthetic monthly case data for forecast tests.
#
# The generator produces seasonal + trend + noise case counts for a small
# number of GIDs with NaN-free covariates and a categorical geo column.
# History is long enough (>= 48 months before any forecast origin) for
# lags-based models such as xgboost (input_chunk_length=48).
from __future__ import annotations

import numpy as np
import pandas as pd


SEASON_LENGTH = 12
MIN_HISTORY_MONTHS = 60  # >= input_chunk_length (48) plus slack


def make_forecast_df(
    n_months: int = 60,
    n_gid: int = 2,
    start: str = "2016-01",
    seed: int = 0,
    with_covariates: bool = True,
    gid_prefix: str = "X",
) -> pd.DataFrame:
    """Return a monthly case DataFrame with the columns models expect.

    Columns: Date (period[M]), GID_1, GID_2 (categorical), future, Cases,
    Log_Cases, and covariates (tmin, prec) that are NaN-free.

    Fits are kept fast by forecasting only a short trailing window (last ~6
    months; >= max horizon for multi-horizon tests) against a modest history.
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start, periods=n_months, freq="M")
    bases = np.linspace(20.0, 60.0, n_gid)
    rows = []
    for gi, base in enumerate(bases):
        gid_1 = f"{gid_prefix}.{gi + 1}_1"
        gid_2 = f"{gid_prefix}.{gi + 1}.1_2"
        for i, d in enumerate(idx):
            seasonal = 1 + 0.5 * np.sin(2 * np.pi * i / SEASON_LENGTH)
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
