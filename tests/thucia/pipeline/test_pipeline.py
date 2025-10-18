import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from thucia.core import models
from thucia.core.cases import cases_per_month
from thucia.core.cases import check_covars_for_nans
from thucia.core.cases import check_index_combinations
from thucia.core.cases import read_db
from thucia.core.cases import write_db
from thucia.core.geo import (
    pad_admin2,
)
from thucia.core.models import filter_admin1
from thucia.core.models import interpolate_missing_dates
from thucia.core.models import run_model
from thucia.core.models import set_historical_na_to_zero
from thucia.core.models.utils import sanitise_covariates


path = (Path(__file__).parent / "test_data").resolve()

logging.basicConfig(level=logging.INFO)


def test_read_write():
    df = pd.read_csv(path / "cases.csv")
    write_db(df, path / "cases")
    tdf = read_db(path / "cases")
    assert len(df) == len(tdf)


def test_cases_per_month():
    # Thucia DataFrame
    tdf = read_db(path / "cases")  # Line-list data

    # Aggregate cases per month
    initial_case_count = len(tdf)
    logging.info("Initial case count: %d", initial_case_count)
    tdf = cases_per_month(tdf)

    # Mock geo data for lookup testing
    n = len(tdf["GID_2"].unique())
    gdf = pd.DataFrame(
        {
            "GID_1": ["GID_000"] * n,
            "GID_2": [f"GID_{i:02d}" for i in range(1, n + 1)],
            "NAME_1": ["Admin1"] * n,
            "NAME_2": [f"Admin2_{i}" for i in range(1, n + 1)],
            "geometry": [None] * n,
        }
    )
    with patch("thucia.core.geo.get_admin2_list", return_value=gdf):
        # Ensure all Admin-2 regions included for covariate maps
        tdf = pad_admin2(tdf)

    # Add placeholders for future months
    last_date = tdf["Date"].max()
    n_months = 12
    future_dates = pd.period_range(
        start=last_date + 1,  # add 1 Period
        periods=n_months,
        freq=last_date.freq,
    )
    df = tdf.df  # convert to pandas DataFrame for edits
    df["future"] = False
    df_last_date = df[df["Date"] == last_date].copy()
    for date in future_dates:
        df_future = df_last_date.copy()
        df_future["Date"] = date
        df_future["Cases"] = np.nan
        df_future["future"] = True
        df = pd.concat([df, df_future], ignore_index=True)
    df.sort_values(by=["Date", "GID_2"], inplace=True)

    processed_case_count = df["Cases"].sum()
    assert initial_case_count == processed_case_count, (
        "Case count mismatch after processing."
    )

    write_db(df, path / "cases_per_month")


@pytest.mark.skip
def test_merge_covariates():
    pass


def test_model_fitting():
    # Add (random) covariate data
    tdf = read_db(path / "cases_per_month")
    df = tdf.df
    df["tmin"] = np.arange(len(df)) % 12
    df["tmax"] = np.arange(len(df)) % 6
    df["prec"] = np.arange(len(df)) % 3

    # Model parameters
    horizon = 12  # <-- select horizon here (### Must be at least 12 for SARIMA ###)
    model = models.timesfm  # <-- select model here
    start_date = pd.Period("2020-01", freq="M")  # <-- select start date here
    retrain = False  # <-- retrain at each forecast date

    # Filter to selected admin-1 regions and clean data
    gid_1 = ["GID1"]  # lookup_gid1(iso3=iso3, admin1_names=adm1)
    df = filter_admin1(df, gid_1)
    df = interpolate_missing_dates(df)
    df = set_historical_na_to_zero(df)

    # Load and merge covariates (treat all as past covariates)
    case_col = "Log_Cases"
    df["Log_Cases"] = np.log1p(df["Cases"])
    df["tmax_lag_0"] = df.groupby("GID_2", observed=False)["tmax"].shift(0)
    df["tmin_lag_0"] = df.groupby("GID_2", observed=False)["tmin"].shift(0)
    df["prec_lag_0"] = df.groupby("GID_2", observed=False)["prec"].shift(0)
    df["log_cases_lag_1"] = df.groupby("GID_2", observed=False)[case_col].shift(1)

    covariate_cols = [
        "tmax_lag_0",  # tmax
        "tmin_lag_0",  # tmin
        "prec_lag_0",  # prec
        "log_cases_lag_1",  # previous cases
    ]
    df = df[["Date", "GID_1", "GID_2", "future", "Cases", case_col] + covariate_cols]

    # Sanitise covariates (NaN replacement, seasonal mean, forward fill, back fill)
    df = sanitise_covariates(df, covariate_cols, start_date)

    # Sanity checks - missing values and missing index combinations (Date, GID_2)
    check_covars_for_nans(df, covariate_cols)
    check_covars_for_nans(df[~df["future"]], [case_col])
    check_index_combinations(df, group_cols=["Date", "GID_2"])

    # Run model
    name = f"{model.__name__}_h{horizon}"
    logging.info("Running model: %s", name)
    df_model = run_model(
        name,
        model,
        df=df,
        path=path,
        save_samples=False,
        save_quantiles=False,
        # Model parameters
        model_kwargs={
            "start_date": start_date,
            "gid_1": gid_1,
            "horizon": horizon,
            "covariate_cols": covariate_cols,
            "retrain": retrain,
        },
    )
    logging.info("Model fitting complete.")

    check_index_combinations(  # horizon count will vary at edges
        df_model, group_cols=["Date", "GID_2", "quantile"]
    )

    assert "prediction" in df_model.columns, "Predictions not found in output."
    assert df_model["prediction"].notnull().all(), "Predictions contain null values."
