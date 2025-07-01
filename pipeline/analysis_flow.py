from pathlib import Path

from thucia.core import models
from thucia.core.cases import cases_per_month
from thucia.core.cases import read_nc
from thucia.core.cases import run_job
from thucia.core.cases import write_nc
from thucia.core.geo import add_incidence_rate
from thucia.core.geo import lookup_gid1
from thucia.core.geo import merge_sources
from thucia.core.geo import pad_admin2
from thucia.core.logging import enable_logging
from thucia.core.models import run_model
# from prefect.futures import wait
# from thucia.core.wrappers import flow


enable_logging()


# @flow(name="Dengue forecast pipeline")
def run_pipeline(iso3: str, adm1: list[str] | None = None):
    path = (Path("data") / "cases" / iso3).resolve()

    if True:
        # Process case data
        run_job(["python", str(path / "load_cases.py")])
        df = read_nc(path / "cases.nc")

        # Aggregate cases per month
        df = cases_per_month(df)
        df = pad_admin2(df)  # Ensure all Admin-2 regions included for covariate maps

        # ###### we need to add predictors for future months in order to predict cases

        # Add predictors for future months
        import pandas as pd
        import numpy as np

        last_date = df["Date"].max()
        n_months = 6
        future_dates = pd.date_range(start=last_date, periods=n_months + 1, freq="ME")[
            1:
        ]

        df_last_date = df[df["Date"] == last_date].copy()
        for date in future_dates:
            df_future = df_last_date.copy()
            df_future["Date"] = date
            df_future["Cases"] = np.nan
            df = pd.concat([df, df_future], ignore_index=True)
        df.sort_values(by=["Date", "GID_2"], inplace=True)

        write_nc(df, path / "cases_per_month.nc")

        # ##############################################################################

        df = read_nc(path / "cases_per_month.nc")  # ###

        # Get geographic, demographic and climatological data
        df_wc = merge_sources(df, ["worldclim.*"])  # slow one
        df_sp = merge_sources(df, ["edo.spi6"])
        df_on = merge_sources(df, ["noaa.oni"])
        df_wp = merge_sources(df, ["worldpop.pop_count"])

        # this approach should be more workflow-friendly
        df_sources = [df_wc, df_sp, df_on, df_wp]
        for dfs in df_sources:
            unique_cols = list(set(dfs.columns) - set(df.columns))
            for col in unique_cols:
                df[col] = dfs[col]

        # Update sources
        #  Need a function that takes a dataframe and updates the sources with most
        #  recent observations, or replaces predictions with observations

        # Add incidence rate
        df = add_incidence_rate(df)
        write_nc(df, path / "cases_with_climate.nc")

    df = read_nc(path / "cases_with_climate.nc")

    # Run models in parallel (submit tasks and wait)
    gid_1 = lookup_gid1(adm1, iso3=iso3) if adm1 else None
    # model_tasks = [
    (run_model("baseline", models.baseline, df, path, gid_1=gid_1),)
    (run_model("climate", models.climate, df, path, gid_1=gid_1),)
    (run_model("sarima", models.sarima, df, path, gid_1=gid_1),)
    (run_model("tcn", models.tcn, df, path, gid_1=gid_1, retrain=False),)
    # ]
    # Await forecasts
    # wait(model_tasks)

    # TODO: Ensembles and scoring


# Some pre-defined pipelines...


def run_peru_northwest():
    run_pipeline("PER", ["Piura", "Tumbes", "Lambayeque"])


def run_mex_oaxaca():
    run_pipeline(iso3="MEX", adm1=["Oaxaca"])


if __name__ == "__main__":
    run_peru_northwest()
    # run_mex_oaxaca()
