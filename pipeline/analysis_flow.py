from pathlib import Path

import numpy as np
import pandas as pd
from thucia.core import models
from thucia.core.cases import cases_per_month
from thucia.core.cases import r2
from thucia.core.cases import read_nc
from thucia.core.cases import run_job
from thucia.core.cases import write_nc
from thucia.core.geo import add_incidence_rate
from thucia.core.geo import lookup_gid1
from thucia.core.geo import merge_sources
from thucia.core.geo import pad_admin2
from thucia.core.logging import enable_logging
from thucia.core.models import run_model
from thucia.core.models.ensemble import create_ensemble
from thucia.viz import plot_ensemble_weights_over_time
# from prefect.futures import wait
# from thucia.core.wrappers import flow


enable_logging()


# @flow(name="Dengue forecast pipeline")
def run_pipeline(iso3: str, adm1: list[str] | None = None):
    path = (Path("data") / "cases" / iso3).resolve()
    df = read_nc(path / "cases.nc")

    if False:
        # Aggregate cases per month
        if "Cases" not in df.columns:
            df = cases_per_month(
                df
            )  # ### Make this more dynamic - check for Cases and aggregate based on month anyway ###
        df = pad_admin2(df)  # Ensure all Admin-2 regions included for covariate maps

        # ###### we need to add predictors for future months in order to predict cases

        # Add predictors for future months

        last_date = df["Date"].max()
        n_months = 6
        future_dates = pd.date_range(start=last_date, periods=n_months + 1, freq="ME")[
            1:
        ]

        df["future"] = False
        df_last_date = df[df["Date"] == last_date].copy()
        for date in future_dates:
            df_future = df_last_date.copy()
            df_future["Date"] = date
            df_future["Cases"] = np.nan
            df_future["future"] = True
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

        # This approach is more workflow-friendly
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

    if False:
        # Run models in parallel (submit tasks and wait)
        gid_1 = lookup_gid1(adm1, iso3=iso3) if adm1 else None
        run_model("baseline", models.baseline, df, path, gid_1=gid_1)
        run_model("inla", models.inla, df, path, gid_1=gid_1)
        run_model("sarima", models.sarima, df, path, gid_1=gid_1)
        run_model(
            "tcn", models.tcn, df, path, gid_1=gid_1, retrain=False
        )  # example model param
        run_model("timesfm", models.timesfm, df, path, gid_1=gid_1)

    if True:
        # Ensembles and scoring
        model_names = ["baseline", "inla", "sarima", "tcn", "timesfm"]
        model_tasks = []
        for model in model_names:  # Read in model forecasts
            model_tasks.append(read_nc(str(path / f"{model}_cases_quantiles.nc")))
        df, ensemble_weights = create_ensemble(model_tasks, model_names=model_names)
        write_nc(df, path / "ensemble_cases_quantiles.nc")
        ensemble_weights.to_csv(path / "ensemble_weights.csv")

        # Report R2 statistic
        model_list = ["baseline", "inla", "sarima", "tcn", "timesfm", "ensemble"]
        for model in model_list:
            df_model = read_nc(str(path / f"{model}_cases_quantiles.nc"))
            r2_model = r2(
                df_model,
                "prediction",
                "Cases",
                transform=np.log1p,
                df_filter={"quantile": 0.50},
            )
            print(f"{model} (R^2): {r2_model:.3f}")

        # Plot ensemble weights
        w = pd.read_csv("data/cases/PER/ensemble_weights.csv")
        w = w.loc[:, ~w.columns.str.contains("^Unnamed")]
        w["Date"] = pd.to_datetime(w["Date"], format="%Y-%m-%d")
        plot_ensemble_weights_over_time(
            w,
            date_col="Date",
            title=f"Ensemble Weights Over Time for {iso3}",
            ma_window=7,
        )


# Some pre-defined pipelines...


def run_peru_northwest():
    iso3 = "PER"
    # Custom loading script for Peru's case data
    # path = (Path("data") / "cases" / iso3).resolve()
    # run_job(["python", str(path / "load_cases.py")])
    # Run the pipeline
    run_pipeline(iso3, ["Piura", "Tumbes", "Lambayeque"])


def run_mex_oaxaca():
    iso3 = "MEX"
    # Custom loading script for Mexico's case data
    path = (Path("data") / "cases" / iso3).resolve()
    run_job(["python", str(path / "load_cases.py")])
    # Run the pipeline
    run_pipeline(iso3=iso3, adm1=["Oaxaca"])


def run_bra_acra():
    iso3 = "BRA"
    # Custom loading script for Brazil's case data
    # path = (Path("data") / "cases" / iso3).resolve()
    # run_job(
    #     [
    #         "python",
    #         str(path / "load_cases.py"),
    #         "--states",
    #         "Acre",
    #         "Amazonas",
    #         "--ey_start",
    #         "2020",
    #         "--ew_start",
    #         "1",
    #     ],
    #     cwd=path,
    # )
    # Run the pipeline
    run_pipeline(iso3=iso3, adm1=["Acre", "Amazonas"])


if __name__ == "__main__":
    run_peru_northwest()
    # run_mex_oaxaca()
    # run_bra_acra()
