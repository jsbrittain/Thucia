import logging
from pathlib import Path

import numpy as np
import pandas as pd
from thucia.core import models
from thucia.core.cases import cases_per_month
from thucia.core.cases import prepare_embeddings
from thucia.core.cases import r2
from thucia.core.cases import read_nc
from thucia.core.cases import rmse
from thucia.core.cases import run_job
from thucia.core.cases import wis
from thucia.core.cases import write_nc
from thucia.core.geo import add_incidence_rate
from thucia.core.geo import lookup_gid1
from thucia.core.geo import merge_sources
from thucia.core.geo import pad_admin2
from thucia.core.logging import enable_logging
from thucia.core.models import run_model
from thucia.core.models.ensemble import create_ensemble
from thucia.core.models.utils import residual_regression
from thucia.viz import plot_ensemble_weights_over_time
# from prefect.futures import wait
# from thucia.core.wrappers import flow


enable_logging(level=logging.DEBUG)


# @flow(name="Dengue forecast pipeline")
def run_pipeline(iso3: str, adm1: list[str] | None = None):
    path = (Path("data") / "cases" / iso3).resolve()
    df = read_nc(path / "cases.nc")

    pdfm_filename = path / "embeddings.nc"
    pdfm_df = prepare_embeddings(pdfm_filename) if pdfm_filename.exists() else None

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
        n_months = 12
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

    if True:
        horizon = 1
        model = models.tcn

        from thucia.core.models.utils import (
            filter_admin1,
            interpolate_missing_dates,
            set_historical_na_to_zero,
        )
        from thucia.core.models.utils.covariates import prepare_covariates

        gid_1 = lookup_gid1(iso3=iso3, admin1_names=adm1)
        df = filter_admin1(df, gid_1)
        df = interpolate_missing_dates(df)
        df = set_historical_na_to_zero(df)
        # df, case_col, covariate_cols = prepare_covariates(df)

        # Load and merge covariates (past covariates only; tcn is PastCovariates model)
        if False:
            df["Log_Cases"] = np.log1p(df["Cases"])
            case_col = "Log_Cases"

            covariate_cols = [
                "LAG_1_LOG_CASES",
                "LAG_1_tmin_roll_2",
                "LAG_1_prec_roll_2",
            ]
            df["LAG_1_LOG_CASES"] = df.groupby("GID_2")["Log_Cases"].shift(1).fillna(0)
            df["LAG_1_tmin_roll_2"] = (
                df.groupby("GID_2")["tmin"]
                .shift(1)
                .rolling(window=2, min_periods=1)
                .mean()
                .fillna(0)
            )
            df["LAG_1_prec_roll_2"] = (
                df.groupby("GID_2")["prec"]
                .shift(1)
                .rolling(window=2, min_periods=1)
                .mean()
                .fillna(0)
            )

        if True:
            # df["Cases"] = df["Cases"].fillna(0)  # # #############################
            df["Log_Cases"] = np.log1p(df["Cases"])
            case_col = "Log_Cases"

            # tmin, lags 0-4
            df["tmax_lag_0"] = df.groupby("GID_2")["tmax"].shift(0)
            df["tmax_lag_1"] = df.groupby("GID_2")["tmax"].shift(1)
            df["tmax_lag_2"] = df.groupby("GID_2")["tmax"].shift(2)
            df["tmax_lag_3"] = df.groupby("GID_2")["tmax"].shift(3)
            df["tmax_lag_4"] = df.groupby("GID_2")["tmax"].shift(4)

            # tmax, lags 0-4
            df["tmin_lag_0"] = df.groupby("GID_2")["tmin"].shift(0)
            df["tmin_lag_1"] = df.groupby("GID_2")["tmin"].shift(1)
            df["tmin_lag_2"] = df.groupby("GID_2")["tmin"].shift(2)
            df["tmin_lag_3"] = df.groupby("GID_2")["tmin"].shift(3)
            df["tmin_lag_4"] = df.groupby("GID_2")["tmin"].shift(4)

            # precip, lags 0-6
            df["prec_lag_0"] = df.groupby("GID_2")["prec"].shift(0)
            df["prec_lag_1"] = df.groupby("GID_2")["prec"].shift(1)
            df["prec_lag_2"] = df.groupby("GID_2")["prec"].shift(2)
            df["prec_lag_3"] = df.groupby("GID_2")["prec"].shift(3)
            df["prec_lag_4"] = df.groupby("GID_2")["prec"].shift(4)
            df["prec_lag_5"] = df.groupby("GID_2")["prec"].shift(5)
            df["prec_lag_6"] = df.groupby("GID_2")["prec"].shift(6)

            # SPI-6, lags 0-6 (check availability)
            df["spi6_lag_0"] = df.groupby("GID_2")["SPI6"].shift(0)
            df["spi6_lag_1"] = df.groupby("GID_2")["SPI6"].shift(1)
            df["spi6_lag_2"] = df.groupby("GID_2")["SPI6"].shift(2)
            df["spi6_lag_3"] = df.groupby("GID_2")["SPI6"].shift(3)
            df["spi6_lag_4"] = df.groupby("GID_2")["SPI6"].shift(4)
            df["spi6_lag_5"] = df.groupby("GID_2")["SPI6"].shift(5)
            df["spi6_lag_6"] = df.groupby("GID_2")["SPI6"].shift(6)

            # ONI, lags 0-6 (check availability)
            df["oni_lag_0"] = df.groupby("GID_2")["TotalONI"].shift(0)
            df["oni_lag_1"] = df.groupby("GID_2")["TotalONI"].shift(1)
            df["oni_lag_2"] = df.groupby("GID_2")["TotalONI"].shift(2)
            df["oni_lag_3"] = df.groupby("GID_2")["TotalONI"].shift(3)
            df["oni_lag_4"] = df.groupby("GID_2")["TotalONI"].shift(4)
            df["oni_lag_5"] = df.groupby("GID_2")["TotalONI"].shift(5)
            df["oni_lag_6"] = df.groupby("GID_2")["TotalONI"].shift(6)

            # ONI 6-month avg, lags 0-1 (i.e. past 6 months and past 7-12 months)
            df["oni_6m_lag_0"] = (
                df.groupby("GID_2")["TotalONI"].shift(0).rolling(window=6).mean()
            )
            df["oni_6m_lag_1"] = (
                df.groupby("GID_2")["TotalONI"].shift(6).rolling(window=6).mean()
            )

            # ONI 12-month avg, lag 0
            df["oni_12m_lag_0"] = (
                df.groupby("GID_2")["TotalONI"].shift(0).rolling(window=12).mean()
            )

            # Population, lag 0
            df["pop_lag_0"] = df.groupby("GID_2")["pop_count"].shift(0)

            # (log) case count, lags 1-2 (only required for iterative models)
            df["log_cases_lag_1"] = df.groupby("GID_2")["Log_Cases"].shift(1)
            df["log_cases_lag_2"] = df.groupby("GID_2")["Log_Cases"].shift(2)

            covariate_cols = [
                "tmax_lag_0",  # 'tmax_lag_1', 'tmax_lag_2', 'tmax_lag_3', 'tmax_lag_4',
                "tmin_lag_0",  # 'tmin_lag_1', 'tmin_lag_2', 'tmin_lag_3', 'tmin_lag_4',
                "prec_lag_0",  # 'prec_lag_1', 'prec_lag_2', 'prec_lag_3', 'prec_lag_4', 'prec_lag_5', 'prec_lag_6',
                "spi6_lag_0",  # 'spi6_lag_1', 'spi6_lag_2', 'spi6_lag_3', 'spi6_lag_4', 'spi6_lag_5', 'spi6_lag_6',
                "oni_lag_0",  # 'oni_lag_1', 'oni_lag_2', 'oni_lag_3', 'oni_lag_4', 'oni_lag_5', 'oni_lag_6',
                "oni_6m_lag_0",  # 'oni_6m_lag_1',
                "oni_12m_lag_0",
                "pop_lag_0",
                "log_cases_lag_1",  # 'log_cases_lag_2',
            ]

        df = df[["Date", "GID_2", "future", "Cases", case_col] + covariate_cols]

        for c in covariate_cols:
            df[c] = df[c].fillna(0)
        #     df[c] = (df[c] - df[c].mean())/df[c].std()

        # Transform covariates by PCA and select top 10 components
        if False:
            keep_components = 8
            from sklearn.decomposition import PCA

            df_covs = df[["Date", "GID_2"] + covariate_cols].copy()
            logging.info("Fit PCA")
            pca = PCA(n_components=min(keep_components, len(covariate_cols)))
            logging.info("Transform covariates by PCA")
            covs_transformed = pca.fit_transform(df_covs[covariate_cols])
            df = df.drop(columns=covariate_cols)
            covariate_cols = [f"PC{i + 1}" for i in range(covs_transformed.shape[1])]
            df[covariate_cols] = covs_transformed
            df = df[["Date", "GID_2", "future", "Cases", case_col] + covariate_cols]

        name = f"{model.__name__}_h{horizon}"
        logging.info("Running model: %s", name)
        run_model(
            name,
            model,
            df=df,
            path=path,
            # Model parameters
            start_date="2015-01-01",
            # end_date=None,
            gid_1=gid_1,
            horizon=horizon,
            # case_col=case_col,
            covariate_cols=covariate_cols,
        )

    if False:
        # Ensembles and scoring
        model_names = ["baseline", "inla", "sarima", "tcn", "timesfm"]
        model_tasks = []
        for model in model_names:  # Read in model forecasts
            model_tasks.append(read_nc(str(path / f"{model}_cases_quantiles.nc")))
        df, ensemble_weights = create_ensemble(model_tasks, model_names=model_names)
        write_nc(df, path / "ensemble_cases_quantiles.nc")
        ensemble_weights.to_csv(path / "ensemble_weights.csv")

    if True:
        logging.info("Reporting statistics for all models")
        # model_list = ["baseline", "inla", "sarima", "tcn", "timesfm", "ensemble"]
        model_list = [name]
        for model in model_list:
            df_model = read_nc(str(path / f"{model}_cases_quantiles.nc"))
            r2_model = r2(
                df_model,
                "prediction",
                "Cases",
                transform=np.log1p,
                df_filter={"quantile": 0.50},
            )
            rmse_model = rmse(
                df_model,
                "prediction",
                "Cases",
                transform=np.log1p,
                df_filter={"quantile": 0.50},
            )
            rmse_model = np.expm1(rmse_model)
            wis_model = wis(
                df_model,
                "prediction",
                "Cases",
                transform=np.log1p,
                df_filter={"quantile": 0.50},
            )
            print(
                f"{model} (WIS): {wis_model:.3f}, (R^2): {r2_model:.3f}, (RMSE): {rmse_model:.3f}"
            )

    if False:
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
    run_pipeline(iso3=iso3, adm1=["São Paulo"])  # , "Minas Gerais"])


if __name__ == "__main__":
    run_peru_northwest()
    # run_mex_oaxaca()
    # run_bra_acra()
