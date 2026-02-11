import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from thucia.core import models
from thucia.core.cases import aggregate_cases
from thucia.core.cases import cases_per_month
from thucia.core.cases import check_index_combinations
from thucia.core.cases import prepare_embeddings
from thucia.core.cases import r2_score
from thucia.core.cases import read_db
from thucia.core.cases import read_nc
from thucia.core.cases import rmse_score
from thucia.core.cases import run_job
from thucia.core.cases import wis
from thucia.core.cases import write_db
from thucia.core.cases import write_nc
from thucia.core.geo import add_incidence_rate
from thucia.core.geo import lookup_gid1
from thucia.core.geo import merge_sources
from thucia.core.geo import pad_admin2
from thucia.core.logging import enable_logging
from thucia.core.models import run_model
from thucia.core.models.ensemble import create_ensemble
from thucia.core.models.utils import add_residual_quantiles
from thucia.core.models.utils import quantiles
from thucia.core.models.utils import residual_regression
from thucia.core.models.utils import sanitise_covariates
from thucia.viz import plot_ensemble_weights_over_time
# from thucia.core.cases import r2
# from thucia.core.cases import rmse
# from thucia.core.models import filter_admin1
# from thucia.core.models import interpolate_missing_dates
# from thucia.core.models import set_historical_na_to_zero


enable_logging(level=logging.DEBUG)


class Steps:
    cases_per_month = False
    plot_cases_per_month = False
    cases_per_week = False
    plot_cases_per_week = False
    cases_per_day = False
    plot_cases_per_day = False
    merge_covariates = False
    prepare_model_inputs = False
    model_fitting = False

    timesfm_quantiles = False
    pdfm_residual_regression = False
    ensemble_creation = False
    model_statistics = False
    regression_calculation = False
    regression_plot = False
    r2_rmse_wis_plots = False
    r2_distribution_plot = False
    r2_rmse_wis_lineplot = False

    report_outliers = False
    report_static_covars = False

    plot_ensemble_weights = False
    plot_model_predictions = False
    plot_model_predictions_admin2 = False
    plot_horizon_lines = False
    plot_ensemble_weights = False
    plot_state_image = False
    plot_covars_admin2 = False
    plot_pdfm_stats = False

    diagnostic_heatmap = False
    diagnostic_sarima = False
    diagnostic_r2_map = False


def run_pipeline(
    path: str,
    iso3: str,
    adm1: list[str] | None = None,
    model: str = "",
    retrain: bool = False,
    steps: Steps = Steps(),
):
    path = (Path(path)).resolve()
    model_filename = model
    cutoff_date = pd.to_datetime("2025-08-31")

    if steps.cases_per_month:  # === Initial data processing block (cases_per_month)
        tdf = read_db(path / "cases")

        # Aggregate cases per month
        if "Cases" not in tdf.columns:
            initial_case_count = len(tdf)
        else:
            initial_case_count = tdf["Cases"].sum()
        tdf = cases_per_month(tdf, cutoff_date=cutoff_date)
        tdf = pad_admin2(tdf)  # Ensure all Admin-2 regions included for covariate maps

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
        if initial_case_count != processed_case_count:
            if not cutoff_date:
                logging.warning(
                    "Case count changed from %d to %d after processing.",
                    initial_case_count,
                    processed_case_count,
                )
            else:
                logging.info(
                    "Case count changed from %d to %d after processing, probably "
                    "due to the cutoff date specified: %s.",
                    initial_case_count,
                    processed_case_count,
                    cutoff_date.strftime("%Y-%m-%d"),
                )
        else:
            logging.info("Case count verified: %d", processed_case_count)

        write_db(df, path / "cases_per_month")

    if steps.plot_cases_per_month:
        tdf = read_db(path / "cases_per_month")
        import matplotlib.pyplot as plt

        df = tdf[["Date", "Cases"]].groupby(["Date"])["Cases"].sum()
        df.index = df.index.to_timestamp(how="end")
        plt.figure(figsize=(10, 6))
        plt.plot(df)
        plt.title(f"Monthly Cases for {iso3}")
        plt.xlabel("Date")
        plt.ylabel("Cases")
        plt.show()

    if steps.cases_per_week:  # === Initial data processing block (cases_per_week)
        tdf = read_db(path / "cases")

        # Aggregate cases per month
        if "Cases" not in tdf.columns:
            initial_case_count = len(tdf)
        else:
            initial_case_count = tdf["Cases"].sum()
        # tdf = aggregate_cases(tdf, freq="W-SUN")  # W-SUN = ISO week (Mon-Sun)
        tdf = aggregate_cases(
            tdf, cutoff_date=cutoff_date, freq="W-SAT"
        )  # W-SAT = CDC epi-week (Sun-Sat)
        tdf = pad_admin2(tdf)  # Ensure all Admin-2 regions included for covariate maps

        # Add placeholders for future months
        df = tdf.df  # convert to pandas DataFrame for edits
        if True:
            last_date = tdf["Date"].max()
            n_months = 12
            future_dates = pd.period_range(
                start=last_date + 1,  # add 1 Period
                periods=n_months,
                freq=last_date.freq,
            )
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
        if initial_case_count != processed_case_count:
            if not cutoff_date:
                logging.warning(
                    "Case count changed from %d to %d after processing.",
                    initial_case_count,
                    processed_case_count,
                )
            else:
                logging.info(
                    "Case count changed from %d to %d after processing, probably "
                    "due to the cutoff date specified: %s.",
                    initial_case_count,
                    processed_case_count,
                    cutoff_date.strftime("%Y-%m-%d"),
                )

        write_db(df, path / "cases_per_week")

    if steps.plot_cases_per_week:
        tdf = read_db(path / "cases_per_week")
        import matplotlib.pyplot as plt

        df = tdf[["Date", "Cases"]].groupby(["Date"])["Cases"].sum()
        df.index = df.index.to_timestamp(how="end")
        plt.figure(figsize=(10, 6))
        plt.plot(df, "r")
        plt.title(f"Weekly Cases for {iso3}")
        plt.xlabel("Date")
        plt.ylabel("Cases")
        plt.show()

    if steps.cases_per_day:  # === Initial data processing block (cases_per_day)
        tdf = read_db(path / "cases")

        # Aggregate cases per month
        if "Cases" not in tdf.columns:
            initial_case_count = len(tdf)
        else:
            initial_case_count = tdf["Cases"].sum()
        tdf = aggregate_cases(
            tdf, cutoff_date=cutoff_date, freq="D"
        )  # Daily aggregation
        tdf = pad_admin2(tdf)  # Ensure all Admin-2 regions included for covariate maps

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
        if initial_case_count != processed_case_count:
            if not cutoff_date:
                logging.warning(
                    "Case count changed from %d to %d after processing.",
                    initial_case_count,
                    processed_case_count,
                )
            else:
                logging.info(
                    "Case count changed from %d to %d after processing, probably "
                    "due to the cutoff date specified: %s.",
                    initial_case_count,
                    processed_case_count,
                    cutoff_date.strftime("%Y-%m-%d"),
                )

        write_db(df, path / "cases_per_day")

    if steps.plot_cases_per_day:
        tdf = read_db(path / "cases_per_day")
        import matplotlib.pyplot as plt

        df = tdf[["Date", "Cases"]].groupby(["Date"])["Cases"].sum()
        df.index = df.index.to_timestamp(how="end")
        plt.figure(figsize=(10, 6))
        plt.plot(df)
        plt.title(f"Daily Cases for {iso3}")
        plt.xlabel("Date")
        plt.ylabel("Cases")
        plt.show()

    if steps.merge_covariates:  # === Merge covariates block (cases_with_climate)
        # Read in cases per month
        tdf = read_db(path / "cases_per_month")
        df = tdf.df  # convert to pandas DataFrame

        # df = df[df['Date'] <= pd.Period("2020-06", freq="M")]

        # Get geographic, demographic and climatological data
        df_wc = merge_sources(df, ["worldclim.*"])  # slow one
        write_db(df_wc, path / "cases_with_climate_worldclim")  # store intermediates
        df_sp = merge_sources(df, ["edo.spi6"])
        write_db(df_sp, path / "cases_with_climate_spi")
        df_on = merge_sources(df, ["noaa.oni"])
        write_db(df_on, path / "cases_with_climate_oni")
        df_wp = merge_sources(df, ["worldpop.pop_count"])
        write_db(df_wp, path / "cases_with_climate_worldpop")

        # This approach is more workflow-friendly
        df_sources = [df_wc, df_sp, df_on, df_wp]
        for dfs in df_sources:
            unique_cols = list(set(dfs.columns) - set(df.columns))
            for col in unique_cols:
                df = df.merge(
                    dfs[["GID_2", "Date", col]], on=["GID_2", "Date"], how="left"
                )

        # Compute Incidence Rate
        df = add_incidence_rate(df)
        write_db(df, path / "cases_with_climate")

    if steps.model_fitting:  # === Prepare model inputs
        # Model parameters
        horizons = [1, 6, 12]  # <-- select horizons here
        model_admin_level = 2  # <-- geo region to fit and predict model
        start_date = pd.Period("2010-01", freq="M")  # <-- select start date here

        try:
            model = getattr(models, model_filename)
        except AttributeError:
            raise ValueError(f"Model '{model}' not found in thucia.core.models")

        multivariate = False
        test_reduced_size = False

        retrain = 12 if retrain else False  # retrain every 12th step (if enabled)

        # Training date is specified before preparing model inputs to ensure training
        # data is sanitised (which can alter the dataframe; we cannot currently write to
        # a Thucia database in-place)
        train_end_date = pd.Period("2019-12", freq="M")  # <-- select train end date

        # Read cases and covariate merged data
        tdf = read_db(path / "cases_with_climate")

        # Filter to selected admin-1 regions and clean data
        gid_1 = lookup_gid1(iso3=iso3, admin1_names=adm1)
        df = tdf.df
        # df = filter_admin1(df, gid_1)

        df = df[df["GID_2"] == "BRA.25.565_2"]
        # df = df[(df['GID_2'] == 'BRA.25.565_2') | (df['GID_2'] == 'BRA.25.564_2')]
        # df = df[df['GID_1'] == 'BRA.25_1']

        # df['Date'] = df['Date'].dt.to_timestamp(how='end').dt.normalize().astype('<M8[ns]')
        # breakpoint()

        # df = interpolate_missing_dates(df)
        # df = set_historical_na_to_zero(df)

        # Load and merge covariates (treat all as past covariates)
        case_col = "Log_Cases"
        df["Log_Cases"] = np.log1p(df["Cases"])
        df["tmax_lag_0"] = df.groupby("GID_2", observed=False)["tmax"].shift(0)
        df["tmax_lag_1"] = df.groupby("GID_2", observed=False)["tmax"].shift(1)
        df["tmax_lag_2"] = df.groupby("GID_2", observed=False)["tmax"].shift(2)
        df["tmin_lag_0"] = df.groupby("GID_2", observed=False)["tmin"].shift(0)
        df["tmin_lag_1"] = df.groupby("GID_2", observed=False)["tmin"].shift(1)
        df["tmin_lag_2"] = df.groupby("GID_2", observed=False)["tmin"].shift(2)
        df["prec_lag_0"] = df.groupby("GID_2", observed=False)["prec"].shift(0)
        df["prec_lag_1"] = df.groupby("GID_2", observed=False)["prec"].shift(1)
        df["prec_lag_2"] = df.groupby("GID_2", observed=False)["prec"].shift(2)
        df["spi6_lag_0"] = df.groupby("GID_2", observed=False)["SPI6"].shift(0)
        df["oni_lag_0"] = df.groupby("GID_2", observed=False)["TotalONI"].shift(0)
        df["oni_6m_lag_0"] = (
            df.groupby("GID_2", observed=False)["TotalONI"]
            .shift(0)
            .rolling(window=6)
            .mean()
        )
        df["oni_12m_lag_0"] = (
            df.groupby("GID_2", observed=False)["TotalONI"]
            .shift(0)
            .rolling(window=12)
            .mean()
        )

        df["pop_lag_0"] = (
            df.groupby("GID_2", observed=False)["pop_count"].shift(0)
            # .rolling(window=3)
            # .mean()
        )
        df["log_cases_lag_1"] = df.groupby("GID_2", observed=False)["Log_Cases"].shift(
            1
        )
        df["log_cases_12m_ma"] = (
            df.groupby("GID_2", observed=False)["Log_Cases"]
            .shift(1)
            .rolling(window=12)
            .mean()
        )
        df["log_cases_24m_ma"] = (
            df.groupby("GID_2", observed=False)["Log_Cases"]
            .shift(1)
            .rolling(window=24)
            .mean()
        )
        df["log_cases_48m_ma"] = (
            df.groupby("GID_2", observed=False)["Log_Cases"]
            .shift(1)
            .rolling(window=48)
            .mean()
        )
        df["log_cases_12m_cum"] = (
            df.groupby("GID_2", observed=False)["Log_Cases"]
            .shift(1)
            .rolling(window=12)
            .sum()
        )
        df["log_cases_cum"] = (
            df.groupby("GID_2", observed=False)["Log_Cases"].shift(1).cumsum()
        )
        df["log_cases_slope"] = df["log_cases_lag_1"].diff()
        df["log_cases_slope_6m"] = df["log_cases_lag_1"].diff().rolling(window=6).mean()
        df["log_cases_slope_12m"] = (
            df["log_cases_lag_1"].diff().rolling(window=12).mean()
        )
        df["log_cases_12m_trend"] = (
            df.groupby("GID_2", observed=False)["Log_Cases"]
            .shift(1)
            .rolling(window=12)
            .apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True)
        )
        df["sin_month"] = np.sin(2 * np.pi * df["Date"].dt.month / 12)
        df["cos_month"] = np.cos(2 * np.pi * df["Date"].dt.month / 12)

        covariate_cols = [
            "tmax_lag_0",  # tmax
            # "tmax_lag_1",
            # "tmax_lag_2",
            "tmin_lag_0",  # tmin
            # "tmin_lag_1",
            # "tmin_lag_2",
            "prec_lag_0",  # prec
            # "prec_lag_1",
            # "prec_lag_2",
            "spi6_lag_0",  # spi6
            "oni_lag_0",  # oni
            "oni_6m_lag_0",
            "oni_12m_lag_0",
            "pop_lag_0",
            "log_cases_lag_1",  # previous cases
            # 'log_cases_12m_ma',  # previous cases 12m moving average
            # 'log_cases_24m_ma',
            # 'log_cases_48m_ma',
            # 'log_cases_12m_cum',  # previous cases 12m cumulative
            # 'log_cases_cum',
            # 'log_cases_slope',
            # 'log_cases_slope_6m',
            # 'log_cases_slope_12m',
            #  'log_cases_12m_trend',  # previous cases 36m trend
            # 'sin_month',
            # 'cos_month',
        ]
        df = df[
            ["Date", "GID_1", "GID_2", "future", "Cases", case_col] + covariate_cols
        ]

        # Sanitise covariates (NaN replacement, seasonal mean, forward fill, back fill)
        # df = sanitise_covariates(df, covariate_cols, None)  # "2019-01-01")  # train_end_date)
        # df = sanitise_covariates(df, covariate_cols, "2019-01-01")
        df = sanitise_covariates(df, covariate_cols, train_end_date)
        # df = df.ffill().bfill()

        write_db(df, path / "model_input_data")

        # return

        #
        # ===
        #
        tdf = read_db(path / "model_input_data")

        # Sanity check - missing values and missing index combinations (Date, GID_2)
        if tdf[covariate_cols].isnull().any().any():
            missing = tdf[tdf[covariate_cols + [case_col]].isnull().any(axis=1)]
            raise ValueError(f"Missing covariate values in data:\n{missing}")
        if tdf[~tdf["future"]][case_col].isnull().any():
            missing = tdf[~tdf["future"]][tdf[case_col].isnull()]
            raise ValueError(f"Missing case values in data:\n{missing}")
        check_index_combinations(tdf, ["Date", "GID_2"])

        # Model name and path
        name = f"{model.__name__}"
        db_file = path / f"{name}_cases_quantiles.duckdb"

        # Model parameters
        model_kwargs = {
            "start_date": start_date,
            "gid_1": gid_1,
            "horizons": horizons,
            "case_col": case_col,
            "covariate_cols": covariate_cols,
            "model_admin_level": model_admin_level,  # 0=country, 1=state, 2=municipality
            "db_file": db_file,  # override target database file
        }
        if model in [
            models.tcn,
            models.tft,
            models.nbeats,
            models.nhits,
            models.xgboost,
            models.chronos,
        ]:
            model_kwargs.update(
                {
                    "train_end_date": train_end_date,
                    "retrain": retrain,
                    "multivariate": multivariate,
                }
            )
            if model != models.chronos:
                model_kwargs.update({"num_samples": 200})

        if test_reduced_size:
            # ### TEMPORARY TESTING FILTER ###
            df = tdf.df
            df = df[df["GID_1"] == "BRA.11_1"]
            # df = df[
            #     (df["GID_1"] == "BRA.3_1")  # 16 municipalities
            #     | (df["GID_1"] == "BRA.23_1")  # 15 municipalities
            # ]
            write_db(df, path / "model_input_data_test")
            tdf = read_db(path / "model_input_data_test")
            del df
            logging.warning(f"Limiting to provinces: {tdf['GID_2'].unique().to_list()}")

        # Run model
        logging.info("Running model: %s", name)
        run_model(
            name,
            model,
            df=tdf.df,  # Pass in either df or tdf
            path=path,  # TODO: assign quantile/samples files in run_model()
            # Model parameters
            model_kwargs=model_kwargs,
        )
        logging.info("Model fitting complete.")

    if (
        steps.timesfm_quantiles
    ):  # === Convert TimesFM deterministic output to quantiles by residual sampling
        filename = model_filename
        df_model = read_db(path / f"{filename}_cases_quantiles").df
        if "quantile" in df_model.columns:
            # Reduce to median if distribution provided
            df_model = df_model[df_model["quantile"] == 0.5]
        df_model = add_residual_quantiles(
            df_model,
            window=None,
            min_history=2,
            quantile_levels=quantiles,
            db_file=path / f"{filename}_cases_quantiles_q.duckdb",
        )

    if steps.pdfm_residual_regression:  # === Residual regression with PDFM embeddings
        filename = model_filename
        method = "pinball"  # 'ridge', 'mlp', 'pinball'
        sliding_window = 24  # None for expanding window

        df_model = read_db(path / f"{filename}_cases_quantiles").df

        pdfm_filename = path / "embeddings.nc"
        if not pdfm_filename.exists():
            raise FileNotFoundError(f"PDFM embeddings file not found: {pdfm_filename}")
        pdfm_df = prepare_embeddings(pdfm_filename)

        # Match provinces with embeddings
        provinces = df_model["GID_2"].unique().tolist()
        pdfm_df = pdfm_df[pdfm_df["GID_2"].isin(provinces)]
        pdfm_df = pdfm_df[~pdfm_df["GID_2"].duplicated()]
        df_model = df_model[df_model["GID_2"].isin(pdfm_df["GID_2"])]

        mode = "standard"
        if mode == "standard":
            logging.info(f"Residual regression with all PDFM features ({mode})")
            df_model_pdfm = residual_regression(
                df_model, pdfm_df, method=method, window=sliding_window
            )
            write_db(
                df_model_pdfm, path / f"{filename}_pdfmrr_{method}_cases_quantiles"
            )
        elif mode == "noise":
            df_model = df_model[df_model["horizon"] == 6]  # <-- select horizon here
            np.random.seed(42)
            for rep in range(100):
                # Replace features with noise
                pdfm_df.loc[:, pdfm_df.columns.str.contains("feature")] = (
                    np.random.normal(
                        0,
                        1,
                        size=pdfm_df.loc[
                            :, pdfm_df.columns.str.contains("feature")
                        ].shape,
                    )
                )

                logging.info(
                    f"Residual regression with all PDFM features ({mode} rep {rep})"
                )
                df_model_pdfm = residual_regression(df_model, pdfm_df, method=method)
                write_db(
                    df_model_pdfm,
                    path / f"{filename}_pdfmrr_{mode}{rep}_cases_quantiles",
                )
        else:
            raise ValueError(f"Unknown PDFM residual regression mode: {mode}")

        if False:
            # Aggregated Search Trends
            logging.info(
                "Residual regression with PDFM features: Aggregated Search Trends"
            )
            cols = ["GID_2", *[f"feature{k}" for k in range(128)]]
            df_model_pdfm = residual_regression(df_model, pdfm_df[cols], method="ridge")
            write_db(df_model_pdfm, path / f"{filename}_pdfmrr1_cases_quantiles")

            # Maps and Busyness
            logging.info("Residual regression with PDFM features: Maps and Busyness")
            cols = ["GID_2", *[f"feature{128 + k}" for k in range(128)]]
            df_model_pdfm = residual_regression(df_model, pdfm_df[cols], method="ridge")
            write_db(df_model_pdfm, path / f"{filename}_pdfmrr2_cases_quantiles")

            # Weather & Air Quality
            logging.info(
                "Residual regression with PDFM features: Weather & Air Quality"
            )
            cols = ["GID_2", *[f"feature{256 + k}" for k in range(74)]]
            df_model_pdfm = residual_regression(df_model, pdfm_df[cols], method="ridge")
            write_db(df_model_pdfm, path / f"{filename}_pdfmrr3_cases_quantiles")

    if steps.diagnostic_heatmap:  # === Diagnostic heatmap (dates/regions)
        # TimesFM outliers:
        #  BRA.25.24_2, BRA.25.383_2

        # Image plot GID vs Date of predictions to find outliers
        filename = model_filename
        horizon = 1
        df = read_db(path / f"{filename}_cases_quantiles.duckdb")
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 6))
        df_pivot = df[(df["quantile"] == 0.5) & (df["horizon"] == horizon)].pivot(
            index="GID_2", columns="Date", values="prediction"
        )
        sns.heatmap(df_pivot, cmap="viridis")
        plt.title(f"Heatmap of Predictions (Median Quantile, Horizon={horizon})")
        plt.xlabel("Date")
        plt.ylabel("GID_2")
        # Show all GIDs
        plt.yticks(
            ticks=np.arange(len(df_pivot.index)), labels=df_pivot.index, rotation=0
        )
        plt.show()

    if steps.ensemble_creation:  # === Create ensemble of models
        # Ensembles and scoring
        model_names = [
            "sarima_h12_pdfmrr",
            "tcn_h12_pdfmrr",
            "tft_h12_pdfmrr",
            "xgboost_h12_pdfmrr",
            "nbeats_h12_pdfmrr",
            "timesfm2_h12_pdfmrr",
        ]
        horizon = 12

        for h in range(1, horizon + 1):
            logging.info(f"Creating ensemble for horizon: {h}")
            model_tasks = []
            for model in model_names:  # Read in model forecasts
                logging.info(f"Reading model: {model}")
                mdf = read_nc(str(path / f"{model}_cases_quantiles.nc"))
                mdf = mdf[mdf["horizon"] == h]
                model_tasks.append(mdf)
                del mdf
            logging.info("Creating ensemble")
            df, ensemble_weights = create_ensemble(model_tasks, model_names=model_names)
            logging.info("Writing ensemble output")
            write_nc(df, path / f"ensemble2_h{h}_pdfm_cases_quantiles.nc")
            ensemble_weights.to_csv(path / f"ensemble2_h{h}_pdfm_weights.csv")

        # Collate predictions into one dataframe
        df_all = []
        for h in range(1, horizon + 1):
            df = read_nc(str(path / f"ensemble2_h{h}_pdfm_cases_quantiles.nc"))
            df["horizon"] = h
            df_all.append(df)
        df_all = pd.concat(df_all, ignore_index=True)
        write_nc(df_all, path / "ensemble2_pdfm_cases_quantiles.nc")

    if steps.model_statistics:  # === Reporting statistics for all models
        logging.info("Reporting statistics for all models")

        # Report R2 statistic
        save_them = True
        model_list = [
            "sarima_h12",
            "sarima_h12_pdfmrr",
            "tcn_h12",
            "tcn_h12_pdfmrr",
            # "tft_h12",
            # "tft_h12_pdfmrr",
            "xgboost_h12",
            "xgboost_h12_pdfmrr",
            "nbeats_h12",
            "nbeats_h12_pdfmrr",
            "timesfm_h12",
            "timesfm_h12_pdfmrr",
            # "ensemble2_h12",
            # "ensemble2_h12_pdfmrr",
        ]
        horizons = [1, 3, 6, 12]

        # threshold = 1e6

        # out = []
        # out_all = []
        for model in model_list:
            df_model0 = read_db(str(path / f"{model}_cases_quantiles"))
            wis_models = []

            for h in horizons:
                logging.info(f"Calculating statistics for model: {model}, horizon: {h}")
                # Filter once
                df_model = df_model0[df_model0["horizon"] == h].copy()  # index into tdf
                # if False:  # threshold > 0:
                #     gid2_outliers = (
                #         df_model[
                #             df_model["prediction"] > threshold
                #         ]["GID_2"].unique()
                #     )
                #     if len(gid2_outliers) > 0:
                #         logging.warning(
                #             f"Model {model} - Horizon {h} - Predictions > {threshold} for GIDs: {gid2_outliers}"
                #         )
                #         for gid in gid2_outliers:
                #             df_model["prediction"] = df_model.apply(
                #                 lambda row: np.nan
                #                 if row["GID_2"] == gid
                #                 else row["prediction"],
                #                 axis=1,
                #             )
                df_model["prediction"] = np.log1p(df_model["prediction"])
                df_model["Cases"] = np.log1p(df_model["Cases"])
                # Compute metrics
                wis_model = wis(
                    df_model,
                    "prediction",
                    "Cases",
                    transform=None,
                    df_filter=None,
                )
                wis_model["horizon"] = h
                wis_models.append(wis_model)
                # breakpoint()
                # # Median-only metrics
                # df_model = df_model[df_model["quantile"] == 0.5]
                # # Compute metrics
                # r2_flat = r2(
                #     df_model,
                #     "prediction",
                #     "Cases",
                #     group_col=None,
                #     transform=None,
                #     df_filter=None,
                # ).astype(float)
                # r2_model = r2(
                #     df_model,
                #     "prediction",
                #     "Cases",
                #     group_col="GID_2",
                #     transform=None,
                #     df_filter=None,
                # ).astype(float)
                # rmse_flat = rmse(
                #     df_model,
                #     "prediction",
                #     "Cases",
                #     group_col=None,
                #     transform=None,
                #     df_filter=None,
                # ).astype(float)
                # rmse_model = rmse(
                #     df_model,
                #     "prediction",
                #     "Cases",
                #     group_col="GID_2",
                #     transform=None,
                #     df_filter=None,
                # ).astype(float)
                # out.append(
                #     {
                #         "model": model,
                #         "horizon": h,
                #         "r2_flat": r2_flat,
                #         "r2_mean": r2_model.mean(),
                #         "r2_std": r2_model.std(),
                #         "r2_q25": r2_model.quantile(0.25),
                #         "r2_q50": r2_model.quantile(0.50),
                #         "r2_q75": r2_model.quantile(0.75),
                #         "rmse_flat": np.expm1(rmse_flat),
                #         "rmse_mean": np.expm1(rmse_model).mean(),
                #         "rmse_std": np.expm1(rmse_model).std(),
                #         "rmse_q25": np.expm1(rmse_model).quantile(0.25),
                #         "rmse_q50": np.expm1(rmse_model).quantile(0.50),
                #         "rmse_q75": np.expm1(rmse_model).quantile(0.75),
                #         "wis_mean": wis_model['WIS'].mean(),
                #         "wis_std": wis_model['WIS'].std(),
                #         "wis_q25": wis_model['WIS'].quantile(0.25),
                #         "wis_q50": wis_model['WIS'].quantile(0.50),
                #         "wis_q75": wis_model['WIS'].quantile(0.75),
                #     }
                # )
                # out_all.append(
                #     {
                #         "model": model,
                #         "horizon": h,
                #         "r2": r2_model.values,
                #         "rmse_flat": np.expm1(rmse_model).values,
                #         "wis": wis_model.values,
                #     }
                # )

            wis_model = pd.concat(wis_models, ignore_index=True)
            wis_model.to_csv(path / f"{model}_wis.csv", index=False)

        #     print(pd.DataFrame(out))
        # df = pd.DataFrame(out)
        # df_all = pd.DataFrame(out_all)
        # print(df)
        # if save_them:
        #     df.to_csv(path / "model_performance.csv", index=False)
        #     df_all.to_csv(path / "model_performance_all.csv", index=False)
        # else:
        #     breakpoint()

    if steps.plot_ensemble_weights:  # === Plot ensemble weights over time
        # Plot ensemble weights
        w = pd.read_csv("data/cases/BRA/ensembles2/ensemble2_h12_weights.csv")
        w.rename(columns={"timesfm2_h12": "timesfm_h12"}, inplace=True)

        w = w.loc[:, ~w.columns.str.contains("^Unnamed")]
        w["Date"] = pd.to_datetime(w["Date"], format="%Y-%m-%d")
        plot_ensemble_weights_over_time(
            w,
            date_col="Date",
            title=f"Ensemble Weights Over Time for {iso3}",
            ma_window=1,
        )

    if steps.report_outliers:
        # Report outlier predictions
        filename = model_filename

        df_model = read_db(path / f"{filename}_cases_quantiles").df

        threshold = 100e6
        for h in range(1, 13):
            nan_gids = (
                df_model[(df_model["horizon"] == h) & (df_model["prediction"].isna())][
                    "GID_2"
                ]
                .unique()
                .tolist()
            )
            print(f"Horizon {h} - NaN predictions for GIDs: {nan_gids}")

            th_gids = (
                df_model[
                    (df_model["horizon"] == h) & (df_model["prediction"] > threshold)
                ]["GID_2"]
                .unique()
                .tolist()
            )
            print(f"Horizon {h} - Predictions > {threshold} for GIDs: {th_gids}")

    if steps.report_static_covars:
        # Based on current model input data
        df_model = read_db(path / "model_input_data").df

        reject = ["Date", "GID_1", "GID_2", "future", "Cases"]
        covariate_cols = set(df_model.columns.tolist()) - set(reject)
        geo_regions = df_model["GID_2"].unique().tolist()
        static_covars = []
        static_geo = []
        for gid in geo_regions:
            df = df_model[df_model["GID_2"] == gid]
            mask = df["Date"] < "2020-01"
            for c in covariate_cols:
                unique_vals = df[mask][c].nunique()
                if unique_vals <= 1:
                    print(f"  --> Static covariate detected: {gid} {c}")
                    static_covars.append(c)
                    static_geo.append(gid)
        static_covars = set(static_covars)
        print(f"Static covariates detected: {static_covars}")
        static_geo = set(static_geo)
        print(f"Geo regions with static covariates: {static_geo}")

    if (
        steps.plot_model_predictions
    ):  # === Plot model predictions against observed cases
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # transform = lambda x: x
        transform = np.log1p

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(12, 6))

        def plot_model_predictions(models_to_plot, horizon, transform=None):
            def _transform(x):
                return x

            transform = transform if transform is not None else _transform

            # Get case count from cases_with_climate
            filestem = models_to_plot[list(models_to_plot.keys())[0]]
            df_model = read_db(str(path / "cases_with_climate")).df
            print(df_model)
            df_model = df_model[df_model["GID_2"] == "BRA.25.565_2"]
            if isinstance(df_model["Date"].dtype, pd.PeriodDtype):
                # Plot at month end (matches end-of-month timestamp at, e.g. 2019-01-31)
                df_model["Date"] = (
                    df_model["Date"].dt.to_timestamp() + pd.offsets.MonthEnd()
                )
            df_plot = df_model
            df_plot = (
                df_plot.groupby("Date")
                .agg({"Cases": "sum"})
                .apply(transform)
                .reset_index()
            )
            plt.plot(
                df_plot["Date"],
                df_plot["Cases"],
                color="black",
                linewidth=2,
                label="Observed",
            )

            # # Get case count from first model
            # filestem = models_to_plot[list(models_to_plot.keys())[0]]
            # df_model = read_db(str(path / f"{filestem}_cases_quantiles"))
            # print(df_model)
            # df_model = df_model[df_model['GID_2'] == 'BRA.25.565_2']
            # if isinstance(df_model['Date'].dtype, pd.PeriodDtype):
            #     # Plot at month end (matches end-of-month timestamp at, e.g. 2019-01-31)
            #     df_model['Date'] = df_model['Date'].dt.to_timestamp() + pd.offsets.MonthEnd()
            # df_plot = df_model[df_model["quantile"] == 0.5]
            # if "horizon" in df_plot:
            #     df_plot = df_plot[df_plot["horizon"] == horizon]
            # df_plot = (
            #     df_plot.groupby("Date")
            #     .agg({"Cases": "sum", "prediction": "sum"})
            #     .apply(transform)
            #     .reset_index()
            # )
            # plt.fill_between(
            #     df_plot["Date"],
            #     0,
            #     df_plot["Cases"],
            #     color="grey",
            #     alpha=0.5,
            #     label="Observed",
            # )
            # plt.plot(
            #     df_plot["Date"],
            #     transform(df_plot["Cases"]),
            #     color="black",
            #     linewidth=2,
            #     label="Observed",
            # )

            for model, filestem in models_to_plot.items():
                print(model)
                print(filestem)
                try:
                    df_model = read_db(str(path / f"{filestem}_cases_quantiles")).df
                    df_model = df_model[df_model["GID_2"] == "BRA.25.565_2"]
                    if isinstance(df_model["Date"].dtype, pd.PeriodDtype):
                        # Plot at month end (matches end-of-month timestamp at, e.g. 2019-01-31)
                        df_model["Date"] = (
                            df_model["Date"].dt.to_timestamp() + pd.offsets.MonthEnd()
                        )
                except FileNotFoundError:
                    print(f"File not found for model {filestem}, skipping...")
                    plt.plot(0, 0, label=model)
                    continue
                df_plot = df_model[
                    (df_model["quantile"] == 0.5) & (df_model["horizon"] == horizon)
                ]
                df_plot_q05 = df_model[
                    (df_model["quantile"] == 0.05) & (df_model["horizon"] == horizon)
                ]
                df_plot_q95 = df_model[
                    (df_model["quantile"] == 0.95) & (df_model["horizon"] == horizon)
                ]
                # Take average over all provinces
                plt.fill_between(
                    df_plot_q05.groupby("Date")["prediction"].sum().index,
                    transform(df_plot_q05.groupby("Date")["prediction"].sum().values),
                    transform(df_plot_q95.groupby("Date")["prediction"].sum().values),
                    alpha=0.3,
                    label=f"IQR {model}",
                )
                df_plot = (
                    df_plot.groupby("Date")
                    .agg({"Cases": "sum", "prediction": "sum"})
                    .apply(transform)
                    .reset_index()
                )
                plt.plot(df_plot["Date"], df_plot["prediction"], label=model)
                # Calculate WIS
                wis_model = wis(
                    df_model[
                        (df_model["horizon"] == horizon)
                        & (df_model["Date"] >= "2024-01-01")
                        & (df_model["Cases"].notna())
                    ],
                    "prediction",
                    "Cases",
                    transform=transform,
                    df_filter=None,
                )["WIS"].mean()  # avg over provinces
                plt.title(f"{horizon}-months ahead, WIS={wis_model}")
            # plt.ylim(0, 5e5)

        if False:
            # Plot models against each other
            models_to_plot = {
                "SARIMA": "sarima_h12",
                "TCN": "tcn_h12",
                "TFT": "tft_h12",
                "XGBoost": "xgboost_h12",
                "N-BEATS": "nbeats_h12",
                "TimesFM": "timesfm_h12",
                "Ensemble": "ensemble_h12",
            }
        else:
            # Plot models with/without PDFM residual regression
            filestem = "tcn"
            models_to_plot = {
                # "timesfm": "timesfm_h12",
                # "timesfm_pdfm": "timesfm_h12_pdfmrr",
                # 'tcn': 'tcn_orig_h12',
                model_filename: model_filename,
                # 'nbeats (orig)': 'nbeats_orig_h12',
                # 'nhits_pdfm_ridge': 'nhits_pdfmrr_ridge',
                # 'nhits_pdfm_pinball': 'nhits_pdfmrr_pinball',
            }

        plt.subplot(3, 1, 1)
        plot_model_predictions(models_to_plot, horizon=1, transform=transform)
        ax = plt.gca()
        ax.tick_params(axis="x", which="both", labelbottom=False)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        # plt.xlim([pd.to_datetime("2016-01-01"), pd.to_datetime("2026-06-01")])
        # plt.title("1-month ahead")

        plt.subplot(3, 1, 2)
        plot_model_predictions(models_to_plot, horizon=6, transform=transform)
        ax = plt.gca()
        ax.tick_params(axis="x", which="both", labelbottom=False)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        # plt.title("6-months ahead")
        # plt.xlim([pd.to_datetime("2016-01-01"), pd.to_datetime("2026-06-01")])
        plt.legend()
        plt.ylabel("Total cases")

        plt.subplot(3, 1, 3)
        plot_model_predictions(models_to_plot, horizon=12, transform=transform)
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        # plt.xlim([pd.to_datetime("2016-01-01"), pd.to_datetime("2026-06-01")])
        # plt.title("12-months ahead")

        plt.show()

    if steps.plot_model_predictions_admin2:
        admin2s = [
            "BRA.13.483_2"
        ]  # , 'BRA.13.679_2', 'BRA.21.152_2', 'BRA.21.86_2', 'BRA.25.24_2', 'BRA.25.383_2', 'BRA.6.112_2', 'BRA.9.176_2', 'BRA.9.72_2']
        # admin2s = ["BRA.13.679_2"]

        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if model_filename:
            models_to_plot = {
                model_filename.split("_")[0].upper(): model_filename,
            }
        else:
            raise ValueError("Please specify a model filename to plot.")

        def plot_model_predictions(models_to_plot, horizon, admin2):
            # Get case count from first model
            filestem = models_to_plot[list(models_to_plot.keys())[0]]
            df_model = read_db(str(path / f"{filestem}_cases_quantiles")).df
            df_model = df_model[df_model["GID_2"] == admin2]
            df_model["Date"] = df_model["Date"].dt.to_timestamp(how="end")
            df_plot = df_model[df_model["quantile"] == 0.5]
            # df_plot['Cases'] = np.log1p(df_plot['Cases'])
            if "horizon" in df_plot:
                df_plot = df_plot[df_plot["horizon"] == horizon]
            df_plot = (
                df_plot.groupby("Date")
                .agg({"Cases": "sum", "prediction": "sum"})
                .reset_index()
            )
            plt.fill_between(
                df_plot["Date"],
                0,
                df_plot["Cases"],
                color="grey",
                alpha=0.5,
                label="Observed",
            )
            plt.plot(
                df_plot["Date"],
                df_plot["Cases"],
                color="black",
                linewidth=2,
                label="Observed",
            )

            plt.sca(plt.gca().twinx())
            for model, filestem in models_to_plot.items():
                try:
                    df_model = read_db(str(path / f"{filestem}_cases_quantiles")).df
                    df_model = df_model[df_model["GID_2"] == admin2]
                    df_model["Date"] = df_model["Date"].dt.to_timestamp(how="end")
                except FileNotFoundError:
                    print(f"File not found for model {filestem}, skipping...")
                    plt.plot(0, 0, label=model)
                    continue
                df_plot = df_model[
                    (df_model["quantile"] == 0.5) & (df_model["horizon"] == horizon)
                ]
                # df_plot['prediction'] = np.log1p(df_plot['prediction'])
                # Take average over all provinces
                df_plot = (
                    df_plot.groupby("Date")
                    .agg({"Cases": "sum", "prediction": "sum"})
                    .reset_index()
                )
                plt.plot(df_plot["Date"], df_plot["prediction"], label=model)

        for admin2 in admin2s:
            sns.set_theme(style="whitegrid")
            plt.figure(figsize=(12, 6))

            xlims = [pd.to_datetime("2020-01-01"), pd.to_datetime("2026-06-01")]
            # xlims = [pd.to_datetime("2020-06-01"), pd.to_datetime("2021-12-01")]

            plt.subplot(3, 1, 1)
            plot_model_predictions(models_to_plot, horizon=1, admin2=admin2)
            ax = plt.gca()
            ax.tick_params(axis="x", which="both", labelbottom=False)
            ax.xaxis.set_major_locator(mdates.YearLocator())
            plt.xlim(xlims)
            plt.title(f"1-month ahead (Admin2: {admin2}")

            plt.subplot(3, 1, 2)
            plot_model_predictions(models_to_plot, horizon=6, admin2=admin2)
            ax = plt.gca()
            ax.tick_params(axis="x", which="both", labelbottom=False)
            ax.xaxis.set_major_locator(mdates.YearLocator())
            plt.title("6-months ahead")
            plt.xlim(xlims)
            plt.legend()
            plt.ylabel("Total cases")

            plt.subplot(3, 1, 3)
            plot_model_predictions(models_to_plot, horizon=12, admin2=admin2)
            ax = plt.gca()
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            plt.xlim(xlims)
            plt.title("12-months ahead")

            plt.show()

    if steps.plot_covars_admin2:
        import matplotlib.pyplot as plt

        tdf = read_db(path / "model_input_data")
        # tdf = read_db(path / "cases_with_climate")
        df = tdf.df

        # df = df[df['GID_2'] == 'BRA.13.483_2']  # Example admin2
        # df = df[df['GID_2'] == 'BRA.13.4_2']  # Example admin2
        df = df[df["GID_2"] == "BRA.13.679_2"]  # Example admin2

        plt.figure(figsize=(12, 6))
        exclude = ["Date", "GID_1", "GID_2", "future", "Cases"]
        covars = set(df.columns.tolist()) - set(exclude)
        # covars = ["pop_lag_0", "spi6_lag_0"]
        for ix, c in enumerate(covars):
            ax = plt.subplot(len(covars), 1, ix + 1)
            # if df[c].dtype in [np.float32, np.float64, np.int32, np.int64]:
            # ax.plot(df["Date"].dt.to_timestamp(how="end"), df[c], label=c)
            ax.plot(df[c], label=c)
            ax.set_ylabel(c, rotation=45)
        plt.title(f"Covariates for Admin2: {df['GID_2'].iloc[0]}")
        plt.show()

    if steps.plot_horizon_lines:  # === Plot horizon lines for one model
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Predicted vs Observed
        filestem = "tft_h12"
        plt.figure(figsize=(6, 6))
        df_model = read_nc(str(path / f"{filestem}_cases_quantiles.nc"))
        for horizon in range(1, 13):
            df_plot = df_model[
                (df_model["quantile"] == 0.5) & (df_model["horizon"] == horizon)
            ]
            df_plot = (
                df_plot.groupby("Date")
                .agg({"Cases": "mean", "prediction": "mean"})
                .reset_index()
            )
            plt.plot(df_plot["Date"], df_plot["prediction"], label=f"h={horizon}")
        plt.xlabel("Date")
        plt.ylabel("Predicted Cases")
        plt.grid()
        plt.legend()
        plt.show()

    if steps.regression_calculation:  # Regression calculation for plots
        save_them = False
        model_list = [
            "sarima_h12",
            "sarima_h12_pdfmrr",
            "tcn_h12",
            "tcn_h12_pdfmrr",
            # "tft_h12",
            # "tft_h12_pdfmrr",
            "xgboost_h12",
            "xgboost_h12_pdfmrr",
            "nbeats_h12",
            "nbeats_h12_pdfmrr",
            "timesfm_h12",
            "timesfm_h12_pdfmrr",
            # "ensemble_h12",
            # "ensemble_h12_pdfmrr",
        ]
        horizon = 12

        out = []

        def transform(x):
            return np.log1p(x)

        for model in model_list:
            df_model0 = read_db(str(path / f"{model}_cases_quantiles"))
            for h in range(1, horizon + 1):
                logging.info(f"Calculating statistics for model: {model}, horizon: {h}")
                df_plot = df_model0[
                    (df_model0["quantile"] == 0.5) & (df_model0["horizon"] == h)
                ].copy()
                df_plot = (  # Average over provinces
                    df_plot.groupby("Date")
                    .agg({"Cases": "mean", "prediction": "mean"})
                    .reset_index()
                )
                if transform:
                    df_plot["Cases"] = transform(df_plot["Cases"])
                    df_plot["prediction"] = transform(df_plot["prediction"])
                r2_model = r2_score(df_plot["Cases"], df_plot["prediction"])
                rmse_model = np.expm1(
                    rmse_score(df_plot["Cases"], df_plot["prediction"])
                )
                out.append(
                    pd.DataFrame(
                        {
                            "model": [model],
                            "horizon": [h],
                            "r2": [r2_model],
                            "rmse": [rmse_model],
                        }
                    )
                )
        out_df = pd.concat(out, ignore_index=True)
        if save_them:
            out_df.to_csv(path / "model_performance_regression.csv", index=False)
        else:
            breakpoint()
        print(out_df)

    if steps.regression_plot:  # Regression plot of predicted vs observed (base vs pdfm)
        filestem0 = "ensemble2"

        import seaborn as sns
        import matplotlib.pyplot as plt

        models_to_plot = {
            filestem0: f"{filestem0}_h12",
            f"{filestem0}_pdfm": f"{filestem0}_h12_pdfmrr",
        }

        models_to_plot = {
            "timesfm": "timesfm2_h12",
            "timesfm_pdfm": "timesfm2_h12_pdfmrr",
        }
        models_to_plot = {
            "ensemble": "ensemble2_h12",
            "ensemble_pdfm": "ensemble2_h12_pdfmrr",
        }

        metrics = pd.read_csv(path / "model_performance_all.csv")
        metrics["r2"] = metrics["r2"].apply(
            lambda s: np.fromstring(s.strip("[]"), sep=" ")
        )
        metrics = metrics[metrics["model"].isin(models_to_plot.values())]

        def transform(x):
            return np.log1p(x)

        plt.figure(figsize=(6, 6))
        for ix, horizon in enumerate([1, 6, 12]):
            # Predicted vs Observed
            plt.subplot(3, 2, 2 * ix + 1)
            r2_model = {}
            for model, filestem in models_to_plot.items():
                df_model = read_nc(str(path / f"{filestem}_cases_quantiles.nc"))
                # provinces = df_model['GID_2'].unique().tolist()[:5]
                # df_model = df_model[df_model["GID_2"].isin(provinces)]
                df_plot = df_model[
                    (df_model["quantile"] == 0.5) & (df_model["horizon"] == horizon)
                ]
                df_plot = (  # Average over provinces
                    df_plot.groupby("Date")
                    .agg({"Cases": "mean", "prediction": "mean"})
                    .reset_index()
                )
                if transform:
                    df_plot["Cases"] = transform(df_plot["Cases"])
                    df_plot["prediction"] = transform(df_plot["prediction"])
                r2_model[model] = r2_score(df_plot["Cases"], df_plot["prediction"])
                # Each sample is the obs/pred for one Date, avg over provinces
                plt.scatter(
                    df_plot["Cases"],
                    df_plot["prediction"],
                    alpha=0.5,
                    label=f"{model} (R²={r2_model[model]:.3f})",
                )
            print(r2_model)
            max_val = max(plt.xlim()[1], plt.ylim()[1])
            if max_val:
                plt.plot([0, max_val], [0, max_val], color="black", linestyle="--")
                plt.xlim(0, max_val)
                plt.ylim(0, max_val)
            plt.xlabel("Observed Cases (log)")
            plt.ylabel("Predicted Cases (log)")
            # plt.title("Average over Provinces")
            plt.axis("equal")
            plt.axis("square")
            plt.grid()
            plt.legend()

            # R2 distribution
            plt.subplot(3, 2, 2 * ix + 2)
            m_df = pd.DataFrame(
                {
                    "model": np.repeat(
                        metrics["model"][metrics["horizon"] == horizon].values,
                        metrics["r2"][metrics["horizon"] == horizon].apply(len).values,
                    ),
                    "r2": np.concatenate(
                        metrics["r2"][metrics["horizon"] == horizon].values
                    ),
                }
            )
            m_df.loc[m_df["model"] == "timesfm2_h12", "model"] = "timesfm_h12"
            m_df.loc[m_df["model"] == "timesfm2_h12_pdfmrr", "model"] = (
                "timesfm_h12_pdfm"
            )
            m_df.loc[m_df["model"] == "ensemble2_h12", "model"] = "ensemble_h12"
            m_df.loc[m_df["model"] == "ensemble2_h12_pdfmrr", "model"] = (
                "ensemble_h12_pdfm"
            )
            sns.histplot(
                data=m_df,
                x="r2",
                hue="model",
                element="step",
                stat="density",
                common_norm=False,
            )
            plt.xlim(0, 1)
            plt.xlabel("R² by Province")
            plt.tight_layout()

        plt.show()

    if steps.r2_distribution_plot:  # === Plot R2 distribution boxplot
        # R2 distribution plot
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        sns.set_theme(style="whitegrid")
        metrics = pd.read_csv(path / "model_performance.csv")
        plt.figure(figsize=(8, 6))
        plt.subplot(1, 1, 1)
        metric = "r2_q50"
        sns.boxplot(data=metrics, x="horizon", y=metric, hue="model")
        plt.ylim(0.0, 1.0)
        plt.title("R² by Forecast Horizon")
        plt.xlabel("Horizon (months)")
        plt.ylabel("R²")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid()
        plt.tight_layout()
        plt.show()

    if steps.r2_rmse_wis_lineplot:  # === Plot R2, RMSE, WIS line plots
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        sns.set_theme(style="whitegrid")
        metrics = pd.read_csv(path / "model_performance.csv")

        metrics.loc[metrics["model"] == "timesfm2_h12", "model"] = "timesfm_h12"
        metrics.loc[metrics["model"] == "timesfm2_h12_pdfmrr", "model"] = (
            "timesfm_h12_pdfmrr"
        )
        metrics.loc[metrics["model"] == "ensemble2_h12", "model"] = "ensemble_h12"
        metrics.loc[metrics["model"] == "ensemble2_h12_pdfmrr", "model"] = (
            "ensemble_h12_pdfmrr"
        )

        plt.figure(figsize=(6, 10))

        plt.subplot(3, 1, 1)
        metric = "r2_q50"
        sns.lineplot(data=metrics, x="horizon", y=metric, hue="model", marker="o")
        plt.ylim(0.0, 1.0)
        plt.title("R² by Forecast Horizon")
        plt.xlabel("Horizon (months)")
        plt.ylabel("R²")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid()

        plt.subplot(3, 1, 2)
        metric = "rmse_q50"
        sns.lineplot(data=metrics, x="horizon", y=metric, hue="model", marker="o")
        plt.ylim(0, metrics[metric].max() * 1.1)
        plt.title("RMSE by Forecast Horizon")
        plt.xlabel("Horizon (months)")
        plt.ylabel("RMSE (cases)")
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # legend off
        plt.legend().remove()
        plt.grid()

        plt.subplot(3, 1, 3)
        metric = "wis_q50"
        sns.lineplot(data=metrics, x="horizon", y=metric, hue="model", marker="o")
        plt.ylim(0, metrics[metric].max() * 1.1)
        plt.title("WIS by Forecast Horizon")
        plt.xlabel("Horizon (months)")
        plt.ylabel("WIS")
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.legend().remove()
        plt.grid()
        plt.tight_layout()
        plt.show()

        # Similar style, but taking the difference of pdfmrr vs base
        model_pairs = [
            ("sarima_h12", "sarima_h12_pdfmrr", "SARIMA"),
            ("tcn_h12", "tcn_h12_pdfmrr", "TCN"),
            ("tft_h12", "tft_h12_pdfmrr", "TFT"),
            ("xgboost_h12", "xgboost_h12_pdfmrr", "XGBoost"),
            ("nbeats_h12", "nbeats_h12_pdfmrr", "N-BEATS"),
            ("timesfm_h12", "timesfm_h12_pdfmrr", "TimesFM"),
            ("ensemble_h12", "ensemble_h12_pdfmrr", "Ensemble"),
        ]
        plt.figure(figsize=(6, 10))
        plt.subplot(3, 1, 1)
        maxdiff = 0
        for base, pdfm, label in model_pairs:
            df_base = metrics[metrics["model"] == base]
            df_pdfm = metrics[metrics["model"] == pdfm]
            if df_base.empty or df_pdfm.empty:
                continue
            metric = "r2_q50"
            maxdiff = max(
                maxdiff, abs(df_pdfm[metric].values - df_base[metric].values).max()
            )
            plt.plot(
                df_pdfm["horizon"],
                df_pdfm[metric].values - df_base[metric].values,
                marker="o",
                label=label,
            )
        plt.ylim(-maxdiff * 1.1, maxdiff * 1.1)
        plt.plot([1, 12], [0, 0], color="black", linestyle="--")
        plt.title("ΔR² with - without PDFM")
        plt.xlabel("Horizon (months)")
        plt.ylabel("ΔR²")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid()
        plt.subplot(3, 1, 2)
        maxdiff = 0
        for base, pdfm, label in model_pairs:
            df_base = metrics[metrics["model"] == base]
            df_pdfm = metrics[metrics["model"] == pdfm]
            if df_base.empty or df_pdfm.empty:
                continue
            metric = "rmse_q50"
            maxdiff = max(
                maxdiff, abs(df_pdfm[metric].values - df_base[metric].values).max()
            )
            plt.plot(
                df_base["horizon"],
                df_pdfm[metric].values - df_base[metric].values,
                marker="o",
                label=label,
            )
        plt.ylim(-maxdiff * 1.1, maxdiff * 1.1)
        plt.plot([1, 12], [0, 0], color="black", linestyle="--")
        plt.title("ΔRMSE with - without PDFM")
        plt.xlabel("Horizon (months)")
        plt.ylabel("ΔRMSE (cases)")
        plt.legend().remove()
        plt.grid()
        plt.subplot(3, 1, 3)
        maxdiff = 0
        for base, pdfm, label in model_pairs:
            df_base = metrics[metrics["model"] == base]
            df_pdfm = metrics[metrics["model"] == pdfm]
            if df_base.empty or df_pdfm.empty:
                continue
            metric = "wis_q50"
            maxdiff = max(
                maxdiff, abs(df_pdfm[metric].values - df_base[metric].values).max()
            )
            plt.plot(
                df_base["horizon"],
                df_pdfm[metric].values - df_base[metric].values,
                marker="o",
                label=label,
            )
        plt.ylim(-maxdiff * 1.1, maxdiff * 1.1)
        plt.plot([1, 12], [0, 0], color="black", linestyle="--")
        plt.title("ΔWIS with - without PDFM")
        plt.xlabel("Horizon (months)")
        plt.ylabel("ΔWIS")
        plt.legend().remove()
        plt.grid()
        plt.tight_layout()
        plt.show()

    if steps.diagnostic_sarima:  # === SARIMA diagnostic image plot over regions
        # SARIMA outliers:
        #  GID-165 / BRA.25.24_2 (early-mid 2021)

        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        horizon = 12

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(12, 6))
        filestem = "sarima_h12"
        df_model = read_nc(str(path / f"{filestem}_cases_quantiles.nc"))
        df_plot = df_model[
            (df_model["quantile"] == 0.5) & (df_model["horizon"] == horizon)
        ]
        df_plot["prediction"] = np.log1p(df_plot["prediction"])
        df_plot = (
            df_plot.groupby(["Date", "GID_2"])
            .agg({"Cases": "mean", "prediction": "mean"})
            .reset_index()
        )
        # Pivot for heatmap
        df_pivot = df_plot.pivot(index="GID_2", columns="Date", values="prediction")
        plt.imshow(df_pivot, aspect="auto", cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Predicted Cases")
        plt.xticks(
            ticks=np.arange(0, len(df_pivot.columns), 6),
            labels=[date.strftime("%Y-%m") for date in df_pivot.columns[::6]],
            rotation=45,
        )
        plt.title(f"SARIMA Predictions Over Time for {iso3}")
        plt.xlabel("Date")
        plt.ylabel("GID_2")
        plt.tight_layout()
        plt.show()

    if steps.diagnostic_r2_map:  # === R2 map of difference between base and pdfm models
        # Map difference of R2 by region, base vs pdfm
        import matplotlib.pyplot as plt
        from thucia.viz.maps import choropleth, hex_cartogram

        model = "timesfm2_h12"

        df = pd.read_csv(path / "model_performance_all.csv")

        # WARNING: Assumes the order of GID_2 is consistent between performance data
        # (unlabelled), and the model data (both pdfm and base). Should add labels to
        # performance data and merge on column.
        def prepare_map_df(df, model, horizon, threshold=False):
            df_base = df[(df["model"] == model) & (df["horizon"] == horizon)]
            df_pdfm = df[
                (df["model"] == f"{model}_pdfmrr") & (df["horizon"] == horizon)
            ]

            # evaluate string to array
            base_r2 = (
                df_base["r2"]
                .apply(lambda s: np.fromstring(s.strip("[]"), sep=" "))
                .iloc[0]
            )
            pdfm_r2 = (
                df_pdfm["r2"]
                .apply(lambda s: np.fromstring(s.strip("[]"), sep=" "))
                .iloc[0]
            )
            diff_r2 = pdfm_r2 - base_r2

            df_model = read_nc(str(path / f"{model}_cases_quantiles.nc"))
            df = pd.DataFrame(
                {
                    "GID_2": df_model["GID_2"].unique(),
                    "r2": diff_r2,
                }
            )
            if threshold:
                df["r2"] = (df["r2"] >= 0).astype(float) * 2 - 1  # +ve/-ve
            df["GID_1"] = df["GID_2"].apply(lambda s: s.rsplit(".", 1)[0]) + "_1"
            return df

        fig, axs = plt.subplots(3, 2, figsize=(12, 6))

        # Horizon 1
        df_h1 = prepare_map_df(df, model, horizon=1)
        choropleth(
            df_h1,
            ax=axs[0][0],
            admin_level=2,
            value_col="r2",
            cmap="viridis",
            aggregation="sum",
        )
        hex_cartogram(
            df_h1,
            ax=axs[0][1],
            admin_level=2,
            value_col="r2",
            side_length_km=20.0,
            cmap="viridis",
            aggregation="sum",
        )

        # Horizon 6
        df_h6 = prepare_map_df(df, model, horizon=6)
        choropleth(
            df_h6,
            ax=axs[1][0],
            admin_level=2,
            value_col="r2",
            cmap="viridis",
            aggregation="sum",
        )
        hex_cartogram(
            df_h6,
            ax=axs[1][1],
            admin_level=2,
            value_col="r2",
            side_length_km=20.0,
            cmap="viridis",
            aggregation="sum",
        )

        # Horizon 12
        df_h12 = prepare_map_df(df, model, horizon=12)
        choropleth(
            df_h12,
            ax=axs[2][0],
            admin_level=2,
            value_col="r2",
            cmap="viridis",
            aggregation="sum",
        )
        hex_cartogram(
            df_h12,
            ax=axs[2][1],
            admin_level=2,
            value_col="r2",
            side_length_km=20.0,
            cmap="viridis",
            aggregation="sum",
        )

        plt.show()

    if steps.plot_state_image:
        tdf = read_db(path / "cases_per_month")
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = tdf.df

        # Aggregate by State
        df = df.groupby(["GID_2", "Date"]).agg({"Cases": "sum"}).reset_index()
        df["no_cases"] = df["Cases"] == 0

        plt.figure(figsize=(10, 6))
        sns.heatmap(
            df.pivot(index="GID_2", columns="Date", values="no_cases"),
            cmap="viridis",
        )
        plt.show()

    if steps.plot_pdfm_stats:
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from thucia.viz.maps import choropleth, hex_cartogram

        model_name = "tcn_h12"

        start_date = pd.Period("2020-01")

        fig, axs = plt.subplots(4, 3, figsize=(12, 6))

        def read_wis(filename):
            df = pd.read_csv(filename)
            df["Date"] = pd.PeriodIndex(df["Date"], freq="M")
            df = df[df["Date"] >= start_date]
            df = df[df["Cases"].notna()]
            return df

        df_model = read_wis(path / f"{model_name}_wis.csv")
        df_pdfm = read_wis(path / f"{model_name}_pdfmrr_wis.csv")

        # Merge WIS based on GID_2, Date and horizon
        df_diff = pd.merge(
            df_pdfm,
            df_model,
            on=["GID_2", "Date", "horizon"],
            suffixes=("_pdfm", "_base"),
        )
        df_diff.rename(columns={"Cases_pdfm": "Cases"}, inplace=True)
        df_diff.drop(columns=["Cases_base"], inplace=True)
        df_diff["Cases"] = np.expm1(df_diff["Cases"])  # LogCases -> Cases
        df_diff["Cases"] = df_diff["Cases"] / 1e4  # Cases -> /10k
        df_diff["WIS_diff"] = df_diff["WIS_pdfm"] - df_diff["WIS_base"]
        gid_parts = df_diff["GID_2"].str.split(".", n=2)
        df_diff["GID_1"] = (
            gid_parts.str[0]
            + "."
            + gid_parts.str[1]
            + gid_parts.str[2].str.split("_").str[1]
        )

        # Plot cases count
        choropleth(
            df_diff[df_diff["horizon"] == 1],
            ax=axs[0][0],
            admin_level=2,
            value_col="Cases",
            cmap="viridis",
            aggregation="sum",
            edgecolor=None,
            linewidth=0,
        )
        for spine in axs[0][0].spines.values():
            spine.set_visible(False)
        axs[0][0].set_xticks([])
        axs[0][0].set_yticks([])
        axs[0][0].set_ylabel("Total Cases (x10k)")
        axs[0][1].axis("off")
        axs[0][2].axis("off")

        for col, horizon in enumerate([1, 6, 12]):
            df_h = df_diff[df_diff["horizon"] == horizon]

            # WIS plot
            choropleth(
                df_h,
                ax=axs[col + 1][0],
                admin_level=2,
                value_col="WIS_pdfm",
                cmap="viridis",
                aggregation="mean",
                edgecolor=None,
                linewidth=0,
            )
            for spine in axs[col + 1][0].spines.values():
                spine.set_visible(False)
            axs[col + 1][0].set_xticks([])
            axs[col + 1][0].set_yticks([])
            axs[col + 1][0].set_ylabel(f"WIS w/PDFM (h={horizon})")

            # WIS difference plot
            choropleth(
                df_h,
                ax=axs[col + 1][1],
                admin_level=2,
                value_col="WIS_diff",
                cmap="RdYlGn",
                symmetric_cmap=True,
                aggregation="mean",
                edgecolor=None,
                linewidth=0,
            )
            for spine in axs[col + 1][1].spines.values():
                spine.set_visible(False)
            axs[col + 1][1].set_xticks([])
            axs[col + 1][1].set_yticks([])
            axs[col + 1][1].set_ylabel("ΔWIS (PDFM - Base)")

            # Scatter plot of WIS
            df_gid = (
                df_h.groupby("GID_2")
                .agg({"WIS_base": "mean", "WIS_pdfm": "mean"})
                .reset_index()
            )
            p = axs[col + 1][2]
            p.scatter(
                np.log1p(df_gid["WIS_base"]),
                np.log1p(df_gid["WIS_pdfm"]),
                alpha=0.5,
                label="WIS",
            )
            max_val = max(p.get_xlim()[1], p.get_ylim()[1])
            if max_val:
                p.plot([0, max_val], [0, max_val], color="black", linestyle="--")
                p.set_xlim(0, max_val)
                p.set_ylim(0, max_val)
            p.set_xlabel("$log(WIS_{Base})$")
            if col == 2:
                p.set_ylabel("$log(WIS_{PDFM})$")
            p.set_aspect("equal")
            p.set_box_aspect(1)

            # hex_cartogram(
            #     df_diff,
            #     ax=axs[row][1],
            #     admin_level=2,
            #     value_col="WIS_delta",
            #     side_length_km=20.0,
            #     cmap="viridis",
            #     aggregation="mean",
            # )

        plt.show()


def run_bra_acra(steps):
    iso3 = "BRA"
    # Custom loading script for Brazil's case data
    if Path("data/cases/BRA/cases.duckdb").exists():
        print("Brazil case data already exists, skipping download.")
    else:
        path = (Path("data") / "cases" / iso3).resolve()
        run_job(
            [
                "python",
                str(path / "load_cases.py"),
                "--states",
                "São Paulo",
                "--ey_start",
                "2020",
                "--ew_start",
                "1",
            ],
            cwd=path,
        )
    # Run the pipeline
    run_pipeline(iso3=iso3, adm1=["São Paulo"])
    # BRA.7_1 = 1 district = Distrito Federal
    # BRA.23_1 = 15 districts = Roraima
    # BRA.25_1 = 645 districts = São Paulo


def run_mex(steps):
    iso3 = "MEX"
    run_pipeline(iso3=iso3, adm1=None, steps=steps)


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    # Path
    parser.add_argument(
        "--path",
        type=str,
        help="Path to data directory",
    )
    # Country
    parser.add_argument(
        "--iso",
        type=str,
        default="BRA",
        help="ISO3 country code (default: BRA)",
    )
    # ADM1 regions (optional)
    parser.add_argument(
        "--adm1",
        type=str,
        default=None,
        help="ADM1 regions to process, comma-separated (default: all)",
    )
    # Parameters
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to run (optional)",
    )
    # Steps
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        help="Pipeline steps to run, comma-separated (default: all)",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Whether to retrain models",
    )
    args = parser.parse_args()

    # Define steps
    steps = Steps()
    if args.steps != "all":
        steps_list = args.steps.split(",")
        for step in steps_list:
            if hasattr(steps, step):
                setattr(steps, step, True)
            else:
                raise ValueError(f"Invalid step: {step}")

    # Run the appropriate pipeline
    run_pipeline(
        path=args.path,
        iso3=args.iso,
        adm1=args.adm1.split(",") if args.adm1 else None,
        model=args.model,
        retrain=args.retrain,
        steps=steps,
    )
