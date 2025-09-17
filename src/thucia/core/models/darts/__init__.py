import logging
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from darts import TimeSeries


class BaseSamples:
    def __init__(
        self,
        df: pd.DataFrame,
        case_col: str = "Cases",
        date_col="Date",
        geo_col="GID_2",
        covariate_cols: Optional[List[str]] = None,
        horizon=1,
        num_samples=1000,
        start_date=None,
    ):
        self.df = df
        self.case_col = case_col
        self.date_col = date_col
        self.geo_col = geo_col
        self.covariate_cols = covariate_cols or []
        self.horizon = horizon
        self.num_samples = num_samples
        self.start_date = start_date
        self.freqstr = "ME"  # df[date_col].dt.freqstr

        # Parameters and functionality provided by subclasses
        self.sampling_method = None
        self.model = self.build_model()

    def build_model(self):
        raise NotImplementedError

    def flatten_bt_to_quantiles(self, bt, gids, date_col="Date"):
        """
        Flatten bt[x][y][z] -> pd.DataFrame
        Adds columns: [date_col, "gid", "origin_idx", "horizon", "quantile", "prediction"]
        """
        quantiles = [
            0.01,
            0.025,
            0.05,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
            0.975,
            0.99,
        ]
        frames = []
        for x, gid in enumerate(gids):
            logging.info(
                f"Converting samples to quantiles, region {gid} ({x + 1} / {len(gids)})"
            )
            for y, origin_list in enumerate(bt[x]):  # forecast origins
                for z, ts in enumerate(origin_list):  # horizon steps
                    df = ts.quantiles_df(quantiles).reset_index()
                    # melt quantile columns to long format
                    df = df.melt(
                        id_vars=[date_col],
                        var_name="quantile",
                        value_name="prediction",
                    )
                    # extract numeric quantile from col name (e.g. Log_Cases_0.1)
                    # nb index-0 is first capture group
                    df["quantile"] = (
                        df["quantile"]
                        .str[::-1]
                        .str.extract(r"^(.+?)\_")[0]
                        .str[::-1]
                        .astype(float)
                    )
                    df[self.geo_col] = gid
                    df["origin_date"] = ts.start_time() - (z + 1) * ts.freq
                    df["horizon"] = z + 1  # 1-based horizon
                    frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def get_cases_covs(self, df):
        logging.info("Preparing time series for each region.")
        cases = []
        past_covs = []
        all_gids = df[self.geo_col].unique()
        for gid, gid_df in df.groupby(self.geo_col):
            # target series
            ts = TimeSeries.from_dataframe(
                gid_df,
                time_col=self.date_col,
                value_cols=[self.case_col],
                fill_missing_dates=True,
                freq=self.freqstr,
            ).astype(np.float32)

            # static covariate: 1xK one-hot with stable column order
            one_hot = pd.get_dummies(
                pd.Categorical([gid], categories=all_gids)  # fixes width/order
            )
            ts = ts.with_static_covariates(one_hot)

            # past covariates for this gid
            pc = TimeSeries.from_dataframe(
                gid_df,
                time_col=self.date_col,
                value_cols=self.covariate_cols,
                fill_missing_dates=True,
                freq=self.freqstr,
            ).astype(np.float32)

            cases.append(ts)
            past_covs.append(pc)

        return cases, past_covs

    def historical_predictions(
        self,
        retrain: bool = True,  # only turn off for faster testing
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        logging.info("Starting historical predictions.")

        # Past data only for training
        df = self.df[~self.df["future"]].copy()
        df.fillna(0, inplace=True)
        df = df.sort_values(by=["GID_2", "Date"])

        # Determine start date
        start_date = pd.Timestamp(start_date) if start_date else self.start_date
        if not start_date:
            start_date = df[self.date_col].min()
        end_date = pd.Timestamp(end_date) if end_date else df[self.date_col].max()

        all_gids = df[self.geo_col].unique()

        cases = []
        past_covs = []

        # Prepare cases and covariates
        cases, past_covs = self.get_cases_covs(df)

        logging.info("Initial model fit.")
        self.model.fit(
            series=cases,
            past_covariates=past_covs if self.covariate_cols else None,
            epochs=150,  # quick train
            verbose=False,
        )

        logging.info("Generating historical forecasts.")
        bt = self.model.historical_forecasts(
            series=cases,
            past_covariates=past_covs,
            start=start_date,
            forecast_horizon=self.horizon,
            stride=1,
            retrain=retrain,
            last_points_only=False,
            verbose=False,
            num_samples=self.num_samples,
        )

        if not isinstance(bt, list):
            bt = [bt]

        # bt[x][y][z].to_dataframe() = horizon z, time y, region x
        #  provides a DataFrame with cols Log_Cases_s<n> for n in num_samples

        df_preds = self.flatten_bt_to_quantiles(bt, all_gids, date_col=self.date_col)
        return df_preds

    def forecast_using_past_covariances(self, horizon: int, train=True) -> pd.DataFrame:
        if train or not self.model._fit_called:
            df = self.df[~self.df["future"]].copy()
            df.fillna(0, inplace=True)
            cases, past_covs = self.get_cases_covs(df)
            self.model.fit(
                series=cases,
                past_covariates=past_covs if self.covariate_cols else None,
                epochs=150,  # quick train
                verbose=False,
            )

        # Full data including future dates
        df = self.df.copy()
        # Covariates cannot be NA
        df.fillna(value={c: 0 for c in self.covariate_cols}, inplace=True)

        # Ensure dataframe is sorted for correct indexing
        df = df.sort_values(by=["GID_2", "Date"])
        all_gids = df[self.geo_col].unique().tolist()
        all_dates = df[self.date_col].unique().tolist()
        forecast_dates = df[df["future"]][self.date_col].unique().tolist()
        origin_date = max(set(all_dates) - set(forecast_dates))

        # Loop over forecast dates ('future' = True) and make iterative predictions
        df_preds = []
        for horizon_index, date in enumerate(forecast_dates):
            logging.info(f"Forecasting for date {date}")
            horizon = horizon_index + 1

            # Prepare cases and covariates
            cases, past_covs = self.get_cases_covs(df)

            # First index where date_col == date
            ix = all_dates.index(date)

            # Use historical_forecasts so that covariates are considered
            # and samples are returned
            bt = self.model.historical_forecasts(
                series=[c[: ix + 1] for c in cases],
                past_covariates=[p[: ix + 1] for p in past_covs],
                start=date,
                forecast_horizon=1,
                stride=1,
                retrain=False,
                last_points_only=False,
                verbose=False,
                num_samples=self.num_samples,
            )

            if not isinstance(bt, list):
                bt = [bt]

            preds = self.flatten_bt_to_quantiles(bt, all_gids, date_col=self.date_col)
            preds["origin_date"] = origin_date  # reset origin date
            preds["horizon"] = horizon  # adjust horizon
            df_preds.append(preds)  # store predictions

            # Then insert prediction into cases column for next iterative update
            for gid in all_gids:
                ypred = preds[
                    (preds[self.geo_col] == gid) & (preds["quantile"] == 0.5)
                ]["prediction"].iloc[0]
                df.loc[
                    (df[self.date_col] == date) & (df[self.geo_col] == gid),
                    self.case_col,
                ] = ypred

        # Can reset future Cases to NA here, but df is a local copy so discard
        # df.loc[df['future'], self.case_col] = np.nan

        return pd.concat(df_preds, ignore_index=True)
