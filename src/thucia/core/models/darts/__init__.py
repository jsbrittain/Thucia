import logging
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from thucia.core.models.utils import sample_to_quantiles_vec

torch.set_float32_matmul_precision(
    "medium"
)  # medium=bfloat, high=tfloat, highest=float32


class DartsBase:
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

        self.quantiles = [
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

        # Parameters and functionality provided by subclasses
        self.sampling_method = None
        self.model = self.build_model()

    def build_model(self):
        raise NotImplementedError

    def get_cases(self, future=None, target_gids: Optional[List[str]] = None):
        if future is None:
            # Use all data
            df = self.df
        elif future:
            df = self.df[self.df["future"]]
        else:
            df = self.df[~self.df["future"]]

        if target_gids is None:
            target_gids = df["GID_2"].unique()

        target_list = []
        covar_list = []
        for gid in target_gids:
            gdf = df[df["GID_2"] == gid]

            ts = TimeSeries.from_dataframe(
                gdf,
                time_col="Date",
                value_cols=["Log_Cases"],
                fill_missing_dates=True,
                freq="ME",
            ).astype(np.float32)

            cov = TimeSeries.from_dataframe(
                gdf,
                time_col="Date",
                value_cols=self.covariate_cols,
                fill_missing_dates=True,
                freq="ME",
            ).astype(np.float32)

            target_list.append(ts)
            covar_list.append(cov)

        return target_list, covar_list, target_gids

    def historical_predictions(
        self,
        retrain: bool = True,  # only turn off for faster testing
        start_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        df = self.df

        # Model pre-fit
        target_gids = df["GID_2"].unique()
        logging.info("Pre-fit (where needed)")
        self.pre_fit(target_gids=target_gids)

        # Now include future data for forecasting, ensuring the same GID mapping
        target_list, covar_list, _ = self.get_cases(target_gids=target_gids)

        rows = []
        for ts, cov, gid in zip(target_list, covar_list, target_gids):
            logging.info(f"Forecasting for GID_2 {gid}")
            tic = pd.Timestamp.now()
            bt = self.historical_forecasts(
                ts, cov, start_date=start_date, retrain=retrain
            )
            # bt = [time]series[horizon][1][samples]

            # process horizons separately
            for h in range(self.horizon):
                vals = np.concat([TimeSeries.all_values(t)[h, :, :] for t in bt])
                dates = np.array([t.time_index[h] for t in bt])

                out = pd.DataFrame(vals)
                out["Date"] = dates
                out = out.melt(
                    id_vars="Date",
                    var_name="sample",
                    value_name="prediction",
                )
                out["sample"] = out["sample"].astype(int)
                out["GID_2"] = gid

                if len(out) > 1:
                    # samples to quantiles
                    out = (
                        pd.concat(
                            {
                                k: sample_to_quantiles_vec(
                                    np.expm1(g["prediction"]).clip(
                                        lower=0
                                    ),  # transform before quantiles
                                    self.quantiles,
                                )
                                for k, g in out.groupby(["Date", "GID_2"])
                            },
                            names=["Date", "GID_2"],
                        )
                        .reset_index()
                        .rename(columns={"value": "prediction"})
                        .drop(columns=["level_2"])
                    )
                    out["horizon"] = h + 1  # 1-based horizon
                else:
                    out["quantile"] = 0.5
                    out["horizon"] = h + 1  # 1-based horizon
                    out["prediction"] = np.expm1(out["prediction"]).clip(lower=0)
                    out = out.drop(columns=["sample"])

                rows.append(out)

            toc = pd.Timestamp.now()
            logging.info(f"Region {gid} done in {toc - tic}")

        preds = pd.concat(rows, ignore_index=True)

        # Merge Cases back in to preds
        preds = preds.merge(
            df[["Date", "GID_2", "Log_Cases"]],
            on=["Date", "GID_2"],
            how="left",
        )
        preds["Cases"] = np.expm1(preds["Log_Cases"])

        return preds

    # def forecast_using_past_covariances(
    #     self,
    #     start_date: Optional[pd.Timestamp] = None,
    # ) -> pd.DataFrame:
    #     # Full data including future dates
    #     target_list, covar_list, target_gids = self.get_cases(future=False)

    #     # Ensure dataframe is sorted for correct indexing
    #     df = df.sort_values(by=["GID_2", "Date"])
    #     all_gids = df[self.geo_col].unique().tolist()
    #     all_dates = df[self.date_col].unique().tolist()
    #     forecast_dates = df[df["future"]][self.date_col].unique().tolist()
    #     origin_date = max(set(all_dates) - set(forecast_dates))

    #     # Loop over forecast dates ('future' = True) and make iterative predictions
    #     df_preds = []
    #     for horizon_index, date in enumerate(forecast_dates):
    #         logging.info(f"Forecasting for date {date}")
    #         horizon = horizon_index + 1

    #         # Prepare cases and covariates
    #         cases, past_covs = self.get_cases_covs(df)

    #         # First index where date_col == date
    #         ix = all_dates.index(date)

    #         # Use historical_forecasts so that covariates are considered
    #         # and samples are returned
    #         bt = self.model.historical_forecasts(
    #             series=[c[: ix + 1] for c in cases],
    #             past_covariates=[p[: ix + 1] for p in past_covs],
    #             start=date,
    #             forecast_horizon=1,
    #             stride=1,
    #             retrain=False,
    #             last_points_only=False,
    #             verbose=False,
    #             num_samples=self.num_samples,
    #         )

    #         if not isinstance(bt, list):
    #             bt = [bt]

    #         preds = self.flatten_bt_to_quantiles(bt, all_gids, date_col=self.date_col)
    #         preds["origin_date"] = origin_date  # reset origin date
    #         preds["horizon"] = horizon  # adjust horizon
    #         df_preds.append(preds)  # store predictions

    #         # Then insert prediction into cases column for next iterative update
    #         for gid in all_gids:
    #             ypred = preds[
    #                 (preds[self.geo_col] == gid) & (preds["quantile"] == 0.5)
    #             ]["prediction"].iloc[0]
    #             df.loc[
    #                 (df[self.date_col] == date) & (df[self.geo_col] == gid),
    #                 self.case_col,
    #             ] = ypred

    #     # Can reset future Cases to NA here, but df is a local copy so discard
    #     # df.loc[df['future'], self.case_col] = np.nan

    #     return pd.concat(df_preds, ignore_index=True)
