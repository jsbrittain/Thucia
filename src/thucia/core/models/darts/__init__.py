import logging
from pathlib import Path
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from thucia.core.cases import align_date_types
from thucia.core.fs import DataFrame
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
        db_file: str | Path | None = None,
        train_start_date=None,
        train_end_date=None,
    ):
        self.df = df
        self.case_col = case_col
        self.date_col = date_col
        self.geo_col = geo_col
        self.covariate_cols = covariate_cols or []
        self.horizon = horizon
        self.num_samples = num_samples
        self.db_file = Path(db_file) if db_file else None

        if self.db_file:
            logging.debug(f"Darts model initialized with file store: {self.db_file}")
        else:
            logging.debug("Darts model initialized without file store.")

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

        # Training period
        if train_start_date is None:
            train_start_date = pd.Timestamp.min
        if train_end_date is None:
            train_end_date = pd.Timestamp.max
        train_start_date = max(
            align_date_types(train_start_date, self.df[self.date_col]),
            self.df[self.date_col].min(),
        )
        train_end_date = min(
            align_date_types(train_end_date, self.df[self.date_col]),
            self.df[self.date_col].max(),
        )
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date

        # Parameters and functionality provided by subclasses
        self.sampling_method = None
        self.model = self.build_model()

    # Child classes must override this method to provide concrete functionality
    def build_model(self):
        raise NotImplementedError

    def get_cases(
        self,
        future=None,
        target_gids: Optional[List[str]] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ):
        if future is None:
            # Use all data
            df = self.df
        elif future:
            df = self.df[self.df["future"]]
        else:
            df = self.df[~self.df["future"]]

        if target_gids is None:
            target_gids = df["GID_2"].unique()

        if start_date is None:
            start_date = df[self.date_col].min()
        if end_date is None:
            end_date = df[self.date_col].max()

        start_date = align_date_types(start_date, df[self.date_col])
        end_date = align_date_types(end_date, df[self.date_col])

        target_list = []
        covar_list = []
        for gid in target_gids:
            gdf = df[
                (df["GID_2"] == gid)
                & (df["Date"] >= start_date)
                & (df["Date"] <= end_date)
            ]
            freq = gdf["Date"].dtype.freq.freqstr
            # darts requires timestamp
            gdf = gdf.assign(Date=gdf["Date"].dt.to_timestamp(how="end"))

            ts = TimeSeries.from_dataframe(
                gdf,
                time_col="Date",
                value_cols=["Log_Cases"],
                fill_missing_dates=True,
                freq=freq,
            ).astype(np.float32)

            cov = TimeSeries.from_dataframe(
                gdf,
                time_col="Date",
                value_cols=self.covariate_cols,
                fill_missing_dates=True,
                freq=freq,
            ).astype(np.float32)

            target_list.append(ts)
            covar_list.append(cov)

        return target_list, covar_list, target_gids

    def historical_predictions(
        self,
        *,
        train_per_region: bool = False,
        retrain: bool = True,  # only turn off for faster testing
        start_date: pd.Timestamp | None = None,
    ) -> DataFrame | pd.DataFrame:
        """
        Pre-fits on all regions, then generates historical forecasts for each region
        separately.
        """
        # Training and forecasting loop
        if train_per_region:
            # Output (Thucia DataFrame if db_file specified, pd.DataFrame otherwise)
            tdf = (
                DataFrame(db_file=Path(self.db_file), new_file=True)
                if self.db_file
                else []
            )
            gid_list = self.df["GID_2"].unique().tolist()
            for ix, gid in enumerate(gid_list):
                logging.info(f"Processing GID_2: {gid}...")
                tic = pd.Timestamp.now()
                df_gid = self.df[self.df["GID_2"] == gid].copy()

                # predictions are always pd.DataFrame
                tdf.append(
                    self._historical_predictions_onepass(
                        df=df_gid,
                        retrain=retrain,
                        start_date=start_date,
                    )
                )

                # Estimate time remaining
                toc = pd.Timestamp.now()
                logging.info(f"Completed GID_2: {gid} in {toc - tic}.")
                estimated_time_remaining = (toc - tic) * (len(gid_list) - ix - 1)
                logging.info(f"Estimated time remaining: {estimated_time_remaining}.")

            if isinstance(tdf, pd.DataFrame):
                tdf = pd.concat(tdf).reset_index()

            return tdf
        else:
            return self._historical_predictions_onepass(
                retrain=retrain,
                start_date=start_date,
            )

    def _historical_predictions_onepass(
        self,
        *,
        df: pd.DataFrame | None = None,
        retrain: bool = True,  # only turn off for faster testing
        start_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Pre-fits on all regions, then generates historical forecasts for each region
        separately.
        """
        df = df if df is not None else self.df  # use provided df, fallback to self.df

        if start_date is None:
            start_date = df["Date"].min()

        # Convert floats to 32-bit precision
        float_cols = df.select_dtypes(include="float").columns
        df[float_cols] = df[float_cols].astype(np.float32)

        # Model pre-fit
        target_gids = df["GID_2"].unique()
        self.pre_fit(target_gids=target_gids)

        # Now include future data for forecasting, ensuring the same GID mapping
        target_list, covar_list, _ = self.get_cases(target_gids=target_gids)

        rows = []
        for ts, cov, gid in zip(target_list, covar_list, target_gids):
            logging.info(f"Forecasting for GID_2 {gid}")
            tic = pd.Timestamp.now()
            start_date_timestamp = start_date.to_timestamp(how="end")
            try:
                bt = self.historical_forecasts(
                    ts, cov, start_date=start_date_timestamp, retrain=retrain
                )
                # bt = [time]series[horizon][1][samples]
            except ValueError as e:
                logging.warning(
                    f"Failed to fit for GID_2 {gid} (msg: {e}), skipping..."
                )
                continue

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

        # Ensure Date is in original format
        freq = df["Date"].dtype.freq.freqstr[0]
        preds["Date"] = preds["Date"].dt.to_period(freq)

        # Merge Cases back in to preds
        preds = preds.merge(
            df[["Date", "GID_2", "Log_Cases"]],
            on=["Date", "GID_2"],
            how="left",
        )
        # Restore GID categories
        preds["GID_2"] = pd.Categorical(
            preds["GID_2"],
            categories=df["GID_2"].cat.categories,
            ordered=df["GID_2"].cat.ordered,
        )
        # Return Cases to original scale
        preds["Cases"] = np.expm1(preds["Log_Cases"])

        return preds
