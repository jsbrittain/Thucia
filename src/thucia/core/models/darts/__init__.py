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
from thucia.core.models.utils import quantiles as default_quantiles
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
        horizons=[1],
        num_samples: int | None = None,
        db_file: str | Path | None = None,
        train_start_date=None,
        train_end_date=None,
        multivariate: bool = False,
        quantiles: List[float] | None = None,
    ):
        self.df = df
        self.case_col = case_col
        self.date_col = date_col
        self.geo_col = geo_col
        self.covariate_cols = covariate_cols or []
        self.horizons = horizons
        self.num_samples = num_samples or 1000
        self.db_file = Path(db_file) if db_file else None
        self.multivariate = multivariate
        self.quantiles = quantiles or default_quantiles
        self.fit_delta = False

        if self.multivariate and "GID_2_codes" not in self.covariate_cols:
            self.covariate_cols.append("GID_2_codes")  # added in get_cases

        if self.db_file:
            logging.debug(f"Darts model initialized with file store: {self.db_file}")
        else:
            logging.debug("Darts model initialized without file store.")

        if len(self.quantiles) == self.num_samples:
            raise ValueError(
                "num_samples cannot be equal to the number of quantiles; "
                "please set num_samples to None or a different value."
            )

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

        # Skip training / forecasting in GID_2s with all zeros in training period
        self.identify_noincidence_regions()

        # Parameters and functionality provided by subclasses
        self.sampling_method = None

    def identify_noincidence_regions(self):
        # Reject GID_2 with all zeros in training period
        def has_nonzero_cases(gdf):
            gdf_train = gdf[
                (gdf[self.date_col] >= self.train_start_date)
                & (gdf[self.date_col] <= self.train_end_date)
            ]
            return (gdf_train[self.case_col] > 0).any()

        self.valid_gids = (
            self.df.groupby(self.geo_col, observed=False)
            .filter(has_nonzero_cases)[self.geo_col]
            .unique()
        )
        self.rejected_gids = set(self.df[self.geo_col].unique()) - set(self.valid_gids)
        if self.rejected_gids:
            logging.info(
                f"Excluding {len(self.rejected_gids)} GID_2s with no incidence in "
                "training period."
            )

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

        # Add GID2 as numeric code
        if self.multivariate:
            if "GID_2_codes" not in df.columns:
                df = df.assign(GID_2_codes=df["GID_2"].cat.codes)
            if "GID_2_codes" not in self.covariate_cols:
                self.covariate_cols.append("GID_2_codes")

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
            freq = "ME"
            # darts requires timestamp
            gdf["Date"] = (
                gdf["Date"].dt.to_timestamp(how="end").dt.normalize().astype("<M8[ns]")
            )

            if self.fit_delta:
                gdf["Log_Cases"] = gdf["Log_Cases"].diff()

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

    def _historical_predictions_per_region(
        self,
        *,
        tdf_out: DataFrame,
        horizon: int,
        retrain: bool = True,  # only turn off for faster testing
        start_date: pd.Timestamp | None = None,
        geo_col: str = "GID_1",
    ) -> DataFrame | pd.DataFrame:
        gid_list = self.df[geo_col].unique().tolist()
        for ix, gid in enumerate(gid_list):
            logging.info(f"Processing {geo_col}: {gid}...")
            tic = pd.Timestamp.now()
            df_gid = self.df[self.df[geo_col] == gid].copy()

            # predictions are always pd.DataFrame
            self._historical_predictions_onepass(
                df=df_gid,
                tdf_out=tdf_out,
                retrain=retrain,
                start_date=start_date,
                horizon=horizon,
            )

            # Estimate time remaining
            toc = pd.Timestamp.now()
            logging.info(f"Completed {geo_col}: {gid} in {toc - tic}.")
            estimated_time_remaining = (toc - tic) * (len(gid_list) - ix - 1)
            logging.info(f"Estimated time remaining: {estimated_time_remaining}.")
        return tdf_out

    def historical_predictions(
        self,
        *,
        model_admin_level: int | None = None,  # 0=country, 1=state, 2=municipality
        retrain: bool = True,  # only turn off for faster testing
        start_date: pd.Timestamp | None = None,
    ) -> DataFrame | pd.DataFrame:
        """
        Pre-fits on all regions, then generates historical forecasts for each region
        separately.
        """
        model_admin_level = model_admin_level if model_admin_level is not None else 0
        logging.info(
            f"Generating historical predictions at admin level {model_admin_level}."
        )

        tdf = (
            DataFrame(db_file=Path(self.db_file), new_file=True)
            if self.db_file
            else DataFrame()  # fallback to in-memory DataFrame
        )

        if model_admin_level == 0:  # Train on entire country
            for horizon in self.horizons:
                self._historical_predictions_onepass(
                    tdf_out=tdf,
                    retrain=retrain,
                    start_date=start_date,
                    horizon=horizon,
                )

        elif model_admin_level == 1:  # Train per state (GID_1)
            for horizon in self.horizons:
                self._historical_predictions_per_region(
                    tdf_out=tdf,
                    retrain=retrain,
                    start_date=start_date,
                    geo_col="GID_1",
                    horizon=horizon,
                )
        elif model_admin_level == 2:  # Train per municipality (GID_2)
            for horizon in self.horizons:
                self._historical_predictions_per_region(
                    tdf_out=tdf,
                    retrain=retrain,
                    start_date=start_date,
                    geo_col="GID_2",
                    horizon=horizon,
                )
        else:
            raise ValueError(
                f"model_admin_level must be 0, 1, or 2 (got '{model_admin_level}')."
            )

        return tdf

    def _extract_horizons(
        self,
        bt,
        gid,
    ):
        # process horizons separately
        rows = []
        for h in self.horizons:
            vals = np.concat([TimeSeries.all_values(t)[h, :, :] for t in bt])
            dates = np.array([t.time_index[h] for t in bt])

            out = pd.DataFrame(vals)
            out["Date"] = dates
            out = out.melt(
                id_vars="Date",
                var_name="sample",  # may be quantiles, renamed later
                value_name="prediction",
            )
            out["sample"] = out["sample"].astype(int)
            out["GID_2"] = gid

            if not self.sampling_method or self.sampling_method == "samples":
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
                                for k, g in out.groupby(
                                    ["Date", "GID_2"], observed=False
                                )
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
            elif self.sampling_method == "quantiles":
                out = out.rename(columns={"sample": "quantile"})
                out["quantile"] = out["quantile"].apply(lambda x: self.quantiles[x])
                out["prediction"] = np.expm1(out["prediction"]).clip(lower=0)
                out["horizon"] = h + 1  # 1-based horizon
            else:
                raise ValueError(f"Unknown sampling_method '{self.sampling_method}'")

            rows.append(out)
        return rows

    def _merge_cases(
        self,
        df: pd.DataFrame,
        preds: pd.DataFrame,
    ) -> pd.DataFrame:
        # Ensure Date is in original format
        freq = df["Date"].dtype.freq.freqstr[0]
        if not pd.api.types.is_period_dtype(preds["Date"]):
            # Coerce to period
            preds["Date"] = preds["Date"].dt.to_period(freq)
        elif preds["Date"].dtype.freq.freqstr != freq:
            # Coalesce frequency
            preds["Date"] = preds["Date"].dt.asfreq(freq)

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
        preds["Cases"] = np.expm1(preds["Log_Cases"]).clip(lower=0)
        return preds

    def _historical_predictions_onepass(
        self,
        *,
        horizon: int,
        df: pd.DataFrame | None = None,
        tdf_out: DataFrame,
        retrain: bool = True,  # only turn off for faster testing
        start_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Pre-fits on all regions, then generates historical forecasts for each region
        separately.
        """
        df = df if df is not None else self.df  # use provided df, fallback to self.df
        self.model = self.build_model(horizon=horizon)

        if start_date is None:
            start_date = df["Date"].min()

        # Convert floats to 32-bit precision
        float_cols = df.select_dtypes(include="float").columns
        df[float_cols] = df[float_cols].astype(np.float32)

        # Model pre-fit
        all_target_gids = df["GID_2"].unique()
        target_gids = [gid for gid in all_target_gids if gid not in self.rejected_gids]
        self.pre_fit(target_gids=target_gids)

        # Now include future data for forecasting, ensuring the same GID mapping
        target_list, covar_list, _ = self.get_cases(target_gids=target_gids)

        if self.multivariate:
            logging.info(f"Forecasting for {len(target_gids)} regions (multivariate)")
            tic = pd.Timestamp.now()
            start_date_timestamp = start_date.to_timestamp(how="end")
            try:
                bt = self.historical_forecasts(
                    target_list,
                    covar_list,
                    start_date=start_date_timestamp,
                    retrain=retrain,
                    horizon=horizon,
                )
                # for multivariate, bt is a list per time-series:
                # bt = [gid][time]series[horizon][1][samples]
            except ValueError as e:
                logging.warning(
                    f"Failed to fit for GID_2 {target_gids} (multivariate) "
                    f"(msg: {e}), skipping..."
                )
                self.rejected_gids.update(target_gids)
            for ix, gid in enumerate(target_gids):
                out = bt[ix].to_dataframe().reset_index(names="Date")
                out = out.melt(
                    id_vars="Date",
                    var_name="quantile",
                    value_name="prediction",
                )
                out["quantile"] = (
                    out["quantile"].str.split(".").str[-1].astype(float) / 1000
                )
                out["horizon"] = horizon
                out["GID_2"] = gid
                out["prediction"] = np.expm1(out["prediction"]).clip(lower=0)
                # print(out)
                # print(out[out['quantile'] == 0.5])
                # tdf_out.append(out)
                tdf_out.append(self._merge_cases(df, out))
            toc = pd.Timestamp.now()
            logging.info(f"Regions {target_gids} done in {toc - tic}")
        else:
            for ts, cov, gid in zip(target_list, covar_list, target_gids):
                logging.info(f"Forecasting for GID_2 {gid}")
                tic = pd.Timestamp.now()
                start_date_timestamp = start_date.to_timestamp(how="end")
                try:
                    bt = self.historical_forecasts(
                        ts,
                        cov,
                        gid=gid,
                        start_date=start_date_timestamp,
                        retrain=retrain,
                        horizon=horizon,
                    )
                    # for univariate, bt relates to a single time-series:
                    # bt = [time]series[horizon][1][samples]
                except ValueError as e:
                    logging.warning(
                        f"Failed to fit for GID_2 {gid} (msg: {e}), skipping..."
                    )
                    self.rejected_gids.add(gid)
                    continue
                out = bt.to_dataframe().reset_index(names="Date")
                out = out.melt(
                    id_vars="Date",
                    var_name="quantile",
                    value_name="prediction",
                )
                out["quantile"] = (
                    out["quantile"].str.split(".").str[-1].astype(float) / 1000
                )
                out["horizon"] = horizon
                out["GID_2"] = gid
                if self.fit_delta:
                    out["prediction"] = out["prediction"].cumsum()
                out["prediction"] = np.expm1(out["prediction"]).clip(lower=0)
                # print(out)
                # print(out[out['quantile'] == 0.5])
                # tdf_out.append(out)
                tdf_out.append(self._merge_cases(df, out))
                toc = pd.Timestamp.now()
                logging.info(f"Region {gid} done in {toc - tic}")

        # Substitute back all-zero regions
        #
        # Re-compute target_gids to account for new entried due to fitting errors; we do
        # not use self.rejected_gids directly as it is not specific to the geographic
        # region being assessed.
        # target_gids = [gid for gid in all_target_gids if gid not in self.rejected_gids]
        # if self.rejected_gids:
        #     logging.info(
        #         f"Adding all-zero predictions for {len(self.rejected_gids)} "
        #         "GID_2s with no incidence in training period, or estimation errors..."
        #     )
        #     logging.info(f"Rejected GID_2s: {self.rejected_gids}")
        #     dates = df["Date"].unique()
        #     dates = dates[dates >= start_date]
        #     dates = dates.to_timestamp(how="end")
        #     dates = np.sort(dates)
        # for gid in set(all_target_gids) - set(target_gids):
        #     logging.info(f"Adding all-zero predictions for GID_2 {gid}...")
        #     # Create rows with predictions equal to zero
        #     out = pd.DataFrame(
        #         [
        #             (d, gid, q, 0, h + 1)
        #             for d in dates
        #             for h in range(self.horizon)
        #             for q in self.quantiles
        #         ],
        #         columns=["Date", "GID_2", "quantile", "prediction", "horizon"],
        #     )
        #     # Remove predictions outside forecasting range (to match other regions)
        #     for ix, date in enumerate(dates):
        #         if ix < self.horizon:
        #             # Remove predictions before horizon available
        #             out = out[~((out["Date"] == date) & (out["horizon"] > ix + 1))]
        #         if len(dates) - ix <= self.horizon:
        #             # Remove predictions after data available
        #             out = out[
        #                 ~(
        #                     (out["Date"] == date)
        #                     & (out["horizon"] <= self.horizon - len(dates) + ix)
        #                 )
        #             ]
        #     # Add to database
        #     tdf_out.append(self._merge_cases(df, out))

        return tdf_out
