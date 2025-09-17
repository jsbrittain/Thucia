import logging
from typing import Optional

import numpy as np
import pandas as pd
from timesfm import TimesFm
from timesfm import TimesFmCheckpoint
from timesfm import TimesFmHparams

from .utils import filter_admin1
from .utils import interpolate_missing_dates
from .utils import set_historical_na_to_zero

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None


class TimeSFMSamples:
    def __init__(
        self,
        timeseries_df: pd.DataFrame,
        case_col: str,
        dynamic_numerical_cols: list[str] = [],
        dynamic_categorical_cols: list[str] = [],
        static_numerical_covariates: list[str] = [],
        static_categorical_cols: list[str] = [],
        group_col: str = "GID_2",
        per_core_batch_size: int = 32,
        num_layers: int = 50,
        context_len: int = 32,
        horizon_len: int = 1,
        num_samples: int = 1000,
        pdfm_df: Optional[pd.DataFrame] = None,  # index = GID_2, cols = embedding dims
        pdfm_pca_dim: Optional[int] = None,  # if set, apply PCA to reduce pdfm_df
    ):
        self.df = timeseries_df.copy()
        self.case_col = case_col
        self.dynamic_numerical_cols = dynamic_numerical_cols
        self.dynamic_categorical_cols = dynamic_categorical_cols
        self.static_numerical_covariates = static_numerical_covariates
        self.static_categorical_cols = static_categorical_cols
        self.group_col = group_col
        self.per_core_batch_size = per_core_batch_size
        self.num_layers = num_layers
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.num_samples = num_samples
        self.pdfm_df = (
            pdfm_df.set_index(group_col)
            if (pdfm_df is not None and group_col in pdfm_df.columns)
            else pdfm_df
        )
        self.pdfm_pca_dim = pdfm_pca_dim

        self.model = TimesFm(
            hparams=TimesFmHparams(
                backend="cpu",
                context_len=self.context_len,
                horizon_len=self.horizon_len,
                num_layers=self.num_layers,
                per_core_batch_size=self.per_core_batch_size,
                use_positional_embedding=True,
            ),
            checkpoint=TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
            ),
        )

    def sample_from_quantiles(
        self,
        quantiles: np.ndarray,
        quantile_levels: np.ndarray,
        num_samples: int = 1000,
    ):
        """
        Simulate samples from a discrete quantile forecast using inverse transform
        sampling.

        Args:
            quantiles: np.array of shape [num_quantiles], e.g., at 10%-90%
            quantile_levels: np.array of quantile probabilities, e.g., [0.1, ..., 0.9]
            num_samples: Number of samples to generate
        Returns:
            np.array of samples
        """
        uniform_samples = np.random.uniform(0, 1, size=num_samples)
        return np.interp(uniform_samples, quantile_levels, quantiles)

    @property
    def use_covariates(self) -> bool:
        return bool(
            self.dynamic_numerical_cols
            or self.dynamic_categorical_cols
            or self.static_numerical_covariates
            or self.static_categorical_cols
        )

    def predict(self, sigma: float = 0.0) -> pd.DataFrame:
        df = self.df.copy()
        if len(df) < (self.context_len + self.horizon_len):
            logging.warning(f"Insufficient data (n={len(df)}) for TimeSFM model.")
            df["sample"] = np.nan
            df["prediction"] = np.nan
            return df

        # ---------- POOL OVER GROUPS ----------
        # Ensure deterministic group order
        df = df.sort_values([self.group_col, "Date"])
        groups = []
        for g, gdf in df.groupby(self.group_col, sort=False):
            # Need at least context_len + horizon_len observations for each group
            if len(gdf) < (self.context_len + self.horizon_len):
                logging.warning(f"Skipping {g}: not enough history ({len(gdf)})")
                continue
            groups.append((g, gdf.reset_index(drop=True)))
        if not groups:
            logging.warning("No groups have sufficient history.")
            df["sample"] = np.nan
            df["prediction"] = np.nan
            return df
        gid_vocab = [gid for gid, _ in groups]
        gid_to_idx = {gid: i for i, gid in enumerate(gid_vocab)}

        N = len(groups)

        # Base univariate inputs (exclude horizon)
        inputs = [
            gdf[self.case_col].iloc[: -self.horizon_len].tolist() for _, gdf in groups
        ]

        # Frequency per series (assume monthly=1 as in your code)
        freq = [1] * N

        if self.use_covariates:
            # Dynamic covariates: list per series
            dynamic_numerical = {}
            for col in self.dynamic_numerical_cols:
                dynamic_numerical[col] = [gdf[col].tolist() for _, gdf in groups]

            dynamic_categorical = {}
            for col in self.dynamic_categorical_cols:
                dynamic_categorical[col] = [gdf[col].tolist() for _, gdf in groups]

            # Static covariates: one value (or vector) per series
            static_numerical = {}
            for col in self.static_numerical_covariates:
                static_numerical[col] = np.array(
                    [gdf[col].iloc[-1] for _, gdf in groups]
                )

            # (A) Static categorical, aligned to gid order
            static_categorical = {}
            if self.static_categorical_cols != ["GID_2"]:
                raise NotImplementedError(
                    "Only 'GID_2' supported as static categorical."
                )
            static_categorical["region_id"] = np.asarray(
                [gid_to_idx[gid] for gid, _ in groups], dtype=np.int64
            )

            # (B) Add PDFM embeddings as static real, aligned to gid order
            if self.pdfm_df is not None:
                # ensure index by group_col
                pdfm_df = self.pdfm_df
                if pdfm_df.index.name != self.group_col:
                    # if it's not indexed yet
                    if self.group_col in pdfm_df.columns:
                        pdfm_df = pdfm_df.set_index(self.group_col)
                    else:
                        raise ValueError(
                            "pdfm_df must have index or column named group_col "
                            "(e.g., 'GID_2')."
                        )

                # Reindex to current group order; will raise if any missing
                try:
                    E = pdfm_df.reindex(gid_vocab).to_numpy(dtype=np.float32, copy=True)
                except Exception as e:
                    raise ValueError(
                        f"Embeddings missing for some groups. Expected {gid_vocab}"
                    ) from e

                # Optional PCA
                if self.pdfm_pca_dim is not None:
                    if PCA is None:
                        raise ImportError(
                            "scikit-learn not available for PCA; set pdfm_pca_dim=None "
                            "or install scikit-learn."
                        )
                    pca = PCA(n_components=self.pdfm_pca_dim, random_state=0)
                    # fit on available groups; for production, fit once and persist
                    E = pca.fit_transform(E).astype(np.float32)

                for j in range(E.shape[1]):
                    static_numerical[f"pdfm_{j}"] = E[:, j].astype(np.float32)  # (N,)

            # ---- TimesFM pooled forecast with covariates ----
            forecasts, _ols = self.model.forecast_with_covariates(
                inputs=inputs,
                dynamic_numerical_covariates=(
                    dynamic_numerical if dynamic_numerical else None
                ),
                dynamic_categorical_covariates=(
                    dynamic_categorical if dynamic_categorical else None
                ),
                static_numerical_covariates=(
                    static_numerical if static_numerical else None
                ),
                static_categorical_covariates=(
                    static_categorical if static_categorical else None
                ),
                freq=freq,
                xreg_mode="xreg + timesfm",
            )

            # forecasts: list of length N, each is array of length horizon_len
            out_rows = []
            for (gid, gdf), fvec in zip(groups, forecasts):
                # dates for the horizon in this series
                horizon_dates = gdf["Date"].iloc[-self.horizon_len :]
                for date, mu in zip(horizon_dates, fvec):
                    samples = np.random.normal(
                        loc=mu,
                        scale=sigma,
                        size=self.num_samples,
                    )
                    out_rows.extend(
                        {
                            "Date": date,
                            self.group_col: gid,
                            "sample": i,
                            "prediction": s,
                        }
                        for i, s in enumerate(samples)
                    )
            pred_df = pd.DataFrame(out_rows)

        else:
            # ---- No covariates: pooled vanilla TimesFM ----
            forecasts, quantile_forecasts = self.model.forecast(
                inputs=inputs,
                freq=freq,
            )

            out_rows = []
            for (gid, gdf), fvec, qvec in zip(groups, forecasts, quantile_forecasts):
                horizon_dates = gdf["Date"].iloc[-self.horizon_len :]
                for nstep, (date, _mu) in enumerate(zip(horizon_dates, fvec)):
                    samples = self.sample_from_quantiles(
                        quantiles=qvec[nstep][1:],  # drop mean at index 0
                        quantile_levels=self.model.hparams.quantiles,
                        num_samples=self.num_samples,
                    )
                    out_rows.extend(
                        {
                            "Date": date,
                            self.group_col: gid,
                            "sample": i,
                            "prediction": s,
                        }
                        for i, s in enumerate(samples)
                    )
            pred_df = pd.DataFrame(out_rows)

        # Merge back onto original rows (keep per-group dates aligned)
        merged = df.merge(pred_df, on=["Date", self.group_col], how="left")
        logging.debug("Finished pooled forecast.")
        return merged


def timesfm(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp = pd.Timestamp.min,
    end_date: str | pd.Timestamp = pd.Timestamp.max,
    gid_1: list[str] | None = None,
    method: str = "historical",  # historical / predict
    include_covariates: bool = True,
    horizon: int = 1,
    pdfm_df: Optional[pd.DataFrame] = None,
    case_col: str = "Log_Cases",
    covariate_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    logging.info("Starting TimeSFM model...")

    # df = df.copy()
    # df = filter_admin1(df, gid_1=gid_1)
    # df = interpolate_missing_dates(df, start_date, end_date)
    # df = set_historical_na_to_zero(df)

    # # Apply log transform for TimeSFM modeling
    # df["Log_Cases"] = np.log1p(df["Cases"])

    # # Pre-compute covariates
    # df["MONTH"] = pd.to_datetime(df["Date"]).dt.month.astype(int)
    # df["LAG_1_LOG_CASES"] = df.groupby("GID_2")["Log_Cases"].shift(1)
    # df["tmin_roll_2"] = df.groupby("GID_2")["tmin"].transform(
    #     lambda x: x.rolling(2).mean()
    # )
    # df["LAG_1_tmin_roll_2"] = df.groupby("GID_2")["tmin_roll_2"].shift(1)
    # df["prec_roll_2"] = df.groupby("GID_2")["prec"].transform(
    #     lambda x: x.rolling(2).mean()
    # )
    # df["LAG_1_prec_roll_2"] = df.groupby("GID_2")["prec_roll_2"].shift(1)

    pool_regions = True

    # Prepare embeddings
    if pdfm_df is not None:
        provinces = df["GID_2"].unique()
        pdfm_df = pdfm_df[pdfm_df["GID_2"].isin(provinces)].copy()
        pdfm_df.set_index("GID_2", inplace=True)
        feature_cols = [f"feature{i}" for i in range(330)]
        pdfm_df = pdfm_df[feature_cols]
        # remove non-unique indices from pdfm_df  ### THIS IS A HACK - NEED TO FIX THE SOURCE
        pdfm_df = pdfm_df[~pdfm_df.index.duplicated(keep="first")]
        # remove gid_2 from df if not in embeddings
        df = df[df["GID_2"].isin(pdfm_df.index)]  ### REMOVE DUPLICATES FROM EMBEDDINGS

    # Instantiate TimeSFM model
    timesfm = TimeSFMSamples(
        timeseries_df=df,
        case_col=case_col,
        dynamic_numerical_cols=covariate_cols if include_covariates else None,
        dynamic_categorical_cols=["MONTH"],
        static_categorical_cols=["GID_2"] if pool_regions else [],
        horizon_len=horizon,
        pdfm_df=pdfm_df,
    )

    # Loop over provinces
    all_forecasts = []
    dates = sorted(df["Date"].unique())
    provinces = df["GID_2"].unique()

    if pool_regions:
        logging.info(
            f"Processing province for TimeSFM model ({len(provinces)} provinces pooled)"
        )

        # Initialize the first forecast with NaN
        n_provinces = len(provinces)
        all_forecasts.append(
            pd.DataFrame(
                {
                    "Date": [dates[0]] * n_provinces,
                    "GID_2": provinces,
                    "sample": [0] * n_provinces,
                    "prediction": [np.nan] * n_provinces,  # initial prediction is NaN
                    "Cases": [
                        df.loc[(df["Date"] == dates[0]) & (df["GID_2"] == province)][
                            "Cases"
                        ].values[0]
                        for province in provinces
                    ],
                }
            )
        )

        # Generate successive historical forecasts
        for date in dates[1:]:  # TimeSFM requires at least 2 data points
            logging.info(f"Processing date {date.strftime('%Y-%m-%d')} (all provinces)")
            # Subset data
            subset_df = df.copy()
            subset_df = subset_df[subset_df["Date"] <= date]
            subset_df = subset_df.fillna(
                0
            )  # TimeSFM cannot cope with NaN covariates #################
            timesfm.df = subset_df
            # Fit model
            df_forecast = timesfm.predict(
                sigma=0.1
            )  # ### Estimate sigma from past residuals

            # Verify that Date entries are either all True or all False
            future = set(df_forecast[df_forecast["Date"] == date]["future"])
            if len(future) > 1:
                raise ValueError(
                    "Inconsistent 'future' flags in forecast results for date {date}"
                )
            future = list(future)[0]
            # Overwrite Log_Cases with median prediction if marked 'future'
            if future:
                for province in provinces:
                    df.loc[
                        (df["Date"] == date) & (df["GID_2"] == province), "Log_Cases"
                    ] = np.median(
                        df_forecast.loc[
                            (df_forecast["Date"] == date)
                            & (df_forecast["GID_2"] == province),
                            "prediction",
                        ]
                    )
                logging.debug(f"Backfilling Log_Cases for {date}.")
            # Append to master list
            df_forecast.loc[df_forecast["future"], "Cases"] = np.nan
            all_forecasts.append(
                df_forecast[df_forecast["Date"] == date][
                    ["Date", "GID_2", "sample", "prediction", "Cases"]
                ]
            )
    else:
        for ix, province in enumerate(provinces):
            logging.info(
                f"Processing province {province} ({ix + 1}/{len(provinces)}) "
                "for TimeSFM model"
            )

            # Initialize the first forecast with NaN
            all_forecasts.append(
                pd.DataFrame(
                    {
                        "Date": [dates[0]],
                        "GID_2": [province],
                        "sample": [0],
                        "prediction": [np.nan],  # initial prediction is NaN
                        "Cases": [
                            df.loc[
                                (df["Date"] == dates[0]) & (df["GID_2"] == province)
                            ]["Cases"].values[0]
                        ],
                    }
                )
            )

            # Generate successive historical forecasts
            for date in dates[1:]:  # TimeSFM requires at least 2 data points
                logging.info(
                    f"Processing date {date.strftime('%Y-%m-%d')} for "
                    "province {province}"
                )
                # Subset data
                subset_df = df.copy()
                subset_df = subset_df[subset_df["GID_2"] == province]
                subset_df = subset_df[subset_df["Date"] <= date]
                subset_df = subset_df.fillna(
                    0
                )  # TimeSFM cannot cope with NaN covariates #################
                timesfm.df = subset_df
                # Fit model
                df_forecast = timesfm.predict(
                    sigma=0.1
                )  # ### Estimate sigma from past residuals
                # Overwrite Log_Cases with median prediction if marked 'future'
                if subset_df["future"].iloc[-1]:
                    df.loc[
                        (df["Date"] == date) & (df["GID_2"] == province), "Log_Cases"
                    ] = np.median(
                        df_forecast.loc[
                            (df_forecast["Date"] == date)
                            & (df_forecast["GID_2"] == province),
                            "prediction",
                        ]
                    )
                    logging.debug(f"Backfilling Log_Cases for {date}, {province}.")
                # Append to master list
                df_forecast.loc[df_forecast["future"], "Cases"] = np.nan
                all_forecasts.append(
                    df_forecast[df_forecast["Date"] == date][
                        ["Date", "GID_2", "sample", "prediction", "Cases"]
                    ]
                )

    # Concatenate all province forecasts into a single DataFrame
    forecast_results = pd.concat(all_forecasts, ignore_index=True)

    # Inverse log transform
    forecast_results["prediction"] = np.clip(
        np.expm1(forecast_results["prediction"]),
        a_min=0,
        a_max=None,
    )

    logging.info("TimeSFM model complete.")
    return forecast_results
