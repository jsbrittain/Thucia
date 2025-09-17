import os
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None

torch.set_default_dtype(torch.float32)
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))


ScaleStrategy = Literal["global", "prefix", "per_cutoff"]


class BasePooledSamples:
    """
    Base class that handles:
      - pooling across entities (GID_2)
      - constant-embedding covariates (optionally PCA-compressed)
      - leak-safe scaling strategies
      - global fit + historical forecasts (walk-forward supported)
    Specializations should override build_model() to return a Darts model instance.
    """

    def __init__(
        self,
        timeseries_df: pd.DataFrame,
        target_col: str,
        covariate_cols: List[str],
        horizon: int = 1,
        input_chunk_length: int = 48,
        withheld_months: int = 0,
        num_samples: int = 1000,
        verbose: bool = True,
        pdfm_df: Optional[pd.DataFrame] = None,
        pdfm_pca_dim: Optional[int] = None,
        scale_strategy: ScaleStrategy = "global",  # "global" | "prefix" | "per_cutoff"
        start: Optional[pd.Timestamp] = None,  # used by "prefix" & "per_cutoff"
        train_length: Optional[int] = None,  # optional rolling window length
    ):
        self.df = timeseries_df.copy()
        self.target_col = target_col
        self.covariate_cols = covariate_cols
        self.horizon = int(horizon)
        self.input_chunk_length = int(input_chunk_length)
        self.withheld_months = int(withheld_months)
        self.num_samples = int(num_samples)
        self.verbose = verbose
        self.pdfm_df = pdfm_df.copy() if pdfm_df is not None else None
        self.pdfm_pca_dim = pdfm_pca_dim
        self.embed_prefix = "feature"

        # scaling policy
        self.scale_strategy: ScaleStrategy = scale_strategy
        self.start: Optional[pd.Timestamp] = (
            pd.to_datetime(start) if start is not None else None
        )
        self.train_length = train_length

        # scalers (we may fit them differently depending on strategy)
        self.covar_scaler = Scaler()
        self._prefix_scaler = None  # used when scale_strategy="prefix"

        # storage for pooled series
        self._fitted = False
        self._targets_by_gid: Dict[str, TimeSeries] = {}
        self._covars_by_gid: Dict[str, TimeSeries] = {}

        # model: created by specialization
        self.model = self.build_model()

    # ---------- specialization hook ----------
    def build_model(self):
        raise NotImplementedError

    # ---------- embeddings ----------
    def _maybe_pca_fit(self):
        if (
            self.pdfm_df is not None
            and self.pdfm_pca_dim is not None
            and PCA is not None
        ):
            if self.pdfm_df.index.name != "GID_2" and "GID_2" in self.pdfm_df.columns:
                self.pdfm_df = self.pdfm_df.set_index("GID_2")
            allE = self.pdfm_df.to_numpy(dtype=np.float32)
            self._pca = PCA(n_components=self.pdfm_pca_dim, random_state=0).fit(allE)
            self._pca_fitted = True
        else:
            self._pca_fitted = False

    def _get_embedding_vec(self, gid: str) -> Optional[np.ndarray]:
        if self.pdfm_df is None:
            return None
        pdfm = self.pdfm_df
        if pdfm.index.name != "GID_2":
            if "GID_2" in pdfm.columns:
                pdfm = pdfm.set_index("GID_2")
            else:
                raise ValueError("pdfm_df must have an index or column named 'GID_2'.")
        if gid not in pdfm.index:
            raise ValueError(f"Missing embeddings for GID_2={gid}")
        E = pdfm.loc[[gid]].to_numpy(dtype=np.float32)  # (1, D)
        if self.pdfm_pca_dim is not None:
            if PCA is None:
                raise ImportError("scikit-learn required for pdfm_pca_dim.")
            if not getattr(self, "_pca_fitted", False):
                self._maybe_pca_fit()
            E = self._pca.transform(E).astype(np.float32)  # (1, D_pca)
        return E.squeeze(0)  # (D,)

    # ---------- series building ----------
    def _build_series_lists(
        self,
    ) -> Tuple[List[TimeSeries], List[TimeSeries], List[str]]:
        targets, covars, gids = [], [], []
        embed_cols = None
        self._maybe_pca_fit()

        for gid, gdf in self.df.groupby("GID_2"):
            gdf = gdf.sort_values("Date").copy()
            gdf_train = (
                gdf.iloc[: -self.withheld_months] if self.withheld_months > 0 else gdf
            )

            ts = TimeSeries.from_dataframe(
                gdf_train,
                time_col="Date",
                value_cols=[self.target_col],
                fill_missing_dates=True,
                freq="ME",
            ).astype(np.float32)

            cov = TimeSeries.from_dataframe(
                gdf_train,
                time_col="Date",
                value_cols=self.covariate_cols,
                fill_missing_dates=True,
                freq="ME",
            ).astype(np.float32)

            # constant embeddings as extra covariate channels
            E = self._get_embedding_vec(gid)
            if E is not None:
                if embed_cols is None:
                    embed_cols = [
                        f"{self.embed_prefix}{j}" for j in range(int(E.shape[0]))
                    ]
                emb_df = pd.DataFrame(
                    {c: E[j] for j, c in enumerate(embed_cols)}, index=gdf_train["Date"]
                )
                emb_ts = TimeSeries.from_dataframe(
                    emb_df, fill_missing_dates=True, freq="ME"
                ).astype(np.float32)
                cov = cov.stack(emb_ts)

            if len(ts) >= self.input_chunk_length + 1:
                targets.append(ts)
                covars.append(cov)
                gids.append(gid)

                # store FULL (incl. withheld) with embeddings for later
                ts_full = TimeSeries.from_dataframe(
                    gdf,
                    time_col="Date",
                    value_cols=[self.target_col],
                    fill_missing_dates=True,
                    freq="ME",
                ).astype(np.float32)

                cov_full = TimeSeries.from_dataframe(
                    gdf,
                    time_col="Date",
                    value_cols=self.covariate_cols,
                    fill_missing_dates=True,
                    freq="ME",
                ).astype(np.float32)

                if E is not None:
                    emb_df_full = pd.DataFrame(
                        {c: E[j] for j, c in enumerate(embed_cols)}, index=gdf["Date"]
                    )
                    emb_ts_full = TimeSeries.from_dataframe(
                        emb_df_full, fill_missing_dates=True, freq="ME"
                    ).astype(np.float32)
                    cov_full = cov_full.stack(emb_ts_full)

                self._targets_by_gid[gid] = ts_full
                self._covars_by_gid[gid] = cov_full

        return targets, covars, gids

    # ---------- fitting ----------
    def fit_global(self):
        target_list, covar_list, gids = self._build_series_lists()
        if not target_list:
            raise ValueError("No series have enough length to train.")

        # Scaling strategy during global fit:
        # - "global": fit scaler on all training covariates (may leak if used for OOS backtests)
        # - "prefix"/"per_cutoff": we still fit now to let Darts validate shapes,
        #   but we'll replace the scaler for OOS later.
        covar_list = self.covar_scaler.fit_transform(covar_list)

        self.model.fit(
            series=target_list, past_covariates=covar_list, verbose=self.verbose
        )

        # Pre-transform full covariates with this scaler (useful for "global")
        cov_full_list = [self._covars_by_gid[g] for g in gids]
        cov_full_list = self.covar_scaler.transform(cov_full_list)
        self._covars_by_gid = {g: cov for g, cov in zip(gids, cov_full_list)}

        self._fitted = True

    # ---------- historical forecasts (leak-safe) ----------
    def historical_forecasts_pooled(
        self,
        start: Optional[pd.Timestamp] = None,
        train_length: Optional[int] = None,
        scale_strategy: Optional[ScaleStrategy] = None,
        retrain: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Rolling backtest with proper scaling.

        scale_strategy:
          - "global": use scaler fitted in fit_global()  (fastest; OOS only if your training didn't include the test period)
          - "prefix": fit ONE scaler on covariate prefixes up to `start`, apply it to whole series, retrain=True for pure OOS
          - "per_cutoff": re-fit scaler at each cutoff (strict walk-forward). Implies retrain=True.

        retrain: if None, chosen automatically from scale_strategy
        """
        if not self._fitted:
            self.fit_global()

        start = pd.to_datetime(start) if start is not None else self.start
        train_length = train_length if train_length is not None else self.train_length
        strategy = scale_strategy or self.scale_strategy

        if strategy == "global":
            use_retrain = bool(retrain) if retrain is not None else False
            rows = []
            for gid, ts in self._targets_by_gid.items():
                cov = self._covars_by_gid[gid]
                bt = self.model.historical_forecasts(
                    series=ts,
                    past_covariates=cov,
                    start=start,
                    forecast_horizon=self.horizon,
                    stride=1,
                    retrain=use_retrain,
                    last_points_only=True,
                    verbose=False,
                    num_samples=self.num_samples,
                    train_length=train_length,
                )
                vals = TimeSeries.all_values(bt)  # (T,1,S)
                T_bt, _, S = vals.shape
                df = (
                    pd.DataFrame(vals.reshape(T_bt, S))
                    .assign(Date=bt.time_index, GID_2=gid)
                    .melt(
                        id_vars=["Date", "GID_2"],
                        var_name="sample",
                        value_name="prediction",
                    )
                )
                rows.append(df)
            pred_df = pd.concat(rows, ignore_index=True)

        elif strategy == "prefix":
            if start is None:
                raise ValueError("scale_strategy='prefix' requires a `start` date.")
            # 1) Fit a scaler on covariate prefixes up to start (across all gids)
            prefix_covs = []
            for gid, cov in self._covars_by_gid.items():
                prefix_covs.append(cov.drop_after(start - cov.freq))
            self._prefix_scaler = Scaler()
            prefix_covs_scaled = self._prefix_scaler.fit_transform(prefix_covs)

            # 2) Transform full covariates with this prefix-only scaler
            cov_full_scaled = self._prefix_scaler.transform(
                list(self._covars_by_gid.values())
            )
            cov_scaled_by_gid = {
                g: c for g, c in zip(self._covars_by_gid.keys(), cov_full_scaled)
            }

            # 3) Run historical forecasts with retrain=True (pure OOS) using transformed covariates
            rows = []
            for gid, ts in self._targets_by_gid.items():
                bt = self.model.historical_forecasts(
                    series=ts,
                    past_covariates=cov_scaled_by_gid[gid],
                    start=start,
                    forecast_horizon=self.horizon,
                    stride=1,
                    retrain=True,  # train only on prefix before each cutoff
                    last_points_only=True,
                    verbose=False,
                    num_samples=self.num_samples,
                    train_length=train_length,  # expanding window if None
                )
                vals = TimeSeries.all_values(bt)
                T_bt, _, S = vals.shape
                df = (
                    pd.DataFrame(vals.reshape(T_bt, S))
                    .assign(Date=bt.time_index, GID_2=gid)
                    .melt(
                        id_vars=["Date", "GID_2"],
                        var_name="sample",
                        value_name="prediction",
                    )
                )
                rows.append(df)
            pred_df = pd.concat(rows, ignore_index=True)

        elif strategy == "per_cutoff":
            # Strict walk-forward: fit a new scaler (and model) at each cutoff.
            # We loop ourselves to guarantee per-cutoff scaler fit on prefix only.
            rows = []
            for gid, ts_full in self._targets_by_gid.items():
                cov_full = self._covars_by_gid[gid]
                times = list(ts_full.time_index)
                # choose first cutoff with enough history
                first_idx = max(self.input_chunk_length, 1)
                if start is not None:
                    # advance first_idx to provided start
                    while first_idx < len(times) and times[first_idx] < start:
                        first_idx += 1
                last_cut = len(times) - self.horizon
                for t_idx in range(first_idx, last_cut + 1):
                    cutoff = times[t_idx]
                    # training prefix strictly before cutoff
                    ts_train = ts_full.drop_after(cutoff - ts_full.freq)
                    cov_train = cov_full.drop_after(cutoff - cov_full.freq)
                    if train_length is not None and len(ts_train) > train_length:
                        ts_train = ts_train[-train_length:]
                        cov_train = cov_train.slice_intersect(ts_train)

                    # fit scaler on prefix ONLY and transform
                    step_scaler = Scaler()
                    cov_train_scaled = step_scaler.fit_transform([cov_train])[0]

                    # re-init a fresh untrained model with same hyperparams
                    model_step = self.build_model()
                    model_step.fit(
                        series=[ts_train],
                        past_covariates=[cov_train_scaled],
                        verbose=False,
                    )

                    # transform full covs with step scaler (safe: stats from prefix)
                    cov_full_scaled = step_scaler.transform([cov_full])[0]

                    # get one-step (at this cutoff) H-ahead forecast
                    yhat = model_step.historical_forecasts(
                        series=ts_full,
                        past_covariates=cov_full_scaled,
                        start=cutoff,
                        forecast_horizon=self.horizon,
                        stride=1,
                        retrain=False,
                        last_points_only=True,
                        verbose=False,
                        num_samples=self.num_samples,
                    )
                    vals = yhat.all_values().reshape(-1, self.num_samples)  # (1,S)
                    df = (
                        pd.DataFrame(vals)
                        .assign(Date=yhat.time_index, GID_2=gid)
                        .melt(
                            id_vars=["Date", "GID_2"],
                            var_name="sample",
                            value_name="prediction",
                        )
                    )
                    rows.append(df)
            pred_df = pd.concat(rows, ignore_index=True)

        else:
            raise ValueError(f"Unknown scale_strategy: {strategy}")

        # invert log if needed
        if self.target_col.lower().startswith("log"):
            pred_df["prediction"] = np.clip(np.expm1(pred_df["prediction"]), 0, None)
        return pred_df

    # ---------- final forecasts ----------
    def final_forecasts(self) -> pd.DataFrame:
        if not self._fitted:
            self.fit_global()

        rows = []
        for gid, ts in self._targets_by_gid.items():
            cov = self._covars_by_gid[gid]
            fut = self.model.predict(
                n=self.horizon,
                series=ts,
                past_covariates=cov,
                num_samples=self.num_samples,
                verbose=False,
            )
            vals = TimeSeries.all_values(fut)
            last_vals = vals[-1, 0, :]
            last_date = fut.time_index[-1]
            rows.append(
                pd.DataFrame(
                    {
                        "Date": [last_date] * self.num_samples,
                        "sample": np.arange(self.num_samples, dtype=int),
                        "prediction": last_vals,
                        "GID_2": gid,
                    }
                )
            )
        out = pd.concat(rows, ignore_index=True)
        if self.target_col.lower().startswith("log"):
            out["prediction"] = np.clip(np.expm1(out["prediction"]), 0, None)
        return out


def data_prep(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp = pd.Timestamp.min,
    end_date: str | pd.Timestamp = pd.Timestamp.max,
    gid_1: Optional[List[str]] = None,
    horizon: int = 1,
    pdfm_df: Optional[pd.DataFrame] = None,
    scale_strategy: ScaleStrategy = "prefix",  # default to leak-safe fast mode
    start: Optional[pd.Timestamp] = None,  # required for "prefix"
    train_length: Optional[int] = None,
) -> pd.DataFrame:
    if gid_1 is not None:
        df = df[df["GID_1"].isin(gid_1)]

    start_date = max(pd.to_datetime(start_date), df["Date"].min())
    end_date = min(pd.to_datetime(end_date), df["Date"].max())

    date_range = pd.date_range(start=start_date, end=end_date, freq="ME")
    date_range += pd.offsets.MonthEnd(0)
    df = df.copy()
    df.loc[:, "Date"] = pd.to_datetime(df["Date"]) + pd.offsets.MonthEnd(0)

    df = (
        df.groupby(["Date", "GID_2"])
        .agg({"Cases": "sum", "tmin": "mean", "prec": "mean"})
        .reset_index()
    )
    multi_index = pd.MultiIndex.from_product(
        [df["GID_2"].unique(), date_range], names=["GID_2", "Date"]
    )
    df = df.set_index(["GID_2", "Date"]).reindex(multi_index).reset_index()

    # target + covariates
    df["Cases"] = df["Cases"].fillna(0)
    df["Log_Cases"] = np.log1p(df["Cases"])
    covariates = ["LAG_1_LOG_CASES", "LAG_1_tmin_roll_2", "LAG_1_prec_roll_2"]
    df["LAG_1_LOG_CASES"] = df.groupby("GID_2")["Log_Cases"].shift(1).fillna(0)
    df["LAG_1_tmin_roll_2"] = (
        df.groupby("GID_2")["tmin"].shift(1).rolling(window=2).mean().fillna(0)
    )
    df["LAG_1_prec_roll_2"] = (
        df.groupby("GID_2")["prec"].shift(1).rolling(window=2).mean().fillna(0)
    )

    # float32
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].astype(np.float32)

    # optional embeddings
    if pdfm_df is not None:
        provinces = df["GID_2"].unique()
        pdfm_df = pdfm_df[pdfm_df["GID_2"].isin(provinces)].copy()
        pdfm_df.set_index("GID_2", inplace=True)
        feat_cols = [c for c in pdfm_df.columns if c.startswith("feature")]
        pdfm_df = pdfm_df[feat_cols]
        pdfm_df = pdfm_df[~pdfm_df.index.duplicated(keep="first")]
        df = df[df["GID_2"].isin(pdfm_df.index)]

    return df, covariates
