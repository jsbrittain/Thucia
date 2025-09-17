import logging
import os
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.utils.likelihood_models import GaussianLikelihood  # or QuantileRegression

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None

torch.set_default_dtype(torch.float32)
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))  # safe


class TFTSamples:
    def __init__(
        self,
        timeseries_df: pd.DataFrame,
        target_col: str,
        covariate_cols: List[str],
        horizon: int = 1,
        withheld_months: int = 12,
        num_samples: int = 1000,
        input_chunk_length: int = 48,
        verbose: bool = True,
        pdfm_df: Optional[pd.DataFrame] = None,
        pdfm_pca_dim: Optional[int] = None,
    ):
        self.df = timeseries_df.copy()
        self.target_col = target_col
        self.covariate_cols = covariate_cols
        self.horizon = horizon
        self.withheld_months = withheld_months
        self.num_samples = num_samples
        self.input_chunk_length = input_chunk_length
        self.verbose = verbose
        self.pdfm_df = pdfm_df.copy() if pdfm_df is not None else None
        self.pdfm_pca_dim = pdfm_pca_dim
        self.embed_prefix = "feature"

        # global scalers (fit across list of series)
        self.covar_scaler = Scaler()
        self.target_scaler = None  # optional

        # ---- TFT model (torch forecasting model) ----
        # Notes:
        # - probabilistic by default; keep a likelihood to enable sampling
        # - we only feed past_covariates here, matching your existing pipeline
        self.model = TFTModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=1,
            hidden_size=16,
            lstm_layers=1,
            num_attention_heads=4,
            dropout=0.2,  # MC dropout adds stochasticity as well
            likelihood=GaussianLikelihood(),
            random_state=42,
            n_epochs=150,
            batch_size=64,
            force_reset=True,
            add_relative_index=True,  # gives a simple future encoder even without explicit future covs
        )

        self._fitted = False
        self._targets_by_gid: Dict[str, TimeSeries] = {}
        self._covars_by_gid: Dict[str, TimeSeries] = {}

    # ---- single builder (the old_working version is removed per request) ----
    def _build_series_lists(
        self,
    ) -> Tuple[List[TimeSeries], List[TimeSeries], List[str]]:
        targets, covars, gids = [], [], []
        embed_cols = None

        # Precompute PCA once if requested
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

        for gid, gdf in self.df.groupby("GID_2"):
            gdf = gdf.sort_values("Date").copy()
            gdf_train = (
                gdf.iloc[: -self.withheld_months] if self.withheld_months > 0 else gdf
            )

            # target
            ts = TimeSeries.from_dataframe(
                gdf_train,
                time_col="Date",
                value_cols=[self.target_col],
                fill_missing_dates=True,
                freq="ME",
            ).astype(np.float32)

            # base covariates
            cov = TimeSeries.from_dataframe(
                gdf_train,
                time_col="Date",
                value_cols=self.covariate_cols,
                fill_missing_dates=True,
                freq="ME",
            ).astype(np.float32)

            # add constant embeddings as extra covariate channels
            E = self._get_embedding_vec(gid)
            if E is not None:
                if embed_cols is None:
                    D = int(E.shape[0])
                    embed_cols = [f"{self.embed_prefix}{j}" for j in range(D)]
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

                # store FULL series (incl. withheld) for backtests
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
                raise ImportError("scikit-learn is required for pdfm_pca_dim.")
            if not hasattr(self, "_pca_fitted"):
                allE = pdfm.to_numpy(dtype=np.float32)
                self._pca = PCA(n_components=self.pdfm_pca_dim, random_state=0).fit(
                    allE
                )
                self._pca_fitted = True
            E = self._pca.transform(E).astype(np.float32)
        return E.squeeze(0)

    def fit_global(self):
        target_list, covar_list, gids = self._build_series_lists()
        if not target_list:
            raise ValueError("No series have enough length to train.")

        # Optionally scale target
        # self.target_scaler = Scaler()
        # target_list = self.target_scaler.fit_transform(target_list)

        # global fit of covariate scaler (recommended for pooled training)
        covar_list = self.covar_scaler.fit_transform(covar_list)

        self.model.fit(
            series=target_list,
            past_covariates=covar_list,
            verbose=self.verbose,
        )

        # transform all stored FULL covariates once
        cov_full_list = [self._covars_by_gid[g] for g in gids]
        cov_full_list = self.covar_scaler.transform(cov_full_list)
        self._covars_by_gid = {g: cov for g, cov in zip(gids, cov_full_list)}

        self._fitted = True

    def historical_forecasts_pooled(self) -> pd.DataFrame:
        if not self._fitted:
            self.fit_global()

        rows = []
        for gid, ts in self._targets_by_gid.items():
            cov = self._covars_by_gid[gid]

            bt = self.model.historical_forecasts(
                series=ts,
                past_covariates=cov,  # already scaled
                forecast_horizon=self.horizon,
                stride=1,
                retrain=False,
                last_points_only=True,
                verbose=False,
                num_samples=self.num_samples,  # sampling thanks to likelihood
            )

            vals = TimeSeries.all_values(bt)  # (T_bt, 1, S)
            T_bt, _, S = vals.shape
            vals = vals.reshape(T_bt, S)
            dates = bt.time_index

            out = pd.DataFrame(vals)
            out["Date"] = dates
            out = out.melt(id_vars="Date", var_name="sample", value_name="prediction")
            out["GID_2"] = gid
            rows.append(out)

        pred_df = pd.concat(rows, ignore_index=True)
        if self.target_col.lower().startswith("log"):
            pred_df["prediction"] = np.clip(
                np.expm1(pred_df["prediction"]), a_min=0, a_max=None
            )
        return pred_df

    def final_forecasts(self) -> pd.DataFrame:
        """Samples only for the final horizon point per province."""
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
            vals = TimeSeries.all_values(fut)  # (H, 1, S)
            last_vals = vals[-1, 0, :]
            last_date = fut.time_index[-1]
            out = pd.DataFrame(
                {
                    "Date": [last_date] * self.num_samples,
                    "sample": np.arange(self.num_samples, dtype=int),
                    "prediction": last_vals,
                    "GID_2": gid,
                }
            )
            rows.append(out)

        pred_df = pd.concat(rows, ignore_index=True)
        if self.target_col.lower().startswith("log"):
            pred_df["prediction"] = np.clip(np.expm1(pred_df["prediction"]), 0, None)
        return pred_df


def extract_oos_for_adapter_from_model(
    model_obj,
    horizon: int = 1,
    train_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build OOS residual training tables from a fitted pooled model (TFTSamples).
    Returns:
        pred_train_df: Date, GID_2, y_pred  (same scale as model_obj.target_col)
        y_df:          Date, GID_2, y_true  (same scale as model_obj.target_col)
    """
    if not getattr(model_obj, "_fitted", False):
        model_obj.fit_global()

    rows_pred, rows_true = [], []

    for gid, ts in model_obj._targets_by_gid.items():
        cov = model_obj._covars_by_gid[gid]
        bt = model_obj.model.historical_forecasts(
            series=ts,
            past_covariates=cov,
            forecast_horizon=horizon,
            stride=1,
            retrain=False,
            last_points_only=True,
            verbose=False,
            num_samples=1,  # mean pred for residual fitting
        )

        truth = ts.slice_intersect(bt)
        yhat = bt.all_values().squeeze(axis=1).squeeze(axis=-1)
        ytru = truth.all_values().squeeze(axis=1)
        dates = bt.time_index

        if train_only and model_obj.withheld_months > 0:
            cutoff = ts.time_index[-model_obj.withheld_months]
            keep = dates < cutoff
            yhat, ytru, dates = yhat[keep], ytru[keep], dates[keep]

        if len(dates) == 0:
            continue

        rows_pred.append(pd.DataFrame({"Date": dates, "GID_2": gid, "y_pred": yhat}))
        rows_true.append(pd.DataFrame({"Date": dates, "GID_2": gid, "y_true": ytru}))

    pred_train_df = (
        pd.concat(rows_pred, ignore_index=True)
        if rows_pred
        else pd.DataFrame(columns=["Date", "GID_2", "y_pred"])
    )
    y_df = (
        pd.concat(rows_true, ignore_index=True)
        if rows_true
        else pd.DataFrame(columns=["Date", "GID_2", "y_true"])
    )

    pred_train_df["Date"] = pd.to_datetime(pred_train_df["Date"])
    y_df["Date"] = pd.to_datetime(y_df["Date"])
    pred_train_df["GID_2"] = pred_train_df["GID_2"].astype(str)
    y_df["GID_2"] = y_df["GID_2"].astype(str)
    return pred_train_df, y_df


def tft(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp = pd.Timestamp.min,
    end_date: str | pd.Timestamp = pd.Timestamp.max,
    gid_1: list[str] | None = None,
    horizon: int = 1,
    pdfm_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    logging.info("Starting TFT forecasting pipeline...")

    if gid_1 is not None:
        df = df[df["GID_1"].isin(gid_1)]

    start_date = max(pd.to_datetime(start_date), df["Date"].min())
    end_date = min(pd.to_datetime(end_date), df["Date"].max())

    date_range = pd.date_range(start=start_date, end=end_date, freq="ME")
    date_range += pd.offsets.MonthEnd(0)
    df.loc[:, "Date"] = pd.to_datetime(df["Date"]) + pd.offsets.MonthEnd(0)

    df = (
        df.groupby(["Date", "GID_2"])
        .agg(
            {
                "Cases": "sum",
                "tmin": "mean",
                "prec": "mean",
            }
        )
        .reset_index()
    )
    multi_index = pd.MultiIndex.from_product(
        [df["GID_2"].unique(), date_range], names=["GID_2", "Date"]
    )
    df = df.set_index(["GID_2", "Date"]).reindex(multi_index).reset_index()

    # target
    df["Cases"] = df["Cases"].fillna(0)
    df["Log_Cases"] = np.log1p(df["Cases"])

    # past covariates (TFT also supports future_covariates; we keep past-only here)
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

    # optional embeddings (constant features)
    if pdfm_df is not None:
        provinces = df["GID_2"].unique()
        pdfm_df = pdfm_df[pdfm_df["GID_2"].isin(provinces)].copy()
        pdfm_df.set_index("GID_2", inplace=True)
        feature_cols = [c for c in pdfm_df.columns if c.startswith("feature")]
        pdfm_df = pdfm_df[feature_cols]
        pdfm_df = pdfm_df[~pdfm_df.index.duplicated(keep="first")]
        df = df[df["GID_2"].isin(pdfm_df.index)]

    tft_obj = TFTSamples(
        timeseries_df=df,
        target_col="Log_Cases",
        covariate_cols=covariates,
        horizon=horizon,
        withheld_months=12,
        num_samples=1000,
        pdfm_df=pdfm_df,  # embeddings appended as channels in past_covariates
        # pdfm_pca_dim=5,
    )

    predictions = tft_obj.historical_forecasts_pooled()
    merged = df.merge(predictions, on=["Date", "GID_2"], how="left")

    logging.info("TFT forecasting complete.")
    return merged
