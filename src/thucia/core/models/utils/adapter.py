import logging
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def _ensure_gid_index(
    predictors_df: pd.DataFrame, gid_col: str = "GID_2"
) -> pd.DataFrame:
    if predictors_df.index.name == gid_col:
        return predictors_df
    if gid_col in predictors_df.columns:
        return predictors_df.set_index(gid_col)
    raise ValueError(f"predictors_df must have index or column named '{gid_col}'.")


def _align_embeddings(
    predictors_df: pd.DataFrame, gids: List[str]
) -> Tuple[np.ndarray, List[str]]:
    missing = [g for g in gids if g not in predictors_df.index]
    if missing:
        raise ValueError(
            f"Embeddings missing for {len(missing)} gid(s). First few: {missing[:5]}"
        )
    M = predictors_df.reindex(gids)
    return M.to_numpy(dtype=np.float32), list(M.columns)


def _prepare_fit_table(
    dfm: pd.DataFrame,
    use_cols: Tuple[str, str, str] = ("Date", "GID_2", "prediction"),
    y_col: str = "Cases",
    train_mask: Optional[pd.Series] = None,
    cutoff_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    date_col, gid_col, yhat_col = use_cols
    if train_mask is not None:
        dfm = dfm[train_mask.values]
    if cutoff_date is not None:
        dfm = dfm[dfm[date_col] <= pd.to_datetime(cutoff_date)]
    if dfm.empty:
        raise ValueError("No rows available to fit adapter after masking/cutoff.")
    dfm["residual"] = dfm[y_col].astype(np.float32) - dfm[yhat_col].astype(np.float32)
    return dfm


# ---------- Base ----------


@dataclass
class _EmbeddingSpace:
    X: np.ndarray  # (N, D) embeddings aligned to gid_order
    gid_order: List[str]  # length N


class AdapterBase:
    """
    Base class. Subclasses implement _fit_impl(X, y) and _predict_impl(X) for residuals.
    """

    def __init__(
        self,
        predictors_df: pd.DataFrame,
        gid_col: str = "GID_2",
        standardize_y: bool = False,
    ):
        self.predictors_raw = _ensure_gid_index(predictors_df, gid_col=gid_col)
        self.gid_col = gid_col
        self.standardize_y = standardize_y

        self._space: Optional[_EmbeddingSpace] = None
        self._y_mean: Optional[float] = None
        self._y_std: Optional[float] = None
        self._fitted: bool = False

    # ---- public API ----
    def fit(
        self,
        df: pd.DataFrame,
        use_cols: Tuple[str, str, str] = ("Date", "GID_2", "prediction"),
        y_col: str = "Cases",
        train_mask: Optional[pd.Series] = None,
        cutoff_date: Optional[pd.Timestamp] = None,
        transform=None,
    ):
        if transform:
            df[y_col] = transform(df[y_col])
            df["prediction"] = transform(df["prediction"])

        dfm = _prepare_fit_table(df, use_cols, y_col, train_mask, cutoff_date)
        gid_order = sorted(dfm[self.gid_col].unique().tolist())
        X_raw, _ = _align_embeddings(self.predictors_raw, gid_order)

        X = X_raw

        # one row per timepoint: replicate per gid embedding to match residual rows
        gid_to_idx = {g: i for i, g in enumerate(gid_order)}
        idxs = dfm[self.gid_col].map(gid_to_idx).to_numpy()
        X_fit = X[idxs]  # (T_total, D)
        y = dfm["residual"].to_numpy(np.float32)  # (T_total,)

        if self.standardize_y:
            self._y_mean = float(y.mean())
            self._y_std = float(y.std() + 1e-8)
            y_std = (y - self._y_mean) / self._y_std
        else:
            y_std = y

        self._space = _EmbeddingSpace(X=X, gid_order=gid_order)
        self._fit_impl(X_fit, y_std)  # <- subclass
        self._fitted = True

        X_fit = X_fit[~np.isnan(y_std),]
        y_std = y_std[~np.isnan(y_std)]

        # quick train fit score (in-sample)
        # pred_train = self._predict_impl(X_fit)
        # r2 = 1.0 - np.sum((y_std - pred_train) ** 2) / np.sum(
        #     (y_std - y_std.mean()) ** 2
        # )
        # logging.info("[Adapter] In-sample R^2 on residuals: %.3f", r2)

    def bias_for_gid(self, gid: str) -> float:
        if not self._fitted or self._space is None:
            return 0.0
        try:
            i = self._space.gid_order.index(gid)
        except ValueError:
            raise ValueError(
                f"GID_2={gid} unknown to adapter; was it included during fit?"
            )
        x = self._space.X[i : i + 1]  # (1,D)
        pred = self._predict_impl(x)[0]
        if self.standardize_y:
            pred = pred * (self._y_std or 1.0) + (self._y_mean or 0.0)
        return float(pred)

    def apply(
        self,
        pred_df: pd.DataFrame,
        date_col: str = "Date",
        gid_col: str = "GID_2",
        pred_col: str = "prediction",
        out_col: str = "prediction",
    ) -> pd.DataFrame:
        """
        Adds per-GID bias to ALL rows (works for per-sample long DataFrames too).
        """
        if not self._fitted:
            logging.warning(
                "Adapter not fitted; returning input unchanged with copied column."
            )
            out = pred_df
            out[out_col] = out[pred_col]
            return out

        # vectorized: map gid->bias
        map_bias = {g: self.bias_for_gid(g) for g in self._space.gid_order}
        out = pred_df
        out[out_col] = out[pred_col] + out[gid_col].map(map_bias).astype(np.float32)
        return out

    # ---- subclass hooks ----
    def _fit_impl(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ---------- Ridge adapter (simple & strong baseline) ----------


class RidgeAdapter(AdapterBase):
    def __init__(self, *args, alpha: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self._ridge = None

    def _fit_impl(self, X: np.ndarray, y: np.ndarray):
        X = X[~np.isnan(y),]
        y = y[~np.isnan(y)]

        self._ridge = Ridge(alpha=self.alpha, fit_intercept=True, random_state=0)
        self._ridge.fit(X, y)

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        return self._ridge.predict(X).astype(np.float32)


# ---------- Tiny MLP adapter (optional) ----------

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    nn = None


class MLPAdapter(AdapterBase):
    def __init__(
        self,
        *args,
        hidden: int = 64,
        l2: float = 1e-4,
        epochs: int = 200,
        lr: float = 1e-3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hidden = hidden
        self.l2 = l2
        self.epochs = epochs
        self.lr = lr
        self._mlp = None
        self._in_dim = None

    def _build_model(self, in_dim: int):
        m = nn.Sequential(
            nn.Linear(in_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )
        for p in m.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return m

    def _fit_impl(self, X: np.ndarray, y: np.ndarray):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for MLPAdapter.")
        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

        self._in_dim = X.shape[1]
        self._mlp = self._build_model(self._in_dim)
        opt = torch.optim.Adam(self._mlp.parameters(), lr=self.lr, weight_decay=self.l2)
        loss_fn = nn.MSELoss()

        self._mlp.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            pred = self._mlp(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            opt.step()

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        self._mlp.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X.astype(np.float32))
            out = self._mlp(X_t).squeeze(1).cpu().numpy().astype(np.float32)
        return out


def residual_regression(df_model, df_predictors, method):
    # Prepare embeddings
    if df_predictors is None:
        logging.warning("Model predictors not provided. Skipping adapter.")
        return df_model

    provinces = df_model["GID_2"].unique()
    df_predictors = df_predictors[df_predictors["GID_2"].isin(provinces)]
    df_predictors.set_index("GID_2", inplace=True)
    feature_cols = [c for c in df_predictors.columns if c.startswith("feature")]
    df_predictors = df_predictors[feature_cols]
    assert set(df_predictors.index.unique()) == set(df_model["GID_2"]), (
        "Model and embeddings must contain the same geographic codes."
    )

    if method == "ridge":
        adapter = RidgeAdapter(
            predictors_df=df_predictors,
            standardize_y=False,
            alpha=2.0,
        )
    elif method == "mlp":
        adapter = MLPAdapter(
            predictors_df=df_predictors,
            standardize_y=False,
        )
    else:
        raise Exception(f"Unrecognised method requested: {method}")

    log_transform = True
    df_work = df_model.copy()
    if log_transform:
        df_work["Cases"] = np.log1p(df_work["Cases"])
        df_work["prediction"] = np.log1p(df_work["prediction"])

    # Ensure a horizon column exists (kept from original behavior)
    if "horizon" not in df_work.columns:
        df_work["horizon"] = 1
    if "quantile" not in df_work.columns:
        df_work["quantile"] = 0.5

    collated = []
    for horizon in df_work["horizon"].unique():
        logging.info(f"Fitting adapter (expanding window) for horizon {horizon}")
        dfh_allq = df_work[df_work["horizon"] == horizon]
        for q in sorted(dfh_allq["quantile"].unique()):
            dfh = dfh_allq[dfh_allq["quantile"] == q].copy()
            # Expanding window per date (strictly causal)
            out_slices = []
            # Convert once to Timestamps for robust comparisons
            dates = pd.to_datetime(dfh["Date"])
            unique_dates = np.sort(dates.unique())

            for d in unique_dates:
                mask_fit = dates < d
                mask_apply = dates == d

                df_fit = dfh.loc[mask_fit].copy()
                df_apply = dfh.loc[mask_apply].copy()

                if df_fit.empty:
                    # Nothing to fit yet; keep predictions as-is for the first date
                    out_slices.append(df_apply)
                    continue

                try:
                    adapter.fit(df=df_fit, transform=None)
                except ValueError as e:
                    logging.warning(f"Adapter failed to fit at date {d}: {e}")
                    out_slices.append(df_apply)
                    continue

                pred_df_corrected = adapter.apply(df_apply, out_col="prediction")
                df_apply.loc[:, "prediction"] = pred_df_corrected["prediction"]
                out_slices.append(df_apply)

            if out_slices:
                df_out = pd.concat(out_slices, axis=0)
                collated.append(df_out)

    df_out_all = pd.concat(collated, axis=0)

    if log_transform:
        df_out_all["Cases"] = np.expm1(df_out_all["Cases"])
        df_out_all["prediction"] = np.expm1(df_out_all["prediction"])

    # Return in the original row order
    # df_out_all = df_out_all.loc[df_model.index]

    # Sort
    df_out_all = df_out_all.sort_values(
        by=["Date", "GID_2", "horizon", "quantile"]
    ).reset_index(drop=True)

    return df_out_all
