import logging
from collections import deque
from typing import List
from typing import Sequence

import numpy as np
import pandas as pd


def add_residual_quantiles(
    df: pd.DataFrame,
    *,
    date_col: str = "Date",
    gid_col: str = "GID_2",
    y_col: str = "Cases",
    pred_col: str = "prediction",
    horizon_col: str = "horizon",
    quantile_levels: Sequence[float] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
    window: int = None,  # None => expanding; integer => moving window size
    min_history: int = 20,
    pool_across_gids_if_sparse: bool = True,
) -> pd.DataFrame:
    """
    Post-hoc predictive distribution from deterministic forecasts via residual quantiles.
    Returns a long dataframe with columns:
      [Date, GID_2, horizon, quantile, prediction]
    """
    if horizon_col not in df.columns:
        # df = df.copy()
        df[horizon_col] = 1
    else:
        # df = df.copy()
        ...

    # Defensive parsing; invalid parse -> NaT (we'll drop those)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # If your point predictions have NaNs, we can’t recover—surface early.
    if df[pred_col].isna().all():
        # Nothing to do—return empty result to make the problem obvious.
        return pd.DataFrame(
            columns=[date_col, gid_col, horizon_col, "quantile", pred_col]
        )

    # Keep only rows with valid dates and predictions
    df = df[~df[date_col].isna()]  # .copy()

    for h in sorted(df[horizon_col].unique()):
        logging.info(f"Processing horizon {h}")
        out_records = []

        dfh = df[df[horizon_col] == h].copy()
        # Sort deterministically for causal pass
        dfh = dfh.sort_values([date_col, gid_col], kind="mergesort")

        hist_by_gid: dict[str, deque] = {}
        pooled_hist: deque = deque()

        dates = dfh[date_col].values
        unique_dates = np.unique(dates)

        for d in unique_dates:
            day_mask = dates == d
            df_apply = dfh.loc[day_mask]

            preds_today = df_apply[pred_col].to_numpy(dtype=float)
            gids_today: List[str] = df_apply[gid_col].astype(str).tolist()

            # ---- APPLY STEP (use only past residuals; history buffers track that) ----
            for i, g in enumerate(gids_today):
                # fallback is the point prediction (never NaN if input isn’t)
                base = preds_today[i]

                ghist = hist_by_gid.get(g, deque())
                if len(ghist) >= min_history:
                    resids = np.fromiter(ghist, dtype=float)
                elif pool_across_gids_if_sparse and len(pooled_hist) >= min_history:
                    resids = np.fromiter(pooled_hist, dtype=float)
                else:
                    resids = None

                if resids is not None and resids.size > 0:
                    # quantiles of residuals + base prediction
                    qs = np.quantile(resids, quantile_levels, method="linear")
                    preds_q = base + qs
                else:
                    # not enough history yet -> use the base prediction at all quantiles
                    preds_q = np.full(len(quantile_levels), base, dtype=float)

                for q, v in zip(quantile_levels, preds_q):
                    out_records.append(
                        {
                            date_col: pd.to_datetime(d),
                            gid_col: g,
                            horizon_col: h,
                            "quantile": float(q),
                            pred_col: float(v),
                            y_col: df_apply[y_col].iloc[i],
                        }
                    )

            # ---- UPDATE STEP: after applying to day d, reveal truth at d and update histories ----
            if y_col in df_apply.columns:
                # If y is missing for some rows, skip those residuals
                y_vals = df_apply[y_col].to_numpy(dtype=float)
                valid_mask = ~np.isnan(y_vals) & ~np.isnan(preds_today)
                for g, r in zip(
                    np.array(gids_today)[valid_mask], (y_vals - preds_today)[valid_mask]
                ):
                    qbuf = hist_by_gid.get(g)
                    if qbuf is None:
                        qbuf = deque()
                        hist_by_gid[g] = qbuf
                    qbuf.append(float(r))
                    pooled_hist.append(float(r))
                    if window is not None:
                        while len(qbuf) > window:
                            qbuf.popleft()
                        while len(pooled_hist) > window:
                            pooled_hist.popleft()

        out_df = pd.DataFrame.from_records(out_records)

        from thucia.core.cases import write_nc

        write_nc(out_df, f"residual_quantiles_h{h}.nc")

    # Enforce nondecreasing quantiles within each (Date,GID,horizon)
    # def _enforce_monotone(group):
    #     group = group.sort_values("quantile").copy()
    #     group["prediction"] = np.maximum.accumulate(group["prediction"].to_numpy())
    #     return group
    # if not out_df.empty:
    #     out_df = (out_df.groupby([date_col, gid_col, horizon_col], group_keys=False)
    #                      .apply(_enforce_monotone)
    #                      .reset_index(drop=True))

    return out_df
