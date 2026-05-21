import numpy as np
import pandas as pd


def _enforce_monotone_quantiles(qs, preds, tol=1e-8):
    """
    Ensure non-decreasing quantile predictions (in-place like the R fix).
    If a tiny violation <= tol is found, lift the current value up to the previous.
    Larger violations are also corrected (conservative).
    """
    preds = preds.copy()
    for i in range(1, len(preds)):
        if preds[i] + tol < preds[i - 1]:
            preds[i] = preds[i - 1]
    return preds


def _bracher_wis_from_quantiles(y, q_levels, q_preds):
    """
    Bracher et al. (2021) weighted WIS given:
      - y: scalar observation
      - q_levels: sorted array of quantile probs in (0,1)
      - q_preds: corresponding predictions (same order)
    Weights: 0.5 for median (q=0.5); alpha/2 for each central interval (q,1-q), alpha=2q.
    Normalisation: divide by (0.5 + sum(alpha/2)).
    """
    # Build intervals from symmetric pairs around 0.5
    q_levels = np.asarray(q_levels, dtype=float)
    q_preds = np.asarray(q_preds, dtype=float)
    # median
    if 0.5 in q_levels:
        m = q_preds[q_levels.tolist().index(0.5)]
        median_term = 0.5 * abs(y - m)
        median_weight = 0.5
    else:
        # linear interpolate median if missing
        lower_mask = q_levels < 0.5
        upper_mask = q_levels > 0.5
        if not lower_mask.any() or not upper_mask.any():
            return np.nan
        ql = q_levels[lower_mask].max()
        qu = q_levels[upper_mask].min()
        pl = q_preds[q_levels.tolist().index(ql)]
        pu = q_preds[q_levels.tolist().index(qu)]
        m = np.interp(0.5, [ql, qu], [pl, pu])
        median_term = 0.5 * abs(y - m)
        median_weight = 0.5

    # interval components
    interval_term_sum = 0.0
    weight_sum = median_weight
    # use only q<0.5 that have symmetric (1-q)
    for q in q_levels[(q_levels > 0) & (q_levels < 0.5)]:
        q_comp = 1.0 - q
        if q_comp not in q_levels:
            continue
        l = q_preds[q_levels.tolist().index(q)]  # noqa: E741
        u = q_preds[q_levels.tolist().index(q_comp)]
        alpha = 2.0 * q
        # interval score IS_alpha
        under = max(0.0, l - y)
        over = max(0.0, y - u)
        IS = (u - l) + (2.0 / alpha) * (under + over)
        weight = alpha / 2.0
        interval_term_sum += weight * IS
        weight_sum += weight

    if weight_sum == 0:
        return np.nan
    return (median_term + interval_term_sum) / weight_sum


def wis_bracher(
    df: pd.DataFrame,
    group_cols=("GID_2", "Date"),
    quantile_col="quantile",
    pred_col="prediction",
    obs_col="Cases",
    log1p_scale=False,
    clamp_negative_to_zero=True,
    monotonic_fix=True,
):
    """
    Compute Bracher-weighted WIS per group, matching scoringutils::score() for quantile forecasts.
    If a 'model' column exists, include it in group_cols to score per model.
    """
    df = df.copy()
    # basic hygiene
    for c in [quantile_col, pred_col, obs_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # clamp negatives (on raw scale) if desired
    if clamp_negative_to_zero and not log1p_scale:
        df.loc[df[pred_col] < 0, pred_col] = 0.0

    # score per group
    out_rows = []
    gcols = list(group_cols)

    # Index into group_cols, returning index (keys) and group (g) as DataFrame
    for keys, g in df.groupby(gcols, sort=True):
        # Cases (assumed unique in group)
        y = g.iloc[0][obs_col]
        # predictions and levels
        q = g[quantile_col].to_numpy()
        p = g[pred_col].to_numpy()
        # sort by quantile
        order = np.argsort(q)
        q = q[order]
        p = p[order]

        # apply transform consistently
        if log1p_scale:
            y_sc = np.log1p(y)
            # predictions are assumed on count scale; transform for scoring
            p_sc = np.log1p(np.clip(p, a_min=0.0, a_max=None))
        else:
            y_sc = y
            p_sc = p

        # enforce non-decreasing quantile predictions
        if monotonic_fix:
            p_sc = _enforce_monotone_quantiles(q, p_sc, tol=1e-8)

        wis = _bracher_wis_from_quantiles(y_sc, q, p_sc)
        row = dict(zip(gcols, keys if isinstance(keys, tuple) else (keys,)))
        row.update(
            {
                "WIS": wis.astype(float),
                obs_col: y_sc.astype(float),
            }
        )
        out_rows.append(row)

    return pd.DataFrame(out_rows)
