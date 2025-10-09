import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_pinball_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def weighted_pinball_loss(weights, predictions, truths, quantiles):
    weights = np.clip(weights, 0, 1)
    weights = weights / np.sum(weights)

    weighted_preds = np.tensordot(
        weights, predictions, axes=(0, 0)
    )  # [n_samples, n_quantiles]
    loss = 0.0
    for i, q in enumerate(quantiles):
        loss += mean_pinball_loss(truths[:, i], weighted_preds[:, i], alpha=q)
    return loss


def train_ensemble(df, quantiles):
    df = df.dropna(subset=["prediction", "Cases"])
    models = df["model"].unique()
    n_models = len(models)

    # Common GIDs are taken per Date
    common_gids = set(df["GID_2"].unique())
    for model in models:
        common_gids = common_gids.intersection(
            set(df[df["model"] == model]["GID_2"].unique())
        )
    common_gids = sorted(list(common_gids))
    df = df[df["GID_2"].isin(common_gids)].reset_index(drop=True)

    model_preds = []
    cases = []
    for ix, model in enumerate(models):
        model_data = df[df["model"] == model]
        model_preds.append(
            model_data.pivot(
                index=["GID_2", "Date"], columns="quantile", values="prediction"
            ).values
        )
        if ix == 0:
            cases = model_data.pivot(
                index=["GID_2", "Date"], columns="quantile", values="Cases"
            ).values
    try:
        model_preds = np.stack(
            model_preds, axis=0
        )  # [n_models, n_samples, n_quantiles]
    except ValueError:
        logging.warning(
            "Inconsistent prediction shapes among models. Ensure all models have predictions for the same set of (GID_2, Date) pairs and quantiles. Settings weights to equal."
        )
        return dict(zip(models, np.ones(n_models) / n_models))

    init_weights = np.ones(n_models) / n_models
    result = minimize(
        fun=weighted_pinball_loss,
        x0=init_weights,
        args=(model_preds, cases, quantiles),
        bounds=[(0, 1)] * n_models,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        method="SLSQP",
        options={"disp": False},
    )

    if not result.success:
        logger.warning("Optimization did not converge, using equal weights.")
        weights = init_weights
    else:
        weights = result.x

    return dict(zip(models, weights))


def apply_ensemble_weights(predictions_dict, weights_dict):
    models = list(predictions_dict.keys())
    weights = np.array([weights_dict.get(m, 0) for m in models])
    weights = (
        weights / weights.sum()
        if weights.sum() > 0
        else np.ones_like(weights) / len(weights)
    )
    model_preds = np.stack([predictions_dict[m] for m in models], axis=0)

    ensemble_pred = np.tensordot(weights, model_preds, axes=(0, 0))
    return ensemble_pred


def merge_model_dfs(model_tasks, model_names=None):
    """
    Merge list of model result dataframes and tag with model names.
    Assumes: each df has columns [Date, GID_2, quantile, prediction, Cases]
    """
    merged = []
    for i, df_model in enumerate(model_tasks):
        df = df_model.copy()
        model_name = model_names[i] if model_names else f"model_{i}"
        df["model"] = model_name
        merged.append(df)
    return pd.concat(merged, ignore_index=True)


def iterative_training(df, quantiles, testing_dates):
    all_weights = []

    for date in testing_dates:
        logger.info(f"Training ensemble weights for {date}")

        p = df[df["Date"] <= date]
        weights = train_ensemble(p, quantiles)
        weights["Date"] = date
        all_weights.append(weights)

    weights_df = pd.DataFrame(all_weights)
    return weights_df


def apply_weights_to_forecasts(df, weights_df, quantiles, lag=1):
    models = df["model"].unique()
    dfe = df[df["model"] == models[0]].copy()  # structure template
    dfe["model"] = "ensemble"

    # ensure Dates are comparable & sorted
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    weights_df = weights_df.copy()
    weights_df["Date"] = pd.to_datetime(weights_df["Date"])
    dates = sorted(df["Date"].unique())

    for i, date in enumerate(dates):
        # use lagged weights if requested
        w_idx = i - lag
        if w_idx < 0:
            continue
        w_date = dates[w_idx]

        logging.info(f"Applying weights for date {date} (weights from {w_date})")
        weight_row = weights_df[weights_df["Date"] == w_date]
        if weight_row.empty:
            logging.warning(f"No weights found for date {w_date}, skipping.")
            continue

        # reset ONLY the current date's predictions
        mask_date = dfe["Date"] == date
        dfe.loc[mask_date, "prediction"] = 0.0

        df_preds = []
        for model in models:
            if model not in weight_row.columns:
                logging.warning(f"Model {model} not in weights for {w_date}; weight=0.")
                weight = 0.0
            else:
                weight = float(weight_row.iloc[0][model])

            model_preds = df[(df["Date"] == date) & (df["model"] == model)][
                ["GID_2", "quantile", "prediction"]
            ].copy()
            if model_preds.empty:
                continue
            model_preds["prediction"] *= weight
            df_preds.append(model_preds)

        if not df_preds:
            continue

        df_preds = (
            pd.concat(df_preds, ignore_index=True)
            .groupby(["GID_2", "quantile"], as_index=False)["prediction"]
            .sum()
        )

        # merge back only for this date and assign explicitly
        dfe_date = (
            dfe.loc[mask_date, :]
            .drop(columns=["prediction"])
            .merge(df_preds, on=["GID_2", "quantile"], how="left")
        )
        dfe_date["prediction"] = dfe_date["prediction"].fillna(0.0)
        # write back to the same rows
        dfe.loc[mask_date, "prediction"] = dfe_date["prediction"].to_numpy()

    return dfe


def fix_quantile_violations(df, tolerance=1e-8):
    # Assumes df has columns: GID_2, Date, model, quantile, prediction
    df = df.sort_values(["GID_2", "Date", "quantile"])

    def fix_group(g):
        preds = g["prediction"].values.copy()
        for i in range(1, len(preds)):
            if preds[i] < preds[i - 1] and (preds[i - 1] - preds[i]) <= tolerance:
                preds[i] = preds[i - 1]
        g["prediction"] = preds
        return g

    df_fixed = (
        df.groupby(["GID_2", "Date"], group_keys=False)
        .apply(fix_group)
        .reset_index(drop=True)
    )
    return df_fixed


def create_ensemble(model_tasks, model_names=None, apply_quantile_fix=False):
    df_all = merge_model_dfs(model_tasks, model_names)

    # Determine common quantiles and dates
    quantiles = set(df_all["quantile"].unique())
    common_dates = set(df_all["Date"].unique())
    for model in df_all["model"].unique():
        quantiles = quantiles.intersection(
            set(df_all[df_all["model"] == model]["quantile"].unique())
        )
        common_dates = common_dates.intersection(
            set(df_all[df_all["model"] == model]["Date"].unique())
        )
    quantiles = sorted(list(quantiles))
    common_dates = sorted(list(common_dates))
    df_all = df_all[
        df_all["quantile"].isin(quantiles) & df_all["Date"].isin(common_dates)
    ].reset_index(drop=True)

    logging.info(f"Using quantiles: {quantiles}")
    testing_dates = sorted(df_all["Date"].unique())

    logging.info("Starting iterative training of ensemble weights...")
    weights_df = iterative_training(df_all, quantiles, testing_dates)
    logging.info(f"Ensemble weights:\n{weights_df}")

    logging.info("Applying ensemble weights to forecasts...")
    ensemble_df = apply_weights_to_forecasts(df_all, weights_df, quantiles)
    if apply_quantile_fix:
        logging.info("Fixing quantile violations in ensemble predictions...")
        ensemble_df = fix_quantile_violations(ensemble_df)

    logging.info("Ensemble creation complete.")
    return ensemble_df, weights_df
