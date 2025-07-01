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


def train_ensemble(predictions_dict, cases, quantiles):
    models = list(predictions_dict.keys())
    model_preds = np.stack(
        [predictions_dict[m] for m in models], axis=0
    )  # shape: (n_models, n_samples, n_quantiles)
    n_models = len(models)

    assert model_preds.shape[1:] == cases.shape, (
        "Mismatch in prediction and truth shape"
    )

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


def prepare_data_for_date(df, date, quantiles, require_truth=True):
    subset = df[df["Date"] == date]
    GID_2s = subset["GID_2"].unique()
    models = subset["model"].unique()

    cases_list = []
    if require_truth:
        for loc in GID_2s:
            tv = []
            for q in quantiles:
                tv_val = (
                    subset[(subset["GID_2"] == loc) & (subset["quantile"] == q)][
                        "Cases"
                    ]
                    .dropna()
                    .unique()
                )
                if len(tv_val) != 1:
                    raise ValueError(
                        f"True value for GID_2 {loc}, quantile {q} on date {date} missing or duplicated. Found: {tv_val}"
                    )
                tv.append(tv_val[0])
            cases_list.append(tv)
        cases = np.array(cases_list)
    else:
        cases = None  # will be ignored downstream

    predictions_dict = {}
    for model in models:
        preds_list = []
        for loc in GID_2s:
            pred_vals = []
            for q in quantiles:
                pred_val = subset[
                    (subset["GID_2"] == loc)
                    & (subset["model"] == model)
                    & (subset["quantile"] == q)
                ]["prediction"].unique()
                if len(pred_val) != 1:
                    raise ValueError(
                        f"Prediction for model {model}, GID_2 {loc}, quantile {q} on date {date} missing or duplicated"
                    )
                pred_vals.append(pred_val[0])
            preds_list.append(pred_vals)
        predictions_dict[model] = np.array(preds_list)

    return predictions_dict, cases, GID_2s


def iterative_training(df, quantiles, testing_dates):
    all_weights = []
    historical_preds = pd.DataFrame()

    for date in testing_dates:
        logger.info(f"Training ensemble weights for {date}")
        current_preds = df[df["Date"] == date]
        current_preds = current_preds.dropna(subset=["prediction", "Cases"])

        historical_preds = pd.concat(
            [historical_preds, current_preds], ignore_index=True
        )
        training_data = historical_preds.dropna(subset=["prediction", "Cases"])

        predictions_dict, cases, _ = prepare_data_for_date(
            training_data, date, quantiles
        )

        if not predictions_dict:
            logger.warning(
                f"No valid model predictions available for date {date}. Skipping."
            )
            continue

        weights = train_ensemble(predictions_dict, cases, quantiles)
        weights["Date"] = date
        all_weights.append(weights)

    weights_df = pd.DataFrame(all_weights)
    return weights_df


def apply_weights_to_forecasts(df, weights_df, quantiles, lag=1):
    ensemble_forecasts = []
    sorted_dates = sorted(df["Date"].unique())
    weights_df = weights_df.sort_values("Date")

    for i, date in enumerate(sorted_dates):
        lag_index = i - lag
        if lag_index < 0:
            # Use equal weights for initial dates if no lagged weights available
            weights_row = None
        else:
            lag_date = sorted_dates[lag_index]
            weights_row = weights_df[weights_df["Date"] == lag_date]

        if weights_row is None or weights_row.empty:
            # Equal weights fallback
            model_names = df["model"].unique()
            n_models = len(model_names)
            weights = {m: 1 / n_models for m in model_names}
            logger.info(f"Using equal weights for date {date} (no lagged weights)")
        else:
            weights = weights_row.drop("Date", axis=1).iloc[0].to_dict()

        preds_dict, _, GID_2s = prepare_data_for_date(
            df, date, quantiles, require_truth=False
        )
        ensemble_pred = apply_ensemble_weights(preds_dict, weights)

        for i_loc, loc in enumerate(GID_2s):
            for i_q, q in enumerate(quantiles):
                ensemble_forecasts.append(
                    {
                        "GID_2": loc,
                        "Date": date,
                        "quantile": q,
                        "prediction": ensemble_pred[i_loc, i_q],
                    }
                )

    return pd.DataFrame(ensemble_forecasts)


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


def create_ensemble(model_tasks, model_names=None):
    df_all = merge_model_dfs(model_tasks, model_names)

    quantiles = sorted(df_all["quantile"].unique())
    testing_dates = sorted(df_all["Date"].unique())

    weights_df = iterative_training(df_all, quantiles, testing_dates)
    logging.info(f"Ensemble weights:\n{weights_df}")

    ensemble_df = apply_weights_to_forecasts(df_all, weights_df, quantiles)
    ensemble_df = fix_quantile_violations(ensemble_df)

    # Merge observed case data back in
    true_values = (
        df_all[["GID_2", "Date", "quantile", "Cases"]]
        .dropna(subset=["Cases"])
        .drop_duplicates(subset=["GID_2", "Date", "quantile"])
    )
    ensemble_df = ensemble_df.merge(
        true_values, on=["GID_2", "Date", "quantile"], how="left"
    )

    return ensemble_df, weights_df
