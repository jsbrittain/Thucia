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


def train_ensemble(predictions_dict, true_values, quantiles):
    models = list(predictions_dict.keys())
    model_preds = np.stack(
        [predictions_dict[m] for m in models], axis=0
    )  # shape: (n_models, n_samples, n_quantiles)
    n_models = len(models)

    assert model_preds.shape[1:] == true_values.shape, (
        "Mismatch in prediction and truth shape"
    )

    init_weights = np.ones(n_models) / n_models

    result = minimize(
        fun=weighted_pinball_loss,
        x0=init_weights,
        args=(model_preds, true_values, quantiles),
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


def prepare_data_for_date(df, date, quantiles):
    subset = df[df["target_end_date"] == date]
    locations = subset["location"].unique()
    models = subset["model"].unique()

    true_values_list = []
    for loc in locations:
        tv = []
        for q in quantiles:
            tv_val = subset[(subset["location"] == loc) & (subset["quantile"] == q)][
                "true_value"
            ].unique()
            if len(tv_val) != 1:
                raise ValueError(
                    f"True value for location {loc}, quantile {q} on date {date} missing or duplicated"
                )
            tv.append(tv_val[0])
        true_values_list.append(tv)
    true_values = np.array(true_values_list)

    predictions_dict = {}
    for model in models:
        preds_list = []
        for loc in locations:
            pred_vals = []
            for q in quantiles:
                pred_val = subset[
                    (subset["location"] == loc)
                    & (subset["model"] == model)
                    & (subset["quantile"] == q)
                ]["prediction"].unique()
                if len(pred_val) != 1:
                    raise ValueError(
                        f"Prediction for model {model}, location {loc}, quantile {q} on date {date} missing or duplicated"
                    )
                pred_vals.append(pred_val[0])
            preds_list.append(pred_vals)
        predictions_dict[model] = np.array(preds_list)

    return predictions_dict, true_values, locations


def iterative_training(df, quantiles, testing_dates):
    all_weights = []
    historical_preds = pd.DataFrame()

    for date in testing_dates:
        logger.info(f"Training ensemble weights for {date}")
        current_preds = df[df["target_end_date"] == date]
        historical_preds = pd.concat(
            [historical_preds, current_preds], ignore_index=True
        )

        predictions_dict, true_values, _ = prepare_data_for_date(
            historical_preds, date, quantiles
        )
        weights = train_ensemble(predictions_dict, true_values, quantiles)
        weights["target_end_date"] = date
        all_weights.append(weights)

    weights_df = pd.DataFrame(all_weights)
    return weights_df


def apply_weights_to_forecasts(df, weights_df, quantiles, lag=1):
    ensemble_forecasts = []
    sorted_dates = sorted(df["target_end_date"].unique())
    weights_df = weights_df.sort_values("target_end_date")

    for i, date in enumerate(sorted_dates):
        lag_index = i - lag
        if lag_index < 0:
            # Use equal weights for initial dates if no lagged weights available
            weights_row = None
        else:
            lag_date = sorted_dates[lag_index]
            weights_row = weights_df[weights_df["target_end_date"] == lag_date]

        if weights_row is None or weights_row.empty:
            # Equal weights fallback
            model_names = df["model"].unique()
            n_models = len(model_names)
            weights = {m: 1 / n_models for m in model_names}
            logger.info(f"Using equal weights for date {date} (no lagged weights)")
        else:
            weights = weights_row.drop("target_end_date", axis=1).iloc[0].to_dict()

        preds_dict, _, locations = prepare_data_for_date(df, date, quantiles)
        ensemble_pred = apply_ensemble_weights(preds_dict, weights)

        for i_loc, loc in enumerate(locations):
            for i_q, q in enumerate(quantiles):
                ensemble_forecasts.append(
                    {
                        "location": loc,
                        "target_end_date": date,
                        "quantile": q,
                        "prediction": ensemble_pred[i_loc, i_q],
                        "model": "trained_ensemble",
                    }
                )

    return pd.DataFrame(ensemble_forecasts)


def fix_quantile_violations(df, tolerance=1e-8):
    # Assumes df has columns: location, target_end_date, model, quantile, prediction
    df = df.sort_values(["location", "target_end_date", "model", "quantile"])

    def fix_group(g):
        preds = g["prediction"].values.copy()
        for i in range(1, len(preds)):
            if preds[i] < preds[i - 1] and (preds[i - 1] - preds[i]) <= tolerance:
                preds[i] = preds[i - 1]
        g["prediction"] = preds
        return g

    df_fixed = (
        df.groupby(["location", "target_end_date", "model"])
        .apply(fix_group)
        .reset_index(drop=True)
    )
    return df_fixed


def example():
    # Create synthetic example data similar in structure to your R workflow
    quantiles = [0.1, 0.5, 0.9]
    dates = pd.date_range("2023-01-01", periods=5, freq="M")
    locations = ["A", "B"]
    models = ["tcn", "sarima", "timegpt"]

    rows = []
    np.random.seed(42)
    for date in dates:
        for loc in locations:
            true_val_median = np.random.poisson(lam=20)
            true_vals = [
                true_val_median - 3,
                true_val_median,
                true_val_median + 3,
            ]  # approximate quantiles
            for model in models:
                # Add some noise to model predictions around true values
                preds = np.array(true_vals) + np.random.normal(
                    0, 2, size=len(quantiles)
                )
                preds = np.clip(preds, 0, None)
                for q, pred, true_v in zip(quantiles, preds, true_vals):
                    rows.append(
                        {
                            "location": loc,
                            "target_end_date": date,
                            "model": model,
                            "quantile": q,
                            "prediction": pred,
                            "true_value": true_v,
                        }
                    )
    df = pd.DataFrame(rows)

    logger.info("Starting iterative ensemble training...")
    weights_df = iterative_training(
        df, quantiles, sorted(df["target_end_date"].unique())
    )
    logger.info("Weights training completed.")
    print(weights_df)

    logger.info("Applying ensemble weights to forecasts...")
    ensemble_df = apply_weights_to_forecasts(df, weights_df, quantiles)
    ensemble_df = fix_quantile_violations(ensemble_df)
    logger.info("Ensemble forecasts created.")
    print(ensemble_df.head(10))


if __name__ == "__main__":
    example()
