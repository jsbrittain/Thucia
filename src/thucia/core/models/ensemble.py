import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_pinball_loss


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
    # predictions_dict: dict of {model_name: np.ndarray of shape (n_samples, n_quantiles)}
    models = list(predictions_dict.keys())
    model_preds = np.stack(
        [predictions_dict[m] for m in models], axis=0
    )  # shape: (n_models, n_samples, n_quantiles)
    n_models = len(models)

    # Ensure shapes match
    assert model_preds.shape[1:] == true_values.shape, (
        "Mismatch in prediction and truth shape"
    )

    # Initialize equal weights
    init_weights = np.ones(n_models) / n_models

    # Optimize
    result = minimize(
        fun=weighted_pinball_loss,
        x0=init_weights,
        args=(model_preds, true_values, quantiles),
        bounds=[(0, 1)] * n_models,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        method="SLSQP",
    )

    return dict(zip(models, result.x))


def apply_ensemble_weights(predictions_dict, weights_dict):
    models = list(predictions_dict.keys())
    weights = np.array([weights_dict[m] for m in models])
    model_preds = np.stack([predictions_dict[m] for m in models], axis=0)

    # Weighted sum
    ensemble_pred = np.tensordot(weights, model_preds, axes=(0, 0))
    return ensemble_pred


def example():
    # Suppose you have predictions and truths for one date
    quantiles = [0.1, 0.5, 0.9]
    true_vals = np.array([[12, 14, 17]])  # shape: (n_samples, n_quantiles)

    predictions = {
        "tcn": np.array([[10, 13, 16]]),
        "sarima": np.array([[11, 14, 18]]),
        "timegpt": np.array([[12, 15, 20]]),
    }

    # Train ensemble
    weights = train_ensemble(predictions, true_vals, quantiles)
    print("Optimal Weights:", weights)

    # Apply ensemble
    ensemble_pred = apply_ensemble_weights(predictions, weights)
    print("Ensemble Prediction:", ensemble_pred)
