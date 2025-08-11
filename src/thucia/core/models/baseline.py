import logging
from typing import List

import numpy as np
import pandas as pd

from .utils import filter_admin1
from .utils import interpolate_missing_dates
from .utils import set_historical_na_to_zero


class BaselineSamples:
    def __init__(self, inc_diffs: List[float] | np.array, symmetrize: bool = True):
        self.symmetrize = symmetrize  # Whether to symmetrize the baseline about zero
        inc_diffs = np.array(inc_diffs)
        if symmetrize:
            # Concatenate positive and negative incidence differences to give mean zero
            # This removes directional bias from the predictions
            self.baseline = np.concatenate([inc_diffs, -inc_diffs])
        else:
            self.baseline = np.array(inc_diffs)

    def predict(self, last_level: float, horizon: int = 1, nsim: int = 5000):
        """

        Note that if you are using a symmetric baseline, horizon steps are independent,
        therefore if you are only interested in the n-step ahead predictor, it is
        more efficient to simply set horizon=1. This is not true for asymmetric
        baselines since these can drift over time, so steps are not independent.

        Params
        ------
        last_level: float
            The last known case count, used as the base for predictions.
        horizon: int
            The number of time points to predict into the future.
        nsim: int
            The number of samples to draw for each prediction.
        Returns
        -------
        result: np.array
            An array of shape (nsim, horizon) containing the predicted case counts.
        """
        result = np.full((nsim, horizon), np.nan)

        # Create evenly spaced quantile samples
        sampled_inc_diffs = np.quantile(self.baseline, np.linspace(0, 1, nsim))

        # Predictions are repeated for each step towards the horizon, using the median
        # predicted level at each step to the horizon.
        for h in range(horizon):
            sampled = np.random.choice(sampled_inc_diffs, size=nsim, replace=False)
            raw = last_level + sampled

            if self.symmetrize:
                # Maintain centeredness by subtracting the median of the raw predictions
                # This is necessary because the baseline has been symmetrized
                corrected = raw - (np.median(raw) - last_level)
            else:
                corrected = raw
                last_level = np.median(corrected)

            result[:, h] = corrected

        return result


def baseline(
    df: pd.DataFrame,  # DataFrame with columns: Date, GID_1, GID_2, Cases, future
    start_date: str | pd.Timestamp = pd.Timestamp.min,
    end_date: str | pd.Timestamp = pd.Timestamp.max,
    gid_1: list[str] | None = None,
    samples: int = 1000,
    # --- Model parameters ---
    step_ahead: int = 1,
    symmetrize: bool = True,
) -> (
    pd.DataFrame
):  # DataFrame with columns: GID_2, Date, sample, prediction, Cases, future
    logging.info("Starting baseline model...")

    df = df.copy()
    df = filter_admin1(df, gid_1)
    df = interpolate_missing_dates(df, start_date, end_date)
    df = set_historical_na_to_zero(df)

    # Loop over regions
    df_samples = []
    for gid2 in df["GID_2"].unique():
        logging.info(f"Processing region {gid2} for baseline model")
        # Filter data for the current region
        region_data = df[df["GID_2"] == gid2].set_index("Date")

        # Calculate incidence differences
        inc_diffs = region_data["Cases"].diff().fillna(0)  # First diff is always NaN

        # Historical forecasting
        for k in range(2 + step_ahead, len(region_data)):
            # Instantiate model
            model = BaselineSamples(
                inc_diffs[
                    1 : k - step_ahead
                ],  # skip first diff (NaN -> 0), and last diff
                symmetrize=symmetrize,
            )

            # If the previous case was a 'future' (prediction), use the median estimate
            # from the last level, otherwise use the actual case count
            if region_data["future"].iloc[k - step_ahead]:
                last_level = (
                    np.nanmedian(df_samples[-step_ahead]["prediction"])
                    if df_samples
                    else 0
                )
            else:
                last_level = region_data["Cases"].iloc[k - step_ahead]

            predictions = model.predict(
                last_level=last_level,
                # since we are only interested in the n-step ahead predictor, we can
                # shortcut the model by setting the horizon to 1 (in the case of
                # symmetrized baselines only), otherwise sample the full horizon...
                horizon=1 if symmetrize else step_ahead,
                nsim=samples,
            )[:, -1]  # ...taking only the furthest point as the n-step predictor
            predictions = np.clip(predictions, 0, None)

            df_samples.append(
                pd.DataFrame(
                    {
                        "GID_2": [gid2] * samples,
                        "Date": [region_data.index[k]] * samples,
                        "sample": list(range(samples)),
                        "prediction": predictions.tolist(),
                        "Cases": region_data["Cases"].iloc[k],  # actual case count
                    }
                )
            )
    df_predictions = pd.concat(df_samples, ignore_index=True)

    # Ensure all GID_2, Date combinations in df are present in df_predictions
    df_predictions = df_predictions.merge(
        df[["GID_2", "Date", "Cases", "future"]],
        on=["GID_2", "Date", "Cases"],
        how="right",
    )

    logging.info("Baseline model complete.")
    return df_predictions
