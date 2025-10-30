import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from thucia.core.cases import align_date_types
from thucia.core.fs import DataFrame


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
    df: pd.DataFrame,
    start_date: str | pd.Timestamp = pd.Timestamp.min,
    end_date: str | pd.Timestamp = pd.Timestamp.max,
    gid_1: List[str] | None = None,
    horizon: int = 1,
    case_col: str = "Log_Cases",
    covariate_cols: list[str] | None = None,
    retrain: bool = True,  # Retrain after every step (accurate but slow)
    db_file: str | Path | None = None,
    model_admin_level: int = 0,  # Admin level for model training
    num_samples: int | None = None,
    multivariate: bool = True,
    # --- Model parameters ---
    symmetrize: bool = True,
    # --- Unusued parameters (for compatibility with other models) ---
    *args,
    **kwargs,
) -> (
    pd.DataFrame
):  # DataFrame with columns: GID_2, Date, sample, prediction, Cases, future
    logging.info("Starting baseline model...")

    if args:
        logging.warning(f"Unused positional arguments: {args}")
    if kwargs:
        logging.warning(f"Unused keyword arguments: {kwargs}")

    if isinstance(df, DataFrame):
        df = df.df
    else:
        df = df.copy()

    num_samples = num_samples or 1000

    # Determine start and end dates
    if start_date is None:
        start_date = pd.Timestamp.min
    if end_date is None:
        end_date = pd.Timestamp.max
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    start_date = max(
        align_date_types(start_date, df["Date"]),
        df["Date"].min(),
    )
    end_date = min(
        align_date_types(end_date, df["Date"]),
        df["Date"].max(),
    )

    df = df[df["Date"].between(start_date, end_date)]

    # Output dataframe
    tdf = (
        DataFrame(db_file=Path(db_file), new_file=True)
        if db_file
        else DataFrame()  # fallback to in-memory DataFrame
    )

    # Loop over regions
    for gid2 in df["GID_2"].unique():
        logging.info(f"Processing region {gid2} for baseline model")
        # Filter data for the current region
        region_data = df[df["GID_2"] == gid2].set_index("Date")

        # Calculate incidence differences
        inc_diffs = region_data["Cases"].diff().fillna(0)

        df_samples = []
        for step_ahead in range(1, horizon + 1):
            logging.info(f"Processing horizon: {step_ahead}")

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
                    nsim=num_samples,
                )[:, -1]  # ...taking only the furthest point as the n-step predictor
                predictions = np.clip(predictions, 0, None)

                df_samples.append(
                    pd.DataFrame(
                        {
                            "GID_2": [gid2] * num_samples,
                            "Date": [region_data.index[k]] * num_samples,
                            "sample": list(range(num_samples)),
                            "prediction": predictions.tolist(),
                            "horizon": [step_ahead] * num_samples,
                            "Cases": region_data["Cases"].iloc[k],  # actual case count
                        }
                    )
                )
        df_predictions = pd.concat(df_samples, ignore_index=True)
        tdf.append(
            df_predictions.merge(
                df[["GID_2", "Date", "Cases", "future"]],
                on=["GID_2", "Date", "Cases"],
                how="right",
            )
        )

    logging.info("Baseline model complete.")
    return tdf
