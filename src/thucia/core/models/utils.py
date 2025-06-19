import logging
import numpy as np
import pandas as pd

quantiles = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def sample_to_quantiles_vec(samples, quantiles=quantiles):
    samples = np.asarray(samples)
    q_values = np.quantile(samples, quantiles)
    return pd.DataFrame({"quantile": quantiles, "value": q_values})


def samples_to_quantiles(
    df: pd.DataFrame,
    quantiles=quantiles,
    gid_1: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convert samples in a DataFrame to quantiles using groupby for efficiency.
    Assumes 'prediction' contains multiple samples per date per region.
    """
    logging.info("Converting samples to quantiles...")

    # Admin-1 filter
    if gid_1 is not None:
        df = df[df["GID_1"].isin(gid_1)]

    # Group once by region
    results = []
    for gid2, group in df.groupby("GID_2"):
        logging.info(f"Processing region {gid2} for quantiles conversion")
        group = group.sort_values("Date")  # ensure Date order
        dates = group["Date"].unique()

        # Map Date -> predictions for vectorized access
        date_groups = group.groupby("Date")

        # Loop through date positions, skipping boundaries
        for k in range(2, len(dates) - 1):
            date = dates[k]
            predictions = date_groups.get_group(date)["prediction"].values

            # Apply quantile transform
            quantile_values = sample_to_quantiles_vec(
                predictions,
                quantiles,
            )["value"].values

            # Pull Cases once
            cases_val = date_groups.get_group(date)["Cases"].iloc[0]

            results.append(
                pd.DataFrame(
                    {
                        "GID_2": gid2,
                        "Date": date,
                        "quantile": quantiles,
                        "prediction": quantile_values,
                        "Cases": cases_val,
                    }
                )
            )

    logging.info("Quantiles conversion complete.")
    return pd.concat(results, ignore_index=True)
