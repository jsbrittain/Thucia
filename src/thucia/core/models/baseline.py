import logging

import numpy as np
import pandas as pd


class BaselineSamples:
    def __init__(self, inc_diffs, symmetrize=True):
        self.symmetrize = symmetrize
        if symmetrize:
            self.baseline = np.concatenate([inc_diffs, -inc_diffs])
        else:
            self.baseline = np.array(inc_diffs)

    def predict(self, last_inc, horizon=1, nsim=5000):
        result = np.full((nsim, horizon), np.nan)

        # Create evenly spaced quantile samples
        sampled_inc_diffs = np.quantile(self.baseline, np.linspace(0, 1, nsim))

        for h in range(horizon):
            sampled = np.random.choice(sampled_inc_diffs, size=nsim, replace=False)
            raw = last_inc + sampled

            if self.symmetrize:
                corrected = raw - (np.median(raw) - last_inc)
            else:
                corrected = raw

            result[:, h] = corrected

        return result


def baseline(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp = pd.Timestamp.min,
    end_date: str | pd.Timestamp = pd.Timestamp.max,
    gid_1: list[str] | None = None,
) -> pd.DataFrame:
    logging.info("Starting baseline model...")

    # Admin-1 filter
    if gid_1 is not None:
        df = df[df["GID_1"].isin(gid_1)]

    # Determine start and end dates
    start_date = max(pd.to_datetime(start_date), df["Date"].min())
    end_date = min(pd.to_datetime(end_date), df["Date"].max())

    # Interpolate date range, ensuring we don't skip any gaps in the data
    date_range = pd.date_range(start=start_date, end=end_date, freq="ME")

    # Ensure dates are aligned to the end of the month
    date_range += pd.offsets.MonthEnd(0)  # Ensure we reset to the end of the month
    df.loc[:, "Date"] = pd.to_datetime(df["Date"]) + pd.offsets.MonthEnd(0)

    # Combine Cases over Status=Confirmed, Probable
    df = df.groupby(["Date", "GID_2"]).agg({"Cases": "sum"}).reset_index()

    # Interpolate missing dates
    multi_index = pd.MultiIndex.from_product(
        [df["GID_2"].unique(), date_range], names=["GID_2", "Date"]
    )
    df = df.set_index(["GID_2", "Date"]).reindex(multi_index).reset_index()

    df["Cases"] = df["Cases"].fillna(0)
    samples = 1000
    df["samples"] = [np.full(samples, 0)] * len(df)

    # Loop over regions
    df_samples = []
    for gid2 in df["GID_2"].unique():
        logging.info(f"Processing region {gid2} for baseline model")
        # Filter data for the current region
        region_data = df[df["GID_2"] == gid2].set_index("Date")

        # Calculate incidence differences
        inc_diffs = region_data["Cases"].diff().fillna(0)

        # Historical forecasting
        for k in range(2, len(region_data) - 1):
            # Instantiate model
            model = BaselineSamples(inc_diffs[0 : k - 1], symmetrize=True)

            predictions = model.predict(
                last_inc=region_data["Cases"].iloc[k - 1],  # last sample
                horizon=1,
                nsim=samples,
            ).squeeze()
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
    logging.info("Baseline model complete.")

    return pd.concat(df_samples, ignore_index=True)
