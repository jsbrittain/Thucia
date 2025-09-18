import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .adapter import residual_regression as residual_regression

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


def filter_admin1(df: pd.DataFrame, gid_1: str) -> pd.DataFrame:
    """
    Filter DataFrame by admin-1 region.
    """
    if gid_1 is not None:
        df = df[df["GID_1"].isin(gid_1)]
    return df.reset_index(drop=True)


def sanitize_dates_inplace(
    df: pd.DataFrame,
    date_col: str = "Date",
    start_date: pd.Timestamp | str = pd.Timestamp.min,
    end_date: pd.Timestamp | str = pd.Timestamp.max,
) -> pd.DataFrame:
    """
    Ensure the date column is in datetime format and at month-end.
    """
    # Determine start and end dates
    df.loc[:, date_col] = pd.to_datetime(df[date_col]) + pd.offsets.MonthEnd(0)
    start_date = max(
        pd.to_datetime(start_date), df[date_col].min()
    ) + pd.offsets.MonthEnd(0)
    end_date = min(pd.to_datetime(end_date), df[date_col].max()) + pd.offsets.MonthEnd(
        0
    )
    date_range = pd.date_range(start=start_date, end=end_date, freq="ME")
    return date_range


def validate_unique_keys(df: pd.DataFrame, col_names: list[str]) -> None:
    # Check that all (Date, GID_2) combinations are unique
    if df[col_names].duplicated().any():
        raise ValueError(
            "DataFrame contains duplicate (Date, GID_2) combinations. "
            "Ensure that the data is aggregated correctly."
        )


def interpolate_missing_dates(
    df,
    start_date: pd.Timestamp | str = pd.Timestamp.min,
    end_date: pd.Timestamp | str = pd.Timestamp.max,
    date_col: str = "Date",
    gid_col: str = "GID_2",
) -> None:
    # Get date range
    date_range = sanitize_dates_inplace(
        df,
        date_col=date_col,
        start_date=start_date,
        end_date=end_date,
    )
    validate_unique_keys(df, col_names=[date_col, gid_col])

    # Interpolate missing dates
    multi_index = pd.MultiIndex.from_product(
        [df[gid_col].unique(), date_range], names=[gid_col, date_col]
    )
    df = df.set_index([gid_col, date_col]).reindex(multi_index).reset_index()
    return df


def set_nan_zero(df: pd.DataFrame, col: str = "Cases", filter_col: str = "future"):
    if df[filter_col].isna().any():
        logging.warning(
            f"Column {filter_col} contains NaNs. Removing rows from dataset."
        )
        df = df[~df[filter_col].isna()]
        df.loc[:, filter_col] = df[filter_col].astype(bool)
    df.loc[~df[filter_col], col] = df.loc[~df[filter_col], col].fillna(0)


def set_historical_na_to_zero(
    df: pd.DataFrame, col: str = "Cases", filter_col: str = "future"
) -> pd.DataFrame:
    """
    Treat historical NAs as zero for the specified column.
    """
    df = df.copy()
    set_nan_zero(df, col=col, filter_col=filter_col)
    return df


def pca_transform(
    df, covariate_cols, keep_components: Optional[int] = 5, case_col="Cases"
):
    # PCA transform covariates and project to fewer dimensions
    df_covs = df[["Date", "GID_2"] + covariate_cols].copy()
    logging.info("Fit PCA")
    pca = PCA(n_components=min(keep_components, len(covariate_cols)))
    logging.info("Transform covariates by PCA")
    covs_transformed = pca.fit_transform(df_covs[covariate_cols])
    df = df.drop(columns=covariate_cols)
    covariate_cols = [f"PC{i + 1}" for i in range(covs_transformed.shape[1])]
    df[covariate_cols] = covs_transformed
    df = df[["Date", "GID_2", "future", "Cases", case_col] + covariate_cols]
    return df


def sanitise_covariates(df, covariate_cols, start_date):
    # Covariate sanitisation
    for c in covariate_cols:
        # NaN replacement: seasonal mean, forward and back fill
        df[c] = df.groupby(["GID_2", df["Date"].dt.month])[c].transform(
            lambda s: s.fillna(s.mean())
        )
        df[c] = df.groupby("GID_2")[c].ffill().bfill()
        # Standardise using pre- start date values
        mask = df["Date"] < start_date
        mu = df[mask][c].mean()
        sd = np.max([1e-8, df[mask][c].std()])
        df[c] = (df[c] - mu) / sd
        # Sanity check
        if df[c].isna().any():
            raise Exception("NaN found in covariates")
    return df
