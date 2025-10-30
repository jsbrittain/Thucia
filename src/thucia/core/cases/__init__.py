import logging
import re
import subprocess
from itertools import product

import numpy as np
import pandas as pd
from thucia.core.fs import DataFrame
from thucia.core.fs import read_db  # noqa: F401
from thucia.core.fs import read_nc  # noqa: F401
from thucia.core.fs import read_zarr  # noqa: F401
from thucia.core.fs import write_db  # noqa: F401
from thucia.core.fs import write_nc  # noqa: F401
from thucia.core.fs import write_zarr  # noqa: F401

from .wis import wis_bracher


def cases_per_month(*args, **kwargs) -> pd.DataFrame:
    return aggregate_cases(*args, **kwargs, freq="M")


def aggregate_cases(
    df: DataFrame | pd.DataFrame,
    statuses: list[str] | None = None,
    cases_col: str = "Cases",
    cutoff_date: pd.Timestamp | str | None = None,
    fill_column: str | None = "GID_2",
    freq: str = "M",
) -> pd.DataFrame:
    """
    Count how many times each exact row appears, grouped by (period) Date

    Parameters
    ----------
    df : DataFrame | pd.DataFrame
        DataFrame with a 'Date' column and optionally a 'Status' column.
    statuses : list[str], optional
        Subset of Status values to include.
    cases_col : str, optional
    cutoff_date : str, optional
    fill_column : str, optional
        If provided, will fill missing Dates, grouped by 'fill_column'.
    freq: str
        Resampling period for dates. Default is 'M' (month end). Typical options:
        - 'M': Month end
        - 'W-SUN': Week ending Sunday (Mon-Sun) [ISO week]
        - 'W-SAT': Week ending Saturday (Sun-Sat) [CDC epidemiological week]

    Returns
    -------
    DataFrame
        Reference to temporary Thucia DataFrame, with deduplicated rows + 'Cases' count.
    """

    # Convert to pandas DataFrame (for now)
    if isinstance(df, DataFrame):
        df = df.df  # load full pandas DataFrame
    else:
        df = df.copy()  # copy of input DataFrame

    if "Date" not in df.columns:
        raise ValueError("DataFrame must contain a 'Date' column.")

    if statuses is not None:
        if "Status" not in df.columns:
            raise ValueError(
                "DataFrame must contain a 'Status' column to use `statuses`."
            )
        df = df[df["Status"].isin(statuses)].copy()

    cutoff_date = pd.to_datetime(cutoff_date) if cutoff_date is not None else None
    if cutoff_date is not None:
        mask = pd.to_datetime(df["Date"]) <= cutoff_date
        if cases_col in df.columns:
            logging.info(
                f"Removing n={df[~mask][cases_col].sum()} cases occurring "
                f"after cutoff date {cutoff_date.date()}, remaining cases: "
                f"{df[mask][cases_col].sum()}"
            )
        else:
            logging.info(
                f"Removing n={len(df[~mask])} records occurring after "
                f"cutoff date {cutoff_date.date()}, remaining records: "
                f"{len(df[mask])}"
            )
        df = df[mask]

    if cases_col in df.columns:
        incoming_case_count = df[cases_col].sum()
    else:
        incoming_case_count = len(df)

    if not isinstance(df["Date"].dtype, pd.PeriodDtype):
        df["Date"] = pd.to_datetime(df["Date"]).dt.to_period(freq)

    # Prepare columns to group by (exclude Status)
    group_cols = ["Date", "GID_2"]  # df.columns.tolist()
    gid2_to_gid1 = dict(zip(df["GID_2"], df["GID_1"]))
    if "Status" in group_cols:
        group_cols.remove("Status")

    # Group and count cases
    if cases_col not in df.columns:
        df[cases_col] = 1
    grouped = df.groupby(group_cols, as_index=False, observed=False)[cases_col].sum()
    grouped["GID_1"] = grouped["GID_2"].map(gid2_to_gid1)

    if fill_column is None:
        return grouped.sort_values(by="Date").reset_index(drop=True)

    # Build full date range
    full_periods = pd.period_range(
        df["Date"].min(),
        df["Date"].max(),
        freq=df["Date"].dtype.freq,
    )

    if fill_column not in group_cols:
        raise ValueError(
            f"Expected '{fill_column}' column in DataFrame for full coverage."
        )

    unique_fill_values = df[fill_column].unique()

    # Full grid of all fill_column x Date combos
    full_grid = pd.DataFrame(
        list(product(unique_fill_values, full_periods)), columns=[fill_column, "Date"]
    )

    # Merge grouped counts onto full grid
    result = pd.merge(full_grid, grouped, on=[fill_column, "Date"], how="left")

    # Fill cases_col with zeros where missing
    result[cases_col] = result[cases_col].fillna(0).astype(int)

    # Identify descriptive columns to fill (all except fill_column, Date, cases_col, and Status)
    descriptive_cols = [
        col
        for col in df.columns
        if col not in [fill_column, "Date", cases_col, "Status"]
    ]

    # For each descriptive column, build a mapping from fill_column to the unique value,
    # then map/fill in the result
    for col in descriptive_cols:
        # Get unique mapping from fill_column to col value (assumes 1 unique value per fill_column)
        mapping = df.drop_duplicates(subset=[fill_column])[
            [fill_column, col]
        ].set_index(fill_column)[col]
        result[col] = result[fill_column].map(mapping)

    # Sort results
    result = result.sort_values(by=["Date", fill_column]).reset_index(drop=True)

    # Check case counts match
    outgoing_case_count = result[cases_col].sum()
    if incoming_case_count != outgoing_case_count:
        logging.warning(
            f"Case count mismatch: incoming {incoming_case_count}, outgoing {outgoing_case_count}"
        )

    # Convert to Thucia DataFrame and clean up
    return DataFrame(df=result)


def _filter_and_separate(df, pred_col, true_col, transform=None, df_filter: dict = {}):
    if df_filter is None:
        df_filter = {}
    for key, value in df_filter.items():
        df = df[df[key] == value]

    y_true = df[true_col]
    y_pred = df[pred_col]
    kk = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[kk]
    y_pred = y_pred[kk]

    if transform is not None:
        y_true = transform(y_true)
        y_pred = transform(y_pred)
    return y_true, y_pred


def r2_score(y_true, y_pred):
    kk = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[kk]
    y_pred = y_pred[kk]

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2x = 1 - (ss_res / ss_tot)
    return r2x


def r2(df, pred_col, true_col, group_col=None, transform=None, df_filter: dict = {}):
    if not group_col:
        logging.warning(
            "No group_col provided for r2 calculation, returning overall r2 score."
        )
        y_true, y_pred = _filter_and_separate(
            df, pred_col, true_col, transform, df_filter
        )
        return r2_score(y_true, y_pred)

    if "quantile" in df.columns:
        df = df[df["quantile"] == 0.5]
    if df_filter is None:
        df_filter = {}
    for key, value in df_filter.items():
        df = df[df[key] == value]

    r2_gid = pd.DataFrame(columns=[group_col, "R2"])
    groups = df[group_col].unique()
    for group in groups:
        dfg = df[df[group_col] == group].copy()
        r2_gid = pd.concat(
            [
                r2_gid,
                pd.DataFrame(
                    {
                        group_col: [group],
                        "R2": [
                            r2_score(
                                dfg[true_col],
                                dfg[pred_col],
                            )
                        ],
                    }
                ),
            ],
            ignore_index=True,
        )
    return r2_gid["R2"]


def rmse_score(y_true, y_pred):
    kk = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[kk]
    y_pred = y_pred[kk]
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)


def rmse(df, pred_col, true_col, group_col=None, transform=None, df_filter: dict = {}):
    if not group_col:
        logging.warning(
            "No group_col provided for rmse calculation, returning overall rmse score."
        )
        y_true, y_pred = _filter_and_separate(
            df, pred_col, true_col, transform, df_filter
        )
        return rmse_score(y_true, y_pred)

    if "quantile" in df.columns:
        df = df[df["quantile"] == 0.5]
    if df_filter is None:
        df_filter = {}
    for key, value in df_filter.items():
        df = df[df[key] == value]

    rmse_gid = pd.DataFrame(columns=[group_col, "RMSE"])
    groups = df[group_col].unique()
    for group in groups:
        dfg = df[df[group_col] == group].copy()
        rmse_gid = pd.concat(
            [
                rmse_gid,
                pd.DataFrame(
                    {
                        group_col: [group],
                        "RMSE": [
                            rmse_score(
                                dfg[true_col],
                                dfg[pred_col],
                            )
                        ],
                    }
                ),
            ],
            ignore_index=True,
        )
    return rmse_gid["RMSE"]


def wis(df, pred_col, true_col, group_col=None, transform=None, df_filter: dict = {}):
    if group_col is not None:
        logging.warning(
            "Group_col provided for wis calculation, but WIS is computed over all groups. Ignoring group_col."
        )

    df = df[["GID_2", "Date", "quantile", pred_col, true_col]]
    if transform is not None:
        df.loc[:, true_col] = transform(df[true_col])
        df.loc[:, pred_col] = transform(df[pred_col])
    wis = wis_bracher(
        df=df,
        group_cols=("GID_2", "Date"),
        quantile_col="quantile",
        pred_col=pred_col,
        obs_col=true_col,
        log1p_scale=False,
        clamp_negative_to_zero=True,
        monotonic_fix=True,
    )
    # Average over Date
    wis_gid = wis.groupby("GID_2").mean().reset_index()
    return wis_gid


def run_job(cmd: list[str], cwd: str | None = None) -> None:
    """
    Run a command in a subprocess and wait for it to finish.
    """

    result = subprocess.run(cmd, cwd=cwd, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}")


def prepare_pdfm_embeddings(
    pdfm_filename: str | None = None,
    provinces: list[str] | None = None,
) -> pd.DataFrame:
    # Load PDFM embeddings
    return read_nc(pdfm_filename)


def prepare_embeddings(filename: str, embedding_type="pdfm") -> pd.DataFrame:
    if embedding_type == "pdfm":
        # Filter embeddings
        # Dimensions:   0-127 are used to reconstruct the Aggregated Search Trends
        #             128-255 are used to reconstruct the Maps and Busyness
        #             256-329 are used for Weather & Air Quality
        return prepare_pdfm_embeddings(filename)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")


def align_date_types(
    source_dates: pd.Series | pd.Timestamp,
    target_dates: pd.Series,
) -> pd.Series:
    """
    Align the date types of source_dates to match target_dates.
    If target_dates is a PeriodIndex, convert source_dates to PeriodIndex with same freq.
    If target_dates is a DatetimeIndex, convert source_dates to DatetimeIndex.

    Parameters
    ----------
    source_dates : pd.Series | pd.Timestamp
        Series of dates to be aligned.
    target_dates : pd.Series
        Series of dates to align to.

    Returns
    -------
    pd.Series | pd.Timestamp
        Foramt aligned dates.
    """
    if isinstance(source_dates, pd.Series):
        if isinstance(target_dates.dtype, pd.PeriodDtype):
            freq = re.search(r"period\[(.+)\]", str(target_dates.dtype.name)).group(1)
            if isinstance(source_dates.dtype, pd.PeriodDtype):
                # Source is already Period, just ensure same freq
                source_dates = source_dates.dt.asfreq(freq)
            else:
                # Convert to datetime, then period
                source_dates = pd.to_datetime(source_dates).dt.to_period(freq)
        else:
            source_dates = pd.to_datetime(source_dates)
    elif isinstance(source_dates, pd.Timestamp):
        if isinstance(target_dates.dtype, pd.PeriodDtype):
            freq = re.search(r"period\[(.+)\]", str(target_dates.dtype.name)).group(1)
            source_dates = source_dates.to_period(freq)
        else:
            source_dates = pd.to_datetime(source_dates)
    return source_dates


def check_index_combinations(df: pd.DataFrame, group_cols):
    """Check that all combinations of group_cols are present in df."""
    product = []
    for c in group_cols:
        product.append(df[c].unique())
    all_idx = pd.MultiIndex.from_product(product, names=group_cols)
    present_idx = pd.MultiIndex.from_frame(df[group_cols].drop_duplicates())
    missing = all_idx.difference(present_idx)
    if not missing.empty:
        raise ValueError(f"Missing index combinations in data:\n{missing}")


def check_covars_for_nans(df, cols):
    if df[cols].isnull().any().any():
        missing = df[df[cols].isnull().any(axis=1)]
        raise ValueError(f"Missing covariate values in data:\n{missing}")
