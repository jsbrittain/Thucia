import logging
import subprocess
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .wis import wis_bracher


def cases_per_month(*args, **kwargs) -> pd.DataFrame:
    return aggregate_cases(*args, **kwargs, period="M")


def aggregate_cases(
    df: pd.DataFrame,
    statuses: list[str] | None = None,
    fill_column: str | None = "GID_2",
    period: str = "M",
) -> pd.DataFrame:
    """
    Count how many times each exact row appears, grouped by (period) Date

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'Date' column and optionally a 'Status' column.
    statuses : list[str], optional
        Subset of Status values to include.
    fill_column : str, optional
        If provided, will fill missing Dates, grouped by 'fill_column'.
    period : str
        Resampling period for dates. Default is 'M' (month end). Typical options:
        - 'M': Month end
        - 'W-SUN': Week ending Sunday (Mon-Sun) [ISO week]
        - 'W-SAT': Week ending Saturday (Sun-Sat) [CDC epidemiological week]

    Returns
    -------
    pd.DataFrame
        DataFrame with deduplicated rows + 'Cases' count.
    """
    if "Date" not in df.columns:
        raise ValueError("DataFrame must contain a 'Date' column.")

    if statuses is not None:
        if "Status" not in df.columns:
            raise ValueError(
                "DataFrame must contain a 'Status' column to use `statuses`."
            )
        df = df[df["Status"].isin(statuses)].copy()

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.to_period(period)

    # Prepare columns to group by (exclude Status)
    group_cols = df.columns.tolist()
    if "Status" in group_cols:
        group_cols.remove("Status")

    # Group and count cases
    grouped = df.groupby(group_cols).size().reset_index(name="Cases")

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

    # Fill Cases with zeros where missing
    result["Cases"] = result["Cases"].fillna(0).astype(int)

    # Identify descriptive columns to fill (all except fill_column, Date, Cases, and Status)
    descriptive_cols = [
        col for col in df.columns if col not in [fill_column, "Date", "Cases", "Status"]
    ]

    # For each descriptive column, build a mapping from fill_column to the unique value,
    # then map/fill in the result
    for col in descriptive_cols:
        # Get unique mapping from fill_column to col value (assumes 1 unique value per fill_column)
        mapping = df.drop_duplicates(subset=[fill_column])[
            [fill_column, col]
        ].set_index(fill_column)[col]
        result[col] = result[fill_column].map(mapping)

    # Sort results nicely
    result = result.sort_values(by=["Date", fill_column]).reset_index(drop=True)

    return result


def write_nc(
    df: pd.DataFrame | xr.Dataset,
    filename: str = "cases.nc",
    index_col: str | None = "Date",
):
    """
    Write the DataFrame or xarray Dataset to a NetCDF file.

    Parameters
    ----------
    df : pd.DataFrame or xr.Dataset
        DataFrame containing case data.
    filename : str
        Name of the output NetCDF file.
    """

    if isinstance(df, pd.DataFrame):
        if not index_col:
            df = df.set_index(index_col)
        ds = df.to_xarray()
    elif isinstance(df, xr.Dataset):
        ds = df
    else:
        raise TypeError("Input must be a pandas DataFrame or xarray Dataset.")

    ds.to_netcdf(filename, mode="w", format="netcdf4")
    logging.info(f"Data written to {filename}")


def read_nc(filename: str | Path) -> pd.DataFrame:
    """
    Read a NetCDF file into a pandas DataFrame.

    Parameters
    ----------
    filename : str
        Name of the NetCDF file to read.

    Returns
    -------
    pd.DataFrame
        The dataset read from the NetCDF file.
    """
    return xr.open_dataset(str(filename)).to_dataframe().reset_index()


def _filter_and_separate(df, pred_col, true_col, transform=None, df_filter: dict = {}):
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


def r2(df, pred_col, true_col, transform=None, df_filter: dict = {}, horizon: int = 1):
    if "horizon" in df.columns and horizon is not None and df["horizon"].nunique() > 1:
        r2s = [
            r2(
                df[df["horizon"] == h],
                pred_col,
                true_col,
                transform,
                df_filter,
                horizon=None,
            )
            for h in df["horizon"].unique()
        ]
        print(r2s)
        return r2s
    if "quantile" in df.columns:
        df = df[df["quantile"] == 0.5]
    y_true, y_pred = _filter_and_separate(df, pred_col, true_col, transform, df_filter)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2x = 1 - (ss_res / ss_tot)
    return r2x


def rmse(df, pred_col, true_col, transform=None, df_filter: dict = {}):
    y_true, y_pred = _filter_and_separate(df, pred_col, true_col, transform, df_filter)
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)


def wis(df, pred_col, true_col, transform=None, df_filter: dict = {}):
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
    return np.nanmean(wis["WIS"])


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
