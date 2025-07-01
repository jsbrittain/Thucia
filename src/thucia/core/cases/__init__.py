import logging
import subprocess
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def cases_per_month(
    df: pd.DataFrame,
    statuses: list[str] | None = None,
    fill_column: str | None = "GID_2",
) -> pd.DataFrame:
    """
    Count how many times each exact row appears, grouped by Date floored to month-end.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'Date' column and optionally a 'Status' column.
    statuses : list[str], optional
        Subset of Status values to include.
    fill_column : str, optional
        If provided, will fill missing Dates, grouped by 'fill_column'.

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
    df["Date"] = pd.to_datetime(df["Date"]) + pd.offsets.MonthEnd(0)

    # Prepare columns to group by (exclude Status)
    group_cols = df.columns.tolist()
    if "Status" in group_cols:
        group_cols.remove("Status")

    # Group and count cases
    grouped = df.groupby(group_cols).size().reset_index(name="Cases")

    if fill_column is None:
        return grouped.sort_values(by="Date").reset_index(drop=True)

    # Build full date range
    min_date, max_date = df["Date"].min(), df["Date"].max()
    full_dates = pd.date_range(start=min_date, end=max_date, freq="ME")

    if fill_column not in group_cols:
        raise ValueError(
            f"Expected '{fill_column}' column in DataFrame for full coverage."
        )

    unique_fill_values = df[fill_column].unique()

    # Full grid of all fill_column x Date combos
    full_grid = pd.DataFrame(
        list(product(unique_fill_values, full_dates)), columns=[fill_column, "Date"]
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


def write_nc(df: pd.DataFrame | xr.Dataset, filename: str = "cases.nc"):
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
        df = df.set_index("Date")
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


def r2(df, pred_col, true_col, transform=None, df_filter: dict = {}):
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

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = 1 - (ss_res / ss_tot)
    return r2


def run_job(cmd: list[str]) -> None:
    """
    Run a command in a subprocess and wait for it to finish.
    """

    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}")
