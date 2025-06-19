import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def cases_per_month(df: pd.DataFrame, statuses: list[str] | None = None):
    """
    Calculate the number of cases per month.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'date' column with datetime objects.

    Returns
    -------
    pd.Series
        Series with the number of cases per month.
    """
    if "Date" not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column.")

    # Filter by Status otherwise all cases are counted
    if statuses is not None:
        if "Status" not in df.columns:
            raise ValueError("DataFrame must contain a 'Status' column for filtering.")
        df = df[df["Status"].isin(statuses)]

    df["Date"] = df["Date"] + pd.offsets.MonthEnd(0)
    df = df.groupby(list(df.columns)).size().reset_index(name="Cases")
    if "Status" in df.columns:
        df.drop(columns=["Status"], inplace=True)
    return df


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
