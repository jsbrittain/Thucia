import logging
from pathlib import Path

import pandas as pd
import xarray as xr
from platformdirs import user_cache_dir

from .DataFrame import DataFrame as DataFrame  # noqa: F401

appname = "thucia"
appauthor = "global.Health"
cache_folder = user_cache_dir(appname, appauthor)


def get_cache_folder():
    """Returns the path to the cache folder."""
    return cache_folder


def write_nc(
    df: pd.DataFrame | xr.Dataset,
    filename: str,
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
        ds = df.to_xarray()
    elif isinstance(df, xr.Dataset):
        ds = df
    else:
        raise TypeError("Input must be a pandas DataFrame or xarray Dataset.")

    if isinstance(df["Date"], pd.PeriodIndex):
        ds.attrs["period_var"] = "Date"
        ds.attrs["period_freq"] = df.index.freqstr
        ds.attrs["period_anchor"] = "end" if df.index.is_end else "start"

    ds.to_netcdf(filename, mode="w", format="netcdf4")
    logging.info(f"Data written to {filename}")


def write_zarr(
    df: pd.DataFrame | xr.Dataset,
    filename: str,
):
    """
    Write the DataFrame or xarray Dataset to a Zarr file.
    Parameters
    ----------
    df : pd.DataFrame or xr.Dataset
        DataFrame containing case data.
    filename : str
        Name of the output Zarr file.
    """

    if isinstance(df, pd.DataFrame):
        df["Date"] = df["Date"].dt.to_timestamp()
        ds = df.to_xarray()
    elif isinstance(df, xr.Dataset):
        ds = df
    else:
        raise TypeError("Input must be a pandas DataFrame or xarray Dataset.")

    ds.to_zarr(filename, mode="w")
    logging.info(f"Data written to {filename}")


def write_db(
    df: pd.DataFrame,
    filename: str,
    table: str | None = None,
    categorical_cols: list[str] | None = None,
):
    if table is None:
        table = "data"

    # Add extension if missing
    if not Path(filename).suffix:
        filename = Path(filename).with_suffix(".duckdb")

    DataFrame(df=df, db_path=filename, table=table)
    return


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
    ds = xr.open_dataset(str(filename)).to_dataframe().reset_index()

    # Restore PeriodIndex (if applicable)
    if "period_var" in ds.attrs and "period_freq" in ds.attrs:
        period_var = ds.attrs["period_var"]
        period_freq = ds.attrs["period_freq"]
        if period_var in ds.columns:
            ds[period_var] = pd.to_datetime(ds[period_var])
            if ds.attrs.get("period_anchor", "end") == "end":
                ds[period_var] = ds[period_var].dt.to_period(period_freq)
            else:
                ds[period_var] = (
                    ds[period_var]
                    .dt.to_period(period_freq)
                    .asfreq(period_freq, how="start")
                )
    return ds


def read_zarr(filename: str | Path) -> pd.DataFrame:
    """
    Read a Zarr file into a pandas DataFrame.

    Parameters
    ----------
    filename : str
        Name of the Zarr file to read.

    Returns
    -------
    pd.DataFrame
        The dataset read from the Zarr file.
    """
    ds = xr.open_zarr(str(filename)).to_dataframe().reset_index()
    return ds


def read_db(
    filename: str | Path,
) -> DataFrame:
    """
    Returns a reference to a Thucia DataFrame class
    """

    # Add extension if missing
    if Path(filename).suffix == "":
        extensions = [".duckdb", ".nc", ".zarr"]
        for ext in extensions:
            if Path(filename).with_suffix(ext).exists():
                filename = Path(filename).with_suffix(ext)
                break

    if Path(filename).suffix == ".duckdb":
        return DataFrame(str(filename))

    if Path(filename).suffix == ".nc":
        return DataFrame(df=read_nc(filename))

    if Path(filename).suffix == ".zarr":
        return DataFrame(df=read_zarr(filename))

    raise ValueError(
        f"Unsupported file extension: {Path(filename).suffix}. "
        "Supported: .duckdb, .nc, .zarr"
    )
