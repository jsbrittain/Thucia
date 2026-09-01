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


def _df_to_xarray(df: pd.DataFrame) -> xr.Dataset:
    """Convert a DataFrame to an xarray Dataset, encoding Period columns.

    netCDF/Zarr cannot encode a pandas Period dtype. Period columns (e.g. a
    monthly ``Date``) are stored as their end timestamps with the period
    metadata recorded in the Dataset attrs so the read path can restore them.
    """
    df = df.copy()
    period_meta: dict[str, tuple[str, str]] = {}
    for col in df.columns:
        if isinstance(df[col].dtype, pd.PeriodDtype):
            idx = pd.PeriodIndex(df[col])
            period_meta[col] = (idx.freqstr, "end")
            df[col] = idx.to_timestamp(how="end")
    ds = df.to_xarray()
    if period_meta:
        first = next(iter(period_meta))
        ds.attrs["period_var"] = first
        ds.attrs["period_freq"] = period_meta[first][0]
        ds.attrs["period_anchor"] = period_meta[first][1]
    return ds


def _restore_period_column(df: pd.DataFrame, attrs: dict) -> pd.DataFrame:
    """Restore a Period column written by :func:`_df_to_xarray`."""
    period_var = attrs.get("period_var")
    period_freq = attrs.get("period_freq")
    if period_var and period_freq and period_var in df.columns:
        df[period_var] = pd.to_datetime(df[period_var])
        if attrs.get("period_anchor", "end") == "end":
            df[period_var] = df[period_var].dt.to_period(period_freq)
        else:
            df[period_var] = (
                df[period_var]
                .dt.to_period(period_freq)
                .asfreq(period_freq, how="start")
            )
    return df


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

    if isinstance(df, xr.Dataset):
        ds = df
    elif isinstance(df, pd.DataFrame):
        ds = _df_to_xarray(df)
    else:
        raise TypeError("Input must be a pandas DataFrame or xarray Dataset.")

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

    if isinstance(df, xr.Dataset):
        ds = df
    elif isinstance(df, pd.DataFrame):
        ds = _df_to_xarray(df)
    else:
        raise TypeError("Input must be a pandas DataFrame or xarray Dataset.")

    ds.to_zarr(filename, mode="w")
    logging.info(f"Data written to {filename}")


def write_db(
    df: pd.DataFrame | DataFrame,
    filename: str,
    table: str | None = None,
    categorical_cols: list[str] | None = None,
):
    if isinstance(df, DataFrame):
        if df.db_path != ":memory:":
            logging.warning(
                "Input is already a Thucia DataFrame. Writing to a new database file."
            )
        df = df.df

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
    ds = xr.open_dataset(str(filename))
    attrs = dict(ds.attrs)
    df = ds.to_dataframe().reset_index()
    return _restore_period_column(df, attrs)


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
    ds = xr.open_zarr(str(filename))
    attrs = dict(ds.attrs)
    df = ds.to_dataframe().reset_index()
    return _restore_period_column(df, attrs)


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
        "Could not open file, missing or unsupported file extension"
        " (supported: .duckdb, .nc, .zarr)."
    )
