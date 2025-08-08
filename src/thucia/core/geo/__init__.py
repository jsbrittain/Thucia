import logging
import unicodedata
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from rapidfuzz import fuzz
from rapidfuzz import process
from thucia.core.fs import cache_folder

from .plugin_loader import load_plugins

plugins = None


def lookup_gid1(admin1_names: list[str], iso3: str = "PER"):
    """
    Lookup GID_1 codes for the given administrative level 1 names in a DataFrame.

    Parameters:
    admin1_names (list[str]): List of administrative level 1 names to lookup.
    iso3 (str): ISO3 country code.

    Returns:
    list[str]: List of GID_1 codes corresponding to the provided names.
    """
    logging.info(
        f"Looking up GID_1 codes for Admin-1 names: {admin1_names} in {iso3}..."
    )
    gdf = get_admin2_list(iso3)
    gdf = gdf[gdf["NAME_1"].isin(admin1_names)]
    admin1_regions = gdf["GID_1"].unique().tolist()
    return admin1_regions


def remove_accents(text):
    if not isinstance(text, str):
        return text
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def fuzzy_match_one(query, reference_list, threshold=90):
    if not isinstance(query, str):
        return ""
    match, score, _ = process.extractOne(
        query, reference_list, scorer=fuzz.token_sort_ratio
    )
    return match if score >= threshold else ""


def get_admin2_list(iso3: str) -> pd.DataFrame:
    """
    Get a list of administrative level 2 regions for the specified ISO3 country code.

    Parameters:
    iso3 (str): The ISO3 code of the country.

    Returns:
    GeoDataFrame: A GeoDataFrame containing the administrative regions.
    """

    file_path = Path(cache_folder) / "geo" / iso3 / f"gadm41_{iso3}.gpkg"
    if not file_path.exists():
        logging.info("GeoPackage file not found, downloading...")
        url = (
            f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso3.upper()}.gpkg"
        )
        response = requests.get(url)
        if response.status_code == 200:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(response.content)
            logging.info(f"Downloaded GeoPackage file to {file_path}")
        else:
            logging.error(f"Failed to download GeoPackage file from {url}")
    if not file_path.exists():
        raise FileNotFoundError(f"GeoPackage file for {iso3} not found at {file_path}")

    gdf = gpd.read_file(str(file_path), layer="ADM_ADM_2")
    gdf = gdf[["GID_0", "GID_1", "NAME_1", "GID_2", "NAME_2", "HASC_2"]]

    return gdf


def align_admin2_regions(
    df: pd.DataFrame, admin1_col: str, admin2_col: str, iso3: str
) -> pd.DataFrame:
    """
    Aligns the administrative level 2 regions in the DataFrame with the standardized
    names.

    Parameters:
    df (pd.DataFrame): The DataFrame containing administrative regions.
    admin1_col (str): The column name for administrative level 1 regions.
    admin2_col (str): The column name for administrative level 2 regions.
    iso3 (str): The ISO3 code of the country.

    Returns:
    GeoDataFrame: The DataFrame with aligned administrative regions.
    """

    logging.info(f"Initial data contains {len(df)} records.")
    logging.info(
        f"There are {df['ADM1'].nunique()} unique Admin-1 regions and "
        f"{df['ADM2'].nunique()} unique Admin-2 regions."
    )
    logging.info(f"Aligning administrative regions for {iso3}...")

    def homogenise(x):
        return x.apply(remove_accents).str.replace(" ", "", regex=False).str.lower()

    logging.info("Loading administrative regions...")
    ref_names = get_admin2_list(iso3)
    logging.info("Homogenising administrative names...")
    adm1h_ref = homogenise(ref_names["NAME_1"])
    adm2h_ref = homogenise(ref_names["NAME_2"])
    adm1h_df = homogenise(df[admin1_col])
    adm2h_df = homogenise(df[admin2_col])

    # Build a reference mapping table
    ref_map = pd.DataFrame(
        {
            "adm1h": adm1h_ref,
            "adm2h": adm2h_ref,
            "NAME_1": ref_names["NAME_1"].values,
            "NAME_2": ref_names["NAME_2"].values,
        }
    )

    # Ensure main DataFrame has the matching homogenized keys
    df["adm1h"] = adm1h_df
    df["adm2h"] = adm2h_df

    exact = df.merge(ref_map, on=["adm1h", "adm2h"], how="left")
    unmatched = exact[exact["NAME_1"].isna() | exact["NAME_2"].isna()].copy()

    unique_unmatched_adm1 = unmatched[["ADM1", "adm1h"]].drop_duplicates()
    unique_unmatched_adm1["fuzzy_match"] = unique_unmatched_adm1["ADM1"].apply(
        lambda x: fuzzy_match_one(x, ref_names["NAME_1"].tolist())
    )

    unique_unmatched_adm2 = unmatched[["ADM2", "adm2h"]].drop_duplicates()
    unique_unmatched_adm2["fuzzy_match"] = unique_unmatched_adm2["ADM2"].apply(
        lambda x: fuzzy_match_one(x, ref_names["NAME_2"].tolist())
    )

    adm1_fuzzy_map = dict(
        zip(unique_unmatched_adm1["adm1h"], unique_unmatched_adm1["fuzzy_match"])
    )
    adm2_fuzzy_map = dict(
        zip(unique_unmatched_adm2["adm2h"], unique_unmatched_adm2["fuzzy_match"])
    )

    def resolve_fuzzy_admin1(row):
        if pd.isna(row["NAME_1"]) and row["adm1h"] in adm1_fuzzy_map:
            return adm1_fuzzy_map[row["adm1h"]]
        return row["NAME_1"]

    def resolve_fuzzy_admin2(row):
        if pd.isna(row["NAME_2"]) and row["adm2h"] in adm2_fuzzy_map:
            return adm2_fuzzy_map[row["adm2h"]]
        return row["NAME_2"]

    exact["NAME_1"] = exact.apply(resolve_fuzzy_admin1, axis=1)
    exact["NAME_2"] = exact.apply(resolve_fuzzy_admin2, axis=1)

    df[admin1_col] = exact["NAME_1"]
    df[admin2_col] = exact["NAME_2"]

    df.drop(columns=["adm1h", "adm2h"], inplace=True)

    # Add GID_1 and GID_2 columns
    logging.info("Merging with reference names...")
    df = df.merge(
        # Need to include both donating columns, and pairing columns
        ref_names[["GID_1", "GID_2", "NAME_1", "NAME_2"]],
        left_on=[admin1_col, admin2_col],
        right_on=["NAME_1", "NAME_2"],
        how="left",
    )
    df = df.drop(columns=["NAME_1", "NAME_2"])

    df.dropna(inplace=True)

    logging.info("Alignment complete.")
    logging.info(f"After merging with admin2 regions, there are {len(df)} records.")
    logging.info(
        f"After alignment, there are {df['ADM1'].nunique()} unique Admin-1 regions and "
        f"{df['ADM2'].nunique()} unique Admin-2 regions."
    )
    return df


def refresh_plugins(verbose: bool = False) -> None:
    """
    Reloads the plugins to ensure the latest versions are used.
    """
    global plugins
    plugins = load_plugins()
    if verbose:
        print("Source plugins loaded:")
        for ref, plugin in plugins.items():
            print(f" - [{ref}] {plugin.name}")
    logging.info("Plugins loaded: " + ", ".join([plugin for plugin in plugins]))


def merge_geo_sources(df: pd.DataFrame, sources: list[str]) -> pd.DataFrame:
    """
    Add source information to the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to which source information will be added.
    source (list[str]): List of sources to be added. Format: ['origin.field']
                        where project is optional, e.g. ['worldclim.*', 'edo.spi-7']
    """
    global plugins
    if not plugins:
        refresh_plugins()

    # Collate source information
    d_sources = {}
    for source in sources:
        if "." not in source:
            raise ValueError(
                "Source format must be 'origin.field', "
                "e.g. 'worldclim.*' or 'edo.spi6'."
            )
        origin, field = source.split(".")
        if origin not in d_sources:
            d_sources[origin] = []
        d_sources[origin].append(field)

    for source, fields in d_sources.items():
        try:
            source_module = plugins.get(source)
        except KeyError:
            logging.error(f"Source plugin '{source}' not found.")
        df = source_module.merge(df, metrics=fields)

    return df


def add_incidence_rate(
    df: pd.DataFrame,
    target_col: str = "DIR",
    cases_col: str = "Cases",
    pop_col: str = "pop_count",
    cases_per: int = 1e5,
) -> pd.DataFrame:
    df[target_col] = cases_per * df[cases_col] / df[pop_col]
    return df


def convert_to_incidence_rate(
    df: pd.DataFrame,
    df_pop: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the DataFrame to incidence rate format.

    Parameters:
    df (pd.DataFrame): The DataFrame containing cases.
    df_pop (pd.DataFrame): The DataFrame containing population counts with columns

    Returns:
    pd.DataFrame: The DataFrame with an additional column for incidence rate.
    """
    if "pop_count" in df.columns:
        df.drop(columns=["pop_count"], inplace=True)
    df_with_pop = df.merge(
        df_pop[["Date", "GID_2", "pop_count"]],
        on=["Date", "GID_2"],
        how="left",
    )
    cases_per = 1e5
    df_with_pop = add_incidence_rate(
        df_with_pop,
        target_col="Cases",
        cases_col="Cases",
        pop_col="pop_count",
        cases_per=cases_per,
    )
    df_with_pop = add_incidence_rate(
        df_with_pop,
        target_col="prediction",
        cases_col="prediction",
        pop_col="pop_count",
        cases_per=cases_per,
    )
    df_with_pop.drop(columns=["pop_count"], inplace=True)
    return df_with_pop


def pad_admin2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all Admin-2 regions are included in the DataFrame, even those with zero
    cases.
    """
    if "GID_2" not in df.columns or "ADM2" not in df.columns:
        raise ValueError("DataFrame must contain 'GID_2' and 'ADM2' columns.")

    # Get unique Admin-2 regions
    gid0 = df["GID_2"].iloc[0][:3]  # Assuming GID_2 starts with GID-0
    unique_admin2 = df["GID_2"].unique()
    all_admin2 = get_admin2_list(gid0)

    # Find missing Admin-2 regions
    missing_admin2 = set(all_admin2["GID_2"].unique()) - set(unique_admin2)
    unique_dates = df["Date"].unique()

    # Create a DataFrame for missing regions with zero cases
    missing_df = []
    for adm2 in missing_admin2:
        missing_df.append(
            pd.DataFrame(
                {
                    "Date": pd.to_datetime(unique_dates).tolist(),
                    "ADM1": [
                        all_admin2["NAME_1"][all_admin2["GID_2"] == adm2].values[0]
                    ]
                    * len(unique_dates),
                    "ADM2": [
                        all_admin2["NAME_2"][all_admin2["GID_2"] == adm2].values[0]
                    ]
                    * len(unique_dates),
                    "GID_1": [
                        all_admin2["GID_1"][all_admin2["GID_2"] == adm2].values[0]
                    ]
                    * len(unique_dates),
                    "GID_2": [adm2] * len(unique_dates),
                    "Cases": [0] * len(unique_dates),
                }
            )
        )

    # Concatenate the original DataFrame with the missing regions
    result = (
        pd.concat([df, *missing_df], ignore_index=True)
        .sort_values(["Date", "GID_2"])
        .reset_index(drop=True)
    )
    result.sort_values(by=["Date", "GID_2"], inplace=True)
    return result


def merge_sources(df, covars: list[str]) -> None:
    """
    Merge geographic and climatological covariates into the main DataFrame.
    This function is a placeholder for actual merging logic.
    """
    # Schedule source aggregation tasks (could parallelise this)
    for covar in covars:
        df_covar = merge_geo_sources(df, [covar])
        merge_vars = list(
            set(["GID_2", "Date"])
            | (set(df_covar.columns.tolist()) - set(df.columns.tolist()))
        )
        df = df.merge(df_covar[merge_vars], on=["GID_2", "Date"], how="left")
    return df
