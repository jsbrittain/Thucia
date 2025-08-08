import argparse
import logging
from io import BytesIO
from io import StringIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests
from thucia.core.cases import write_nc
from thucia.core.geo import align_admin2_regions

# ======================================================================================

# These data use Brazilian Institute of Geography and Statistics (IBGE) municipality
#  codes to query and identify
# 7 digit SSRRMMM (SS = state, RR = region, MMM = municipality)

# States to query (None or [] for all, noting there are 5571 municipalities in Brazil)
default_states = ["Acre"]  # Acre contains 22 municipalities

# URL for the municipality codes (zip file)
# see https://www.ibge.gov.br/explica/codigos-dos-municipios.php
url = (
    "https://geoftp.ibge.gov.br/organizacao_do_territorio/estrutura_territorial/"
    "divisao_territorial/2024/DTB_2024.zip"
)
# Local filename of the (decompressed) municipality codes
filename = "DTB_2024/RELATORIO_DTB_BRASIL_2024_MUNICIPIOS.xls"


# ======================================================================================


def download_municipality_codes():
    # Download and decompress municipality codes
    if not url:
        raise Exception("Municipality codes are missing and URL is not set")
    logging.info(f"Downloading {url}")
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error downloading file: {response.status_code}")
    with ZipFile(BytesIO(response.content)) as zip_file:
        zip_file.extractall("DTB_2024")
        logging.info(f"Extracted files to {Path(filename).parent}")


def read_municipality_codes():
    if not Path(filename).exists():
        download_municipality_codes()
    df_municipalities = pd.read_excel(
        filename,
        header=6,
        sheet_name="DTB_Municípios",
        usecols=[
            "Nome_UF",  # State name
            "Código Município Completo",  # Municipality code
            "Nome_Município",  # Municipality name
        ],
    )
    df_municipalities.rename(
        columns={
            "Código Município Completo": "geocode",
            "Nome_Município": "municipality",
            "Nome_UF": "state",
        },
        inplace=True,
    )
    return df_municipalities


def query_state(geocode, ey_start=None, ew_start=None, ey_end=None, ew_end=None):
    # Service URL for dengue cases in Brazil
    #  https://info.dengue.mat.br/services/api     - includes a data dictionary
    #  https://info.dengue.mat.br/services/api/doc - API documentation
    url = "https://info.dengue.mat.br/api/alertcity"

    # Default epidemiological year and week to last 12 months if not provided
    current_date = pd.Timestamp.now().isocalendar()
    ey_start = ey_start or current_date.year - 1
    ew_start = ew_start or max(1, current_date.week - 1)
    ey_end = ey_end or current_date.year
    ew_end = ew_end or current_date.week

    # All parameters are mandatory
    search_filter = {
        "geocode": geocode,  # municipality code (IBGE)
        "disease": "dengue",  # dengue | chikungunya | zika
        "format": "json",  # json | csv
        "ey_start": ey_start,  # epidemiological year start
        "ew_start": ew_start,  # epidemiological week (1-53) start
        "ey_end": ey_end,  # ...end
        "ew_end": ew_end,  # ...some years can have 52, others 53
    }
    full_url = (
        url + "?" + "&".join([f"{key}={value}" for key, value in search_filter.items()])
    )
    logging.info(
        f"Querying state {geocode} from {ey_start}W{ew_start} to {ey_end}W{ew_end} "
        f"with url: {full_url}"
    )

    response = requests.get(full_url)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code}")

    # Response may still contain an error message encoded in JSON format
    if isinstance(response.json(), dict) and "error" in response.json():
        raise Exception(f"API Error: {response.json()['error_message']}")

    # Read data into a DataFrame
    df = pd.read_json(StringIO(response.text))
    if df.empty:
        raise Exception("No data returned from the API")

    # Convert epi-week to month/year
    df["week_start"] = pd.to_datetime(df["data_iniSE"], unit="ms")
    df["year"] = df["week_start"].dt.year
    df["month"] = df["week_start"].dt.month

    # Convert week_start to end-of-month dates
    df["end_of_month"] = df["week_start"] + pd.offsets.MonthEnd(0)
    df = df[["casos", "end_of_month"]]

    # Aggregate by month
    df = df.groupby("end_of_month").sum().reset_index()

    df["geocode"] = geocode

    return df


def query_states(geocodes, *args, **kwargs):
    df = []
    for ix, geocode in enumerate(geocodes, start=1):
        logging.info(f"Querying state {geocode} ({ix} / {len(geocodes)})")
        df_state = query_state(geocode, *args, **kwargs)
        df_state["geocode"] = geocode
        df.append(df_state)
    df = pd.concat(df, ignore_index=True)
    return df


def add_state_info(df, df_geocodes):
    # Add state information to the DataFrame
    df = df.merge(
        df_geocodes[["geocode", "state", "municipality"]], on="geocode", how="left"
    )
    if df.isnull().values.any():
        logging.warning("Some geocodes could not be matched with state information")
    return df


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Load dengue cases from Brazil by municipality."
    )
    parser.add_argument(
        "--states",
        nargs="*",
        default=default_states,
        help="List of states to query (default: all states)",
    )
    parser.add_argument(
        "--ey_start",
        type=int,
        default=None,
        help="Epidemiological year start (default: last year)",
    )
    parser.add_argument(
        "--ew_start",
        type=int,
        default=None,
        help="Epidemiological week start (default: last week)",
    )
    args = parser.parse_args()
    states = args.states if args.states else None
    ey_start = args.ey_start
    ew_start = args.ew_start

    logging.info(
        f"Loading dengue cases for states: {states}, "
        f"from epidemiological year {ey_start} and week {ew_start}"
    )

    # Display municipality count for each state
    df_geocodes = read_municipality_codes()
    print(df_geocodes.groupby("state").size().reset_index(name="municipality_count"))

    # Query dengue cases for the specified states
    if states:
        df_geocodes = df_geocodes[df_geocodes["state"].isin(states)]
    df = query_states(
        df_geocodes["geocode"].unique(),
        ey_start=ey_start,
        ew_start=ew_start,
    )
    df = add_state_info(df, df_geocodes)
    df.drop(columns=["geocode"], inplace=True)
    df.rename(
        columns={
            "state": "ADM1",
            "municipality": "ADM2",
            "end_of_month": "Date",
            "casos": "Cases",
        },
        inplace=True,
    )
    df = align_admin2_regions(df, "ADM1", "ADM2", iso3="BRA")
    write_nc(df)  # ADM1, ADM2, (Status, Cases), Date, GID_1, GID_2
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = main()
