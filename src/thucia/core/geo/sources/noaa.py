import io
import logging

import pandas as pd
import requests
from pandas.tseries.offsets import MonthEnd
from thucia.core.geo.plugin_base import SourceBase


class NOAA(SourceBase):
    ref = "noaa"
    name = "Oceanic Niño Index (ONI) - NOAA"

    ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

    def __init__(self):
        self._oni_df = self._load_oni()

    def _load_oni(self) -> pd.DataFrame:
        logging.info("Downloading ONI data from NOAA...")
        response = requests.get(self.ONI_URL)
        response.raise_for_status()
        df = pd.read_fwf(io.StringIO(response.text), colspecs="infer", header=0)

        # Assign columns and melt
        season_to_month = {
            "DJF": 1,
            "JFM": 2,
            "FMA": 3,
            "MAM": 4,
            "AMJ": 5,
            "MJJ": 6,
            "JJA": 7,
            "JAS": 8,
            "ASO": 9,
            "SON": 10,
            "OND": 11,
            "NDJ": 12,
        }
        df["Date"] = df["SEAS"].map(season_to_month)
        df["Date"] = pd.to_datetime(
            df["Date"].astype(str) + "-" + df["YR"].astype(str), format="%m-%Y"
        ) + MonthEnd(0)

        df.drop(columns=["SEAS", "YR"], inplace=True)
        df.rename(
            columns={
                "TOTAL": "TotalONI",
                "ANOM": "AnomONI",
            },
            inplace=True,
        )
        return df

    def merge(
        self,
        df: pd.DataFrame,
        metrics: list[str] | None = None,
        measures: list[str] | None = None,
    ) -> pd.DataFrame:
        logging.info("Merging ONI data with case data by Date...")
        df_merged = df.merge(self._oni_df, on="Date", how="left")
        logging.info("Merge complete.")
        return df_merged
