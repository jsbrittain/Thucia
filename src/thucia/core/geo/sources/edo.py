import logging
import pandas as pd

import requests
import zipfile
import io

from pathlib import Path
from thucia.core.fs import cache_folder
from thucia.core.geo.stats import raster_stats_gid2

from thucia.core.geo.plugin_base import SourceBase


class EDO(SourceBase):
    ref = "edo"
    name = "European Drought Observatory (EDO)"

    def get_filename(self, year):
        # Check for file in cache, download if not, and return as a DataFrame

        dirstem = Path(cache_folder) / "climate"
        filestem = "spc06_m_gdo_{year}0101_m_300_z01.tif"

        tif_file = Path(dirstem) / filestem.format(year=year)

        if not tif_file.exists():
            # Download file and place in the cache
            url_template = (
                "https://drought.emergency.copernicus.eu/data/"
                "Drought_Observatories_datasets/"
                "GDO_CHIRPS_Standardized_Precipitation_Index_SPI6/"
                "ver3-0-0/"
                "spc06_m_gdo_{year}0101_{year}1201_m.zip"
            )

            url = url_template.format(
                year=year,
            )
            logging.info(f"Downloading EDO data from {url}...")
            response = requests.get(url)
            if response.status_code != 200:
                raise FileNotFoundError(f"Failed to download {url}")
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(dirstem)
            if not tif_file.exists():
                raise FileNotFoundError(
                    f"Raster file {tif_file} not found after extraction."
                )

        return tif_file

    def merge(
        self,
        df: pd.DataFrame,
        metrics: list[str] | None = None,
        measures: list[str] | None = None,
    ) -> pd.DataFrame:
        logging.info("Merging EDO data with case data...")

        # Get unique GID_2 and Date combinations
        unique_gid2_dates = df[["GID_2", "Date"]].drop_duplicates()

        # Read and merge mean climate data per region for each Date
        stats = []
        for date in unique_gid2_dates["Date"].unique():
            date_df = unique_gid2_dates[unique_gid2_dates["Date"] == date]
            gid_2s = date_df["GID_2"].tolist()

            # Read the corresponding raster file for the date
            try:
                tif_file = self.get_filename(date.year)
            except FileNotFoundError as e:
                logging.warning(f"Raster file for {date} not found: {e}")
                continue

            # Calculate zonal statistics for the GID_2 regions
            stat = raster_stats_gid2(tif_file, gid_2s)
            stat = stat[stat["mean"].notna()]
            stat["Date"] = date
            stats.append(stat)

        stats = pd.concat(stats, ignore_index=True)
        col_map = {"mean": "SPI6"}
        stats.rename(columns=col_map, inplace=True)

        # Merge with the original DataFrame
        df = df.merge(
            stats[["GID_2", "Date", *col_map.values()]],
            on=["GID_2", "Date"],
            how="left",
        )

        logging.info("EDO data merged.")
        return df
