import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests
from thucia.core.fs import cache_folder
from thucia.core.geo.plugin_base import SourceBase
from thucia.core.geo.stats import raster_stats_gid2


class WorldPop(SourceBase):
    ref = "worldpop"
    name = "WorldPop"

    def get_filename(self, metric, gid_1, year):
        # Check for file in cache, download if not, and return as a DataFrame

        dirstem = Path(cache_folder) / "geo"
        dirstem.mkdir(parents=True, exist_ok=True)

        if metric == "pop_count":
            max_year = 2020
            if year > max_year:
                logging.warning(
                    f"Year {year} exceeds maximum available year {max_year} for "
                    "population count data. Switching to unconstrained estimates."
                )
                metric = "pop_estimate_unconstrained"

        match metric:
            case "pop_count":
                filestem = "{gid1}_ppp_{year}_1km_Aggregated.tif"
                url_template = (
                    "https://data.worldpop.org/GIS/Population/"
                    "Global_2000_2020_1km/{year}/{GID1}/"
                    "{gid1}_ppp_{year}_1km_Aggregated.tif"
                )
            case "pop_estimate_constrained":
                filestem = "{gid1}_pop_{year}_CN_1km_R2024B_UA_v1.tif"
                url_template = (
                    "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2024B/"
                    "{year}/{GID1}/v1/1km_ua/constrained/"
                    "{gid1}_pop_{year}_CN_1km_R2024B_UA_v1.tif"
                )
            case "pop_estimate_unconstrained":
                filestem = "{gid1}_pop_{year}_UC_1km_R2024B_UA_v1.tif"
                url_template = (
                    "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2024B/"
                    "{year}/{GID1}/v1/1km_ua/unconstrained/"
                    "{gid1}_pop_{year}_UC_1km_R2024B_UA_v1.tif"
                )
            case _:
                raise ValueError(f"Unsupported metric: {metric}")

        # Local file name
        tif_file = Path(dirstem) / filestem.format(gid1=gid_1.lower(), year=year)

        if not tif_file.exists():
            # Download file and place in the cache
            url = url_template.format(
                gid1=gid_1.lower(),
                GID1=gid_1.upper(),
                year=year,
            )
            logging.info(f"Downloading WorldPop data from {url}...")
            response = requests.get(url)
            if response.status_code != 200:
                raise FileNotFoundError(f"Failed to download {url}")
            with open(tif_file, "wb") as f:
                f.write(response.content)
            if not tif_file.exists():
                raise FileNotFoundError(
                    f"Raster file {tif_file} not found after extraction."
                )

        return tif_file

    @lru_cache(maxsize=500)
    def get_cached_stats(self, tif_file, gid_2s, stats):
        return raster_stats_gid2(tif_file, list(gid_2s), stats=list(stats))

    def merge(
        self,
        df: pd.DataFrame,
        metrics: list[str] | None = None,
        measures: list[str] | None = None,
    ) -> pd.DataFrame:
        logging.info("Merging population data with case data...")

        if not metrics:
            metrics = ["pop_count"]
        if metrics == ["*"]:
            metrics = [
                "pop_count",
                "pop_estimate_constrained",
                "pop_estimate_unconstrained",
            ]
        if not measures:
            measures = ["sum"]

        # Get unique GID_2 and Date combinations
        unique_gid2_dates = df[["GID_2", "Date"]].drop_duplicates()
        unique_gid1s = df["GID_1"].str.split(".", expand=True)[0].unique()
        if len(unique_gid1s) != 1:
            raise ValueError("All records must be for the same GID_1 region.")
        gid_1 = unique_gid1s[0]

        for metric in metrics:
            logging.info(f"Merging population data for metric: {metric}")

            # Read and merge mean climate data per region for each Date
            stats = []
            for date in unique_gid2_dates["Date"].unique():
                date_df = unique_gid2_dates[unique_gid2_dates["Date"] == date]
                gid_2s = date_df["GID_2"].tolist()

                # Read the corresponding raster file for the date
                try:
                    tif_file = self.get_filename(metric, gid_1, date.year)
                except FileNotFoundError as e:
                    logging.warning(f"Raster file for {date} not found: {e}")
                    continue

                if tif_file is None:
                    continue

                # Calculate zonal statistics for the GID_2 regions
                stat = self.get_cached_stats(  # cached as pop is per year
                    tif_file, tuple(gid_2s), stats=tuple(["sum"])
                ).copy()
                if len(stat) != len(gid_2s):
                    print(
                        f"Warning: Expected {len(gid_2s)} stats for {date}, got {len(stat)}"
                    )
                stat["sum"] = stat["sum"].fillna(0)  # Ensure no NaN values
                stat["Date"] = date
                stats.append(stat)

            if len(stats) != len(unique_gid2_dates["Date"].unique()):
                logging.warning(
                    f"Some dates did not have data for {metric}. "
                    "Check if the raster files are available."
                )

            stats = pd.concat(stats, ignore_index=True)
            col_map = {f"{measure}": f"{metric}_{measure}" for measure in measures}
            stats.rename(columns=col_map, inplace=True)

            # Merge with the original DataFrame
            df = df.merge(
                stats[["GID_2", "Date", *col_map.values()]],
                on=["GID_2", "Date"],
                how="left",
            )

            # Simpify 'sum' column names
            if "sum" in measures:
                df.rename(columns={f"{metric}_sum": f"{metric}"}, inplace=True)
            logging.info(f"Merged {metric} data with {len(stats)} records.")

        logging.info("Climate data merged.")
        return df
