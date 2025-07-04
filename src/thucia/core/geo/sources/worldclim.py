import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import rasterio
import requests
from thucia.core.fs import cache_folder
from thucia.core.geo.plugin_base import SourceBase
from thucia.core.geo.stats import raster_stats_gid2


class WorldClim(SourceBase):
    ref = "worldclim"
    name = "WorldClim"

    def _get_filename_cru(self, metric, year, month):
        # Check for file in cache, download if not, and return as a DataFrame

        dirstem = Path(cache_folder) / "climate"
        filestem = "wc2.1_cruts4.09_2.5m_{metric}_{year}-{month:02d}.tif"

        tif_file = Path(dirstem) / filestem.format(
            metric=metric, year=year, month=month
        )

        if not tif_file.exists():
            # CRU-TS version
            cru_ts_version = "4.09"
            max_year = 2024  # CRU-TS 4.09 supports records up to 2024

            # Download file and place in the cache
            url_template = (
                "https://geodata.ucdavis.edu/climate/worldclim/2_1/hist/"
                "cts{cts_version}/"
                "wc2.1_cruts4.09_2.5m_{metric}_{year_start}-{year_end}.zip"
            )
            # Year range is by decade
            year_start = year - (year % 10)
            year_end = min(year_start + 9, max_year)

            url = url_template.format(
                cts_version=cru_ts_version,
                metric=metric,
                year_start=year_start,
                year_end=year_end,
            )
            logging.info(f"Downloading WorldClim data from {url}...")
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

    def _get_filename_forecast(self, metric, year, month, source="ACCESS-CM2"):
        # Check for file in cache, download if not, and return as a DataFrame

        logging.warning(
            f"WorldClim data not available for {year}-{month:02d} ---"
            " using '{source}' forecast data."
        )

        res = "2.5m"
        ssp = "ssp245"

        dirstem = Path(cache_folder) / "climate"
        filestem_month = "wc2.1_{res}_{metric}_{source}_{ssp}_{year}-{month:02d}.tif"
        filestem_year = "wc2.1_{res}_{metric}_{source}_{ssp}_{year}.tif"

        tif_file_month = Path(dirstem) / filestem_month.format(
            res=res, metric=metric, source=source, ssp=ssp, year=year, month=month
        )
        tif_file_year = Path(dirstem) / filestem_year.format(
            res=res, metric=metric, source=source, ssp=ssp, year=year
        )

        if not tif_file_year.exists():
            # Download file and place in the cache
            url_template = (
                "https://geodata.ucdavis.edu/cmip6/{res}/{source}/{ssp}/"
                "wc2.1_{res}_{metric}_{source}_{ssp}_{year_start}-{year_end}.tif"
            )
            # Year range is by decade
            year_start = year - (year % 10)
            year_end = year_start + 9

            url = url_template.format(
                res=res,
                source=source,
                ssp=ssp,
                metric=metric,
                year_start=year_start,
                year_end=year_end,
            )
            logging.info(f"Downloading WorldClim data from {url}...")
            response = requests.get(url)
            if response.status_code != 200:
                raise FileNotFoundError(f"Failed to download {url}")
            with open(tif_file_year, "wb") as f:
                f.write(response.content)

        if not tif_file_month.exists():
            # Extract the month band from the year file
            with rasterio.open(tif_file_year) as src:
                band = src.read(int(month))
                profile = src.profile.copy()
                profile.update(count=1)
                with rasterio.open(tif_file_month, "w", **profile) as dst:
                    dst.write(band, 1)

        if not tif_file_month.exists():
            raise FileNotFoundError(
                f"Raster file {tif_file_month} not found after extraction."
            )

        return tif_file_month

    def get_filename(self, metric, year, month):
        try:
            return self._get_filename_cru(metric, year, month)
        except FileNotFoundError:
            pass
        try:
            return self._get_filename_cru(metric, year, month)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Raster file for {metric} in {year}-{month:02d} not found."
            )

    def merge(
        self,
        df: pd.DataFrame,
        metrics: list[str] | None = None,
        measures: list[str] | None = None,
    ) -> pd.DataFrame:
        logging.info("Merging climate data with case data...")

        if not metrics or metrics == ["*"]:
            metrics = ["tmin", "tmax", "prec"]
        if not measures:
            measures = ["mean"]

        # Get unique GID_2 and Date combinations
        unique_gid2_dates = df[["GID_2", "Date"]].drop_duplicates()

        for metric in metrics:
            logging.info(f"Merging climate data for metric: {metric}")

            # Read and merge mean climate data per region for each Date
            stats = []
            for date in unique_gid2_dates["Date"].unique():
                date_df = unique_gid2_dates[unique_gid2_dates["Date"] == date]
                gid_2s = date_df["GID_2"].tolist()

                # Read the corresponding raster file for the date
                try:
                    tif_file = self.get_filename(metric, date.year, date.month)
                except FileNotFoundError as e:
                    logging.warning(f"Raster file for {date} not found: {e}")
                    continue

                # Calculate zonal statistics for the GID_2 regions
                stat = raster_stats_gid2(tif_file, gid_2s)
                stat = stat[stat["mean"].notna()]
                stat["Date"] = date
                stats.append(stat)

            stats = pd.concat(stats, ignore_index=True)
            col_map = {f"{measure}": f"{metric}_{measure}" for measure in measures}
            stats.rename(columns=col_map, inplace=True)

            # Merge with the original DataFrame
            df = df.merge(
                stats[["GID_2", "Date", *col_map.values()]],
                on=["GID_2", "Date"],
                how="left",
            )

            # Simpify 'mean' column names
            if "mean" in measures:
                df.rename(columns={f"{metric}_mean": f"{metric}"}, inplace=True)
            logging.info(f"Merged {metric} data with {len(stats)} records.")

        logging.info("Climate data merged.")
        return df
