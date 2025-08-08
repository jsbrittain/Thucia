import inspect
import logging
from pathlib import Path

import pandas as pd
import thucia
from thucia.core.cases import write_nc
from thucia.core.containers import build_container
from thucia.core.containers import run_in_container
from thucia.core.fs import cache_folder


def inla(
    df: pd.DataFrame,
    gid_1: list[str] | None = None,
) -> pd.DataFrame:
    # Determine country code
    iso3s = df["GID_2"].apply(lambda x: x.split(".")[0]).unique()
    if len(iso3s) != 1:
        raise ValueError("Input DataFrame must contain data for a single ISO3 country.")
    iso3 = iso3s[0]

    # File paths
    climate_path = Path(inspect.getfile(inla)).parent / "inla"
    data_path = (
        Path(inspect.getfile(thucia)).parent.parent.parent / "data" / "cases" / iso3
    )
    gadm_path = Path(cache_folder) / "geo" / iso3
    gadm_filename = f"gadm41_{iso3}.gpkg"

    # Build container
    tag = "thucia/model/climate:latest"
    platform = "linux/amd64"
    build_container(
        str(climate_path),
        tag,
        dockerfile="Dockerfile",
        platform=platform,
    )

    # Run climate model in container
    logging.info(f"Running climate model in container {tag} with platform {platform}.")
    run_in_container(
        image=tag,
        command=[
            "Rscript",
            "climate.R",
            "--input",
            "/data/cases_with_climate.nc",
            "--gadm",
            f"/geo/{gadm_filename}",
            "--output",
            "/data/climate_cases_samples.csv",
            "--samples",
            "1000",
            "--admin1",
            ",".join(gid_1) if gid_1 else "",
        ],
        platform=platform,
        volumes={
            str(data_path): {"bind": "/data", "mode": "rw"},
            str(gadm_path): {"bind": "/geo", "mode": "ro"},
        },
    )

    # Convert output (csv) to NetCDF
    csv_file = data_path / "climate_cases_samples.csv"
    nc_file = csv_file.with_suffix(".nc")
    df = pd.read_csv(str(csv_file))
    df["Date"] = pd.to_datetime(df["Date"])
    write_nc(df, str(nc_file))

    return df
