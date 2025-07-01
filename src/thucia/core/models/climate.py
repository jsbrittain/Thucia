import inspect
import logging
import os
from pathlib import Path

import pandas as pd
import thucia
from thucia.core.cases import write_nc
from thucia.core.containers import build_container
from thucia.core.containers import run_in_container

MAX_WORKERS = int(os.environ.get("THUCIA_CLIMATE_MAX_WORKERS", os.cpu_count()))


def climate(
    df: pd.DataFrame,
    gid_1: list[str] | None = None,
) -> pd.DataFrame:
    climate_path = Path(inspect.getfile(climate)).parent / "climate"
    data_path = (
        Path(inspect.getfile(thucia)).parent.parent.parent / "data" / "cases" / "PER"
    )
    tag = "thucia/model/climate:latest"
    platform = "linux/amd64"
    build_container(
        str(climate_path),
        tag,
        dockerfile="Dockerfile",
        platform=platform,
    )

    logging.info(f"Running climate model in container {tag} with platform {platform}.")
    run_in_container(
        image=tag,
        command=[
            "Rscript",
            "climate.R",
            "--input",
            "/data/cases_with_climate.nc",
            "--output",
            "/data/climate_cases_samples.csv",
            "--samples",
            "1000",
            "--admin1",
            ",".join(gid_1) if gid_1 else "",
        ],
        platform=platform,
        volumes={str(data_path): {"bind": "/data", "mode": "rw"}},
    )

    # Convert output (csv) to NetCDF
    csv_file = data_path / "climate_cases_samples.csv"
    nc_file = csv_file.with_suffix(".nc")
    df = pd.read_csv(str(csv_file))
    df["Date"] = pd.to_datetime(df["Date"])
    write_nc(df, str(nc_file))

    return df
