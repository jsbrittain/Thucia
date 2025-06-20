import logging

import pandas as pd
from thucia.core.containers import run_in_container


def climate(
    df: pd.DataFrame,
    gid_1: list[str] | None = None,
) -> pd.DataFrame:
    for date in df["date"].unique():
        logging.info(f"Launching climate container for date: {date}")
        run_in_container(
            # image="docker.io/library/r-base:latest",
            image="climate:latest",
            command=["Rscript", "-e", "print('hello')"],
            volumes={
                # "/local/path": {"bind": "/scripts", "mode": "ro"}
            },
        )
