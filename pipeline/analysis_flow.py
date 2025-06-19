from pathlib import Path

from prefect.futures import wait
from thucia.flow import models
from thucia.flow.cases import cases_per_month
from thucia.flow.cases import read_nc
from thucia.flow.geo import add_incidence_rate
from thucia.flow.geo import lookup_gid1
from thucia.flow.geo import merge_geo_sources
from thucia.flow.geo import pad_admin2
from thucia.flow.logging import enable_prefect_logging_redirect
from thucia.flow.models import run_model
from thucia.flow.wrappers import flow


enable_prefect_logging_redirect()


@flow(name="Dengue forecast pipeline")
def run_pipeline(iso3: str, adm1: list[str] | None = None):
    gid_1 = lookup_gid1(adm1, iso3=iso3) if adm1 else None
    path = Path("data") / "cases" / iso3

    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = read_nc(path / "cases.nc")

    df = cases_per_month(df)
    df = pad_admin2(df)
    df = merge_geo_sources(
        df,
        [
            "worldclim.*",  # tmin, tmax, prec
            "edo.spi6",  # SPI6
            "noaa.oni",  # TotalONI, AnomONI
            "worldpop.pop_count",  # counts to 2020, unconstrained estimates beyond
        ],
    )
    df = add_incidence_rate(df)

    # Run models in parallel (submit tasks and wait)
    model_tasks = [
        run_model.submit("baseline", models.baseline, df, path, gid_1=gid_1),
        # run_model.submit('climate', models.climate, df, path, gid_1=gid_1),
        run_model.submit("sarima", models.sarima, df, path, gid_1=gid_1),
        run_model.submit("tcn", models.tcn, df, path, gid_1=gid_1, retrain=False),
        # run_model.submit('timegpt', timegpt.tcn, df, path, gid_1=gid_1),
    ]
    # Await results
    wait(model_tasks)


# Some pre-defined pipelines...


def run_peru_northwest():
    run_pipeline("PER", ["Piura", "Tumbes", "Lambayeque"])


def run_mex_oaxaca():
    run_pipeline(iso3="MEX", adm1=["Oaxaca"])


if __name__ == "__main__":
    run_peru_northwest()
