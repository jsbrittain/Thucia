import logging
import argparse
import numpy as np

from pathlib import Path
from thucia.viz import plot_all_admin2
from thucia.core.cases import read_nc, cases_per_month, write_nc, r2
from thucia.core.models import baseline, climate, sarima, tcn, samples_to_quantiles
from thucia.core.geo import (
    lookup_gid1,
    merge_geo_sources,
    pad_admin2,
    add_incidence_rate,
    convert_to_incidence_rate,
)


logging.basicConfig(level=logging.INFO)


def run_pipeline(iso3: str, adm1: list[str] | None = None):
    # Lookup GID-1 for the specified admin-1 regions
    gid_1 = lookup_gid1(adm1, iso3=iso3) if adm1 else None
    logging.info(f"Using GID-1 regions: {gid_1}")

    def file_path(filename):
        return Path("data") / "cases" / iso3 / filename

    # Read case data
    if not file_path("cases.nc").exists():
        logging.info(
            "Cases data not found, please run the data preparation script first."
        )
        return

    logging.info("Loading cases data...")
    df = read_nc(file_path("cases.nc"))

    # Monthly aggregation and save
    if Path(file_path("cases_per_month.nc")).exists():
        logging.info("Cases per month data already exists, loading from cache...")
        df = read_nc(file_path("cases_per_month.nc"))
    else:
        df = cases_per_month(df)
        write_nc(df, file_path("cases_per_month.nc"))

    # Ensure all Admin-2 regions are included, even with zero cases
    # This increasing processing but allows us to plot all regions
    df = pad_admin2(df)

    # Merge climate data
    if Path(file_path("cases_with_climate.nc")).exists():
        logging.info("Cases with climate data already exist, loading from cache...")
        df = read_nc(file_path("cases_with_climate.nc"))
    else:
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
        write_nc(df, file_path("cases_with_climate.nc"))

    # We use ONI for El Nino, but it would be nice to show here how to substitute local
    # coastal temperature data if available, such as ICEN for Peru.

    # Baseline model (and save)
    if Path(file_path("baseline_cases_samples.nc")).exists():
        logging.info("Baseline samples already exist, loading from cache...")
        df_baseline_cases_samples = read_nc(file_path("baseline_cases_samples.nc"))
        df_baseline_cases_quantiles = read_nc(file_path("baseline_cases_quantiles.nc"))
        df_baseline_dir_samples = read_nc(file_path("baseline_dir_samples.nc"))
        df_baseline_dir_quantiles = read_nc(file_path("baseline_dir_quantiles.nc"))
    else:
        # Case analysis
        df_baseline_cases_samples = baseline(df, gid_1=gid_1)
        write_nc(df_baseline_cases_samples, file_path("baseline_cases_samples.nc"))
        df_baseline_cases_quantiles = samples_to_quantiles(df_baseline_cases_samples)
        write_nc(df_baseline_cases_quantiles, file_path("baseline_cases_quantiles.nc"))
        # Dengue Incidence Rates
        df_baseline_dir_samples = convert_to_incidence_rate(
            df_baseline_cases_samples, df
        )
        write_nc(df_baseline_dir_samples, file_path("baseline_dir_samples.nc"))
        df_baseline_dir_quantiles = samples_to_quantiles(df_baseline_dir_samples)
        write_nc(df_baseline_dir_quantiles, file_path("baseline_dir_quantiles.nc"))

    breakpoint()

    # Climate model
    if Path(file_path("climate_cases_samples.nc")).exists():
        logging.info("Climate samples already exist, loading from cache...")
        # df_climate_samples = read_nc(file_path("climate_samples.nc"))
        # df_climate_quantiles = read_nc(file_path("climate_quantiles.nc"))
    else:
        df_climate_samples = climate(df, gid_1=gid_1)
        write_nc(df_climate_samples, file_path("climate_cases_samples.nc"))
        df_climate_quantiles = samples_to_quantiles(df_climate_samples)
        write_nc(df_climate_quantiles, file_path("climate_cases_quantiles.nc"))

    breakpoint()

    # SARIMA model
    if Path(file_path("sarima_cases_samples.nc")).exists():
        logging.info("SARIMA samples already exist, loading from cache...")
        df_sarima_cases_samples = read_nc(file_path("sarima_cases_samples.nc"))
        df_sarima_cases_quantiles = read_nc(file_path("sarima_cases_quantiles.nc"))
    else:
        df_sarima_cases_samples = sarima(df, gid_1=gid_1)
        write_nc(df_sarima_cases_samples, file_path("sarima_cases_samples.nc"))
        df_sarima_cases_quantiles = samples_to_quantiles(df_sarima_cases_samples)
        write_nc(df_sarima_cases_quantiles, file_path("sarima_cases_quantiles.nc"))

    # TCN model
    if Path(file_path("tcn_cases_samples.nc")).exists():
        logging.info("TCN samples already exist, loading from cache...")
        df_tcn_cases_samples = read_nc(file_path("tcn_cases_samples.nc"))
        df_tcn_cases_quantiles = read_nc(file_path("tcn_cases_quantiles.nc"))
    else:
        df_tcn_cases_samples = tcn(df, gid_1=gid_1, retrain=False)
        write_nc(df_tcn_cases_samples, file_path("tcn_cases_samples.nc"))
        df_tcn_cases_quantiles = samples_to_quantiles(df_tcn_cases_samples)
        write_nc(df_tcn_cases_quantiles, file_path("tcn_cases_quantiles.nc"))

    # TimeGPT model
    if Path(file_path("timegpt_cases_samples.nc")).exists():
        logging.info("TimeGPT samples already exist, loading from cache...")
    else:
        logging.info("Skipping TimeGPT model (not implemented)...")
        # df_timegpt_samples = timegpt(df, gid_1=gid_1)
        # write_nc(df_timegpt_samples, file_path("timegpt_samples.nc"))
        # df_timegpt_quantiles = samples_to_quantiles(df_timegpt_samples)
        # write_nc(df_timegpt_quantiles, file_path("timegpt_quantiles.nc"))

    # # Province timeseries
    # plot_all_admin2(df_baseline_cases_quantiles, metric='quantiles')
    plot_regions = False
    r2_baseline = r2(
        df_baseline_cases_quantiles,
        "prediction",
        "Cases",
        transform=np.log1p,
        df_filter={"quantile": 0.50},
    )
    print(f"R-squared: {r2_baseline:.3f}")
    if plot_regions:
        plot_all_admin2(
            df_baseline_cases_quantiles,
            transform=np.log1p,
        )

    r2_sarima = r2(
        df_sarima_cases_quantiles,
        "prediction",
        "Cases",
        transform=np.log1p,
        df_filter={"quantile": 0.50},
    )
    print(f"R-squared: {r2_sarima:.3f}")
    if plot_regions:
        plot_all_admin2(
            df_sarima_cases_quantiles,
            transform=np.log1p,
        )

    r2_tcn = r2(
        df_tcn_cases_quantiles,
        "prediction",
        "Cases",
        transform=np.log1p,
        df_filter={"quantile": 0.50},
    )
    print(f"R-squared: {r2_tcn:.3f}")
    if plot_regions:
        plot_all_admin2(
            df_tcn_cases_quantiles,
            transform=np.log1p,
        )

    breakpoint()


def main():
    parser = argparse.ArgumentParser(
        description="Run analysis pipeline for dengue cases."
    )
    parser.add_argument(
        "--iso3",
        type=str,
        default="PER",
        help="ISO3 country code (default: 'PER')",
    )
    parser.add_argument(
        "--adm1",
        type=str,
        nargs="*",
        default=None,  # example: --adm1 "Piura" "Tumbes" "Lambayeque"
        help="List of admin-1 regions (default: None)",
    )
    args = parser.parse_args()
    run_pipeline(
        iso3=args.iso3,
        adm1=args.adm1,
    )


# Some pre-defined pipelines...


# To call: python -m pipeline.analysis.run_peru_northwest
def run_peru_northwest():
    run_pipeline("PER", ["Piura", "Tumbes", "Lambayeque"])


# To call: python -m pipeline.analysis.run_mex_oaxaca
def run_mex_oaxaca():
    run_pipeline(iso3="MEX", adm1=["Oaxaca"])


if __name__ == "__main__":
    main()
