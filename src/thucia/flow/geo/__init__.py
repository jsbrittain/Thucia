import logging

from prefect.futures import wait
from thucia.core import geo
from thucia.flow.wrappers import task


@task
def lookup_gid1(*args, **kwargs):
    return geo.lookup_gid1(*args, **kwargs)


@task
def remove_accents(*args, **kwargs):
    return geo.remove_accents(*args, **kwargs)


@task
def fuzzy_match_one(*args, **kwargs):
    return geo.fuzzy_match_one(*args, **kwargs)


@task
def get_admin2_list(*args, **kwargs):
    return geo.get_admin2_list(*args, **kwargs)


@task
def align_admin2_regions(*args, **kwargs):
    return geo.align_admin2_regions(*args, **kwargs)


@task
def refresh_plugins(*args, **kwargs):
    return geo.refresh_plugins(*args, **kwargs)


@task
def merge_geo_sources(*args, **kwargs):
    return geo.merge_geo_sources(*args, **kwargs)


@task
def add_incidence_rate(*args, **kwargs):
    return geo.add_incidence_rate(*args, **kwargs)


@task
def convert_to_incidence_rate(*args, **kwargs):
    return geo.convert_to_incidence_rate(*args, **kwargs)


@task
def pad_admin2(*args, **kwargs):
    return geo.pad_admin2(*args, **kwargs)


@task
def merge_sources(df, covars: list[str]) -> None:
    """
    Merge geographic and climatological covariates into the main DataFrame.
    This function is a placeholder for actual merging logic.
    """
    # Schedule source aggregation tasks
    df_covars = []
    for covar in covars:
        df_covars.append(merge_geo_sources.submit(df, [covar]))
    wait(df_covars)
    # Merge data sources
    for df_covar in df_covars:
        merge_vars = list(
            set(["GID_2", "Date"])
            | (set(df_covar.result().columns.tolist()) - set(df.columns.tolist()))
        )
        logging.info(
            f"Merging covariate {df_covar.result().columns.tolist()} into main DataFrame."
        )
        logging.info(f"Merge variables: {merge_vars}")
        df = df.merge(df_covar.result()[merge_vars], on=["GID_2", "Date"], how="left")
    return df
