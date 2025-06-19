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
