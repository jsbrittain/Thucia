from thucia.core import cases
from thucia.flow.wrappers import task


@task
def cases_per_month(*args, **kwargs):
    return cases.cases_per_month(*args, **kwargs)


@task
def write_nc(*args, **kwargs):
    return cases.write_nc(*args, **kwargs)


@task
def read_nc(*args, **kwargs):
    return cases.read_nc(*args, **kwargs)


@task
def r2(*args, **kwargs):
    return cases.r2(*args, **kwargs)


@task
def run_job(*args, **kwargs):
    return cases.run_job(*args, **kwargs)
