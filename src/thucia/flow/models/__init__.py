from thucia.core import models
from thucia.flow.wrappers import task


@task
def baseline(*args, **kwargs):
    return models.baseline(*args, **kwargs)


@task
def climate(*args, **kwargs):
    return models.climate(*args, **kwargs)


@task
def sarima(*args, **kwargs):
    return models.sarima(*args, **kwargs)


@task
def tcn(*args, **kwargs):
    return models.tcn(*args, **kwargs)


@task
def samples_to_quantiles(*args, **kwargs):
    return models.samples_to_quantiles(*args, **kwargs)


@task
def quantiles(*args, **kwargs):
    return models.quantiles(*args, **kwargs)


@task
def run_model(*args, **kwargs):
    return models.run_model(*args, **kwargs)
