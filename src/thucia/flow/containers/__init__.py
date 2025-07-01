from thucia.flow import containers
from thucia.flow.wrappers import task


@task
def get_available_container_runtime():
    containers.get_available_container_runtime()


@task
def build_in_docker(*args, **kwargs):
    containers.build_in_docker(*args, **kwargs)


@task
def run_in_docker(*args, **kwargs):
    containers.run_in_docker(*args, **kwargs)


@task
def build_in_podman(*args, **kwargs):
    containers.build_in_podman(*args, **kwargs)


@task
def run_in_podman(*args, **kwargs):
    containers.run_in_podman(*args, **kwargs)


@task
def build_container(*args, **kwargs):
    containers.build_container(*args, **kwargs)


@task
def run_in_container(*args, **kwargs):
    containers.run_in_container(*args, **kwargs)
