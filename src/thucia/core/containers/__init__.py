import os
import docker
import logging

from podman import PodmanClient
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=1)
def get_available_container_runtime() -> str | None:
    try:
        docker.from_env()
        return "docker"
    except docker.errors.DockerException:
        pass
    try:
        uid = os.getuid()
        PodmanClient(base_url=f"unix:///run/user/{uid}/podman/podman.sock")
        return "podman"
    except RuntimeError:
        pass
    return None


def build_in_docker(path, tag, dockerfile, remove=True):
    logging.info("Building image in Docker")
    client = docker.from_env()
    image, build_logs = client.images.build(
        path=path,
        tag=tag,
        dockerfile=dockerfile,
        rm=remove,
    )
    for log in build_logs:
        if 'stream' in log:
            logging.info(log['stream'].strip())
    return image


def run_in_docker(image, command, volumes={}, remove=True):
    logging.info("Running container in Docker")
    client = docker.from_env()
    container = client.containers.run(
        image=image,
        command=command,
        volumes=volumes,
        remove=remove,
        detach=False,
    )
    exitcode = 0
    return exitcode, container.decode()


def build_in_podman(path, tag, dockerfile, remove=True):
    logging.info("Building image in Podman")
    uid = os.getuid()
    with PodmanClient(base_url=f"unix:///run/user/{uid}/podman/podman.sock") as client:
        image, build_logs = client.images.build(
            path=path,
            tag=tag,
            dockerfile=dockerfile,
            rm=remove,
        )
        for log in build_logs:
            logging.info(log.decode().strip())
    logging.info("Image built successfully")
    return image


def run_in_podman(image, command, volumes={}, remove=True):
    logging.info("Running container in Podman")
    uid = os.getuid()
    with PodmanClient(base_url=f"unix:///run/user/{uid}/podman/podman.sock") as client:
        container = client.containers.create(
            image=image,
            command=command,
            volumes=volumes,
            remove=remove,
        )
        container.start()
        exitcode = container.wait()
        logs = b"".join(container.logs()).decode()
        if exitcode == 0:
            logging.info("Container ran successfully")
            return exitcode, logs
        raise RuntimeError(f"Podman container failed with exit code {exitcode}: {logs}")


def build_container(path, tag, dockerfile="Dockerfile", remove=True):
    runtime = get_available_container_runtime()
    if runtime == "docker":
        logging.info("Using Docker")
        return build_in_docker(path, tag, dockerfile, remove)
    elif runtime == "podman":
        logging.info("Using Podman")
        return build_in_podman(path, tag, dockerfile, remove)
    else:
        raise RuntimeError("Neither Docker nor Podman is installed.")


def run_in_container(*args, **kwargs):
    runtime = get_available_container_runtime()
    if runtime == "docker":
        logging.info("Using Docker")
        exitcode, logs = run_in_docker(*args, **kwargs)
    elif runtime == "podman":
        logging.info("Using Podman")
        exitcode, logs = run_in_podman(*args, **kwargs)
    else:
        raise RuntimeError("Neither Docker nor Podman is installed.")
    return exitcode, logs
