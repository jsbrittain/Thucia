import logging
import os
from functools import lru_cache

import docker
from podman import PodmanClient


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


def build_in_docker(path, tag, dockerfile="Dockerfile", remove=True, *args, **kwargs):
    logging.info("Building image in Docker")
    client = docker.from_env()
    image, build_logs = client.images.build(
        path=path,
        tag=tag,
        dockerfile=dockerfile,
        rm=remove,
        *args,
        **kwargs,
    )
    for log in build_logs:
        if "stream" in log:
            logging.info(log["stream"].strip())
    return image


# def run_in_docker_attached(image, command, volumes={}, remove=True, *args, **kwargs):
#     logging.info("Running container in Docker")
#     client = docker.from_env()
#     container = client.containers.run(
#         image=image,
#         command=command,
#         volumes=volumes,
#         remove=remove,
#         detach=False,
#         *args,
#         **kwargs,
#     )
#     exitcode = 0
#     return exitcode, container.decode()


def run_in_docker(image, command, volumes={}, remove=True, *args, **kwargs):
    logging.info("Running container in Docker")
    client = docker.from_env()

    container = client.containers.run(
        image=image,
        command=command,
        volumes=volumes,
        remove=False,  # Needed to stream logs
        detach=True,  # Needed to stream logs
        *args,
        **kwargs,
    )

    try:
        # Stream logs line by line
        for line in container.logs(stream=True, follow=True):
            print(line.decode().rstrip())

        # Wait for container to finish and get exit code
        result = container.wait()
        exitcode = result.get("StatusCode", 1)

        logs = container.logs().decode()
    finally:
        if remove:
            container.remove()

    return exitcode, logs


def build_in_podman(path, tag, dockerfile="Dockerfile", remove=True, *args, **kwargs):
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


# def run_in_podman_attached(image, command, volumes={}, remove=True, *args, **kwargs):
#     logging.info("Running container in Podman")
#     uid = os.getuid()
#     with PodmanClient(base_url=f"unix:///run/user/{uid}/podman/podman.sock") as client:
#         container = client.containers.create(
#             image=image,
#             command=command,
#             volumes=volumes,
#             remove=remove,
#             *args,
#             **kwargs,
#         )
#         container.start()
#         exitcode = container.wait()
#         logs = b"".join(container.logs()).decode()
#         if exitcode == 0:
#             logging.info("Container ran successfully")
#             return exitcode, logs
#         raise RuntimeError(f"Podman container failed with exit code {exitcode}: {logs}")


def run_in_podman(image, command, volumes={}, remove=True, *args, **kwargs):
    logging.info("Running container in Podman")
    uid = os.getuid()
    with PodmanClient(base_url=f"unix:///run/user/{uid}/podman/podman.sock") as client:
        container = client.containers.create(
            image=image,
            command=command,
            volumes=volumes,
            remove=remove,
            *args,
            **kwargs,
        )
        container.start()

        # Stream logs line by line
        logs = []
        for line in container.logs(stream=True, follow=True):
            decoded_line = line.decode().rstrip()
            print(decoded_line)  # or logging.info(decoded_line)
            logs.append(decoded_line)

        # Wait for the container to finish
        exitcode = container.wait()

        if exitcode == 0:
            logging.info("Container ran successfully")
            return exitcode, "\n".join(logs)

        raise RuntimeError(
            f"Podman container failed with exit code {exitcode}:\n" + "\n".join(logs)
        )


def build_container(*args, **kwargs):
    runtime = get_available_container_runtime()
    if runtime == "docker":
        logging.info("Using Docker")
        return build_in_docker(*args, **kwargs)
    elif runtime == "podman":
        logging.info("Using Podman")
        return build_in_podman(*args, **kwargs)
    else:
        raise RuntimeError("No supported container runtime found.")


def run_in_container(*args, **kwargs):
    runtime = get_available_container_runtime()
    if runtime == "docker":
        logging.info("Using Docker")
        exitcode, logs = run_in_docker(*args, **kwargs)
    elif runtime == "podman":
        logging.info("Using Podman")
        exitcode, logs = run_in_podman(*args, **kwargs)
    else:
        raise RuntimeError("No supported container runtime found.")
    return exitcode, logs
