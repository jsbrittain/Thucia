# Probing tests for the container-runtime module, using mocks so no Docker or
# Podman runtime is required. Exercises runtime auto-detection, the
# RuntimeError when no runtime is available, and the docker/podman run paths'
# exit-code handling. The real `test_containers.py` still builds/runs an alpine
# image when a runtime is present.
import types

import pytest
import thucia.core.containers as C
from docker.errors import DockerException


@pytest.fixture(autouse=True)
def clear_runtime_cache():
    C.get_available_container_runtime.cache_clear()
    yield
    C.get_available_container_runtime.cache_clear()


def _no_daemon():
    # docker.from_env() raises DockerException when no daemon is reachable.
    return lambda *a, **k: (_ for _ in ()).throw(DockerException("unable to connect"))


class _PodmanUnavailable:
    def __init__(self, base_url=None):
        raise RuntimeError("no podman socket")


class _PodmanAvailable:
    def __init__(self, base_url=None):
        self.base_url = base_url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_default_auto_detect_docker(monkeypatch):
    monkeypatch.setattr(C.docker, "from_env", lambda *a, **k: "client")
    assert C.get_available_container_runtime() == "docker"


def test_auto_detect_podman_when_no_docker(monkeypatch):
    monkeypatch.setattr(C.docker, "from_env", _no_daemon())
    monkeypatch.setattr(C, "PodmanClient", _PodmanAvailable)
    assert C.get_available_container_runtime() == "podman"


def test_auto_detect_podman_uses_user_socket(monkeypatch):
    import os

    monkeypatch.setattr(C.docker, "from_env", _no_daemon())
    seen = {}

    class Client:
        def __init__(self, base_url=None):
            seen["url"] = base_url

    monkeypatch.setattr(C, "PodmanClient", Client)
    C.get_available_container_runtime()
    assert seen["url"] == f"unix:///run/user/{os.getuid()}/podman/podman.sock"


def test_no_runtime_returns_none(monkeypatch):
    monkeypatch.setattr(C.docker, "from_env", _no_daemon())
    monkeypatch.setattr(C, "PodmanClient", _PodmanUnavailable)
    assert C.get_available_container_runtime() is None


def test_build_container_raises_without_runtime(monkeypatch):
    monkeypatch.setattr(C.docker, "from_env", _no_daemon())
    monkeypatch.setattr(C, "PodmanClient", _PodmanUnavailable)
    with pytest.raises(RuntimeError, match="No supported container runtime"):
        C.build_container(path=".", tag="t:latest")


def test_run_in_container_raises_without_runtime(monkeypatch):
    monkeypatch.setattr(C.docker, "from_env", _no_daemon())
    monkeypatch.setattr(C, "PodmanClient", _PodmanUnavailable)
    with pytest.raises(RuntimeError, match="No supported container runtime"):
        C.run_in_container(image="img", command=["true"])


class _FakeContainer:
    def __init__(self, exitcode=0, output=b"hello world", wait_int=False):
        self._exitcode = exitcode
        self._output = output
        self._wait_int = wait_int
        self.removed = False

    def logs(self, stream=False, follow=False):
        if stream or follow:
            return iter([self._output])
        return self._output

    def wait(self):
        # docker containers.wait() returns a StatusCode dict; podman returns an
        # int.
        if self._wait_int:
            return self._exitcode
        return {"StatusCode": self._exitcode}

    def remove(self):
        self.removed = True


class _FakeDockerClient:
    def __init__(self, container):
        self.container = container

    @property
    def containers(self):
        return types.SimpleNamespace(run=lambda *a, **k: self.container)


def test_run_in_docker_success(monkeypatch):
    container = _FakeContainer(exitcode=0, output=b"hello world")
    client = _FakeDockerClient(container)
    monkeypatch.setattr(
        C, "docker", types.SimpleNamespace(from_env=lambda *a, **k: client)
    )
    exitcode, output = C.run_in_docker(image="img", command=["true"])
    assert exitcode == 0
    assert output == "hello world"
    assert container.removed is True


def test_run_in_docker_nonzero_exit(monkeypatch):
    container = _FakeContainer(exitcode=1, output=b"oops")
    client = _FakeDockerClient(container)
    monkeypatch.setattr(
        C, "docker", types.SimpleNamespace(from_env=lambda *a, **k: client)
    )
    exitcode, output = C.run_in_docker(image="img", command=["false"])
    assert exitcode == 1
    assert output == "oops"


def test_run_in_container_docker_success(monkeypatch):
    container = _FakeContainer(exitcode=0, output=b"hello world")
    client = _FakeDockerClient(container)
    monkeypatch.setattr(
        C, "docker", types.SimpleNamespace(from_env=lambda *a, **k: client)
    )
    monkeypatch.setattr(C, "get_available_container_runtime", lambda: "docker")
    exitcode, output = C.run_in_container(image="img", command=["true"])
    assert exitcode == 0
    assert output == "hello world"


def _podman_client(container):
    class Client:
        def __init__(self, base_url=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    client = Client()
    client.containers = types.SimpleNamespace(create=lambda *a, **k: container)
    return client


def test_run_in_podman_success(monkeypatch):
    from thucia.core.containers import run_in_podman

    container = _FakeContainer(exitcode=0, output=b"podman ok", wait_int=True)
    container.start = lambda: None
    monkeypatch.setattr(C, "PodmanClient", lambda *a, **k: _podman_client(container))
    exitcode, output = run_in_podman(image="img", command=["true"])
    assert exitcode == 0
    assert output == "podman ok"


def test_run_in_podman_nonzero_raises(monkeypatch):
    from thucia.core.containers import run_in_podman

    container = _FakeContainer(exitcode=2, output=b"boom", wait_int=True)
    container.start = lambda: None
    monkeypatch.setattr(C, "PodmanClient", lambda *a, **k: _podman_client(container))
    with pytest.raises(RuntimeError, match="exit code 2"):
        run_in_podman(image="img", command=["false"])
