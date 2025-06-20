from pathlib import Path
from thucia.core.containers import build_container, run_in_container

def test_containers():
    build_container(
        path=Path(__file__).parent,
        tag="thucia-test/alpine:latest",
        dockerfile="Dockerfile",
        remove=True
    )
    result, output = run_in_container(
        image="thucia-test/alpine:latest",
        command=["sh", "-c", "echo hello"],
        volumes={}
    )
    assert result == 0
    assert output == 'hello'
