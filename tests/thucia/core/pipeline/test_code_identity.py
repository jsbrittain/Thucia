# Probe tests for implementation identity ("which Thucia produced this?").
#
# The pin by content digest must work with no git repository (wheel installs),
# the commit must never be fabricated for a package installed into a repo's own
# venv, and a dirty tree must not pass a commit pin.
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
from thucia.core.fs import read_run_metadata
from thucia.core.pipeline import fit_model
from thucia.core.pipeline import PipelineConfig
from thucia.core.pipeline.provenance import _source_digest
from thucia.core.pipeline.provenance import check_expected_code
from thucia.core.pipeline.provenance import code_document
from thucia.core.pipeline.provenance import package_digest
from thucia.core.pipeline.provenance import repo_state
from thucia.core.pipeline.provenance import verify_code_identity

GIT = shutil.which("git")


def _inputs(n_periods: int = 24) -> pd.DataFrame:
    idx = pd.period_range("2016-01", periods=n_periods, freq="M")
    cases = np.tile(np.arange(float(n_periods)), 2)
    df = pd.DataFrame(
        {
            "Date": idx.repeat(2),
            "GID_1": ["G.1_1"] * 2 * n_periods,
            "GID_2": ["G.1.1_2", "G.1.2_2"] * n_periods,
            "future": [False] * 2 * n_periods,
            "Cases": cases,
            "Log_Cases": np.log1p(cases),
        }
    )
    df["GID_1"] = df["GID_1"].astype("category")
    df["GID_2"] = df["GID_2"].astype("category")
    return df


def _config(tmp_path) -> PipelineConfig:
    return PipelineConfig(
        path=tmp_path,
        start_date=pd.Period("2017-06", freq="M"),
        horizons=[1],
        num_samples=50,
    )


def _git(*args, cwd) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


@pytest.fixture
def source_repo(tmp_path):
    """A throwaway repo laid out like Thucia (``src/thucia``)."""
    if GIT is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "repo"
    package = repo / "src" / "thucia"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("X = 1\n")
    (package / "core.py").write_text("Y = 2\n")
    (repo / "pyproject.toml").write_text("[project]\nname = 'thucia'\n")
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "initial", cwd=repo)
    return repo, package


# ---------------------------------------------------------------------------
# Content digest
# ---------------------------------------------------------------------------


def test_source_digest_tracks_content(source_repo):
    _, package = source_repo
    before = _source_digest(package)
    assert before.startswith("sha256:")

    (package / "core.py").write_text("Y = 3\n")
    assert _source_digest(package) != before

    # a new file changes it too
    (package / "extra.py").write_text("Z = 4\n")
    assert _source_digest(package) not in (before, None)


def test_source_digest_ignores_bytecode_and_generated_version(source_repo):
    _, package = source_repo
    before = _source_digest(package)

    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "core.cpython-313.pyc").write_bytes(b"noise")
    (package / "_version.py").write_text("__version__ = '0.1.dev1+gabc'\n")

    # bytecode and the setuptools-scm generated module are build artefacts, not
    # implementation: the digest must not move when either appears
    assert _source_digest(package) == before


def test_source_digest_none_without_sources(tmp_path):
    assert _source_digest(tmp_path) is None


def test_package_digest_basis_is_sources_for_a_directory(source_repo):
    _, package = source_repo
    digest, basis = package_digest(package)
    assert basis == "sources"
    assert digest.startswith("sha256:")


def test_package_digest_none_for_a_missing_package(tmp_path):
    assert package_digest(tmp_path / "not_a_package") == (None, None)


# ---------------------------------------------------------------------------
# Commit detection
# ---------------------------------------------------------------------------


def test_repo_state_reads_commit_and_clean_tree(source_repo):
    repo, package = source_repo
    state = repo_state(package)

    assert state is not None
    assert state["id"] == _git("rev-parse", "HEAD", cwd=repo)
    assert state["short"] == state["id"][:7]
    assert state["dirty"] is False


def test_repo_state_flags_a_dirty_tree(source_repo):
    _, package = source_repo
    (package / "core.py").write_text("Y = 99\n")

    assert repo_state(package)["dirty"] is True


def test_repo_state_ignores_untracked_files(source_repo):
    _, package = source_repo
    (package / "scratch.py").write_text("temporary\n")

    assert repo_state(package)["dirty"] is False


def test_repo_state_does_not_borrow_an_enclosing_repo(source_repo, tmp_path):
    # A wheel installed into a project's own .venv sits *inside* the repo but is
    # not the checked-out source: claiming the repo's commit would be fabricated.
    repo, _ = source_repo
    installed = repo / ".venv" / "lib" / "python3.13" / "site-packages" / "thucia"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text("X = 1\n")

    assert repo_state(installed) is None


def test_repo_state_none_outside_a_repository(tmp_path):
    package = tmp_path / "src" / "thucia"
    package.mkdir(parents=True)
    assert repo_state(package) is None


# ---------------------------------------------------------------------------
# The recorded document and verification
# ---------------------------------------------------------------------------


def test_code_document_describes_the_running_install():
    doc = code_document()

    assert doc["source"] in {"editable", "local", "git", "index", "source", "url"}
    assert doc["digest"].startswith("sha256:")
    assert doc["digest_basis"] in {"sources", "record"}
    assert doc["distribution"]["name"] == "Thucia"
    # the numerics stack that can change output under an identical commit
    for dep in ("numpy", "pandas", "duckdb"):
        assert dep in doc["deps"]


def test_verify_identity_accepts_the_running_implementation():
    assert verify_code_identity(code_document()) == {}


def test_verify_identity_reports_a_changed_digest():
    recorded = code_document()
    recorded["digest"] = "sha256:deadbeef"

    mismatches = verify_code_identity(recorded)
    assert set(mismatches) == {"digest"}
    assert mismatches["digest"][0] == "sha256:deadbeef"


def test_verify_identity_skips_unobserved_fields():
    # an index install records no commit; that is "not applicable", not a mismatch
    recorded = {**code_document(), "commit": None}
    assert "commit.id" not in verify_code_identity(recorded)


def test_verify_identity_compares_short_and_full_commits():
    current = {**code_document(), "commit": {"id": "a" * 40, "dirty": False}}
    short = {**current, "commit": {"id": "a" * 7, "dirty": False}}

    assert verify_code_identity(short, current=current) == {}


def test_verify_identity_can_compare_dependencies():
    current = {**code_document(), "deps": {"numpy": "2.0.0"}}
    recorded = {**code_document(), "deps": {"numpy": "1.0.0"}}

    assert "deps.numpy" not in verify_code_identity(recorded, current=current)
    mismatches = verify_code_identity(recorded, current=current, include_deps=True)
    assert mismatches["deps.numpy"] == ("1.0.0", "2.0.0")


# ---------------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------------


def test_version_pin_matches_and_mismatches():
    version = code_document()["distribution"]["version"]

    check_expected_code(expect_version=version)  # matching pin is a no-op
    with pytest.raises(ValueError, match="does not match the pinned"):
        check_expected_code(expect_version="0.0.0-not-installed")


def test_commit_pin_requires_a_source_checkout():
    doc = code_document()
    if doc["commit"] is not None:
        # running from a checkout: pin the commit, and expect a dirty-tree refusal
        # only when the tree actually is dirty
        if doc["commit"]["dirty"]:
            with pytest.raises(ValueError, match="working tree is dirty"):
                check_expected_code(expect_commit=doc["commit"]["id"])
        else:
            check_expected_code(expect_commit=doc["commit"]["id"])
        with pytest.raises(ValueError, match="does not match the pinned"):
            check_expected_code(expect_commit="0" * 40)
    else:
        with pytest.raises(ValueError, match="carries no git commit"):
            check_expected_code(expect_commit="0" * 40)


def test_commit_pin_refuses_an_index_install(monkeypatch):
    import thucia.core.pipeline.provenance as provenance

    index_doc = {**code_document(), "source": "index", "commit": None}
    monkeypatch.setattr(provenance, "code_document", lambda: index_doc)

    with pytest.raises(ValueError, match="carries no git commit"):
        check_expected_code(expect_commit="0" * 40)


def test_fit_model_pins_to_a_saved_artefact(tmp_path):
    df = _inputs()
    fit_model(df, "baseline", _config(tmp_path))
    saved = tmp_path / "baseline_cases_quantiles.duckdb"

    # the artefact records the implementation that produced it
    assert read_run_metadata(saved)["code"]["digest"] == code_document()["digest"]

    fit_model(df, "baseline", _config(tmp_path), expect=saved)  # matches

    tampered = read_run_metadata(saved)
    tampered["code"]["digest"] = "sha256:deadbeef"
    with pytest.raises(ValueError, match="does not match the pinned artefact"):
        fit_model(df, "baseline", _config(tmp_path), expect=tampered)


def test_fit_model_refuses_a_wrong_version_pin(tmp_path):
    with pytest.raises(ValueError, match="does not match the pinned"):
        fit_model(_inputs(), "baseline", _config(tmp_path), expect_version="0.0.0-nope")
