# Run provenance for fitted models.
#
# Every fitted quantile artefact can carry the parameters that produced it:
# which model (and with which per-run configuration), the effective pipeline
# config, a digest of the inputs, the running implementation and the software
# environment. The documents are plain JSON-able mappings so they can be written
# into the DuckDB file's run-metadata table without a schema change every time a
# config field is added.
from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import subprocess
from datetime import datetime
from datetime import timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any
from typing import Mapping

import pandas as pd
from thucia.core.fs.metadata import data_digest
from thucia.core.fs.metadata import jsonable
from thucia.core.models._meta import ModelSelection
from thucia.core.pipeline.config import PipelineConfig

#: Columns fed to the input digest when present (order, geo unit, outcome).
_DIGEST_COLUMNS = ("Date", "Cases")

#: Files excluded from the source digest. `_version.py` is the conventional
#: setuptools-scm generated module: it changes per *build* of the same commit,
#: so including it would make the digest build-dependent rather than
#: implementation-dependent.
_DIGEST_EXCLUDE_NAMES = frozenset({"_version.py"})

#: Packages whose versions can change forecast output under an identical Thucia
#: implementation (probabilistic backends and the numerics stack).
_DEPENDENCY_PROBES = ("darts", "torch", "statsmodels", "numpy", "pandas", "duckdb")


def selection_document(selection: ModelSelection) -> dict[str, Any]:
    """How this run's model was selected and configured."""
    return {
        "name": selection.name,
        "label": selection.label,
        "overrides": dict(selection.overrides),
        "params": dict(selection.params),
    }


def _regions_document(regions: Any) -> Any:
    """Describe a region map without embedding geometry in the metadata."""
    if regions is None:
        return None
    if isinstance(regions, pd.DataFrame):
        return {
            "kind": "frame",
            "rows": int(len(regions)),
            "columns": [str(c) for c in regions.columns],
        }
    return {"kind": "path", "path": jsonable(Path(regions))}


def config_document(config: PipelineConfig) -> dict[str, Any]:
    """The effective ``PipelineConfig`` (per-model overrides applied)."""
    doc: dict[str, Any] = {}
    for f in dataclasses.fields(config):
        value = getattr(config, f.name)
        if f.name == "regions":
            doc["regions"] = _regions_document(value)
        else:
            doc[f.name] = jsonable(value)
    return doc


def inputs_document(
    df: pd.DataFrame,
    *,
    geo_col: str,
    case_col: str = "Cases",
) -> dict[str, Any]:
    """Shape and structural digest of the frame the model was fitted on."""
    doc: dict[str, Any] = {
        "rows": int(len(df)),
        "geo_units": int(df[geo_col].nunique()) if geo_col in df.columns else None,
    }
    if "Date" in df.columns:
        doc["periods"] = int(df["Date"].nunique())
        doc["start"] = jsonable(df["Date"].min())
        doc["end"] = jsonable(df["Date"].max())
    observed = df
    if "future" in df.columns:
        observed = df[~df["future"]]
    doc["digest"] = data_digest(observed, [*_DIGEST_COLUMNS, geo_col, case_col])
    return doc


# ---------------------------------------------------------------------------
# Implementation identity (which Thucia is running?)
# ---------------------------------------------------------------------------


def _package_root() -> Path | None:
    """Directory of the imported ``thucia`` package (None if unavailable)."""
    try:
        import thucia
    except Exception:  # pragma: no cover - import failures are reported elsewhere
        return None
    path = getattr(thucia, "__file__", None)
    return Path(path).resolve().parent if path else None


def _source_digest(root: Path) -> str | None:
    """Content digest of the ``*.py`` files under ``root`` (sorted, path-tagged)."""
    entries = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts or path.name in _DIGEST_EXCLUDE_NAMES:
            continue
        rel = path.relative_to(root).as_posix()
        entries.append((rel, hashlib.sha256(path.read_bytes()).hexdigest()))
    if not entries:
        return None
    digest = hashlib.sha256()
    for rel, file_hash in entries:
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(file_hash.encode())
    return f"sha256:{digest.hexdigest()}"


def _record_digest() -> str | None:
    """Fallback digest from the installed distribution's ``RECORD`` hashes.

    Used when the package is not an ordinary directory (e.g. a zipped install).
    The basis differs from a source digest, so it is recorded alongside it.
    """
    try:
        dist = importlib_metadata.distribution("thucia")
    except importlib_metadata.PackageNotFoundError:
        return None
    pairs = sorted(
        (str(entry), entry.hash.value)
        for entry in (dist.files or [])
        if entry.hash is not None
    )
    if not pairs:
        return None
    digest = hashlib.sha256()
    for path, file_hash in pairs:
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update(file_hash.encode())
    return f"sha256:{digest.hexdigest()}"


def package_digest(root: Path | None = None) -> tuple[str | None, str | None]:
    """``(digest, basis)`` for the running package; ``(None, None)`` if unknown.

    A fingerprint of the imported code rather than of a version string, so it
    identifies the implementation even for an editable checkout with a dirty
    working tree — and it needs no git repository. When no usable package
    directory can be found (e.g. a zipped install) the installed distribution's
    ``RECORD`` hashes are used instead and the basis says so.
    """
    explicit = root is not None
    root = _package_root() if root is None else root
    if root is not None and root.is_dir():
        digest = _source_digest(root)
        if digest is not None:
            return digest, "sources"
    elif explicit:
        return None, None
    digest = _record_digest()
    return digest, ("record" if digest else None)


def _git(*args: str, cwd: Path) -> str | None:
    """Run a git command, returning stdout (``""`` when empty, None on failure)."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def repo_state(root: Path | None) -> dict[str, Any] | None:
    """Commit and dirty flag for a source checkout, or None for an install.

    The work tree is only accepted when the running package *is* the checked-out
    source (``<toplevel>/src/thucia`` or ``<toplevel>/thucia``). Without that
    guard a wheel installed into a project-local ``.venv`` would inherit the
    surrounding repository's commit, which would be a fabricated pin.
    """
    if root is None:
        return None
    toplevel = _git("rev-parse", "--show-toplevel", cwd=root)
    if not toplevel:
        return None
    toplevel_path = Path(toplevel).resolve()
    candidates = {
        (toplevel_path / "src" / "thucia").resolve(),
        (toplevel_path / "thucia").resolve(),
    }
    if root.resolve() not in candidates:
        return None

    commit = _git("rev-parse", "HEAD", cwd=toplevel_path)
    if commit is None:
        return None
    status = _git("status", "--porcelain", "--untracked-files=no", cwd=toplevel_path)
    return {
        "id": commit,
        "short": commit[:7],
        "describe": _git("describe", "--tags", "--always", cwd=toplevel_path),
        # None (not False) when the state could not be read: a clean claim must
        # be earned, never assumed.
        "dirty": None if status is None else bool(status),
    }


def _direct_url() -> dict[str, Any] | None:
    """PEP 610 ``direct_url.json`` for VCS/URL/local installs (absent for indexes)."""
    try:
        dist = importlib_metadata.distribution("thucia")
        text = dist.read_text("direct_url.json")
    except (importlib_metadata.PackageNotFoundError, OSError):
        return None
    if not text:
        return None
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        return None
    return doc if isinstance(doc, dict) else None


def _install_source(
    direct_url: Mapping[str, Any] | None,
    repo: Mapping[str, Any] | None,
    installed: bool,
) -> str:
    """Classify how the running package was installed.

    ``editable`` / ``local`` / ``git`` / ``archive`` / ``url`` come from the
    PEP 610 record; ``index`` is a registry (e.g. PyPI) install, which carries
    no direct URL; ``source`` is an uninstalled source tree on ``sys.path``.
    """
    if direct_url:
        if "vcs_info" in direct_url:
            return "git"
        dir_info = direct_url.get("dir_info") or {}
        if dir_info:
            return "editable" if dir_info.get("editable") else "local"
        if "archive_info" in direct_url:
            return "archive"
        return "url"
    if not installed:
        return "source"
    return "index"


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _dependency_versions() -> dict[str, str | None]:
    return {name: _package_version(name) for name in _DEPENDENCY_PROBES}


def code_document() -> dict[str, Any]:
    """Identify the implementation that is running.

    Prefer ``digest`` (content) over ``distribution.version`` (a claim): the
    version is build-time metadata and goes stale for editable installs, while
    the digest describes the code actually imported. ``commit`` is only
    populated when the running package *is* a git checkout, and is ``None`` for
    wheel/index installs — where the version and digest are the pins.
    """
    root = _package_root()
    digest, basis = package_digest(root)
    direct_url = _direct_url()
    version = _package_version("thucia")
    repo = repo_state(root)
    return {
        "source": _install_source(direct_url, repo, version is not None),
        "distribution": {"name": "Thucia", "version": version} if version else None,
        "commit": repo,
        "direct_url": direct_url,
        "digest": digest,
        "digest_basis": basis,
        "deps": _dependency_versions(),
    }


def environment_document() -> dict[str, Any]:
    """Software environment that produced the artefact (implementation is in ``code``)."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# Pinning / verification
# ---------------------------------------------------------------------------


def _dig(doc: Mapping[str, Any], dotted: str) -> Any:
    node: Any = doc
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _commits_match(left: str, right: str) -> bool:
    """Compare commits by prefix (short vs full SHAs)."""
    return left == right or left.startswith(right) or right.startswith(left)


#: Fields compared by :func:`verify_code_identity`, in reporting order.
_VERIFIED_FIELDS = (
    "source",
    "digest",
    "digest_basis",
    "distribution.name",
    "distribution.version",
    "commit.id",
)


def verify_code_identity(
    recorded: Mapping[str, Any],
    *,
    current: Mapping[str, Any] | None = None,
    include_deps: bool = False,
) -> dict[str, tuple[Any, Any]]:
    """Compare a recorded ``code`` document with the running implementation.

    Returns ``{field: (recorded, current)}`` for every mismatch; an empty dict
    means the artefact was produced by the implementation that is running now.
    Fields the recording could not observe (``None`` — e.g. ``commit.id`` for an
    index install) are skipped rather than reported as mismatches. Set
    ``include_deps`` to also compare the numerics-stack versions.
    """
    current = code_document() if current is None else current
    mismatches: dict[str, tuple[Any, Any]] = {}
    for field in _VERIFIED_FIELDS:
        want, have = _dig(recorded, field), _dig(current, field)
        if want is None or have is None:
            continue
        if field == "commit.id":
            if not _commits_match(str(want), str(have)):
                mismatches[field] = (want, have)
        elif want != have:
            mismatches[field] = (want, have)
    if include_deps:
        for name in _DEPENDENCY_PROBES:
            want = _dig(recorded, f"deps.{name}")
            have = _dig(current, f"deps.{name}")
            if want is not None and have is not None and want != have:
                mismatches[f"deps.{name}"] = (want, have)
    return mismatches


def check_expected_code(
    *,
    expect: str | Path | Mapping[str, Any] | None = None,
    expect_commit: str | None = None,
    expect_version: str | None = None,
) -> None:
    """Raise when the running implementation does not match a requested pin.

    ``expect`` accepts a saved artefact (path) or a metadata mapping holding a
    ``code`` document (as returned by ``read_run_metadata``), and compares the
    full identity. ``expect_commit`` requires a source/git install at that
    commit *with a clean working tree*; ``expect_version`` compares the
    installed distribution version (the pin for a registry install).
    """
    if expect is None and expect_commit is None and expect_version is None:
        return

    current = code_document()

    if expect is not None:
        recorded = _as_code_document(expect)
        mismatches = verify_code_identity(recorded, current=current)
        if mismatches:
            detail = ", ".join(
                f"{field}: recorded {want!r} != current {have!r}"
                for field, (want, have) in mismatches.items()
            )
            raise ValueError(
                f"thucia implementation does not match the pinned artefact ({detail})"
            )

    version = _dig(current, "distribution.version")
    if expect_version is not None and expect_version != version:
        raise ValueError(
            f"thucia version {version!r} does not match the pinned {expect_version!r}"
        )

    if expect_commit is not None:
        commit = _dig(current, "commit.id")
        if commit is None:
            raise ValueError(
                f"pinned commit {expect_commit!r} cannot be checked: this install "
                f"carries no git commit (source: {current.get('source')!r}). "
                "Pin by version or by an artefact's code digest instead."
            )
        if not _commits_match(str(commit), expect_commit):
            raise ValueError(
                f"thucia commit {str(commit)[:7]} does not match the pinned "
                f"{expect_commit[:7]}"
            )
        if _dig(current, "commit.dirty"):
            raise ValueError(
                f"thucia is at the pinned commit {str(commit)[:7]} but the working "
                "tree is dirty; the running code is not that commit."
            )


def _as_code_document(expect: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    """Normalise a path / metadata mapping / code document to a code document."""
    if isinstance(expect, (str, Path)):
        from thucia.core.fs import read_run_metadata

        metadata = read_run_metadata(expect)
    else:
        metadata = expect
    if isinstance(metadata, Mapping) and "code" in metadata:
        code = metadata["code"]
        if isinstance(code, Mapping):
            return code
    if isinstance(metadata, Mapping):
        return metadata
    raise TypeError(
        "expect must be a saved artefact path or a run-metadata mapping, "
        f"got {type(expect).__name__}"
    )


def run_metadata_docs(
    selection: ModelSelection,
    config: PipelineConfig,
    inputs: pd.DataFrame | None = None,
) -> dict[str, Mapping[str, Any]]:
    """Documents to record alongside a fitted model.

    ``selection``, ``config``, ``code`` and ``environment`` are always present;
    ``inputs`` is added when the fitted frame is supplied.
    """
    docs: dict[str, Any] = {
        "selection": selection_document(selection),
        "config": config_document(config),
        "code": code_document(),
        "environment": environment_document(),
    }
    if inputs is not None:
        docs["inputs"] = inputs_document(
            inputs, geo_col=config.geo_col, case_col="Cases"
        )
    return docs
