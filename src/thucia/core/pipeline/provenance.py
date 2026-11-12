# Run provenance for fitted models.
#
# Every fitted quantile artefact can carry the parameters that produced it:
# which model (and with which per-run configuration), the effective pipeline
# config, a digest of the inputs, and the software environment. The documents
# are plain JSON-able mappings so they can be written into the DuckDB file's
# run-metadata table without a schema change every time a config field is added.
from __future__ import annotations

import dataclasses
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


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def environment_document() -> dict[str, Any]:
    """Software environment that produced the artefact."""
    return {
        "thucia": _package_version("thucia"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git": _git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def run_metadata_docs(
    selection: ModelSelection,
    config: PipelineConfig,
    inputs: pd.DataFrame | None = None,
) -> dict[str, Mapping[str, Any]]:
    """Documents to record alongside a fitted model.

    ``selection``, ``config`` and ``environment`` are always present;
    ``inputs`` is added when the fitted frame is supplied.
    """
    docs: dict[str, Any] = {
        "selection": selection_document(selection),
        "config": config_document(config),
        "environment": environment_document(),
    }
    if inputs is not None:
        docs["inputs"] = inputs_document(
            inputs, geo_col=config.geo_col, case_col="Cases"
        )
    return docs
