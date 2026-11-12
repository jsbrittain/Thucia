# Run-metadata serialisation helpers.
#
# Saved artefacts carry a small set of JSON documents describing how they were
# produced (model selection, pipeline config, input digest, environment). These
# helpers normalise arbitrary Python values to JSON-safe ones and compute a
# cheap structural digest of the model inputs, so provenance never depends on
# ad-hoc ``str()``/``repr()`` representations.
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from datetime import date
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Mapping

import numpy as np
import pandas as pd

#: Key used to carry run metadata on a pandas frame (``df.attrs``) and in the
#: netCDF/Zarr Dataset attrs.
METADATA_ATTR = "run_metadata"

#: Name of the DuckDB table holding run metadata (one row per document).
RUN_METADATA_TABLE = "__run_metadata__"

#: Dataset-attr key used to carry run metadata through netCDF/Zarr, which can
#: only store flat attributes.
ATTRS_KEY = "thucia_run_metadata"


def _path_repr(path: Path) -> str:
    """Render a path relative to the working directory when it is below it."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except (ValueError, OSError):
        return str(path)


def _float_repr(value: float) -> float | None:
    # JSON has no NaN/Inf; provenance stores those as null rather than emitting
    # an unparseable document.
    return None if (math.isnan(value) or math.isinf(value)) else value


def jsonable(obj: Any) -> Any:
    """Recursively convert ``obj`` into JSON-serialisable values.

    Handles dataclasses, mappings, ``Path``, ``pd.Period``/``Timestamp`` and
    numpy scalars/arrays. Values with no natural JSON form are preserved as
    ``{"type": ..., "repr": ...}`` rather than raising, so recording provenance
    can never fail during a (potentially long) fit.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return _float_repr(obj)
    if isinstance(obj, Path):
        return _path_repr(obj)
    if isinstance(obj, pd.Period):
        return {"period": str(obj), "freq": obj.freqstr}
    if isinstance(obj, (pd.Timestamp, datetime, date, np.datetime64)):
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [jsonable(v) for v in sorted(obj, key=repr)]
    if isinstance(obj, pd.DataFrame):
        # A region map (or any frame) is recorded structurally: provenance must
        # not embed megabytes of geometry.
        return {
            "kind": "frame",
            "rows": int(len(obj)),
            "columns": [str(c) for c in obj.columns],
        }
    return {"type": type(obj).__name__, "repr": repr(obj)}


def dumps(obj: Any) -> str:
    """Encode a document (or value) as deterministic JSON."""
    return json.dumps(jsonable(obj), sort_keys=True, allow_nan=False)


def loads(text: str) -> Any:
    """Decode a document written by :func:`dumps`."""
    return json.loads(text)


def data_digest(df: pd.DataFrame, columns) -> str | None:
    """Cheap structural digest of ``df[columns]`` (None when no column matches).

    Lets a saved fit be checked against the inputs it was produced from: the
    digest is order-insensitive by construction (row-wise hashes are summed by
    ``hash_pandas_object``) and independent of dtypes that cannot be hashed
    directly (Periods/categoricals are stringified first).
    """
    cols = list(dict.fromkeys(c for c in columns if c in df.columns))
    if not cols:
        return None
    sub = df[cols].copy()
    for col in sub.columns:
        if isinstance(sub[col].dtype, pd.PeriodDtype) or isinstance(
            sub[col].dtype, pd.CategoricalDtype
        ):
            sub[col] = sub[col].astype(str)
    hashed = pd.util.hash_pandas_object(sub, index=False)
    # Sort the row hashes so the digest describes the *set* of observations
    # (a re-sorted input frame is the same fit) while still being sensitive to
    # any value change.
    ordered = np.sort(hashed.to_numpy())
    return hashlib.sha256(ordered.tobytes()).hexdigest()[:16]
