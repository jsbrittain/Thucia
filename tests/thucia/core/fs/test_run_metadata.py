# Probe tests for run metadata stored alongside saved artefacts.
import json

import pandas as pd
import pytest
from thucia.core.fs import DataFrame
from thucia.core.fs import read_db
from thucia.core.fs import read_nc
from thucia.core.fs import read_run_metadata
from thucia.core.fs import write_db
from thucia.core.fs import write_nc
from thucia.core.fs.metadata import data_digest
from thucia.core.fs.metadata import dumps
from thucia.core.fs.metadata import jsonable


def _frame():
    return pd.DataFrame(
        {
            "Date": pd.period_range("2020-01", periods=3, freq="M"),
            "Cases": [1.0, 2.0, 3.0],
        }
    )


def test_metadata_roundtrip_duckdb(tmp_path):
    path = tmp_path / "run.duckdb"
    docs = {
        "selection": {"name": "baseline", "label": None},
        "config": {"horizons": [1], "start_date": "2019-01"},
    }
    write_db(_frame(), path, metadata=docs)

    assert read_run_metadata(path) == docs
    # the wrapper exposes the same documents, and data reads are unaffected
    assert read_db(path).metadata == docs
    assert len(read_db(path)) == 3


def test_missing_metadata_is_empty(tmp_path):
    path = tmp_path / "plain.duckdb"
    write_db(_frame(), path)

    assert read_run_metadata(path) == {}
    assert read_db(path).metadata == {}
    # files that do not exist (yet) read as empty rather than raising
    assert read_run_metadata(tmp_path / "absent.duckdb") == {}


def test_metadata_survives_append(tmp_path):
    path = tmp_path / "run.duckdb"
    write_db(_frame(), path, metadata={"selection": {"name": "baseline"}})

    db = DataFrame(db_path=path)
    db.append(_frame())

    assert len(db) == 6
    assert read_run_metadata(path) == {"selection": {"name": "baseline"}}


def test_metadata_replaced_or_merged(tmp_path):
    path = tmp_path / "run.duckdb"
    write_db(_frame(), path, metadata={"a": 1})

    db = DataFrame(db_path=path)
    db.set_metadata({"b": 2})
    assert db.metadata == {"b": 2}

    db.set_metadata({"c": 3}, replace=False)
    assert db.metadata == {"b": 2, "c": 3}


def test_write_without_metadata_preserves_documents(tmp_path):
    path = tmp_path / "run.duckdb"
    write_db(_frame(), path, metadata={"a": 1})
    write_db(_frame(), path)

    assert read_run_metadata(path) == {"a": 1}


def test_metadata_roundtrip_netcdf(tmp_path):
    path = tmp_path / "run.nc"
    frame = _frame()
    frame.attrs["run_metadata"] = {"selection": {"name": "chronos"}}
    write_nc(frame, path)

    assert read_run_metadata(path) == {"selection": {"name": "chronos"}}
    assert read_nc(path).attrs["run_metadata"] == {"selection": {"name": "chronos"}}


def test_metadata_roundtrip_zarr(tmp_path):
    pytest.importorskip("zarr")
    from thucia.core.fs import read_zarr
    from thucia.core.fs import write_zarr

    path = tmp_path / "run.zarr"
    frame = _frame()
    frame.attrs["run_metadata"] = {"selection": {"name": "baseline"}}
    write_zarr(frame, path)

    assert read_run_metadata(path) == {"selection": {"name": "baseline"}}
    assert read_zarr(path).attrs["run_metadata"] == {"selection": {"name": "baseline"}}


def test_dumps_is_deterministic_and_json_safe():
    doc = {"b": 1, "a": pd.Period("2020-01", freq="M"), "c": float("nan")}
    out = dumps(doc)

    assert out == dumps({"a": doc["a"], "b": 1, "c": None})
    assert json.loads(out)["c"] is None
    assert json.loads(out)["a"] == {"period": "2020-01", "freq": "M"}


def test_jsonable_handles_frames_sets_and_paths(tmp_path):
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    out = jsonable({"regions": frame, "cols": {"b", "a"}, "path": tmp_path})

    assert out["regions"] == {
        "kind": "frame",
        "rows": 2,
        "columns": ["a", "b"],
    }
    assert out["cols"] == ["a", "b"]
    assert out["path"] == str(tmp_path)


def test_data_digest_tracks_values_not_row_order():
    frame = _frame()
    shuffled = frame.iloc[::-1].reset_index(drop=True)

    assert data_digest(frame, ["Date", "Cases"]) == data_digest(
        shuffled, ["Date", "Cases"]
    )
    changed = frame.assign(Cases=frame["Cases"] + 1)
    assert data_digest(frame, ["Date", "Cases"]) != data_digest(
        changed, ["Date", "Cases"]
    )
    assert data_digest(frame, ["not_a_column"]) is None
