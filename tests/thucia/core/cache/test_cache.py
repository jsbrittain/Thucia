import pandas as pd
import pytest
from thucia.core.cache import Cache


@pytest.fixture
def cache_file(tmp_path):
    return tmp_path / "cache.sqlite"


def _records():
    return pd.DataFrame(
        [
            {"metric": "tmax", "Date": "2020-01-01", "GID_2": "A", "mean": 25.5},
            {"metric": "tmax", "Date": "2020-02-01", "GID_2": "A", "mean": 26.0},
            {"metric": "prec", "Date": "2020-01-01", "GID_2": "B", "mean": 100.0},
        ]
    )


def _cache(cache_file):
    return Cache(
        "sqlite",
        cache_file=cache_file,
        columntypes={
            "metric": "TEXT",
            "Date": "TEXT",
            "GID_2": "TEXT",
            "mean": "REAL",
        },
        keys=["metric", "Date", "GID_2"],
    )


def test_roundtrip_add_and_get(cache_file):
    c = _cache(cache_file)
    c.add_records(_records())
    got = c.get_record("tmax", pd.Timestamp("2020-01-01"), "A")
    assert got is not None
    assert got["mean"].tolist() == pytest.approx([25.5])


def test_get_record_missing_returns_none(cache_file):
    c = _cache(cache_file)
    c.add_records(_records())
    assert c.get_record("tmax", pd.Timestamp("1999-01-01"), "A") is None


def test_empty_cache_returns_empty(cache_file):
    c = _cache(cache_file)
    assert c.get_record("tmax", pd.Timestamp("2020-01-01"), "A") is None
    empty = c.get_records({"metric": ["tmax"], "Date": ["2020-01-01"], "GID_2": ["A"]})
    assert empty.empty


def test_get_records_batched_missing_keys_omitted(cache_file):
    c = _cache(cache_file)
    c.add_records(_records())
    got = c.get_records(
        {
            "metric": ["tmax", "tmax"],
            "Date": ["2020-01-01", "2020-02-01"],
            "GID_2": ["A", "Z"],
        }
    )
    # 'Z' has no record -> only the hit row returned
    assert len(got) == 1
    assert got["GID_2"].tolist() == ["A"]


def test_cache_registry_unknown_type(cache_file):
    with pytest.raises(ValueError):
        Cache(
            "not-a-cache",
            cache_file=cache_file,
            columntypes={"metric": "TEXT"},
            keys=["metric"],
        )


def test_sqlite_cache_replaces_on_add(cache_file):
    c = _cache(cache_file)
    c.add_records(_records())
    c.add_records(
        pd.DataFrame(
            [{"metric": "tmax", "Date": "2020-01-01", "GID_2": "A", "mean": 99.0}]
        )
    )
    got = c.get_record("tmax", pd.Timestamp("2020-01-01"), "A")
    assert got["mean"].tolist() == pytest.approx([99.0])
