import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from thucia.core.fs import DataFrame
from thucia.core.fs import read_nc
from thucia.core.fs import write_nc


def test_in_memory():
    # In-memory dataframe
    tdf = DataFrame(
        df=pd.DataFrame(
            {
                "Date": ["2023-01-01", "2023-01-02"],
                "GID_2": ["A", "B"],
            }
        )
    )
    assert len(tdf) == 2


def test_write_df():
    db_file = Path(tempfile.NamedTemporaryFile().name)
    assert not db_file.exists()
    tdf = DataFrame(db_file=db_file)
    tdf.write_df(
        pd.DataFrame(
            {
                "Date": ["2023-01-01", "2023-01-02"],
                "GID_2": ["A", "B"],
            }
        ),
    )
    assert len(tdf) == 2
    assert db_file.exists()
    db_file.unlink()


def test_append_new():
    tdf = DataFrame()
    assert len(tdf) == 0
    tdf.append(
        pd.DataFrame(
            {
                "Date": ["2023-01-03", "2023-01-04"],
                "GID_2": ["C", "D"],
            }
        ),
    )
    assert len(tdf) == 2


def test_append_existing():
    tdf = DataFrame()
    assert len(tdf) == 0
    categories = ["A", "B", "C", "D"]
    tdf.write_df(
        pd.DataFrame(
            {  # Categories must be fully defined on the first write
                "Date": ["2023-01-01", "2023-01-02"],
                "GID_2": pd.Categorical(["A", "B"], categories=categories),
            }
        ),
    )
    assert len(tdf) == 2
    tdf.append(
        pd.DataFrame(
            {
                "Date": ["2023-01-03", "2023-01-04"],
                "GID_2": pd.Categorical(["C", "D"], categories=categories),
            }
        ),
    )
    assert len(tdf) == 4


def test_accessor_col():
    # Test direct accessors on DataFrame object (should load from query)
    tdf = DataFrame(
        df=pd.DataFrame(
            {
                "Date": ["2023-01-01", "2023-01-02"],
                "GID_2": ["A", "B"],
            }
        ),
    )
    assert len(tdf) == 2
    assert (tdf["Date"] == pd.Series(["2023-01-01", "2023-01-02"])).all()
    assert (tdf["GID_2"] == pd.Series(["A", "B"])).all()


def test_accessor_cols():
    # Test direct accessors on DataFrame object (should load from query)
    tdf = DataFrame(
        df=pd.DataFrame(
            {
                "Date": ["2023-01-01", "2023-01-02"],
                "GID_2": ["A", "B"],
            }
        ),
    )
    assert len(tdf) == 2
    assert (tdf["Date", "GID_2"] == tdf.df).all().all()


def test_accessor_slice():
    # Test direct accessors on DataFrame object (should load from query)
    tdf = DataFrame(
        df=pd.DataFrame(
            {
                "Date": ["2023-01-01", "2023-01-02"],
                "GID_2": ["A", "B"],
            }
        ),
    )
    assert len(tdf) == 2
    assert tdf[0:2].equals(tdf.df)


def test_accessor_boolean_vector():
    # Test direct accessors on DataFrame object (should load from query)
    tdf = DataFrame(
        df=pd.DataFrame(
            {
                "Date": ["2023-01-01", "2023-01-02"],
                "GID_2": ["A", "B"],
            }
        ),
    )
    assert len(tdf) == 2
    assert tdf[pd.Series([True, True])].equals(tdf.df)
    assert (
        tdf[pd.Series([True, False])]
        .reset_index(drop=True)
        .equals(tdf.df.iloc[[0]].reset_index(drop=True))
    )
    assert (
        tdf[pd.Series([False, True])]
        .reset_index(drop=True)
        .equals(tdf.df.iloc[[1]].reset_index(drop=True))
    )
    assert len(tdf[pd.Series([False, False])]) == 0


def test_head():
    tdf = DataFrame(df=pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    out = tdf.head(2)
    assert len(out) == 2
    assert out["a"].tolist() == [1, 2]


# --- NetCDF round-trips ---


def test_nc_roundtrip_embedding_frame(tmp_path):
    df = pd.DataFrame(
        {
            "GID_2": ["A", "B"],
            "feature0": [1.0, 2.0],
            "feature1": [0.5, -0.5],
        }
    )
    path = tmp_path / "emb.nc"
    write_nc(df, str(path))
    out = read_nc(str(path))
    for col in ["GID_2", "feature0", "feature1"]:
        assert out[col].tolist() == df[col].tolist()


def test_nc_roundtrip_period_date_column(tmp_path):
    df = pd.DataFrame(
        {
            "Date": pd.period_range("2020-01", periods=3, freq="M").repeat(2),
            "GID_2": ["A", "B"] * 3,
            "Cases": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    path = tmp_path / "cases.nc"
    write_nc(df, str(path))
    out = read_nc(str(path))
    # the Period column must survive the netCDF round-trip
    assert out["Date"].dtype == "period[M]"
    assert (out["Date"].to_numpy() == df["Date"].to_numpy()).all()
    assert len(out) == len(df)
    assert np.allclose(out["Cases"], df["Cases"])


def test_read_nc_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_nc(str(tmp_path / "does_not_exist.nc"))
