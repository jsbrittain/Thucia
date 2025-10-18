import tempfile
from pathlib import Path

import pandas as pd
from thucia.core.fs import DataFrame


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
