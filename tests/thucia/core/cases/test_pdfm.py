# Probing tests for the user-supplied PDFM embeddings path.
# PDFM embeddings are not publicly distributed: the library loads whatever
# `.nc` file the user provides, filters/dedupes it, and feeds it to the
# residual-regression adapter. Real embeddings are mimicked with a synthetic
# frame shaped like the real one (GID_2 + feature0..featureN).
import numpy as np
import pandas as pd
import pytest
from thucia.core.cases import prepare_embeddings
from thucia.core.cases import prepare_pdfm_embeddings
from thucia.core.fs import write_nc
from thucia.core.pipeline import apply_residual_regression
from thucia.core.pipeline import PipelineConfig


def _embeddings_df(n_gid=3):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "GID_1": [f"BRA.{i}_1" for i in range(n_gid)],
            "GID_2": [f"BRA.{i}.{j}_2" for i in range(n_gid) for j in range(1)],
            "ADM1": ["State"] * n_gid,
            "ADM2": [f"Name{i}" for i in range(n_gid)],
            **{
                f"feature{d}": rng.normal(size=n_gid)
                for d in range(4)  # a subset of the real 0..329 dims
            },
        }
    )


@pytest.fixture
def embeddings_nc(tmp_path):
    path = tmp_path / "embeddings.nc"
    write_nc(_embeddings_df(), str(path))
    return str(path)


def test_prepare_pdfm_embeddings_loads_file(embeddings_nc):
    df = prepare_pdfm_embeddings(embeddings_nc)
    assert len(df) == 3
    assert {"GID_1", "GID_2", "ADM1", "ADM2"} <= set(df.columns)
    assert [c for c in df.columns if c.startswith("feature")] == [
        "feature0",
        "feature1",
        "feature2",
        "feature3",
    ]


def test_prepare_pdfm_embeddings_filters_provinces(embeddings_nc):
    want = _embeddings_df()["GID_2"].iloc[:2].tolist()
    df = prepare_pdfm_embeddings(embeddings_nc, provinces=want)
    assert set(df["GID_2"]) == set(want)


def test_prepare_pdfm_embeddings_dedupes_geo_codes(tmp_path):
    df = _embeddings_df(2)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate GID_2
    path = tmp_path / "dup.nc"
    write_nc(df, str(path))
    # Duplicate geo codes are an encoding error: warn and keep the first.
    with pytest.warns(UserWarning, match="encoding error"):
        out = prepare_pdfm_embeddings(str(path))
    assert len(out) == 2
    assert out["GID_2"].duplicated().sum() == 0


def test_prepare_pdfm_embeddings_missing_geo_col(tmp_path):
    df = pd.DataFrame({"ADM2": ["A"], "feature0": [1.0]})  # no GID_2
    path = tmp_path / "bad.nc"
    write_nc(df, str(path))
    with pytest.raises(ValueError, match="GID_2"):
        prepare_pdfm_embeddings(str(path))


def test_prepare_pdfm_embeddings_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_pdfm_embeddings(str(tmp_path / "nope.nc"))


def test_prepare_embeddings_dispatches_to_pdfm(embeddings_nc):
    direct = prepare_pdfm_embeddings(embeddings_nc)
    via_dispatch = prepare_embeddings(embeddings_nc, embedding_type="pdfm")
    pd.testing.assert_frame_equal(direct, via_dispatch)


def test_prepare_embeddings_unknown_type_raises(embeddings_nc):
    with pytest.raises(ValueError, match="Unknown embedding type"):
        prepare_embeddings(embeddings_nc, embedding_type="bert")


def _quantiles_df(n_dates=6, gid_2s=("BRA.0.0_2", "BRA.1.0_2")):
    dates = pd.period_range("2020-01", periods=n_dates, freq="M")
    rows = []
    for d in dates:
        for g in gid_2s:
            rows.append(
                {
                    "Date": d,
                    "GID_2": g,
                    "horizon": 1,
                    "quantile": 0.5,
                    "prediction": 10.0,
                    "Cases": 12.0,
                }
            )
    return pd.DataFrame(rows)


def test_pdfm_embeddings_to_residual_regression_end_to_end(embeddings_nc):
    # The user flow: load their embeddings file, then apply residual regression
    # to a quantile frame (predictions carry a per-GID bias).
    embeddings = prepare_pdfm_embeddings(embeddings_nc)
    df = _quantiles_df(gid_2s=embeddings["GID_2"].tolist())
    out = apply_residual_regression(
        df,
        embeddings,
        PipelineConfig(path=".", horizons=[1]),
        method="ridge",
        geo_col="GID_2",
    )
    assert len(out) == len(df)
    assert set(out["GID_2"]) == set(df["GID_2"])
    assert out["prediction"].notna().all()
    # The regression is over per-GID embeddings, so the corrected prediction
    # should differ from the flat 10.0 input.
    assert (out["prediction"] != 10.0).any()


def test_residual_regression_missing_geo_codes_subsamples(embeddings_nc):
    from thucia.core.models.utils.adapter import residual_regression

    embeddings = prepare_pdfm_embeddings(embeddings_nc)
    # Add a model province that has no embeddings: the regression must warn and
    # continue on the provinces that do have embeddings.
    df = _quantiles_df(gid_2s=embeddings["GID_2"].tolist() + ["BRA.99.99_2"])
    with pytest.warns(UserWarning, match="embeddings"):
        out = residual_regression(df, embeddings, method="ridge", geo_col="GID_2")
    assert set(out["GID_2"]) == set(embeddings["GID_2"])
    assert len(out) < len(df)
    assert out["prediction"].notna().all()


def test_residual_regression_no_provinces_have_embeddings_raises(embeddings_nc):
    from thucia.core.models.utils.adapter import residual_regression

    embeddings = prepare_pdfm_embeddings(embeddings_nc)
    df = _quantiles_df(gid_2s=["BRA.99.99_2", "BRA.98.98_2"])
    with (
        pytest.warns(UserWarning, match="embeddings"),
        pytest.raises(ValueError, match="embeddings"),
    ):
        residual_regression(df, embeddings, method="ridge", geo_col="GID_2")


def test_residual_regression_duplicate_geo_codes_warns(embeddings_nc):
    from thucia.core.models.utils.adapter import residual_regression

    embeddings = prepare_pdfm_embeddings(embeddings_nc)
    # Bypass prepare_pdfm_embeddings dedupe (as a raw file with dupes would):
    dup = pd.concat([embeddings, embeddings.iloc[[0]]], ignore_index=True)
    df = _quantiles_df(gid_2s=embeddings["GID_2"].tolist())
    with pytest.warns(UserWarning, match="encoding error"):
        out = residual_regression(df, dup, method="ridge", geo_col="GID_2")
    assert len(out) == len(df)
    assert out["prediction"].notna().all()
