import numpy as np
import pandas as pd
import pytest
from thucia.core.cases import check_covars_for_nans
from thucia.core.cases import check_index_combinations
from thucia.core.cases import r2
from thucia.core.cases import r2_score
from thucia.core.cases import rmse_score
from thucia.core.cases import wis


def _quantile_frame(gid="A", n=6, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.period_range("2020-01", periods=n, freq="M")
    rows = []
    for i, d in enumerate(dates):
        for q in [0.05, 0.5, 0.95]:
            rows.append(
                {
                    "GID_2": gid,
                    "Date": d,
                    "quantile": q,
                    "prediction": rng.uniform(0, 10),
                    "Cases": 5.0 + i,
                }
            )
    return pd.DataFrame(rows)


def test_r2_score_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert r2_score(y, y) == pytest.approx(1.0)


def test_rmse_score_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse_score(y, y) == pytest.approx(0.0)


def test_r2_wrapper_median_only():
    # r2() must restrict to q=0.5 before scoring
    df = _quantile_frame()
    out = r2(df, "prediction", "Cases", group_col="GID_2")
    assert isinstance(out, pd.DataFrame)
    assert "R2" in out.columns
    assert len(out) == 1


def test_r2_wrapper_overall():
    df = _quantile_frame()
    val = r2(df, "prediction", "Cases", group_col=None)
    assert np.isfinite(val)


def test_r2_constant_target_returns_zero():
    # Zero-variance target must not produce -inf/nan (sklearn convention: 0.0)
    y = np.array([3.0, 3.0, 3.0])
    assert r2_score(y, y) == 0.0
    df = _quantile_frame()
    df["Cases"] = 5.0  # force constant target
    val = r2(df, "prediction", "Cases", group_col=None)
    assert val == 0.0


def test_wis_wrapper_matches_module():
    from thucia.core.cases.wis import wis_bracher

    df = _quantile_frame()
    wrapped = wis(df, "prediction", "Cases", geo_col="GID_2")
    direct = wis_bracher(
        df[["GID_2", "Date", "quantile", "prediction", "Cases"]],
        group_cols=("GID_2", "Date"),
    )
    assert np.allclose(
        wrapped.sort_values("Date")["WIS"].values,
        direct.sort_values("Date")["WIS"].values,
    )


def test_check_index_combinations_raises_on_gap():
    df = pd.DataFrame(
        {
            "Date": pd.period_range("2020-01", periods=2, freq="M").repeat(2),
            "GID_2": ["A", "B"] * 2,
        }
    )
    check_index_combinations(df, ["Date", "GID_2"])  # complete, no error
    gap = df[~((df["Date"] == pd.Period("2020-01", "M")) & (df["GID_2"] == "B"))]
    with pytest.raises(ValueError, match="Missing index combinations"):
        check_index_combinations(gap, ["Date", "GID_2"])


def test_check_covars_for_nans_raises():
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]})
    check_covars_for_nans(df, ["b"])  # fine
    with pytest.raises(ValueError, match="Missing covariate"):
        check_covars_for_nans(df, ["a"])
