# Heavy real-model forecast fits. Run with `uv run pytest -m slow` (excluded
# from the default/CI run via `addopts = "-m 'not slow'"`).
import importlib

import numpy as np
import pandas as pd
import pytest
from forecast_data import assert_forecast_alignment
from forecast_data import COVARIATE_COLS
from forecast_data import make_forecast_df
from thucia.core.cases.wis import wis_bracher
from thucia.core.models import chronos
from thucia.core.models import nbeats
from thucia.core.models import sarima
from thucia.core.models import xgboost
from thucia.core.models.chronos import CHRONOS_QUANTILES
from thucia.core.models.darts.foundation import TIMESFM_QUANTILES
from thucia.core.quantiles import quantiles

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def df():
    return make_forecast_df()


def _as_df(out):
    return out.df if hasattr(out, "df") else out


def test_nbeats_forecast(df):
    out = _as_df(
        nbeats(
            df,
            start_date=pd.Period("2020-07", freq="M"),
            train_start_date=pd.Period("2016-01", freq="M"),
            train_end_date=pd.Period("2020-06", freq="M"),
            geo_parent_filter=None,
            horizons=[1],
            case_col="Log_Cases",
            covariate_cols=COVARIATE_COLS,
            retrain=False,
            db_file=None,
            train_col="GID_2",
            num_samples=20,
            multivariate=False,
        )
    )
    assert sorted(out["quantile"].unique()) == quantiles
    assert np.isfinite(out["prediction"]).all()
    assert (out["prediction"] >= 0).all()


def test_xgboost_multihorizon():
    # output_chunk_length=12 (max horizon) needs >= 48 lags + 12 months of
    # training history, hence the longer fixture.
    df = make_forecast_df(n_periods=96, start="2014-01")
    out = _as_df(
        xgboost(
            df,
            start_date=pd.Period("2020-01", freq="M"),
            train_start_date=pd.Period("2014-01", freq="M"),
            train_end_date=pd.Period("2019-12", freq="M"),
            geo_parent_filter=None,
            horizons=[1, 3, 6, 12],
            case_col="Log_Cases",
            covariate_cols=COVARIATE_COLS,
            retrain=False,
            db_file=None,
            train_col="GID_2",
            num_samples=20,
            multivariate=False,
        )
    )
    assert sorted(out["horizon"].unique()) == [1, 3, 6, 12]
    assert sorted(out["quantile"].unique()) == quantiles
    assert np.isfinite(out["prediction"]).all()


@pytest.mark.network
def test_timesfm3_forecast_requires_torch_licence_and_network(df):
    # TimesFM 3.0 downloads a HuggingFace checkpoint and needs the
    # non-commercial licence acknowledged; opt-in only.
    pytest.importorskip("torch")
    try:
        from thucia.core.models import timesfm3
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"timesfm3 unavailable: {exc}")
    out = _as_df(
        timesfm3(
            df,
            start_date=pd.Period("2020-07", freq="M"),
            geo_parent_filter=None,
            horizons=[1],
            case_col="Log_Cases",
            covariate_cols=COVARIATE_COLS,
            retrain=False,
            db_file=None,
            accept_license=True,
        )
    )
    # Native pre-trained grid, ordered, and scoreable.
    assert sorted(out["quantile"].unique()) == sorted(TIMESFM_QUANTILES)
    assert np.isfinite(out["prediction"]).all()
    med = out[out["quantile"] == 0.5].set_index(["GID_2", "Date"])["prediction"]
    wide = out.pivot_table(
        index=["GID_2", "Date"],
        columns="quantile",
        values="prediction",
        observed=True,
    )
    assert (wide.diff(axis=1).iloc[:, 1:] >= -1e-6).all().all()
    assert np.isfinite(med).all()


@pytest.mark.network
def test_timesfm25_forecast_ignores_covariates(df):
    # TimesFM 2.5 takes no covariates: they are dropped with a warning, and the
    # forecast still covers the 9-level pre-trained grid.
    pytest.importorskip("torch")
    try:
        from thucia.core.models import timesfm25
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"timesfm25 unavailable: {exc}")
    with pytest.warns(UserWarning, match="does not support covariates"):
        out = _as_df(
            timesfm25(
                df,
                start_date=pd.Period("2020-07", freq="M"),
                geo_parent_filter=None,
                horizons=[1],
                case_col="Log_Cases",
                covariate_cols=COVARIATE_COLS,
                retrain=False,
                db_file=None,
            )
        )
    assert sorted(out["quantile"].unique()) == sorted(TIMESFM_QUANTILES)
    assert np.isfinite(out["prediction"]).all()


def test_sarima_weekly_seasonal():
    # SARIMA on a weekly cadence with an explicit 52-period seasonal length;
    # the weekly anchor must be preserved.
    df = make_forecast_df(freq="W-SAT", n_periods=208, start="2016-01-02", n_gid=1)
    out = _as_df(
        sarima(
            df,
            start_date=pd.Period("2019-12-07", freq="W-SAT"),
            geo_parent_filter=None,
            horizons=[1],
            case_col="Log_Cases",
            covariate_cols=COVARIATE_COLS,
            retrain=False,
            db_file=None,
            train_col="GID_2",
            num_samples=20,
            season_length=52,
        )
    )
    assert str(out["Date"].dtype) == "period[W-SAT]"
    assert sorted(out["quantile"].unique()) == quantiles
    assert np.isfinite(out["prediction"]).all()


@pytest.mark.network
def test_chronos_forecast_direct_quantiles(df):
    # Chronos-2 is zero-shot but downloads a HuggingFace checkpoint; opt-in.
    start = pd.Period("2020-07", freq="M")
    out = _as_df(
        chronos(
            df,
            start_date=start,
            train_start_date=pd.Period("2016-01", freq="M"),
            train_end_date=pd.Period("2020-06", freq="M"),
            geo_parent_filter=None,
            horizons=[1],
            case_col="Log_Cases",
            covariate_cols=COVARIATE_COLS,
            retrain=False,
            db_file=None,
            train_col="GID_2",
        )
    )

    # Direct quantile parameters on the documented (canonical) subset.
    assert sorted(out["quantile"].unique()) == sorted(CHRONOS_QUANTILES)
    assert np.isfinite(out["prediction"]).all()
    assert (out["prediction"] >= 0).all()

    # Vector outputs are aligned to the target Date, not the forecast origin.
    assert_forecast_alignment(out, df, start)

    scored = out[out["prediction"].notna()]
    wis = wis_bracher(
        scored[["GID_2", "Date", "quantile", "prediction", "Cases"]],
        group_cols=("GID_2", "Date"),
    )
    assert np.isfinite(wis["WIS"]).all()


@pytest.mark.network
def test_chronos_quantile_subset_and_median_only(df):
    # Requesting a subset of the pre-trained grid keeps exactly those levels,
    # and median_only switches to a deterministic single-level forecast.
    common = dict(
        start_date=pd.Period("2020-07", freq="M"),
        geo_parent_filter=None,
        horizons=[1],
        case_col="Log_Cases",
        covariate_cols=COVARIATE_COLS,
        retrain=False,
        db_file=None,
    )
    subset = _as_df(chronos(df, quantiles=[0.05, 0.5, 0.95], **common))
    assert sorted(subset["quantile"].unique()) == [0.05, 0.5, 0.95]

    median = _as_df(chronos(df, median_only=True, **common))
    assert sorted(median["quantile"].unique()) == [0.5]
    assert np.isfinite(median["prediction"]).all()


@pytest.mark.network
def test_chronos_vector_output_alignment():
    # `last_points_only=True` must keep the *target* step of the horizon vector,
    # dated at origin + horizon (the timesfm-style alignment).
    mod = importlib.import_module("thucia.core.models.chronos")
    df = make_forecast_df(n_periods=60, n_gid=1)
    horizon = 3

    model = mod.ChronosQuantilesDarts(
        df=df,
        case_col="Log_Cases",
        geo_col="GID_2",
        geo_parent="GID_1",
        covariate_cols=COVARIATE_COLS,
        horizons=[1],
        train_start_date=pd.Period("2016-01", freq="M"),
        train_end_date=pd.Period("2020-06", freq="M"),
        quantiles=CHRONOS_QUANTILES,
    )
    model.model = model.build_model(horizon=horizon)
    model.pre_fit(target_gids=[df["GID_2"].iloc[0]])

    ts, cov, _ = model.get_cases()
    start = pd.Period("2020-07", freq="M").to_timestamp(how="end")
    target_aligned = model.historical_forecasts(
        ts[0], cov[0], start_date=start, retrain=False, horizon=horizon
    )
    vectors = model.model.historical_forecasts(
        series=ts[0],
        past_covariates=cov[0],
        forecast_horizon=horizon,
        start=start,
        stride=1,
        retrain=False,
        last_points_only=False,
        predict_likelihood_parameters=True,
        verbose=False,
    )

    assert target_aligned.time_index.tolist() == [
        series.time_index[-1] for series in vectors
    ]
    target_df = target_aligned.to_dataframe()
    for series in vectors:
        assert np.allclose(
            target_df.loc[series.time_index[-1]].to_numpy(),
            series.values()[-1].reshape(-1),
        )
