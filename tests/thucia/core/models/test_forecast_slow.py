# Heavy real-model forecast fits. Run with `uv run pytest -m slow` (excluded
# from the default/CI run via `addopts = "-m 'not slow'"`).
import numpy as np
import pandas as pd
import pytest
from forecast_data import COVARIATE_COLS
from forecast_data import make_forecast_df
from thucia.core.models import nbeats
from thucia.core.models import xgboost
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
            gid_1=None,
            horizons=[1],
            case_col="Log_Cases",
            covariate_cols=COVARIATE_COLS,
            retrain=False,
            db_file=None,
            model_admin_level=2,
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
    df = make_forecast_df(n_months=96, start="2014-01")
    out = _as_df(
        xgboost(
            df,
            start_date=pd.Period("2020-01", freq="M"),
            train_start_date=pd.Period("2014-01", freq="M"),
            train_end_date=pd.Period("2019-12", freq="M"),
            gid_1=None,
            horizons=[1, 3, 6, 12],
            case_col="Log_Cases",
            covariate_cols=COVARIATE_COLS,
            retrain=False,
            db_file=None,
            model_admin_level=2,
            num_samples=20,
            multivariate=False,
        )
    )
    assert sorted(out["horizon"].unique()) == [1, 3, 6, 12]
    assert sorted(out["quantile"].unique()) == quantiles
    assert np.isfinite(out["prediction"]).all()


def test_timesfm_forecast_requires_torch_and_network(df):
    # TimesFM downloads a HuggingFace checkpoint; opt-in only.
    pytest.importorskip("torch")
    try:
        from thucia.core.models import timesfm
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"timesfm unavailable: {exc}")
    out = _as_df(
        timesfm(
            df,
            start_date=pd.Period("2020-07", freq="M"),
            gid_1=None,
            horizons=[1],
            case_col="Log_Cases",
            covariate_cols=COVARIATE_COLS,
            retrain=False,
            db_file=None,
        )
    )
    assert np.isfinite(out["prediction"]).all()
