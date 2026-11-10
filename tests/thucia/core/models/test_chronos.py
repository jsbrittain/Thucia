# Chronos wrapper contract: the darts variant must accept the `horizon` kwarg
# DartsBase passes, request direct quantile parameters (not samples), and use
# target-aligned (`last_points_only=True`) forecasts. These are cheap,
# network-free checks; the real fit lives in test_forecast_slow.py.
import importlib

from forecast_data import COVARIATE_COLS
from forecast_data import make_forecast_df
from thucia.core.models.chronos import CHRONOS_QUANTILES
from thucia.core.models.chronos import ChronosQuantilesDarts

chronos_mod = importlib.import_module("thucia.core.models.chronos")


def _model():
    df = make_forecast_df(n_periods=60, n_gid=1)
    return ChronosQuantilesDarts(
        df=df,
        case_col="Log_Cases",
        geo_col="GID_2",
        geo_parent="GID_1",
        covariate_cols=COVARIATE_COLS,
        horizons=[1],
        quantiles=CHRONOS_QUANTILES,
    )


def test_build_model_accepts_horizon_and_quantiles(monkeypatch):
    captured = {}

    class _FakeChronos2:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(chronos_mod, "Chronos2Model", _FakeChronos2)
    model = _model()
    model.build_model(horizon=6)

    assert captured["input_chunk_length"] == model.input_chunk_length
    assert captured["output_chunk_length"] == 6
    assert captured["likelihood"].quantiles == CHRONOS_QUANTILES


def test_historical_forecasts_requests_direct_target_aligned_quantiles():
    model = _model()
    captured = {}

    class _FakeModel:
        def historical_forecasts(self, **kwargs):
            captured.update(kwargs)
            return "sentinel"

    model.model = _FakeModel()
    out = model.historical_forecasts(
        "targets", "covariates", start_date="2020-07", retrain=False, horizon=3
    )

    assert out == "sentinel"
    assert captured["series"] == "targets"
    assert captured["past_covariates"] == "covariates"
    assert captured["forecast_horizon"] == 3
    assert captured["start"] == "2020-07"
    assert captured["retrain"] is False
    assert captured["last_points_only"] is True
    assert captured["predict_likelihood_parameters"] is True
    assert "num_samples" not in captured
    assert model.sampling_method == "quantiles"
