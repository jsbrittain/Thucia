# Contract tests for the fixed-grid foundation-model wrappers (chronos,
# timesfm25, timesfm3). Cheap and network-free: the darts model is replaced by a
# stub, so only the wrapper logic (quantile validation, median-only mode,
# covariate routing, component contract) is exercised.
import importlib

import pytest
from forecast_data import COVARIATE_COLS
from forecast_data import make_forecast_df
from thucia.core.models.chronos import CHRONOS_QUANTILES
from thucia.core.models.chronos import ChronosQuantiles
from thucia.core.models.darts.foundation import TIMESFM_QUANTILES
from thucia.core.models.timesfm25 import TimesFMQuantiles25
from thucia.core.models.timesfm3 import TimesFMQuantiles3
from thucia.core.quantiles import quantiles as CANONICAL

timesfm3_mod = importlib.import_module("thucia.core.models.timesfm3")

WRAPPERS = {
    "chronos": (ChronosQuantiles, CHRONOS_QUANTILES),
    "timesfm25": (TimesFMQuantiles25, TIMESFM_QUANTILES),
    "timesfm3": (TimesFMQuantiles3, TIMESFM_QUANTILES),
}


def _model(cls, **kwargs):
    df = make_forecast_df(n_periods=60, n_gid=2)
    params = dict(
        df=df,
        case_col="Log_Cases",
        geo_col="GID_2",
        geo_parent="GID_1",
        covariate_cols=list(COVARIATE_COLS),
        horizons=[1],
    )
    if cls is TimesFMQuantiles3:
        params["accept_license"] = True
    params.update(kwargs)
    return cls(**params)


@pytest.mark.parametrize("name", sorted(WRAPPERS))
def test_default_quantiles_are_the_native_grid(name):
    cls, grid = WRAPPERS[name]
    assert _model(cls).quantiles == grid


@pytest.mark.parametrize("name", sorted(WRAPPERS))
def test_requested_subset_is_accepted(name):
    cls, grid = WRAPPERS[name]
    subset = [grid[0], 0.5, grid[-1]]
    model = _model(cls, quantiles=subset)
    assert model.quantiles == sorted(subset)
    assert model.likelihood().quantiles == sorted(subset)


@pytest.mark.parametrize("name", sorted(WRAPPERS))
def test_non_subset_quantiles_raise(name):
    cls, _ = WRAPPERS[name]
    with pytest.raises(ValueError, match=r"not in the pre-trained grid"):
        _model(cls, quantiles=[0.025, 0.5, 0.975])


@pytest.mark.parametrize("name", sorted(WRAPPERS))
def test_median_only_yields_a_single_level(name):
    cls, _ = WRAPPERS[name]
    model = _model(cls, median_only=True)
    assert model.quantiles == [0.5]
    assert model.likelihood().quantiles == [0.5]


@pytest.mark.parametrize("name", sorted(WRAPPERS))
def test_median_only_conflicts_with_quantiles(name):
    cls, grid = WRAPPERS[name]
    with pytest.raises(ValueError, match="mutually exclusive"):
        _model(cls, median_only=True, quantiles=[grid[1]])


@pytest.mark.parametrize("name", sorted(WRAPPERS))
def test_missing_median_warns(name):
    cls, grid = WRAPPERS[name]
    with pytest.warns(UserWarning, match="0.5 is not among"):
        _model(cls, quantiles=[grid[0], grid[-1]])


def test_timesfm3_requires_licence_acceptance():
    df = make_forecast_df(n_periods=60, n_gid=1)
    with pytest.raises(ValueError, match="Non-Commercial License"):
        TimesFMQuantiles3(
            df=df,
            case_col="Log_Cases",
            geo_col="GID_2",
            geo_parent="GID_1",
            covariate_cols=list(COVARIATE_COLS),
            horizons=[1],
        )


def test_timesfm25_drops_covariates_with_a_warning():
    with pytest.warns(UserWarning, match="does not support covariates"):
        model = _model(TimesFMQuantiles25)
    assert model.covariate_cols == []
    assert model.past_covariate_cols == []
    assert model.future_covariate_cols == []


def test_timesfm25_rejects_covariate_routing():
    with pytest.raises(ValueError, match="does not support covariates"):
        _model(TimesFMQuantiles25, future_covariate_cols=["tmin"])


@pytest.mark.parametrize("name", ["chronos", "timesfm3"])
def test_past_covariates_are_the_default_slot(name):
    cls, _ = WRAPPERS[name]
    model = _model(cls)
    assert model.past_covariate_cols == list(COVARIATE_COLS)
    assert model.future_covariate_cols == []

    kwargs = model._covariate_kwargs(_FakeSeries())
    assert "future_covariates" not in kwargs
    assert list(kwargs["past_covariates"].cols) == list(COVARIATE_COLS)


@pytest.mark.parametrize("name", ["chronos", "timesfm3"])
def test_future_all_routes_every_covariate_to_both_slots(name):
    cls, _ = WRAPPERS[name]
    model = _model(cls, future_covariate_cols="all")
    assert model.past_covariate_cols == list(COVARIATE_COLS)
    assert model.future_covariate_cols == list(COVARIATE_COLS)

    kwargs = model._covariate_kwargs(_FakeSeries())
    assert list(kwargs["past_covariates"].cols) == list(COVARIATE_COLS)
    assert list(kwargs["future_covariates"].cols) == list(COVARIATE_COLS)


@pytest.mark.parametrize("name", ["chronos", "timesfm3"])
def test_per_column_covariate_routing(name):
    cls, _ = WRAPPERS[name]
    model = _model(cls, past_covariate_cols=["tmin"], future_covariate_cols=["prec"])
    kwargs = model._covariate_kwargs(_FakeSeries())
    assert list(kwargs["past_covariates"].cols) == ["tmin"]
    assert list(kwargs["future_covariates"].cols) == ["prec"]


@pytest.mark.parametrize("name", ["chronos", "timesfm3"])
def test_unknown_covariate_column_raises(name):
    cls, _ = WRAPPERS[name]
    with pytest.raises(ValueError, match="unknown covariate"):
        _model(cls, future_covariate_cols=["not_a_covariate"])


def test_emitted_quantile_levels_are_checked():
    model = _model(ChronosQuantiles, quantiles=[0.1, 0.5, 0.9])

    class _FakeForecast:
        components = ["Log_Cases_q0.100", "Log_Cases_q0.500", "Log_Cases_q0.900"]

    model.check_quantile_components([_FakeForecast])  # no raise

    class _WrongForecast:
        components = ["Log_Cases_q0.100", "Log_Cases_q0.500"]

    with pytest.raises(RuntimeError, match="darts returned quantile levels"):
        model.check_quantile_components(_WrongForecast())


def test_build_model_passes_likelihood_and_horizon(monkeypatch):
    captured = {}

    class _FakeTimesFM3:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(timesfm3_mod, "TimesFM3Model", _FakeTimesFM3)
    model = _model(TimesFMQuantiles3, quantiles=[0.1, 0.5, 0.9])
    model.build_model(horizon=4)

    assert captured["output_chunk_length"] == 4
    assert captured["accept_license"] is True
    assert captured["likelihood"].quantiles == [0.1, 0.5, 0.9]
    assert captured["input_chunk_length"] == model.input_chunk_length


def test_native_grids_are_inside_the_canonical_grid():
    assert set(TIMESFM_QUANTILES) <= set(CANONICAL)
    assert set(CHRONOS_QUANTILES) <= set(CANONICAL)
    # documented limitations: timesfm has no tails beyond 0.1/0.9, chronos
    # lacks 0.025/0.975.
    assert 0.025 not in TIMESFM_QUANTILES and 0.05 not in TIMESFM_QUANTILES
    assert 0.025 not in CHRONOS_QUANTILES and 0.975 not in CHRONOS_QUANTILES


class _FakeSeries:
    """Minimal TimeSeries stand-in: column selection via ``series[cols]``."""

    def __init__(self, cols=tuple(COVARIATE_COLS)):
        self.cols = list(cols)

    @property
    def components(self):
        return self.cols

    def __getitem__(self, cols):
        return _FakeSeries(cols)
