import importlib
import inspect
import pkgutil

import pytest
import thucia.core.models as models


def _all_module_names():
    # Every module file (including helper libraries) for import-cleanliness checks
    return sorted(
        m.name
        for m in pkgutil.iter_modules(models.__path__)
        if not m.name.startswith("_") and not m.ispkg
    )


def test_models_package_advertises_only_models():
    # __all__ lists the auto-discovered models (per the lazy __getattr__ contract)
    assert len(models.__all__) > 0
    for name in models.__all__:
        assert hasattr(models, name), f"{name} advertised but not resolvable"


def test_each_advertised_model_is_callable():
    for name in models.__all__:
        assert callable(getattr(models, name)), f"{name} is not callable"


def test_unknown_model_raises_attribute_error():
    with pytest.raises(AttributeError):
        models.definitely_not_a_model


def test_model_functions_accept_keyword_contract():
    for name in models.__all__:
        fn = getattr(models, name)
        sig = inspect.signature(fn)
        assert "df" in sig.parameters, f"{name} missing required arg 'df'"


def test_all_modules_import_cleanly():
    for name in _all_module_names():
        importlib.import_module(f"thucia.core.models.{name}")


def test_list_models_matches_all():
    assert models.list_models() == models.__all__


def test_get_model_resolves_callable():
    assert callable(models.get_model("baseline"))


def test_get_model_unknown_raises():
    with pytest.raises(ValueError, match="not_a_model"):
        models.get_model("not_a_model")


# Keyword set passed by thucia.core.pipeline.fit_model / run_backtest to
# run_model() for every model.
COMMON_KWARGS = {
    "start_date": "2020-01",
    "gid_1": None,
    "horizons": [1, 3, 6, 12],
    "case_col": "Log_Cases",
    "covariate_cols": ["x"],
    "model_admin_level": 2,
    "db_file": None,
}
# Extra kwargs added for the darts-based models.
DARTS_KWARGS = {
    "train_end_date": "2022-01",
    "retrain": False,
    "multivariate": False,
    "num_samples": 200,
}
DARTS_MODELS = {"tcn", "tft", "nbeats", "nhits", "xgboost"}
# Models with *args/**kwargs swallow anything; inla is container-based with its own
# interface, so exclude it from the shared forecasting contract.
LOOSE_MODELS = {"baseline", "sarima", "movavg", "timesfm"}


def _model_set():
    return set(models.__all__) - LOOSE_MODELS - {"inla"}


def test_all_models_bind_common_kwargs():
    for name in _model_set():
        sig = inspect.signature(getattr(models, name))
        sig.bind_partial(**COMMON_KWARGS)  # raises TypeError on unknown kwargs


def test_darts_models_bind_full_pipeline_kwargs():
    for name in DARTS_MODELS:
        sig = inspect.signature(getattr(models, name))
        kw = {**COMMON_KWARGS, **DARTS_KWARGS}
        sig.bind_partial(**kw)


def test_chronos_binds_pipeline_kwargs_without_num_samples():
    # chronos has no num_samples parameter; the pipeline deliberately skips it
    sig = inspect.signature(getattr(models, "chronos"))
    kw = {**COMMON_KWARGS, **DARTS_KWARGS}
    kw.pop("num_samples")
    sig.bind_partial(**kw)


def test_horizons_default_is_a_list():
    # All models share a list-form `horizons` default (never a scalar `horizon`)
    for name in models.__all__:
        if name in LOOSE_MODELS or name == "inla":
            continue
        sig = inspect.signature(getattr(models, name))
        p = sig.parameters["horizons"]
        assert p.default == [1], f"{name}.horizons default should be [1]"
