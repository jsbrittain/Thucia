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
