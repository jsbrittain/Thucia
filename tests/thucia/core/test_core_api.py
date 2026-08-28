# Pins the lazy public API contract of `thucia.core` (PEP 562 __getattr__).
# Importing the package must stay cheap: the heavy computation/scheduling
# submodules (pipeline, models, validation, cache, containers) must NOT be
# loaded until a name that needs them is accessed.
#
# Import-time/laziness behavior is asserted in subprocesses so it never mutates
# this interpreter's sys.modules (deleting/re-importing shared modules here
# would invalidate registry/plugin state other tests rely on).
import subprocess
import sys

import pytest


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )


LAZY_SCRIPT = """
import sys
import thucia.core
heavy = [
    m
    for m in (
        "thucia.core.pipeline",
        "thucia.core.models",
        "thucia.core.validation",
        "thucia.core.cache",
        "thucia.core.containers",
    )
    if m in sys.modules
]
print("HEAVY:" + repr(sorted(heavy)))
"""


def test_import_is_lazy():
    r = _run(LAZY_SCRIPT)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "HEAVY:[]"


def test_heavy_submodules_resolve_lazily():
    r = _run(
        """
import sys
import thucia.core as c
assert "thucia.core.models" not in sys.modules, sys.modules
m = c.models
assert m is sys.modules["thucia.core.models"]
assert "thucia.core.validation" not in sys.modules, sys.modules
print("OK")
"""
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "OK"


def test_attribute_access_caches_resolution():
    r = _run(
        """
import thucia.core as c
assert "wis" not in c.__dict__
_ = c.wis
assert "wis" in c.__dict__
print("OK")
"""
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "OK"


def test_all_resolves_to_expected_objects():
    import thucia.core
    from thucia.core import __all__

    resolved = {name: getattr(thucia.core, name) for name in __all__}

    from thucia.core.pipeline import PipelineConfig
    from thucia.core.registry import Registry
    from thucia.core.validation import run_backtest

    assert resolved["PipelineConfig"] is PipelineConfig
    assert resolved["run_backtest"] is run_backtest
    assert resolved["Registry"] is Registry
    assert resolved["models"] is __import__("thucia.core.models", fromlist=["x"])

    import types

    for name, obj in resolved.items():
        assert isinstance(obj, (types.ModuleType, type, types.FunctionType)) or callable(
            obj
        ), f"{name!r} resolved to {obj!r}"


def test_unknown_attribute_raises():
    import thucia.core

    with pytest.raises(AttributeError):
        _ = thucia.core.this_does_not_exist
