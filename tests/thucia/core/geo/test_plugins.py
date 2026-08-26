import pandas as pd
import pytest
from thucia.core.geo import merge_geo_sources
from thucia.core.geo import refresh_plugins
from thucia.core.geo.plugin_base import source_registry
from thucia.core.geo.plugin_loader import load_plugins
from thucia.core.registry import PluginNotFoundError


def test_real_plugins_self_register():
    refresh_plugins()
    names = source_registry.names()
    assert {"worldclim", "edo", "noaa", "worldpop"} <= set(names)
    for name in names:
        cls = source_registry.get(name)
        assert issubclass(cls, object)


def test_merge_geo_sources_unknown_origin_raises():
    with pytest.raises(PluginNotFoundError):
        merge_geo_sources(pd.DataFrame(), ["nope.metric"])


def test_merge_geo_sources_malformed_source_raises():
    with pytest.raises(ValueError, match="origin.field"):
        merge_geo_sources(pd.DataFrame(), ["no-dot-here"])


def test_load_plugins_isolates_broken_module(tmp_path, monkeypatch):
    # A module that raises on import must not stop the other plugins loading.
    pkg = tmp_path / "sources"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "broken.py").write_text("raise RuntimeError('boom')")
    (pkg / "fine.py").write_text(
        "from thucia.core.geo.plugin_base import SourceBase, source_registry\n"
        "@source_registry.register()\n"
        "class Fine(SourceBase):\n"
        "    ref = 'fine'\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        plugins = load_plugins(plugin_dir=pkg, module_stem="sources")
    finally:
        source_registry.unregister("fine")
    assert "broken" not in plugins
    assert "fine" in plugins
