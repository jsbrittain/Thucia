# Probing tests for the cache-root resolution (core/fs/__init__.py).
# The property under test is *lazy* resolution: the cache root must reflect
# THUCIA_CACHE_DIR / set_cache_folder at call time, never a value frozen at
# import (which is what makes the override usable on space-limited HPC homes).
import os
from pathlib import Path

import pytest
import thucia.core.fs as fs
from platformdirs import user_cache_dir


@pytest.fixture(autouse=True)
def _clean_cache_env(monkeypatch):
    """No ambient THUCIA_CACHE_DIR leaks in from the developer's shell."""
    monkeypatch.delenv(fs.CACHE_DIR_ENV, raising=False)
    monkeypatch.setattr(fs, "_cache_folder_override", None)
    yield


def test_default_is_the_platform_cache_dir():
    assert fs.get_cache_folder() == Path(user_cache_dir(fs.appname, fs.appauthor))


def test_env_var_overrides_default(monkeypatch, tmp_path):
    monkeypatch.setenv(fs.CACHE_DIR_ENV, str(tmp_path))
    assert fs.get_cache_folder() == tmp_path


def test_env_var_is_resolved_at_call_time(monkeypatch, tmp_path):
    # The whole point of issue #25: setting the variable after import must work.
    before = fs.get_cache_folder()
    monkeypatch.setenv(fs.CACHE_DIR_ENV, str(tmp_path))
    after = fs.get_cache_folder()
    assert after == tmp_path
    assert after != before


def test_env_var_expands_home(monkeypatch):
    monkeypatch.setenv(fs.CACHE_DIR_ENV, "~/thucia-scratch")
    assert fs.get_cache_folder() == Path("~/thucia-scratch").expanduser().absolute()


def test_env_var_relative_path_is_made_absolute(monkeypatch):
    # A relative root would otherwise silently follow the process CWD.
    monkeypatch.setenv(fs.CACHE_DIR_ENV, "thucia-relative-cache")
    resolved = fs.get_cache_folder()
    assert resolved.is_absolute()
    assert resolved == Path.cwd() / "thucia-relative-cache"


def test_set_cache_folder_beats_the_env_var(monkeypatch, tmp_path):
    env_dir = tmp_path / "env"
    explicit = tmp_path / "explicit"
    monkeypatch.setenv(fs.CACHE_DIR_ENV, str(env_dir))
    assert fs.set_cache_folder(explicit) == explicit
    assert fs.get_cache_folder() == explicit


def test_set_cache_folder_none_restores_env(monkeypatch, tmp_path):
    monkeypatch.setenv(fs.CACHE_DIR_ENV, str(tmp_path))
    fs.set_cache_folder(tmp_path / "explicit")
    assert fs.set_cache_folder(None) == tmp_path
    assert fs.get_cache_folder() == tmp_path


def test_accessor_does_not_create_the_directory(tmp_path):
    target = tmp_path / "not-created"
    assert fs.set_cache_folder(target) == target
    assert fs.get_cache_folder() == target
    assert not target.exists()


def test_legacy_cache_folder_attribute_is_a_str(monkeypatch, tmp_path):
    # Backwards compatibility for `from thucia.core.fs import cache_folder`.
    monkeypatch.setenv(fs.CACHE_DIR_ENV, str(tmp_path))
    assert fs.cache_folder == str(tmp_path)
    assert isinstance(fs.cache_folder, str)


def test_legacy_attribute_tracks_the_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv(fs.CACHE_DIR_ENV, str(tmp_path))
    assert fs.cache_folder == str(tmp_path)
    other = tmp_path / "other"
    monkeypatch.setenv(fs.CACHE_DIR_ENV, str(other))
    assert fs.cache_folder == str(other)


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError, match="no attribute"):
        fs.not_a_real_attribute


def test_public_surface_exposes_the_accessors():
    import thucia.core as core

    assert core.get_cache_folder is fs.get_cache_folder
    assert core.set_cache_folder is fs.set_cache_folder


def test_env_var_name_is_stable():
    # The documented name; changing it is a user-facing break.
    assert fs.CACHE_DIR_ENV == "THUCIA_CACHE_DIR"
    assert os.environ.get(fs.CACHE_DIR_ENV) is None


def test_sources_resolve_the_cache_root_per_instance(monkeypatch, tmp_path):
    """Plugin cache paths must follow the override, not an import-time snapshot."""
    from thucia.core.geo.sources.edo import EDO
    from thucia.core.geo.sources.worldclim import WorldClim

    monkeypatch.setenv(fs.CACHE_DIR_ENV, str(tmp_path / "first"))
    assert WorldClim().cache_file == tmp_path / "first" / "climate" / (
        "worldclim_stats.sqlite"
    )
    assert EDO().cache_file == tmp_path / "first" / "climate" / "edo_stats.sqlite"

    # A different root yields a different cache file for a new instance.
    monkeypatch.setenv(fs.CACHE_DIR_ENV, str(tmp_path / "second"))
    assert WorldClim().cache_file == tmp_path / "second" / "climate" / (
        "worldclim_stats.sqlite"
    )
    assert EDO().cache_file == tmp_path / "second" / "climate" / "edo_stats.sqlite"


def test_gadm_path_follows_the_override(monkeypatch, tmp_path):
    """The GADM lookup used by geo/stats.py must read the overridden root."""
    import thucia.core.geo.stats as stats

    class _ReachedReadFile(Exception):
        pass

    monkeypatch.setenv(fs.CACHE_DIR_ENV, str(tmp_path))

    def fake_read_file(path, layer):
        assert Path(path).parent == tmp_path / "geo" / "X"
        raise _ReachedReadFile(path)

    monkeypatch.setattr(stats.gpd, "read_file", fake_read_file)

    (tmp_path / "geo" / "X").mkdir(parents=True)
    (tmp_path / "geo" / "X" / "gadm41_X.gpkg").touch()

    with pytest.raises(_ReachedReadFile):
        stats.raster_stats_gid2("some.tif", ["X.1.1_2"])
