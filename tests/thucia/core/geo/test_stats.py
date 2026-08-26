# Probing tests for raster zonal-statistics helpers (core/geo/stats.py).
# GADM GeoPackage and raster access are mocked (no network, no real raster).
import pandas as pd
import pytest

import thucia.core.geo.stats as stats


@pytest.fixture
def polygons():
    return pd.DataFrame(
        {
            "GID_0": ["X"] * 2,
            "COUNTRY": ["X"] * 2,
            "GID_1": ["X.1_1"] * 2,
            "NL_NAME_1": [None, None],
            "NAME_1": ["A", "A"],
            "GID_2": ["X.1.1_2", "X.1.2_2"],
            "NL_NAME_2": [None, None],
            "NAME_2": ["One", "Two"],
            "TYPE_2": ["Type"] * 2,
            "ENGTYPE_2": ["Type"] * 2,
            "CC_2": [None, None],
            "HASC_2": ["XA", "XB"],
            "geometry": [None, None],
        }
    )


def test_raster_stats_gid2_returns_kept_columns_with_stats(
    polygons, monkeypatch, tmp_path
):
    gid_2s = ["X.1.1_2", "X.1.2_2"]
    monkeypatch.setattr(stats, "cache_folder", str(tmp_path))
    # create the expected gpkg file so the existence check passes
    gpkg = tmp_path / "geo" / "X" / "gadm41_X.gpkg"
    gpkg.parent.mkdir(parents=True)
    gpkg.touch()

    def fake_read_file(path, layer):
        assert layer == "ADM_ADM_2"
        return polygons

    def fake_zonal_stats(polys, tif, stats=None):
        assert tif == "some.tif"
        assert set(polys["GID_2"]) == set(gid_2s)
        assert stats == ["mean"]
        return [{"mean": 1.0}, {"mean": 2.0}]

    monkeypatch.setattr(stats.gpd, "read_file", fake_read_file)
    monkeypatch.setattr(stats, "zonal_stats", fake_zonal_stats)

    out = stats.raster_stats_gid2("some.tif", gid_2s, stats=["mean"])

    assert "mean" in out.columns
    assert out["mean"].tolist() == [1.0, 2.0]
    # geometry and internal columns are dropped; only keep_columns + stats remain
    assert "geometry" not in out.columns
    assert set(out.columns) == {
        "GID_0",
        "COUNTRY",
        "GID_1",
        "NL_NAME_1",
        "NAME_1",
        "GID_2",
        "NL_NAME_2",
        "NAME_2",
        "TYPE_2",
        "ENGTYPE_2",
        "CC_2",
        "HASC_2",
        "mean",
    }


def test_raster_stats_gid2_multi_stats(polygons, monkeypatch, tmp_path):
    monkeypatch.setattr(stats, "cache_folder", str(tmp_path))
    gpkg = tmp_path / "geo" / "X" / "gadm41_X.gpkg"
    gpkg.parent.mkdir(parents=True)
    gpkg.touch()
    monkeypatch.setattr(stats.gpd, "read_file", lambda *a, **k: polygons)

    def fake_zonal_stats(polys, tif, stats=None):
        assert stats == ["mean", "count"]
        return [{"mean": 1.0, "count": 3}, {"mean": 2.0, "count": 5}]

    monkeypatch.setattr(stats, "zonal_stats", fake_zonal_stats)

    out = stats.raster_stats_gid2("some.tif", ["X.1.1_2", "X.1.2_2"], stats=["mean", "count"])

    assert out["mean"].tolist() == [1.0, 2.0]
    assert out["count"].tolist() == [3, 5]


def test_raster_stats_gid2_mixed_iso3_raises(polygons, monkeypatch, tmp_path):
    monkeypatch.setattr(stats, "cache_folder", str(tmp_path))
    with pytest.raises(ValueError, match="same ISO3"):
        stats.raster_stats_gid2("some.tif", ["X.1.1_2", "Y.1.1_2"])


def test_raster_stats_gid2_missing_gpkg_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(stats, "cache_folder", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="not found"):
        stats.raster_stats_gid2("some.tif", ["X.1.1_2"])
