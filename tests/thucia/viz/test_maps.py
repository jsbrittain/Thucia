# Probing tests for thucia.viz.maps: geometry, adjacency, and hexmap helpers.
# GADM GeoPackage reads are mocked; synthetic shapely geometries drive the
# spatial logic (Agg backend, no network).
import matplotlib
import pytest
from shapely.geometry import Polygon

matplotlib.use("Agg")

import thucia.viz.maps as maps


def make_rect(x0, y0, x1, y1):
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


@pytest.fixture
def geo_gdf():
    """Two touching squares (X.1.1_2, X.1.2_2) + one detached square under a
    second admin-1 region. No case column (GADM has no case data)."""
    import geopandas as gpd

    return gpd.GeoDataFrame(
        {
            "GID_0": ["X"] * 3,
            "GID_1": ["X.1_1", "X.1_1", "X.2_1"],
            "GID_2": ["X.1.1_2", "X.1.2_2", "X.2.1_2"],
            "NAME_1": ["State", "State", "Other"],
            "NAME_2": ["One", "Two", "Three"],
        },
        geometry=[
            make_rect(0, 0, 1, 1),
            make_rect(1, 0, 2, 1),
            make_rect(5, 5, 6, 6),
        ],
        crs="EPSG:4326",
    )


def _value_df(geo_gdf):
    """Data-side frame carrying GID codes + a value column (as real usage)."""
    return geo_gdf.drop(columns=["geometry", "GID_0", "NAME_1", "NAME_2"]).assign(
        Cases=[1, 2, 3]
    )


@pytest.fixture
def mock_read(monkeypatch, tmp_path, geo_gdf):
    monkeypatch.setattr(maps, "cache_folder", str(tmp_path))
    monkeypatch.setattr(maps.gpd, "read_file", lambda *a, **k: geo_gdf)
    return geo_gdf


# --- subset_regions (pure) ---


def test_subset_regions_none_returns_input(geo_gdf):
    out = maps.subset_regions(geo_gdf, "GID_2", None)
    assert len(out) == 3


def test_subset_regions_callable(geo_gdf):
    out = maps.subset_regions(geo_gdf, "GID_2", lambda g: g.endswith("1_2"))
    assert set(out["GID_2"]) == {"X.1.1_2", "X.2.1_2"}


def test_subset_regions_list_direct_match(geo_gdf):
    out = maps.subset_regions(geo_gdf, "GID_2", ["X.1.1_2"])
    assert set(out["GID_2"]) == {"X.1.1_2"}


def test_subset_regions_string_is_listed(geo_gdf):
    out = maps.subset_regions(geo_gdf, "GID_2", "X.1.2_2")
    assert set(out["GID_2"]) == {"X.1.2_2"}


def test_subset_regions_hierarchical_gid2_prefix(geo_gdf):
    # Selecting an admin1 code (one dot) at admin2 level keeps all its regions.
    out = maps.subset_regions(geo_gdf, "GID_2", ["X.1_1"])
    assert set(out["GID_2"]) == {"X.1.1_2", "X.1.2_2"}


# --- adjacency_matrix ---


def test_adjacency_binary_symmetric(geo_gdf, mock_read):
    adj, _ = maps.adjacency_matrix(country="X", admin_level=2, weight="binary")
    # X.1.1_2 and X.1.2_2 touch; the detached one does not
    assert adj.loc["X.1.1_2", "X.1.2_2"] == 1
    assert adj.loc["X.1.1_2", "X.2.1_2"] == 0
    assert adj.loc["X.1.2_2", "X.2.1_2"] == 0
    assert (adj.values == adj.values.T).all()
    assert adj.dtypes.iloc[0] == "int64"


def test_adjacency_rook_subset_of_queen(geo_gdf, mock_read):
    queen, _ = maps.adjacency_matrix(country="X", admin_level=2, contiguity="queen")
    rook, _ = maps.adjacency_matrix(country="X", admin_level=2, contiguity="rook")
    for a, b in zip(queen.columns, rook.columns):
        assert (rook[a] <= queen[a]).all()


def test_adjacency_include_self_sets_diagonal(geo_gdf, mock_read):
    adj, _ = maps.adjacency_matrix(
        country="X", admin_level=2, weight="binary", include_self=True
    )
    assert (adj.values.diagonal() == 1).all()


def test_adjacency_weighted_shared_border_km(geo_gdf, mock_read):
    adj, _ = maps.adjacency_matrix(
        country="X", admin_level=2, weight="shared_border_km", include_self=True
    )
    # touching squares share a real border of positive length (metres)
    assert adj.loc["X.1.1_2", "X.1.2_2"] > 0
    # self diagonal is 0 for weighted
    assert adj.loc["X.1.1_2", "X.1.1_2"] == 0.0
    # non-adjacent stays at fillna default 0.0
    assert adj.loc["X.1.1_2", "X.2.1_2"] == 0.0


def test_adjacency_from_df_infers_country(geo_gdf, mock_read):
    df = geo_gdf.drop(columns=["geometry"])
    adj, _ = maps.adjacency_matrix(df=df, admin_level=2)
    assert adj.loc["X.1.1_2", "X.1.2_2"] == 1


def test_adjacency_invalid_contiguity_raises(geo_gdf):
    with pytest.raises(ValueError, match="contiguity"):
        maps.adjacency_matrix(country="X", admin_level=2, contiguity="bad")


def test_adjacency_invalid_weight_raises(geo_gdf):
    with pytest.raises(ValueError, match="weight"):
        maps.adjacency_matrix(country="X", admin_level=2, weight="bad")


def test_adjacency_no_country_raises():
    with pytest.raises(ValueError, match="country"):
        maps.adjacency_matrix(country=None, admin_level=2)


# --- choropleth / boundary (mock gpd.read_file) ---


def test_boundary_invalid_admin_level_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(maps, "cache_folder", str(tmp_path))
    with pytest.raises(ValueError, match="Unsupported admin_level"):
        maps.boundary(country="X", admin_level=3)


def test_boundary_plots(monkeypatch, tmp_path, geo_gdf):
    monkeypatch.setattr(maps, "cache_folder", str(tmp_path))
    monkeypatch.setattr(maps.gpd, "read_file", lambda *a, **k: geo_gdf)
    fig, ax = matplotlib.pyplot.subplots()
    maps.boundary(country="X", admin_level=1, ax=ax)
    matplotlib.pyplot.close(fig)


def test_choropleth_multiple_countries_raises(monkeypatch, tmp_path):
    import pandas as pd

    monkeypatch.setattr(maps, "cache_folder", str(tmp_path))
    df = pd.DataFrame(
        {"GID_1": ["X.1_1", "Y.1_1"], "GID_2": ["X.1.1_2", "Y.1.1_2"], "Cases": [1, 2]}
    )
    with pytest.raises(ValueError, match="multiple countries"):
        maps.choropleth(df, admin_level=2)


def test_choropleth_invalid_admin_level_raises(monkeypatch, tmp_path):
    import pandas as pd

    monkeypatch.setattr(maps, "cache_folder", str(tmp_path))
    df = pd.DataFrame({"GID_1": ["X.1_1"], "GID_2": ["X.1.1_2"], "Cases": [1]})
    with pytest.raises(ValueError, match="Unsupported admin_level"):
        maps.choropleth(df, admin_level="GID_3")


def test_choropleth_applies_transform_and_colorbar(monkeypatch, tmp_path, geo_gdf):
    monkeypatch.setattr(maps, "cache_folder", str(tmp_path))
    monkeypatch.setattr(maps.gpd, "read_file", lambda *a, **k: geo_gdf)
    df = _value_df(geo_gdf)
    fig, ax = matplotlib.pyplot.subplots()
    maps.choropleth(df, admin_level=2, ax=ax, value_transform=lambda x: x * 100, colorbar=True)
    matplotlib.pyplot.close(fig)


def test_choropleth_symmetric_cmap(monkeypatch, tmp_path, geo_gdf):
    monkeypatch.setattr(maps, "cache_folder", str(tmp_path))
    monkeypatch.setattr(maps.gpd, "read_file", lambda *a, **k: geo_gdf)
    df = _value_df(geo_gdf)
    fig, ax = matplotlib.pyplot.subplots()
    maps.choropleth(df, admin_level=2, ax=ax, symmetric_cmap=True)
    matplotlib.pyplot.close(fig)


# --- hexmap / hex_cartogram ---


def test_hexmap_centroid_method(monkeypatch, tmp_path, geo_gdf):
    monkeypatch.setattr(maps, "cache_folder", str(tmp_path))
    monkeypatch.setattr(maps.gpd, "read_file", lambda *a, **k: geo_gdf)
    out = maps.hexmap(_value_df(geo_gdf), admin_level=2, method="centroid")
    assert "hex_id" in out.columns
    assert "value" in out.columns
    assert out["value"].notna().any()


def test_hexmap_area_method(monkeypatch, tmp_path, geo_gdf):
    monkeypatch.setattr(maps, "cache_folder", str(tmp_path))
    monkeypatch.setattr(maps.gpd, "read_file", lambda *a, **k: geo_gdf)
    out = maps.hexmap(_value_df(geo_gdf), admin_level=2, method="area")
    assert "hex_id" in out.columns
    assert "value" in out.columns
    assert out["value"].notna().any()


def test_hexmap_missing_gid1_raises(monkeypatch, tmp_path):
    import pandas as pd

    monkeypatch.setattr(maps, "cache_folder", str(tmp_path))
    with pytest.raises(ValueError, match="GID_1"):
        maps.hexmap(pd.DataFrame({"GID_2": ["X.1.1_2"]}), admin_level=2)


def test_hexmap_multiple_countries_raises(monkeypatch, tmp_path):
    import pandas as pd

    monkeypatch.setattr(maps, "cache_folder", str(tmp_path))
    df = pd.DataFrame({"GID_1": ["X.1_1", "Y.1_1"], "Cases": [1, 2]})
    with pytest.raises(ValueError, match="multiple countries"):
        maps.hexmap(df, admin_level=1)


def test_hex_cartogram_one_hex_per_gid2(monkeypatch, tmp_path, geo_gdf):
    monkeypatch.setattr(maps, "cache_folder", str(tmp_path))
    monkeypatch.setattr(maps.gpd, "read_file", lambda *a, **k: geo_gdf)
    out = maps.hex_cartogram(_value_df(geo_gdf), admin_level=2)
    # exactly one hex per admin-2 region
    assert len(out) == 3
    assert set(out["GID_2"]) == {"X.1.1_2", "X.1.2_2", "X.2.1_2"}
    assert "centroid_dist_m" in out.columns
    # projected back to WGS84
    assert out.crs is not None and out.crs.to_epsg() == 4326


def test_hex_cartogram_non_gid2_raises(monkeypatch, tmp_path):
    import pandas as pd

    monkeypatch.setattr(maps, "cache_folder", str(tmp_path))
    df = pd.DataFrame({"GID_1": ["X.1_1"], "GID_2": ["X.1.1_2"], "Cases": [1]})
    with pytest.raises(ValueError, match="admin_level=2"):
        maps.hex_cartogram(df, admin_level=1)


def test_hex_cartogram_missing_gid1_raises(monkeypatch, tmp_path):
    import pandas as pd

    monkeypatch.setattr(maps, "cache_folder", str(tmp_path))
    with pytest.raises(ValueError, match="GID_1"):
        maps.hex_cartogram(pd.DataFrame({"GID_2": ["X.1.1_2"]}), admin_level=2)
