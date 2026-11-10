# Probing tests for thucia.viz.scores: score aggregation + faceted choropleths.
import matplotlib
import pandas as pd
import pytest
from shapely.geometry import Polygon

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from thucia.viz import aggregate_scores  # noqa: E402
from thucia.viz import plot_scores  # noqa: E402


def _regions():
    import geopandas as gpd

    return gpd.GeoDataFrame(
        {"region_id": ["A", "B", "C"]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
        ],
        crs="EPSG:4326",
    )


def _scores():
    """Two dates per (geo, horizon) so WIS averaging is observable."""
    rows = [
        ("A", 1, 0.2, 0.1),
        ("A", 1, 0.4, 0.1),
        ("A", 3, 1.0, 0.3),
        ("A", 3, 1.2, 0.3),
        ("B", 1, 0.6, 0.0),
        ("B", 1, 0.8, 0.0),
        ("B", 3, 0.9, -0.1),
        ("B", 3, 1.1, -0.1),
        ("C", 1, 0.5, 0.2),
        ("C", 1, 0.7, 0.2),
        ("C", 3, 0.7, 0.4),
        ("C", 3, 0.9, 0.4),
    ]
    return pd.DataFrame(rows, columns=["district_id", "horizon", "WIS", "R2"]).assign(
        model="m"
    )


def test_aggregate_scores_mean_per_geo_horizon():
    out = aggregate_scores(_scores(), geo_col="district_id")

    a_h1 = out[(out["district_id"] == "A") & (out["horizon"] == 1)].iloc[0]
    a_h3 = out[(out["district_id"] == "A") & (out["horizon"] == 3)].iloc[0]
    assert a_h1["WIS"] == pytest.approx(0.3)
    assert a_h3["WIS"] == pytest.approx(1.1)
    assert a_h1["R2"] == pytest.approx(0.1)  # constant per region/horizon
    assert len(out) == 6  # 3 regions x 2 horizons
    assert set(out["horizon"]) == {1, 3}


def test_aggregate_scores_missing_horizon_defaults_to_one():
    df = _scores().drop(columns="horizon")
    out = aggregate_scores(df, geo_col="district_id")
    assert set(out["horizon"]) == {1}


def test_aggregate_scores_horizon_filter():
    out = aggregate_scores(_scores(), geo_col="district_id", horizons=[3])
    assert set(out["horizon"]) == {3}


def test_aggregate_scores_multiple_models_requires_selection():
    df = pd.concat([_scores(), _scores().assign(model="n")], ignore_index=True)
    with pytest.raises(ValueError, match="multiple models"):
        aggregate_scores(df, geo_col="district_id")

    out = aggregate_scores(df, geo_col="district_id", model="m")
    assert len(out) == 6


def test_aggregate_scores_unknown_metric_raises():
    with pytest.raises(ValueError, match="none of the requested metrics"):
        aggregate_scores(_scores(), geo_col="district_id", metrics=("MAE",))


def test_plot_scores_facets_metrics_and_horizons():
    fig, axes = plot_scores(
        _scores(),
        regions=_regions(),
        geo_col="district_id",
        region_col="region_id",
    )

    assert axes.shape == (2, 2)  # metrics x horizons
    assert "WIS" in axes[0, 0].get_title()
    assert "h=1" in axes[0, 0].get_title()
    assert "R2" in axes[1, 0].get_title()
    assert "h=3" in axes[1, 1].get_title()
    plt.close(fig)


def test_plot_scores_no_matching_horizon_raises():
    with pytest.raises(ValueError, match="no scores to plot"):
        plot_scores(_scores(), regions=_regions(), geo_col="district_id", horizons=[99])


def test_plot_scores_empty_slice_is_blank_but_present():
    # A metric/horizon with no scores must not crash the faceting.
    df = _scores()
    df.loc[df["horizon"] == 3, "R2"] = float("nan")
    fig, axes = plot_scores(
        df,
        regions=_regions(),
        geo_col="district_id",
        region_col="region_id",
    )
    assert axes.shape == (2, 2)
    plt.close(fig)
