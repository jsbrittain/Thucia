# Small-multiple choropleths of per-region forecast scores (WIS / R2).
#
# ``score_model`` / ``run_backtest`` emit one score per region (x horizon), so a
# score map is just a per-region aggregation handed to ``maps.choropleth``. The
# score frame may carry a ``model`` column (e.g. from run_panel); filter it with
# ``model=`` when it holds more than one model.
import matplotlib.pyplot as plt
import pandas as pd

from .maps import choropleth


def aggregate_scores(
    scores: pd.DataFrame,
    *,
    geo_col: str,
    metrics=("WIS", "R2"),
    horizons=None,
    model: str | None = None,
) -> pd.DataFrame:
    """Mean score per ``(geo_col, horizon)``.

    ``WIS`` is averaged over dates; ``R2`` is constant per region/horizon, so
    averaging is a no-op. A missing ``horizon`` column is treated as horizon 1.
    """
    df = scores.copy()
    if geo_col not in df.columns:
        raise ValueError(f"geo_col {geo_col!r} not in scores columns")

    if "model" in df.columns:
        models = df["model"].dropna().unique().tolist()
        if model is not None:
            df = df[df["model"] == model]
        elif len(models) > 1:
            raise ValueError(
                f"scores contain multiple models {models}; pass model= to select one"
            )

    if "horizon" not in df.columns:
        df["horizon"] = 1
    if horizons is not None:
        df = df[df["horizon"].isin(list(horizons))]

    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        raise ValueError(f"none of the requested metrics are in scores: {metrics}")

    return df.groupby([geo_col, "horizon"], as_index=False, observed=False)[
        metrics
    ].mean()


def plot_scores(
    scores: pd.DataFrame,
    *,
    regions,
    geo_col: str,
    region_col: str | None = None,
    metrics=("WIS", "R2"),
    horizons=None,
    model: str | None = None,
    cmap: str = "viridis",
    colorbar_format: str = "%.2f",
    symmetric_metrics=("R2",),
    figsize=None,
):
    """Faceted choropleths: one row per metric, one column per horizon.

    ``regions`` is a GeoPackage path or GeoDataFrame (see ``maps.choropleth``);
    ``geo_col`` is the region-code column in ``scores`` and ``region_col`` its
    key in ``regions`` (defaults to ``geo_col``). Returns ``(fig, axes)``.
    """
    agg = aggregate_scores(
        scores, geo_col=geo_col, metrics=metrics, horizons=horizons, model=model
    )
    metrics = [m for m in metrics if m in agg.columns]
    horizons = sorted(agg["horizon"].unique())
    if not metrics or not horizons:
        raise ValueError("no scores to plot for the requested metrics/horizons")
    nrows, ncols = len(metrics), len(horizons)

    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for i, metric in enumerate(metrics):
        for j, horizon in enumerate(horizons):
            ax = axes[i, j]
            sub = (
                agg[agg["horizon"] == horizon][[geo_col, metric]]
                .dropna(subset=[metric])
                .reset_index(drop=True)
            )
            if sub.empty:
                ax.set_axis_off()
            else:
                choropleth(
                    sub,
                    ax=ax,
                    value_col=metric,
                    aggregation="mean",
                    regions=regions,
                    geo_col=geo_col,
                    region_col=region_col,
                    cmap=cmap,
                    colorbar_format=colorbar_format,
                    symmetric_cmap=metric in symmetric_metrics,
                )
            ax.set_title(f"{metric} (h={horizon})")

    return fig, axes
