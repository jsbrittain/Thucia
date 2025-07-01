import logging
from typing import Callable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from thucia.core.geo import get_admin2_list


def plot_all_admin2(
    df_baseline: pd.DataFrame,
    gid_1: list[str] | None = None,
    limit: int = 20,
    metric: str = "Cases",  # 'Cases'
    measure: str = "quantiles",  # 'samples' or 'quantiles'
    transform: Callable | None = None,  # Optional transformation function
):
    # Determine ISO3 code from the DataFrame
    iso3 = df_baseline["GID_2"].str.split(".").str[0].unique()
    if len(iso3) != 1:
        raise ValueError(f"Expected a single ISO3 code, found {len(iso3)}: {iso3}")
    iso3 = iso3[0]
    admin_list = get_admin2_list(iso3)

    # Admin-1 filter
    if gid_1:
        gid_2s = df_baseline[df_baseline["GID_1"].isin(gid_1)]["GID_2"].unique()
    else:
        gid_2s = df_baseline["GID_2"].unique()
    gid_2names = admin_list[admin_list["GID_2"].isin(gid_2s)]["NAME_2"]

    # Limit sub-plots
    if len(gid_2s) > limit:
        logging.warning(
            f"Too many Admin-2 regions to render ({len(gid_2s)}), limiting to first 20."
        )
        gid_2s = gid_2s[:limit]
        gid_2names = gid_2names[:limit]

    cols = 2
    rows = np.ceil(len(gid_2s) / cols).astype(int)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 8))
    for gid_2, gid_2name, ax in zip(gid_2s, gid_2names, axes.flatten()[: len(gid_2s)]):
        logging.info(f"Generating plot for {gid_2name} ({gid_2})")

        df_subset = df_baseline[df_baseline["GID_2"] == gid_2]
        df_subset = df_subset[df_subset["Date"] >= "2018-01-01"]
        df_subset = df_subset[df_subset["Date"] < "2022-01-01"]

        date = df_subset[df_subset["quantile"] == 0.50]["Date"].values
        cases = df_subset[df_subset["quantile"] == 0.50][metric].values

        if measure == "quantiles":
            q05 = df_subset[df_subset["quantile"] == 0.05]["prediction"].values
            q50 = df_subset[df_subset["quantile"] == 0.50]["prediction"].values
            q95 = df_subset[df_subset["quantile"] == 0.95]["prediction"].values

            # Apply transformation if provided
            if transform:
                q05 = transform(q05)
                q50 = transform(q50)
                q95 = transform(q95)
                cases = transform(cases)

            ax.fill_between(date, q05, q95, color="green", alpha=0.3)
            ax.plot(
                date,
                cases,
                label=metric,
                color="red",
                marker="o",
                linestyle="None",
            )
            ax.plot(date, q50, label=metric)
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.set_title(f"{gid_2name} ({gid_2})")
        elif measure == "samples":
            raise NotImplementedError(
                "Samples measure is not implemented yet. Please use 'quantiles'."
            )
        else:
            raise ValueError(
                f"Unknown measure: {measure}. Use 'quantiles' or 'samples'."
            )

    plt.show()


def plot_ensemble_weights_over_time(
    df, date_col="Date", title="Ensemble Weights Over Time", ma_window=None
):
    """
    Plot ensemble weights over time from a wide-format DataFrame, with optional moving average smoothing.

    Parameters:
        df (pd.DataFrame): DataFrame with one date column and one or more model weight columns.
        date_col (str): Name of the column containing dates.
        title (str): Plot title.
        ma_window (int or None): If set, applies a moving average of the given window size to smooth the lines.
    """
    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' must be a column in the DataFrame")

    # Apply moving average if specified
    df_plot = df.copy()
    if ma_window:
        for col in df_plot.columns:
            if col != date_col:
                df_plot[col] = (
                    df_plot[col].rolling(window=ma_window, min_periods=1).mean()
                )

    # Convert to long format
    df_long = df_plot.melt(id_vars=date_col, var_name="model", value_name="weight")
    df_long = df_long.dropna(subset=["weight"])

    # Plot
    plt.figure(figsize=(12, 6))
    for model in df_long["model"].unique():
        model_data = df_long[df_long["model"] == model]
        plt.plot(model_data[date_col], model_data["weight"], label=model)

    plt.title(title + (f" (MA Window = {ma_window})" if ma_window else ""))
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.legend(title="Model")
    plt.grid(True)

    # Limit x-axis ticks to years only
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()
