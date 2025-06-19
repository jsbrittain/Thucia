import seaborn as sns
import streamlit as st
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from thucia.core.cases import read_nc
from pathlib import Path

FOLDER_PATH = "/Users/jsb/repos/jsbrittain/thucia/data"

st.set_page_config(layout="wide")
st.title("🦟 Dengue Prediction Platform")

st.warning("WARNING - Development build — Do not rely on data.")

# --- Sidebar Controls
countries = {
    "Peru": "PER",
}
metrics = {
    "Cases": "Cases",
}

country_name = st.sidebar.selectbox("Country:", list(countries.keys()))
country = countries[country_name]

metric_name = st.sidebar.selectbox("Metric:", list(metrics.keys()), index=1)
metric = metrics[metric_name]

# --- Load Real Model Data
with st.spinner("Analysing datasets..."):
    filename = Path(FOLDER_PATH) / "cases" / country / "baseline.nc"

    df = read_nc(str(filename))
    df.fillna(0)

    def overlay_plot(data, color, **kwargs):
        ax = plt.gca()

        # Sort data by target_end_date
        data = data.sort_values("Date")

        ax.fill_between(
            data[data["quantile"] == 0.5]["Date"].values,
            data[data["quantile"] == 0.05]["prediction"].values,
            data[data["quantile"] == 0.95]["prediction"].values,
            color="green",
            alpha=0.1,
            label="95% Interval",
        )
        sns.lineplot(
            data=data[data["quantile"] == 0.5],
            x="target_end_date",
            y="prediction",
            ax=ax,
            label="Prediction",
            color="green",
        )
        sns.scatterplot(
            data=data[data["quantile"] == 0.5],
            x="target_end_date",
            y="true_value",
            ax=ax,
            label="Observed",
            color="red",
            s=20,
        )

    # Create FacetGrid
    g = sns.FacetGrid(
        df, col="location", col_wrap=2, height=2.5, aspect=2, sharey=False
    )
    g.map_dataframe(overlay_plot)
    g.set_axis_labels("Date", metric_name)
    # Format x-axis labels to be Year only (no repeat labels)
    for ax in g.axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
    # Figure legend (on second axis only)
    g.axes.flat[1].legend()

    st.pyplot(g.figure)
