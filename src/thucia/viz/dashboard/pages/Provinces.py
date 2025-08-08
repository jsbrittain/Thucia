from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from thucia.core.cases import read_nc

st.set_page_config(layout="wide")
st.title("🦟 Dengue Prediction Platform")

st.warning("DEVELOPMENT BUILD")

# --- Sidebar Controls
countries = {
    "Peru": "PER",
    # "Brazil": "BRA",
    "Mexico": "MEX",
}
metrics = {
    "Cases": "Cases",
}

country_name = st.sidebar.selectbox("Country:", list(countries.keys()))
country = countries[country_name]

metric_name = st.sidebar.selectbox("Metric:", list(metrics.keys()), index=0)
metric = metrics[metric_name]

data_folder = (
    Path(__file__).parent.parent.parent.parent.parent.parent
    / "data"
    / "cases"
    / country
)
data_file_list = list(data_folder.glob("*.nc"))
data_filename = st.sidebar.selectbox(
    "Data file:", [data_file.name for data_file in data_file_list]
)
casedata_filename = str(data_folder / data_filename)

# --- Load Real Model Data
with st.spinner("Loading datasets..."):
    filename = casedata_filename

    df = read_nc(str(filename))
    df.fillna(0)

if "quantile" not in df.columns:
    st.error("The selected dataset does not contain quantile data.")
    st.stop()


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
        x="Date",
        y="prediction",
        ax=ax,
        label="Prediction",
        color="green",
    )
    sns.scatterplot(
        data=data[data["quantile"] == 0.5],
        x="Date",
        y="Cases",
        ax=ax,
        label="Observed",
        color="red",
        s=20,
    )


with st.spinner("Analysing datasets..."):
    # Create FacetGrid
    g = sns.FacetGrid(df, col="GID_2", col_wrap=2, height=2.5, aspect=2, sharey=False)
    g.map_dataframe(overlay_plot)
    g.set_axis_labels("Date", metric_name)
    # Format x-axis labels to be Year only (no repeat labels)
    for ax in g.axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_tick_params(rotation=45)
    # Figure legend (on second axis only)
    g.axes.flat[1].legend()

    st.pyplot(g.figure)
