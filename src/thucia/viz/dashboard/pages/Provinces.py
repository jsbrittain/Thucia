from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from thucia.core.cases import read_db

st.set_page_config(layout="wide")
st.title("🦟 Dengue Prediction Platform")

st.warning("DEVELOPMENT BUILD")

# --- Sidebar Controls
countries = {
    "Peru": "PER",
    "Brazil": "BRA",
    "Brazil-WHO": "BRA_who",
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
data_file_list = (
    list(data_folder.glob("*.duckdb"))
    + list(data_folder.glob("*.nc"))
    + list(data_folder.glob("*.zarr"))
)
data_filename = st.sidebar.selectbox(
    "Data file:", [data_file.name for data_file in data_file_list]
)
if data_filename is None:
    st.error("No data files found.")
    st.stop()
casedata_filename = str(data_folder / data_filename)

horizon = st.select_slider(
    "Horizon:",
    options=list(range(1, 13)),
    value=1,
)

show_interval = st.checkbox("Show Prediction Interval (90%)", value=True)
show_observed = st.checkbox("Show Observed", value=True)

# --- Load Real Model Data
with st.spinner("Loading datasets..."):
    filename = casedata_filename

    df = read_db(str(filename)).df
    # df.fillna(0)

    if "horizon" in df.columns:
        # If horizons are present, we take the 1-step ahead prediction
        df = df[df["horizon"] == horizon]
        df = df.drop(columns=["horizon"])

if "quantile" not in df.columns:
    st.error("The selected dataset does not contain quantile data.")
    st.stop()


def overlay_plot(data, color, **kwargs):
    ax = plt.gca()

    # Sort data by target_end_date
    data = data.sort_values("Date")

    if show_interval:
        ax.fill_between(
            # Take date from same quantiles as plot (there can be missing edge quantiles
            # compared with median quantiles)
            data[data["quantile"] == 0.05]["Date"].values,
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
    if show_observed:
        sns.scatterplot(
            data=data[data["quantile"] == 0.5],
            x="Date",
            y="Cases",
            ax=ax,
            label="Observed",
            color="red",
            s=20,
        )


if df["GID_2"].nunique() > 10:
    st.warning("Restricting view to the first 10 provinces only.")
    df = df[df["GID_2"].isin(df["GID_2"].unique()[:10])]


# Plot cannot cope with period dates
df["Date"] = df["Date"].astype("datetime64[ns]")


with st.spinner("Analysing datasets..."):
    # Create FacetGrid
    try:
        # remove unused categories before creating facet plot
        df["GID_2"] = df["GID_2"].astype(str).astype("category")
        # Plot
        g = sns.FacetGrid(
            df, col="GID_2", col_wrap=2, height=2.5, aspect=2, sharey=False
        )
        g.map_dataframe(overlay_plot)
        g.set_axis_labels("Date", metric_name)
        # Format x-axis labels to be Year only (no repeat labels)
        for ax in g.axes.flat:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.xaxis.set_major_locator(mdates.YearLocator(1))
            ax.xaxis.set_tick_params(rotation=45)
        # Figure legend
        if len(g.axes.flat) > 0:
            g.axes.flat[0].legend()

        st.pyplot(g.figure)
    except Exception as e:
        st.error(f"Error creating plots: {e}")
        st.stop()
