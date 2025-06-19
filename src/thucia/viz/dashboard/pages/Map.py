import numpy as np
import pydeck as pdk
import colorcet as cc
import streamlit as st
import geopandas as gpd

from pathlib import Path
from thucia.core.fs import cache_folder
from thucia.core.cases import read_nc

st.set_page_config(layout="wide")
st.title("Thucia")

st.warning("DEVELOPMENT BUILD")

# --- Sidebar Controls
countries = {
    "Peru": "PER",
    # "Brazil": "BRA",
    "Mexico": "MEX",
}
admin_levels = {
    "Admin 1": "ADM1",
    "Admin 2": "ADM2",
}
basemaps = {
    "Light": "light",
    "Dark": "dark",
    "Satellite": "satellite",
}
colormaps = {
    "CET D8": cc.CET_D8,
    "Fire": cc.fire,
    "Rainbow": cc.rainbow,
    "Blue": cc.blues,
    "Cool": cc.kbc,
}

country_name = st.sidebar.selectbox("Country:", list(countries.keys()))
country = countries[country_name]

admin_level_name = st.sidebar.selectbox("Admin Level:", list(admin_levels.keys()))
admin_level = admin_levels[admin_level_name]

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

# Load geospatial data
geo_filename = Path(cache_folder) / "geo" / country / f"gadm41_{country}.gpkg"
if admin_level == "ADM1":
    geojson = gpd.read_file(geo_filename, layer="ADM_ADM_1")
elif admin_level == "ADM2":
    geojson = gpd.read_file(geo_filename, layer="ADM_ADM_2")
else:
    raise ValueError(f"Unsupported admin level: {admin_level}")


def get_color(score, cmap, alpha=0):
    hex_color = cmap[int(score * (len(cmap) - 1))]
    rgb = tuple(int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    return list(rgb) + [alpha]


def get_colormap():
    cmap = colormaps[selected_cmap]
    return cmap


# --- Date slider
case_data = read_nc(str(data_folder / "cases_with_climate.nc"))  # base dataset

# Now read model predictions and merge with case data
prediction_data = read_nc(casedata_filename)
if "quantile" in prediction_data.columns.tolist():
    # If quantiles are present, we take the median
    prediction_data = prediction_data[prediction_data["quantile"] == 0.5]
    prediction_data = prediction_data.drop(columns=["quantile"])
    case_data = case_data.merge(
        prediction_data[["Date", "GID_2", "prediction"]],
        on=["Date", "GID_2"],
        how="left",
    )
if "sample" in prediction_data.columns.tolist():
    # If samples are present, we take the median
    prediction_data = prediction_data.groupby(["Date", "GID_2"]).median().reset_index()
    case_data = case_data.merge(
        prediction_data[["Date", "GID_2", "prediction"]],
        on=["Date", "GID_2"],
        how="left",
    )

available_dates = sorted(case_data["Date"].unique())
start_date, end_date = st.select_slider(
    "Select a date:",
    options=available_dates,
    format_func=lambda d: d.strftime("%Y-%m-%d"),
    value=(available_dates[-2], available_dates[-1]),
)

# Filter out non-numeric metrics
metrics = case_data.columns.tolist()
metrics = [m for m in metrics if case_data[m].dtype.kind in "biufc"]
metric = st.sidebar.selectbox(
    "Metric:",
    metrics,
    index=metrics.index("Cases") if "Cases" in metrics else 0,
)
if metric in ["Cases", "prediction"]:
    metric_op = "sum"
else:
    metric_op = "mean"

# Aggregate metric over selected date range
case_data = (
    case_data[(case_data["Date"] >= start_date) & (case_data["Date"] <= end_date)][
        [metric, "GID_1"]
    ]
    .groupby(case_data["GID_2"])
    .agg(
        {
            metric: metric_op,
            "GID_1": "first",
        }
    )
    .reset_index()
)

if admin_level == "ADM1":
    # Aggregate over GID_2 for ADM1 level
    case_data = (
        case_data.groupby("GID_1")
        .agg(
            {
                metric: metric_op,
                "GID_2": "first",
            }
        )
        .reset_index()
    )

case_data[case_data[metric] == 0] = (
    np.nan
)  # Set zero values to NaN for better visualization

if admin_level == "ADM1":
    gid_lookup = "GID_1"
elif admin_level == "ADM2":
    gid_lookup = "GID_2"

basemap_name = st.sidebar.selectbox("Basemap style:", list(basemaps.keys()))
basemap = basemaps[basemap_name]

selected_cmap = st.sidebar.selectbox("Colormap:", list(colormaps.keys()))
alpha_value = st.sidebar.slider(
    "Fill Opacity (Alpha)", min_value=0, max_value=255, value=140
)

# --- Load model data
model_data = {
    gid2: cases
    for gid2, cases in zip(case_data[gid_lookup].values, case_data[metric].values)
}
color_white = [255, 255, 255, alpha_value]

# simplify geometry
geojson["geometry"] = geojson["geometry"].simplify(
    tolerance=0.01, preserve_topology=True
)

geojson["data"] = ["-"] * len(geojson)
geojson["fill_color"] = [color_white] * len(geojson)


min_data = np.nanmin(list(model_data.values()))
max_data = np.nanmax(list(model_data.values()))
if max_data > 0:
    color_white = [255, 255, 255, alpha_value]
    for i, line in geojson.iterrows():
        name = line[gid_lookup]
        data = model_data.get(name, np.nan)
        if not np.isnan(data):
            # scale color to data
            normalised_data = (data - min_data) / (max_data - min_data)
            geojson.at[i, "data"] = f"{data}"
            geojson.at[i, "fill_color"] = get_color(
                normalised_data, get_colormap(), alpha_value
            )

# --- Select basemap
minx, miny, maxx, maxy = geojson["geometry"].total_bounds
center_lon = (minx + maxx) / 2
center_lat = (miny + maxy) / 2


def compute_zoom(minx, maxx):
    width_deg = maxx - minx
    return np.clip(8 - np.log2(width_deg), 1, 20)


map_style = {
    "light": "mapbox://styles/mapbox/light-v11",
    "dark": "mapbox://styles/mapbox/dark-v11",
    "satellite": "mapbox://styles/mapbox/satellite-streets-v12",
}[basemap]

# --- Create layer
layer = pdk.Layer(
    "GeoJsonLayer",
    geojson,
    pickable=True,
    auto_highlight=True,
    get_fill_color="fill_color",
    get_line_color=[30, 30, 30],
    line_width_min_pixels=0.5,
)

# --- Tooltip config
tooltip_text = f"{metric}: {{data}}"
if admin_level == "ADM1":
    tooltip = {
        "html": f"<b>{{NAME_1}}</b><br>{tooltip_text}",
        "style": {"backgroundColor": "white", "color": "black"},
    }
elif admin_level == "ADM2":
    tooltip = {
        "html": f"<b>{{NAME_2}}</b><br>{tooltip_text}",
        "style": {"backgroundColor": "white", "color": "black"},
    }

# --- Show map
deck = pdk.Deck(
    layers=[layer],
    map_style=map_style,
    tooltip=tooltip,
)
st.pydeck_chart(deck)
