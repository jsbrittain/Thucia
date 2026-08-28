# Thucia

```{toctree}
:maxdepth: 1
:hidden:

installation
quickstart
pipeline
models
case-sources
geo-covariates
data-layer
validation
cli
api
contributing
```

![Thucia](logo/logo.png)

Thucia is an intelligent disease forecasting platform named in homage to the
ancient historian _Thucydides_, who chronicled the Plague of Athens. Just as he
gave the world its first clinical record of an epidemic, Thucia brings clarity
to modern public health threats through data, modeling, and timely alerts. With
a focus on arboviral diseases like dengue, Zika, and chikungunya, Thucia
empowers governments, researchers, and health organizations with actionable
insight—before outbreaks escalate.

## What Thucia does

Thucia builds **probabilistic case forecasts** for geo-region (admin-2 level)
case data. It is a Python library (`Python >= 3.11`) with a growing set of
reusable, well-tested building blocks:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Area
  - What it provides
* - {doc}`Pipeline <pipeline>`
  - A chain of reusable stages that turn raw line-list case data into scored
    forecasts: aggregation, covariate merging, model-input preparation, model
    fitting, scoring, ensembling, and residual correction.
* - {doc}`Models <models>`
  - Twelve pluggable forecasting models (statistical baselines, ARIMA, and
    deep-learning / foundation models), each described by declarative metadata.
* - {doc}`Case data sources <case-sources>`
  - A plugin registry for ingesting case data. The first driver, Infodengue,
    fetches Brazilian municipality data with the disease (dengue, Zika,
    chikungunya) supplied as a parameter.
* - {doc}`Geo & covariate sources <geo-covariates>`
  - Climate and population covariates (WorldClim, EDO, NOAA, WorldPop) merged
    onto the case grid.
* - {doc}`Data layer <data-layer>`
  - A lazy DuckDB-backed `DataFrame` plus NetCDF/Zarr read/write helpers, with
    period columns and categorical geo codes handled transparently.
* - {doc}`Validation <validation>`
  - Backtesting over the pipeline stages to produce per-horizon WIS/RMSE/R² and
    skill relative to a reference model.
* - {doc}`CLI <cli>`
  - A command-line interface for case aggregation and an interactive Streamlit
    dashboard.
```

## Try it

```console
$ pip install -e .
$ thucia --help
```

See the {doc}`installation` guide and the {doc}`quickstart` to get a forecast
running in a few lines of Python.
