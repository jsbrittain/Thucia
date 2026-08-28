# API Reference

The public API lives under `thucia.core` and resolves lazily: importing
`thucia.core` is cheap, and submodules load on first attribute access.

```{eval-rst}
.. currentmodule:: thucia.core
```

## Pipeline stages

The reusable computation stages extracted from the research pipeline. Each is
a thin, data-in/data-out function configured by
{class}`PipelineConfig <thucia.core.pipeline.PipelineConfig>`.

```{eval-rst}
.. autoclass:: thucia.core.pipeline.PipelineConfig
   :members:

.. autofunction:: thucia.core.pipeline.cases_per_period
.. autofunction:: thucia.core.pipeline.merge_covariates
.. autofunction:: thucia.core.pipeline.prepare_model_inputs
.. autofunction:: thucia.core.pipeline.fit_model
.. autofunction:: thucia.core.pipeline.score_model
.. autofunction:: thucia.core.pipeline.aggregate_quantiles
.. autofunction:: thucia.core.pipeline.build_ensemble
.. autofunction:: thucia.core.pipeline.apply_residual_regression
```

## Models

```{eval-rst}
.. autofunction:: thucia.core.models.list_models
.. autofunction:: thucia.core.models.get_model
.. autofunction:: thucia.core.models.get_model_spec
.. autofunction:: thucia.core.models.run_model
.. autoclass:: thucia.core.models._meta.ModelSpec
   :members:
```

## Data layer

```{eval-rst}
.. autoclass:: thucia.core.fs.DataFrame
   :members:

.. autofunction:: thucia.core.cases.read_db
.. autofunction:: thucia.core.cases.write_db
.. autofunction:: thucia.core.cases.read_nc
.. autofunction:: thucia.core.cases.write_nc
```

## PDFM embeddings

PDFM embeddings are **not publicly distributed** — they must be supplied by
the user as a NetCDF file with one row per admin region: a geo-code column
(such as `GID_2`) plus `feature0`..`feature329` embedding columns. Load a file
with {func}`thucia.core.cases.prepare_pdfm_embeddings` (which can restrict to
a subset of provinces and dedupes geo codes), then feed the frame to
{func}`thucia.core.pipeline.apply_residual_regression` to correct per-region
forecast bias. Provinces without embeddings are dropped with a warning and
regression continues on the rest.

```{eval-rst}
.. autofunction:: thucia.core.cases.prepare_pdfm_embeddings
.. autofunction:: thucia.core.cases.prepare_embeddings
```

## Scoring

```{eval-rst}
.. autofunction:: thucia.core.cases.wis
.. autofunction:: thucia.core.cases.r2
.. autofunction:: thucia.core.cases.rmse
```

## Plugin registry

Covariate sources (WorldClim, EDO, NOAA, WorldPop) and cache backends
self-register through a shared registry primitive.

```{eval-rst}
.. autoclass:: thucia.core.registry.Registry
   :members:

.. autoclass:: thucia.core.registry.PluginNotFoundError
```

```{eval-rst}
.. automodule:: thucia.core.geo.plugin_base
   :members: SourceBase, source_registry
```

## Validation

Backtesting over the pipeline stages: cut the history at a sweep of cutoff
dates, fit on the past only, score the held-out window, and aggregate
WIS/RMSE/R2 per horizon (plus skill relative to a reference model).

```{eval-rst}
.. autofunction:: thucia.core.validation.expand_cutoffs
.. autoclass:: thucia.core.validation.BacktestConfig
.. autofunction:: thucia.core.validation.run_backtest
.. autoclass:: thucia.core.validation.BacktestResult
```
