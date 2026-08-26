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
.. autofunction:: thucia.core.models.run_model
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
