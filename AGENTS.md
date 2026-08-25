# AGENTS.md

Thucia is a disease-forecasting library (arboviral: dengue, Zika, chikungunya) for geo-region case data. Python >=3.11 (CI runs 3.13).

## Commands

- Env: `uv sync --all-extras` then `pip install -e .` (see `dev/README.md`). No `.venv` is committed.
- Tests: `uv run pytest` (CI uses `uv run pytest --cov=thucia --cov-report=term --cov-report=html`). Run `uv run pytest tests/thucia/core/models/test_wis.py::test_perfect_forecast_zero_wis` for a single test.
- Lint/format: `pre-commit` runs `ruff`, `ruff-format`, and `reorder-python-imports` (import ordering is NOT isort/black style). Run `pre-commit run --all-files`.
- Docs are Sphinx (`docs/`), built on ReadTheDocs from `docs/requirements.txt` (myst-parser, sphinx-book-theme).
- CLI: `thucia <command>` (dashboard, cases-per-month, cases-per-week, cases-per-day, plot-*). `thucia dashboard` launches the Streamlit app.

## Architecture

- `src/thucia/core/` is the library. `flow/` wraps Prefect tasks/flows (`flow/wrappers.py`); `viz/` is matplotlib + Streamlit.
- **`flow/` is DEPRECATED and broken** (Prefect-specific; scheduled for rework). `flow/fs` imports a non-existent `thucia.flow.wrapper`; `flow/containers` tasks recurse into themselves; `flow/models` references core symbols that don't exist. Do not extend it; don't rely on it.
- `pipeline/analysis_core.py` is the primary research pipeline: one `Steps` toggle flag per stage (cases-per-month → merge covariates → model fitting → quantiles → PDFM residual regression → stats). `pipeline/analysis_flow.py` is a Prefect variant; most blocks are `if False:`.
- Case data lives per-country under `data/cases/<ISO3>/` with custom loaders (e.g. `data/cases/BRA/load_cases.py`). `data/` is gitignored.

## Data layer (DuckDB)

- `read_db`/`write_db` (re-exported from `thucia.core.fs` and `thucia.core.cases`) default to `.duckdb` files; the extension is appended if missing. `.nc`/`.zarr` are also supported.
- `thucia.core.fs.DataFrame` is a lazy wrapper over a DuckDB table. Access the full frame via `.df`; `write_db`/`DataFrame(df=...)` writes one table named `data`.
- Period columns (`pd.Period`, e.g. monthly case dates) are stored as timestamps + metadata in the `__column_metadata__` table and restored on read — don't bypass the wrapper when writing.
- `GID_1`/`GID_2` are stored as DuckDB ENUMs. When calling `.append()`, all categories must be present on first write.

## Models

- `src/thucia/core/models/` is auto-discovered: each module must expose a callable whose name matches the module name (e.g. `sarima.py` → `sarima(df, ...)`), since `models/__init__.py` imports by name. Helper modules that don't satisfy this contract must be listed in `_HELPER_MODULES` (currently `ensemble`, `quantiles`) so they aren't advertised as models; the discovery also skips subpackages (`darts`, `utils`).
- The models package uses a module-type `__getattribute__` override so `models.<name>` / `getattr(models, name)` always resolves to the model callable — even if `thucia.core.models.<name>` was imported as a submodule first (which would otherwise shadow the function).
- Model entry points share a unified interface: every model takes `df, start_date, end_date, gid_1, horizons=[1] (a list), case_col="Log_Cases", covariate_cols, retrain, db_file, model_admin_level, multivariate, num_samples`. Never introduce a scalar `horizon` param — `pipeline/analysis_core.py` passes `horizons=[1,3,6,12]` to every model, and `run_model()` forwards it. (tft/tide/chronos were broken by this; keep them list-form.)
- The canonical quantile grid lives once in `thucia.core.quantiles` (15 levels) and is re-exported via `thucia.core.models.quantiles`, `thucia.core.models.utils`, and `thucia.core.models.utils.residual_quantiles` — import it, don't redefine it. `chronos` uses a documented 13-level subset (`CHRONOS_QUANTILES`); `quantile_sum_fast` defaults to the canonical grid.
- Model input DataFrames need columns `Date, GID_1, GID_2, future, Cases, Log_Cases` plus covariates. Covariates must be NaN-free — run `sanitise_covariates()` first and assert no NaNs. The geo column is coerced to `categorical` by `DartsBase` (models rely on `.cat` access).
- `run_model()` converts samples→quantiles if needed and writes `{name}_cases_quantiles.duckdb`; the file name derives from the `name` argument (the pipeline passes `model.__name__`).
- Darts-based models (sarima, tcn, tft, nbeats, nhits, tide) share `DartsBase` (`core/models/darts/`); `torch.set_float32_matmul_precision("medium")` is set at import. `timesfm` needs torch (`timesfm[torch]` extra; base deps only pull `timesfm` without torch); chronos is a separate extra.

## Geo & covariates

- Covariate plugins live in `core/geo/sources/` (`worldclim`, `edo`, `noaa`, `worldpop`); each subclasses `SourceBase` and registers a `ref` string. Merge via `merge_sources(df, ["worldclim.*"])`.
- Sources download from geodata.ucdavis.edu and cache under the platformdirs cache dir (`~/.cache/global.Health/thucia/` on macOS), including GADM GeoPackages. WorldClim merging is the slowest pipeline step.
- `pad_admin2()` downloads the full GADM admin-2 list for a country, so geo code tests mock `get_admin2_list` (see `tests/thucia/pipeline/test_pipeline.py`).

## Containers & INLA

- `thucia.core.containers` auto-detects Docker, else the user Podman socket (`unix:///run/user/$UID/podman/podman.sock`). No runtime → `RuntimeError`.
- The INLA model (`core/models/inla.py`) builds and runs an R container (`core/models/inla/Dockerfile`, platform `linux/amd64`).
- `tests/thucia/core/containers/test_containers.py` builds/runs an alpine image and fails if no container runtime is running — skip it locally if Docker/Podman isn't up.

## Tests

- Unit tests under `tests/thucia/core/...` (cases, fs, models, geo, cache). `tests/thucia/pipeline/` uses generated data (`generate_test_data.py` writes `tests/thucia/pipeline/test_data/`); `test_merge_covariates` and `test_model_fitting` are marked `@pytest.mark.skip`.
- The suite was written "probingly": tests assert real correctness properties (WIS analytic values and monotonicity, quantile ordering, ensemble weight normalization, adapter residual shrinkage, cache round-trips) and have surfaced + fixed real bugs. Keep this style: prefer asserting properties over smoke-testing.
- Known hardening already applied (add regression tests if you touch these): `DataFrame.head()` was broken; `sanitise_covariates()` crashed on string/Timestamp `start_date`; `run_model()` had mutable default args; `apply_weights_to_forecasts()` crashed on `pd.Period` dates; `r2_score()` returned `-inf` on constant targets; `wis_bracher()` didn't implement its documented `model` auto-grouping.
- `tests/thucia/core/models/test_models_registry.py` enforces the model contract: every advertised model binds the pipeline's `model_kwargs` (via `inspect.signature().bind_partial`) and uses a list-form `horizons` default. `tests/thucia/core/test_quantiles.py` pins the canonical grid to a single object across all import paths and checks `quantile_sum_fast`'s default matches it.
- Coverage is ~48%. The forecasting model bodies (sarima/tcn/xgboost/... and `DartsBase`), `viz/`, `flow/`, `core/geo/stats.py`, `core/models/utils/covariates.py`, and `cli/steps.py` remain largely untested — fitting darts models needs real synthetic time series and is intentionally not covered by fast unit tests.
- `tests/thucia/core/containers/test_containers.py` builds/runs an alpine image and fails if no container runtime is running — skip it locally if Docker/Podman isn't up.
