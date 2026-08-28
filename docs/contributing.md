# Contributing

Thanks for contributing to Thucia! This page covers the developer workflow:
environment setup, testing, linting, and building these docs.

## Setting up

Thucia uses [uv](https://docs.astral.sh/uv/). Make sure it is installed, then
set up the environment with all extras:

```console
$ uv sync --all-extras
$ uv pip install -e .
```

Python 3.11+ is required (CI runs 3.13).

## Layout

- `src/thucia/core/` — the library. `core/__init__.py` is a **lazy** public
  surface; internal detail lives in the submodules (`pipeline`, `validation`,
  `models`, `cases`, `geo`, `fs`, `cache`, `container`).
- `src/thucia/viz/` — matplotlib + Streamlit dashboard.
- `src/thucia/cli/` — the `thucia` command-line entry point.
- `tests/` — the test suite (see below).
- `docs/` — these Sphinx docs.

## Running the tests

```console
$ uv run pytest                     # fast suite by default
$ uv run pytest --cov=thucia        # with coverage (works locally)
$ uv run pytest -m slow             # heavy real-model fits
```

The suite is configured (`addopts`) to load `-p thucia_pytest_bootstrap`, which
pre-imports `duckdb` before `pytest-cov` starts, because coverage's source
discovery otherwise corrupts `duckdb`'s multi-module extension submodules on
CPython 3.13.

Some tests need a container runtime or network:

```console
$ uv run pytest tests/thucia/core/containers/   # needs Docker/Podman running
$ uv run pytest -m "slow and network"           # timesfm checkpoint download
```

### Test style

Tests are written **probingly** — they assert real correctness properties
(WIS analytic values, quantile ordering, ensemble weight normalisation, cache
round-trips) rather than smoke-testing. When you touch a code path, add a
property-based regression test in the same style.

## Linting & formatting

Pre-commit manages `ruff`, `ruff-format`, and `reorder-python-imports`. Note
that import ordering here is **not** isort/black style.

```console
$ pre-commit run --all-files
```

## Building these docs

The docs are built with Sphinx + Myst-Parser + the Sphinx Book theme, and use
`mermaid` diagrams.

```console
$ uv sync --all-extras                       # pulls in docs/requirements.txt
$ uv run sphinx-build -b html -W --keep-going docs docs/_build/html
```

`-W` treats warnings as errors, so keep the build clean. If
`sphinxcontrib.mermaid` is missing locally, `uv pip install sphinxcontrib-mermaid`
(it is already in `docs/requirements.txt` and installed on ReadTheDocs).

## Architectural conventions

- **Covariate sources** subclass `SourceBase`, set a `ref`, implement
  `merge`, and self-register via `@source_registry.register()` (a shared
  `thucia.core.registry.Registry` that also backs case sources and cache
  backends).
- **Case sources** subclass `CaseSource` and are **disease-generic**: the
  disease is a `fetch(**params)` argument, never a hard-coded constant.
- **Models** live under `core/models/`; a module must expose a callable whose
  name matches the module name. Each model declares a module-level `ModelSpec`
  so the pipeline and backtest can dispatch without name special-casing. Keep
  the `horizons` argument list-form.
- **New public names** must be added to `thucia.core.__all__` and pinned in
  `tests/thucia/core/test_core_api.py`.
