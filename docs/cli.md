# Command-line interface

The `thucia` console entry point exposes case-aggregation commands and the
interactive dashboard. Run `thucia --help` to see the full list.

```console
$ thucia --help
```

## Commands

| Command                  | Purpose                                              |
|--------------------------|------------------------------------------------------|
| `thucia dashboard`       | Launch the Streamlit dashboard (needs the `vis` extra). |
| `thucia cases-per-month` | Aggregate case counts per month and write the result. |
| `thucia cases-per-week`  | Aggregate case counts per week and write the result.  |
| `thucia cases-per-day`   | Aggregate case counts per day and write the result.   |
| `thucia plot-cases-per-month` | Plot monthly case counts.                        |
| `thucia plot-cases-per-week`  | Plot weekly case counts.                         |
| `thucia plot-cases-per-day`   | Plot daily case counts.                          |
| `thucia provenance <file>`    | Show or verify the run metadata in a saved artefact. |

## The dashboard

```console
$ thucia dashboard
```

starts the Streamlit application (`streamlit run <viz>/dashboard/app.py`). It
provides map and per-province forecast explorers. It requires the `vis` extra —
see {doc}`installation`.

## Aggregation & plotting commands

The `cases-per-*` and `plot-cases-per-*` commands share a common set of
`--project`-style options:

```console
$ thucia cases-per-month \
    --project my_project \
    --cases-col Cases \
    --cases-file cases \
    --output-file cases_per_month
```

```{list-table}
:header-rows: 1
:widths: 24 76

* - Option
  - Meaning
* - `--project`
  - Project name (a subfolder under the projects root). Use an empty string to
    use the current directory. Default `"cases"`.
* - `--cases-col`
  - Column name containing the case counts (default `"Cases"`).
* - `--cases-file`
  - Input DB/file label passed to `read_db`.
* - `--output-file`
  - Output DB/file label passed to `write_db` (aggregation commands only).
* - `--projects-root`
  - Optional override for the projects root path.
```

The `plot-cases-per-*` commands additionally accept `--title`, `--date-col`
(column of dates, default `"Date"`), and `--cases-col`. They read the matching
`cases-per-*` output and display a line plot.

## Inspecting run metadata

Every artefact written by `fit_model`/`fit_models` records how it was produced
(see {doc}`data-layer`). `thucia provenance` reads it back without Python:

```console
$ thucia provenance out/baseline_cases_quantiles.duckdb
$ thucia provenance out/baseline_cases_quantiles.duckdb --document selection
$ thucia provenance out/baseline_cases_quantiles.duckdb --verify --include-deps
```

```{list-table}
:header-rows: 1
:widths: 24 76

* - Option
  - Meaning
* - `<file>`
  - Artefact to inspect (`.duckdb`, `.nc` or `.zarr`). Required.
* - `--document <name>`
  - Print only that document: `selection`, `config`, `code`, `inputs` or
    `environment`. Default: all documents.
* - `--json`
  - Emit compact single-line JSON (for piping) instead of indented JSON.
* - `--verify`
  - Compare the recorded `code` document with the implementation running now;
    exits `1` when they differ (mismatched fields go to stderr).
* - `--include-deps`
  - With `--verify`, also compare the numerics-stack versions
    (`darts`/`torch`/`statsmodels`/`numpy`/`pandas`/`duckdb`).
```

Exit codes: `0` success (including a matching `--verify`), `1` no metadata
recorded / implementation mismatch, `2` unknown `--document`.

## Projects

By default, project files are stored under `<package>/src/projects` (relative to
the installed package). Use `--projects-root` to point the CLI at your own
directory, or pass `--project ""` to operate in the current directory.
