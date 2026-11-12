import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import thucia.core.cases as cases
import thucia.core.geo as geo
from thucia.core.fs import read_db
from thucia.core.fs import read_run_metadata
from thucia.core.fs import write_db

PROJECTS_ROOT = Path(__file__).parent.parent.parent.resolve() / "projects"


def launch_dashboard():
    app_path = Path(__file__).parent.parent / "viz" / "dashboard" / "app.py"
    subprocess.run(["streamlit", "run", app_path], check=True)


def cases_per_month(
    project: str = "cases",
    cases_col: str = "Cases",
    cases_file: str = "cases",
    output_file: str = "cases_per_month",
    projects_root: Path | None = None,
):
    # If project specified, use projects root, otherwise use current directory
    projects_root = projects_root or PROJECTS_ROOT
    path = (Path(projects_root) / project).resolve() if project else Path(".").resolve()

    # Read cases data
    tdf = read_db(path / cases_file)

    # Aggregate cases per month
    tdf = cases.cases_per_month(tdf, cases_col=cases_col)
    tdf = geo.ensure_all_regions(
        tdf
    )  # Ensure all Admin-2 regions included for covariate maps

    # Write output
    write_db(tdf, path / output_file)


def plot_cases_per_month(
    project: str = "cases",
    cases_file: str = "cases_per_month",
    projects_root: Path | None = None,
    title: str | None = None,
    date_col: str = "Date",
    cases_col: str = "Cases",
):
    # If project specified, use projects root, otherwise use current directory
    projects_root = projects_root or PROJECTS_ROOT
    path = (Path(projects_root) / project).resolve() if project else Path(".").resolve()

    # Read cases per month data
    tdf = read_db(path / cases_file)

    # Aggregate and plot
    df = tdf[[date_col, cases_col]].groupby([date_col])[cases_col].sum()
    df.index = df.index.to_timestamp(how="end")
    plt.figure(figsize=(10, 6))
    plt.plot(df)
    plt.title(title or f"Monthly Cases for {project}")
    plt.xlabel(date_col)
    plt.ylabel(cases_col)
    plt.show()


def cases_per_week(
    project: str = "cases",
    cases_col: str = "Cases",
    cases_file: str = "cases",
    output_file: str = "cases_per_week",
    projects_root: Path | None = None,
):
    # If project specified, use projects root, otherwise use current directory
    projects_root = projects_root or PROJECTS_ROOT
    path = (Path(projects_root) / project).resolve() if project else Path(".").resolve()

    # Read cases data
    tdf = read_db(path / cases_file)

    # Aggregate cases per week
    tdf = cases.aggregate_cases(tdf, cases_col=cases_col, freq="W")

    # Write output
    write_db(tdf, path / output_file)


def plot_cases_per_week(
    project: str = "cases",
    cases_file: str = "cases_per_week",
    projects_root: Path | None = None,
    title: str | None = None,
    date_col: str = "Date",
    cases_col: str = "Cases",
):
    # If project specified, use projects root, otherwise use current directory
    projects_root = projects_root or PROJECTS_ROOT
    path = (Path(projects_root) / project).resolve() if project else Path(".").resolve()

    # Read cases per week data
    tdf = read_db(path / cases_file)

    # Aggregate and plot
    df = tdf[[date_col, cases_col]].groupby([date_col])[cases_col].sum()
    df.index = df.index.to_timestamp(how="end")
    plt.figure(figsize=(10, 6))
    plt.plot(df)
    plt.title(title or f"Weekly Cases for {project}")
    plt.xlabel(date_col)
    plt.ylabel(cases_col)
    plt.show()


def cases_per_day(
    project: str = "cases",
    cases_col: str = "Cases",
    cases_file: str = "cases",
    output_file: str = "cases_per_day",
    projects_root: Path | None = None,
):
    # If project specified, use projects root, otherwise use current directory
    projects_root = projects_root or PROJECTS_ROOT
    path = (Path(projects_root) / project).resolve() if project else Path(".").resolve()

    # Read cases data
    tdf = read_db(path / cases_file)

    # Aggregate cases per day
    tdf = cases.aggregate_cases(tdf, cases_col=cases_col, freq="D")

    # Write output
    write_db(tdf, path / output_file)


def plot_cases_per_day(
    project: str = "cases",
    cases_file: str = "cases_per_day",
    projects_root: Path | None = None,
    title: str | None = None,
    date_col: str = "Date",
    cases_col: str = "Cases",
):
    # If project specified, use projects root, otherwise use current directory
    projects_root = projects_root or PROJECTS_ROOT
    path = (Path(projects_root) / project).resolve() if project else Path(".").resolve()

    # Read cases per day data
    tdf = read_db(path / cases_file)

    # Aggregate and plot
    df = tdf[[date_col, cases_col]].groupby([date_col])[cases_col].sum()
    df.index = df.index.to_timestamp(how="end")
    plt.figure(figsize=(10, 6))
    plt.plot(df)
    plt.title(title or f"Daily Cases for {project}")
    plt.xlabel(date_col)
    plt.ylabel(cases_col)
    plt.show()


def provenance(
    file: str | Path,
    document: str | None = None,
    as_json: bool = False,
    verify: bool = False,
    include_deps: bool = False,
) -> int:
    """Print, or verify, the run metadata recorded in a saved artefact.

    Returns a process exit code: 0 on success (and on a successful --verify),
    1 when there is no metadata / the implementation does not match, 2 for an
    unknown document name.
    """
    docs = read_run_metadata(file)
    if not docs:
        print(
            f"No run metadata recorded in {file}. Artefacts written by "
            "fit_model/fit_models carry it; older files do not.",
            file=sys.stderr,
        )
        return 1

    if verify:
        # Imported lazily: `thucia.core.pipeline` pulls in the model stack, and
        # the CLI's import path must stay cheap.
        from thucia.core.pipeline.provenance import verify_code_identity

        recorded = docs.get("code") or {}
        if not recorded:
            print(f"No 'code' document recorded in {file}.", file=sys.stderr)
            return 1
        mismatches = verify_code_identity(recorded, include_deps=include_deps)
        if not mismatches:
            print(f"{file}: matches the running implementation")
            return 0
        for field, (want, have) in mismatches.items():
            print(f"{field}: recorded {want!r} != current {have!r}", file=sys.stderr)
        return 1

    if document is not None:
        if document not in docs:
            print(
                f"Unknown document {document!r} in {file}; available: "
                f"{', '.join(sorted(docs))}.",
                file=sys.stderr,
            )
            return 2
        payload = docs[document]
    else:
        payload = docs

    if as_json:
        print(json.dumps(payload, indent=None, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
