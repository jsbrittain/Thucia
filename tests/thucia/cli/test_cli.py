# Probing tests for the CLI: parser/dispatch wiring and the aggregation steps
# against a temp project (GADM mocked, no network).
import matplotlib

matplotlib.use("Agg")

import json

import pandas as pd
import pytest
from thucia import build_parser
from thucia import main
from thucia.cli import CommandsList
from thucia.cli import steps
from thucia.core.cases import read_db
from thucia.core.cases import write_db


@pytest.fixture
def project(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    return d


@pytest.fixture
def admin2_list():
    return pd.DataFrame(
        {
            "GID_1": ["G.1_1"] * 3,
            "GID_2": ["G.1.1_2", "G.1.2_2", "G.1.3_2"],
            "NAME_1": ["State"] * 3,
            "NAME_2": ["A", "B", "C"],
        }
    )


def _line_list(n_months=8, cases_col="Cases"):
    dates = pd.date_range("2020-01-31", periods=n_months, freq="ME").strftime(
        "%Y-%m-%d"
    )
    return pd.DataFrame(
        {
            "Date": dates.tolist() * 2,
            "GID_1": ["G.1_1"] * (2 * n_months),
            "GID_2": (["G.1.1_2"] * n_months) + (["G.1.2_2"] * n_months),
            cases_col: list(range(1, n_months + 1)) * 2,
        }
    )


def test_parser_exposes_all_commands():
    parser = build_parser()
    for cmd in CommandsList:
        # `minimal_argv` supplies any required positional (e.g. a file to inspect)
        parser.parse_args([cmd.name, *cmd.minimal_argv])
    with pytest.raises(SystemExit):
        parser.parse_args(["not-a-command"])


def test_main_requires_a_command():
    with pytest.raises(SystemExit):
        main([])


def test_main_dispatches_cases_per_month(monkeypatch):
    calls = {}

    def fake(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr("thucia.cli.steps.cases_per_month", fake)
    main(["cases-per-month", "--project", "myproj", "--output-file", "out"])
    assert calls["project"] == "myproj"
    assert calls["output_file"] == "out"
    assert calls["cases_file"] == "cases"


def test_main_dispatches_dashboard(monkeypatch):
    calls = []

    def fake_launch():
        calls.append(1)

    monkeypatch.setattr("thucia.cli.steps.launch_dashboard", fake_launch)
    main(["dashboard"])
    assert calls == [1]


def test_cases_per_month_step(project, admin2_list, monkeypatch):
    monkeypatch.setattr("thucia.core.geo.get_admin2_list", lambda iso3: admin2_list)
    write_db(_line_list(), project / "cases")
    steps.cases_per_month(project="proj", projects_root=project.parent)

    out = read_db(project / "cases_per_month").df
    assert out["Date"].dtype == "period[M]"
    assert set(out["GID_2"].unique()) == set(admin2_list["GID_2"])  # padded
    assert out.groupby("GID_2", observed=False)["Cases"].sum().sum() == pytest.approx(
        _line_list()["Cases"].sum()
    )


def test_cases_per_month_step_honors_cases_col(project, admin2_list, monkeypatch):
    monkeypatch.setattr("thucia.core.geo.get_admin2_list", lambda iso3: admin2_list)
    raw = _line_list(cases_col="Reported")
    write_db(raw, project / "cases")
    steps.cases_per_month(
        project="proj",
        projects_root=project.parent,
        cases_col="Reported",
    )
    out = read_db(project / "cases_per_month").df
    assert "Reported" in out.columns  # custom column name preserved
    assert out["Reported"].sum() == pytest.approx(raw["Reported"].sum())


def test_cases_per_week_step(project, admin2_list, monkeypatch):
    monkeypatch.setattr("thucia.core.geo.get_admin2_list", lambda iso3: admin2_list)
    write_db(_line_list(), project / "cases")
    steps.cases_per_week(project="proj", projects_root=project.parent)
    out = read_db(project / "cases_per_week").df
    assert "period[W" in str(out["Date"].dtype)
    assert len(out) > 0


def test_cases_per_day_step(project, admin2_list, monkeypatch):
    monkeypatch.setattr("thucia.core.geo.get_admin2_list", lambda iso3: admin2_list)
    write_db(_line_list(), project / "cases")
    steps.cases_per_day(project="proj", projects_root=project.parent)
    out = read_db(project / "cases_per_day").df
    assert "period[D]" in str(out["Date"].dtype)
    assert len(out) > 0


def test_plot_cases_per_month_runs(project, admin2_list, monkeypatch):
    monkeypatch.setattr("thucia.core.geo.get_admin2_list", lambda iso3: admin2_list)
    write_db(_line_list(), project / "cases")
    steps.cases_per_month(project="proj", projects_root=project.parent)
    steps.plot_cases_per_month(project="proj", projects_root=project.parent)
    # no exception means the plot path executed; figures are in memory
    import matplotlib.pyplot as plt

    assert len(plt.get_fignums()) > 0


def test_plot_cases_per_week_runs(project, admin2_list, monkeypatch):
    monkeypatch.setattr("thucia.core.geo.get_admin2_list", lambda iso3: admin2_list)
    write_db(_line_list(), project / "cases")
    steps.cases_per_week(project="proj", projects_root=project.parent)
    steps.plot_cases_per_week(project="proj", projects_root=project.parent)
    import matplotlib.pyplot as plt

    assert len(plt.get_fignums()) > 0


def test_plot_cases_per_day_runs(project, admin2_list, monkeypatch):
    monkeypatch.setattr("thucia.core.geo.get_admin2_list", lambda iso3: admin2_list)
    write_db(_line_list(), project / "cases")
    steps.cases_per_day(project="proj", projects_root=project.parent)
    steps.plot_cases_per_day(project="proj", projects_root=project.parent)
    import matplotlib.pyplot as plt

    assert len(plt.get_fignums()) > 0


def test_main_dispatches_plot_cases_per_month(monkeypatch):
    calls = {}

    def fake(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr("thucia.cli.steps.plot_cases_per_month", fake)
    main(["plot-cases-per-month", "--project", "p", "--title", "T"])
    assert calls["project"] == "p"
    assert calls["title"] == "T"


def test_main_dispatches_cases_per_week(monkeypatch):
    calls = {}

    def fake(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr("thucia.cli.steps.cases_per_week", fake)
    main(["cases-per-week", "--project", "p"])
    assert calls["project"] == "p"


def test_main_dispatches_plot_cases_per_week(monkeypatch):
    calls = {}

    def fake(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr("thucia.cli.steps.plot_cases_per_week", fake)
    main(["plot-cases-per-week", "--project", "p"])
    assert calls["project"] == "p"


def test_main_dispatches_cases_per_day(monkeypatch):
    calls = {}

    def fake(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr("thucia.cli.steps.cases_per_day", fake)
    main(["cases-per-day", "--project", "p"])
    assert calls["project"] == "p"


def test_main_dispatches_plot_cases_per_day(monkeypatch):
    calls = {}

    def fake(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr("thucia.cli.steps.plot_cases_per_day", fake)
    main(["plot-cases-per-day", "--project", "p"])
    assert calls["project"] == "p"


# ---------------------------------------------------------------------------
# thucia provenance -- inspect/verify the run metadata in a saved artefact
# ---------------------------------------------------------------------------


def _artefact(tmp_path, metadata=None):
    path = tmp_path / "run.duckdb"
    frame = pd.DataFrame({"Date": [1, 2], "Cases": [1.0, 2.0]})
    write_db(frame, path, metadata=metadata)
    return path


def test_main_dispatches_provenance(monkeypatch):
    calls = {}

    def fake(file, **kwargs):
        calls["file"] = file
        calls.update(kwargs)
        return 0

    monkeypatch.setattr("thucia.cli.steps.provenance", fake)
    code = main(["provenance", "run.duckdb", "--verify", "--include-deps"])

    assert code == 0
    assert calls["file"] == "run.duckdb"
    assert calls["verify"] is True
    assert calls["include_deps"] is True
    assert calls["as_json"] is False


def test_provenance_prints_all_documents(tmp_path, capsys):
    path = _artefact(tmp_path, {"selection": {"name": "baseline"}, "config": {"n": 1}})

    assert steps.provenance(str(path)) == 0
    out = capsys.readouterr().out
    assert '"selection"' in out and '"config"' in out
    assert "\n  " in out  # indented by default


def test_provenance_selects_one_document(tmp_path, capsys):
    path = _artefact(tmp_path, {"selection": {"name": "baseline"}, "config": {"n": 1}})

    assert steps.provenance(str(path), document="selection") == 0
    assert json.loads(capsys.readouterr().out) == {"name": "baseline"}


def test_provenance_unknown_document(tmp_path, capsys):
    path = _artefact(tmp_path, {"selection": {"name": "baseline"}})

    assert steps.provenance(str(path), document="nope") == 2
    assert "available: selection" in capsys.readouterr().err


def test_provenance_compact_json(tmp_path, capsys):
    path = _artefact(tmp_path, {"selection": {"name": "baseline", "label": None}})

    assert steps.provenance(str(path), document="selection", as_json=True) == 0
    out = capsys.readouterr().out.strip()
    assert "\n" not in out


def test_provenance_without_metadata(tmp_path, capsys):
    path = _artefact(tmp_path)

    assert steps.provenance(str(path)) == 1
    assert "No run metadata recorded" in capsys.readouterr().err


def test_provenance_verify_matches_the_running_implementation(tmp_path, capsys):
    from thucia.core.pipeline.provenance import code_document

    path = _artefact(tmp_path, {"code": code_document()})

    assert steps.provenance(str(path), verify=True) == 0
    assert "matches the running implementation" in capsys.readouterr().out


def test_provenance_verify_reports_a_mismatch(tmp_path, capsys):
    from thucia.core.pipeline.provenance import code_document

    recorded = code_document()
    recorded["digest"] = "sha256:deadbeef"
    path = _artefact(tmp_path, {"code": recorded})

    assert steps.provenance(str(path), verify=True) == 1
    err = capsys.readouterr().err
    assert "digest" in err and "sha256:deadbeef" in err


def test_provenance_verify_without_a_code_document(tmp_path, capsys):
    path = _artefact(tmp_path, {"selection": {"name": "baseline"}})

    assert steps.provenance(str(path), verify=True) == 1
    assert "No 'code' document" in capsys.readouterr().err
