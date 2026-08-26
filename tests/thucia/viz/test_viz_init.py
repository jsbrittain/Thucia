# Probing tests for thucia.viz.__init__ plotting helpers.
# matplotlib Agg backend; GADM admin list is mocked (no network).
import matplotlib
import pytest

matplotlib.use("Agg")

import pandas as pd

import thucia.viz as viz


@pytest.fixture
def admin2_list():
    return pd.DataFrame(
        {
            "GID_1": ["X.1_1"] * 2,
            "GID_2": ["X.1.1_2", "X.1.2_2"],
            "NAME_2": ["One", "Two"],
        }
    )


def _baseline(measure="quantiles"):
    rows = []
    for gid_2 in ["X.1.1_2", "X.1.2_2"]:
        for q in [0.05, 0.50, 0.95]:
            rows.append(
                {
                    "Date": pd.Timestamp("2020-01-01"),
                    "GID_1": "X.1_1",
                    "GID_2": gid_2,
                    "quantile": q,
                    "prediction": 10.0,
                    "Cases": 5.0,
                }
            )
    return pd.DataFrame(rows)


def test_plot_all_admin2_quantiles(monkeypatch, admin2_list):
    monkeypatch.setattr("thucia.viz.get_admin2_list", lambda iso3: admin2_list)
    # no exception -> happy path rendered on the Agg backend
    viz.plot_all_admin2(_baseline())


def test_plot_all_admin2_multiple_iso3_raises(monkeypatch):
    df = _baseline()
    df["GID_2"] = ["Y.1.1_2"] * 4 + ["Z.1.1_2"] * 2
    with pytest.raises(ValueError, match="single ISO3"):
        viz.plot_all_admin2(df)


def test_plot_all_admin2_samples_not_implemented(monkeypatch, admin2_list):
    monkeypatch.setattr("thucia.viz.get_admin2_list", lambda iso3: admin2_list)
    with pytest.raises(NotImplementedError, match="Samples measure"):
        viz.plot_all_admin2(_baseline(), measure="samples")


def test_plot_all_admin2_unknown_measure_raises(monkeypatch, admin2_list):
    monkeypatch.setattr("thucia.viz.get_admin2_list", lambda iso3: admin2_list)
    with pytest.raises(ValueError, match="Unknown measure"):
        viz.plot_all_admin2(_baseline(), measure="bogus")


def test_plot_all_admin2_gid1_filter(monkeypatch, admin2_list):
    monkeypatch.setattr("thucia.viz.get_admin2_list", lambda iso3: admin2_list)
    viz.plot_all_admin2(_baseline(), gid_1=["X.1_1"])


def test_plot_all_admin2_transform_applied(monkeypatch, admin2_list):
    monkeypatch.setattr("thucia.viz.get_admin2_list", lambda iso3: admin2_list)
    viz.plot_all_admin2(_baseline(), transform=lambda x: x * 2)


def test_plot_ensemble_weights_missing_date_col_raises():
    with pytest.raises(ValueError, match="must be a column"):
        viz.plot_ensemble_weights_over_time(pd.DataFrame({"model_a": [0.5, 0.5]}))


def test_plot_ensemble_weights_ma_window(monkeypatch):
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "model_a": [0.4, 0.6, 0.5],
            "model_b": [0.6, 0.4, 0.5],
        }
    )
    viz.plot_ensemble_weights_over_time(df, ma_window=2)


def test_plot_ensemble_weights_basic(monkeypatch):
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "model_a": [0.5, 0.5],
        }
    )
    viz.plot_ensemble_weights_over_time(df)
