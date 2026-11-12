# Probe tests for configured model selection and its recorded provenance.
import numpy as np
import pandas as pd
import pytest
from thucia.core.fs import read_run_metadata
from thucia.core.models._meta import as_selection
from thucia.core.models._meta import ModelSelection
from thucia.core.models._meta import validate_model_params
from thucia.core.pipeline import fit_model
from thucia.core.pipeline import fit_models
from thucia.core.pipeline import PipelineConfig
from thucia.core.validation import _is_fast


def _inputs(n_periods: int = 24) -> pd.DataFrame:
    idx = pd.period_range("2016-01", periods=n_periods, freq="M")
    cases = np.tile(np.arange(float(n_periods)), 2)
    df = pd.DataFrame(
        {
            "Date": idx.repeat(2),
            "GID_1": ["G.1_1"] * 2 * n_periods,
            "GID_2": ["G.1.1_2", "G.1.2_2"] * n_periods,
            "future": [False] * 2 * n_periods,
            "Cases": cases,
            "Log_Cases": np.log1p(cases),
        }
    )
    df["GID_1"] = df["GID_1"].astype("category")
    df["GID_2"] = df["GID_2"].astype("category")
    return df


def _config(tmp_path, **kwargs) -> PipelineConfig:
    base = dict(
        path=tmp_path,
        start_date=pd.Period("2017-06", freq="M"),
        horizons=[1],
        num_samples=50,
    )
    base.update(kwargs)
    return PipelineConfig(**base)


def test_names_are_normalised_to_selections():
    assert as_selection("baseline") == ModelSelection("baseline")
    selection = ModelSelection("baseline", label="b")
    assert as_selection(selection) is selection
    with pytest.raises(TypeError, match="ModelSelection"):
        as_selection(42)


def test_selection_requires_a_name_and_non_empty_label():
    with pytest.raises(ValueError, match="name"):
        ModelSelection("")
    with pytest.raises(ValueError, match="label"):
        ModelSelection("baseline", label="")


def test_string_and_selection_fit_the_same_model(tmp_path):
    df = _inputs()
    by_name = fit_model(df, "baseline", _config(tmp_path), db_file=None)
    by_selection = fit_model(
        df, ModelSelection("baseline"), _config(tmp_path), db_file=None
    )

    a = by_name.df if hasattr(by_name, "df") else by_name
    b = by_selection.df if hasattr(by_selection, "df") else by_selection
    assert list(a.columns) == list(b.columns)
    assert len(a) == len(b)


def test_label_names_artifacts_and_records_provenance(tmp_path):
    df = _inputs()
    fit_model(df, ModelSelection("baseline", label="baseline_v2"), _config(tmp_path))

    saved = tmp_path / "baseline_v2_cases_quantiles.duckdb"
    assert saved.exists()

    docs = read_run_metadata(saved)
    assert set(docs) == {"selection", "config", "code", "environment", "inputs"}
    assert docs["selection"]["name"] == "baseline"
    assert docs["selection"]["label"] == "baseline_v2"
    assert docs["config"]["horizons"] == [1]
    assert docs["code"]["digest"].startswith("sha256:")
    assert docs["inputs"]["geo_units"] == 2
    assert docs["inputs"]["digest"]


def test_overrides_and_params_reach_the_model(tmp_path, monkeypatch):
    import thucia.core.pipeline.stages as stages
    from thucia.core.models._meta import ModelSpec

    captured = {}

    def _fake_model(df, symmetrize=True, **kwargs):
        captured.update(kwargs)
        captured["symmetrize"] = symmetrize
        return pd.DataFrame(
            {
                "Date": df["Date"],
                "GID_2": df["GID_2"],
                "quantile": 0.5,
                "prediction": 0.0,
                "Cases": df["Cases"],
            }
        )

    spec = ModelSpec(
        name="_fake",
        supports=frozenset({"season_length", "retrain"}),
    )
    monkeypatch.setattr(stages, "get_model", lambda name: _fake_model)
    monkeypatch.setattr(stages.models, "get_model_spec", lambda name: spec)

    fit_model(
        _inputs(),
        ModelSelection(
            "_fake",
            overrides={"season_length": 6, "retrain": True},
            params={"symmetrize": False},
        ),
        _config(tmp_path, season_length=12, retrain=False),
        db_file=None,
    )

    assert captured["season_length"] == 6  # override beats the shared config
    assert captured["retrain"] is True
    assert captured["symmetrize"] is False  # params reach the callable


def test_unknown_override_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown config overrides"):
        fit_model(
            _inputs(),
            ModelSelection("baseline", overrides={"not_a_field": 1}),
            _config(tmp_path),
            db_file=None,
        )


def test_override_the_model_cannot_use_warns(tmp_path):
    # `baseline` declares no support for `quantiles`, so applying it is a no-op
    # and must not fail silently.
    with pytest.warns(UserWarning, match="not used by model"):
        fit_model(
            _inputs(),
            ModelSelection("baseline", overrides={"quantiles": [0.1, 0.5, 0.9]}),
            _config(tmp_path),
            db_file=None,
        )


def test_unknown_param_raises_at_validation():
    import thucia.core.models as models

    selection = ModelSelection("baseline", params={"not_a_kwarg": 1})
    model_fn = models.get_model("baseline")
    with pytest.raises(ValueError, match="unknown model params"):
        validate_model_params(selection, model_fn)

    # a parameter the callable declares explicitly is accepted
    validate_model_params(
        ModelSelection("baseline", params={"symmetrize": False}), model_fn
    )


def test_duplicate_labels_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate model labels"):
        fit_models(
            _inputs(),
            ["baseline", ModelSelection("baseline")],
            _config(tmp_path),
        )


def test_fit_models_returns_one_labelled_frame_each(tmp_path):
    frames = fit_models(
        _inputs(),
        [ModelSelection("baseline", label="a"), ModelSelection("movavg", label="b")],
        _config(tmp_path),
    )

    assert set(frames) == {"a", "b"}
    for frame in frames.values():
        assert "quantile" in frame.columns and len(frame) > 0
    assert (tmp_path / "a_cases_quantiles.duckdb").exists()
    assert (tmp_path / "b_cases_quantiles.duckdb").exists()
    # both persistence paths (model-written and run_model-written) record the run
    for label in ("a", "b"):
        docs = read_run_metadata(tmp_path / f"{label}_cases_quantiles.duckdb")
        assert docs["selection"]["label"] == label


def test_backtest_accepts_a_selection():
    assert _is_fast("baseline") is True
    assert _is_fast(ModelSelection("baseline", overrides={"retrain": True})) is True
    assert _is_fast(ModelSelection("sarima")) is False
