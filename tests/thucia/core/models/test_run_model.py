import numpy as np
import pandas as pd
from thucia.core.models import run_model


def _input_df(n=4):
    return pd.DataFrame(
        {
            "Date": pd.period_range("2020-01", periods=n, freq="M").repeat(2),
            "GID_2": ["A", "B"] * n,
            "future": [False] * (n * 2),
            "Cases": np.arange(n * 2).astype(float),
            "Log_Cases": np.log1p(np.arange(n * 2).astype(float)),
        }
    )


def _quantile_model(df, *a, **k):
    rows = []
    for g in df["GID_2"].unique():
        for _, r in df[df["GID_2"] == g].iterrows():
            for q in [0.1, 0.5, 0.9]:
                rows.append(
                    {
                        "Date": r["Date"],
                        "GID_2": g,
                        "Cases": r["Cases"],
                        "quantile": q,
                        "prediction": r["Cases"] * q,
                        "horizon": 1,
                    }
                )
    return pd.DataFrame(rows)


def test_run_model_writes_quantile_db(tmp_path):
    df = _input_df()
    run_model("probe_q", _quantile_model, df, tmp_path, save_quantiles=True)
    files = {p.name for p in tmp_path.iterdir()}
    assert "probe_q_cases_quantiles.duckdb" in files


def test_run_model_output_name_uses_name_arg(tmp_path):
    # The output file is named from the `name` argument, not the model fn name
    df = _input_df()
    run_model("custom_label", _quantile_model, df, tmp_path, save_quantiles=True)
    files = {p.name for p in tmp_path.iterdir()}
    assert "custom_label_cases_quantiles.duckdb" in files
    assert "_quantile_model_cases_quantiles.duckdb" not in files


def test_run_model_exponentiates_when_only_log_cases(tmp_path):
    def log_only(df, *a, **k):
        rows = []
        for g in df["GID_2"].unique():
            for _, r in df[df["GID_2"] == g].iterrows():
                for q in [0.1, 0.5, 0.9]:
                    rows.append(
                        {
                            "Date": r["Date"],
                            "GID_2": g,
                            "Log_Cases": r["Log_Cases"],
                            "quantile": q,
                            "prediction": r["Log_Cases"] * q,
                            "horizon": 1,
                        }
                    )
        return pd.DataFrame(rows)

    df = _input_df()
    out = run_model("probe_log", log_only, df, tmp_path, save_quantiles=True)
    assert "Cases" in out.columns
    assert "prediction" in out.columns
    # predictions reconstructed on count scale via expm1
    assert out["prediction"].min() >= 0


def test_run_model_missing_both_quantiles_and_samples(tmp_path):
    def nothing(df, *a, **k):
        rows = []
        for g in df["GID_2"].unique():
            for _, r in df[df["GID_2"] == g].iterrows():
                rows.append(
                    {"Date": r["Date"], "GID_2": g, "Cases": r["Cases"], "horizon": 1}
                )
        return pd.DataFrame(rows)

    df = _input_df()
    run_model("probe_none", nothing, df, tmp_path, save_quantiles=True)
    # no quantile/sample output -> nothing saved, returns model output as-is
    assert set(tmp_path.iterdir()) == set()
