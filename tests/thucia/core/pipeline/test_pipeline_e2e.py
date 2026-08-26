# End-to-end pipeline test: raw line-list cases -> fitted, scored forecasts.
# The full stage chain runs on small synthetic data with GADM and covariate
# sources mocked (no network).
import numpy as np
import pandas as pd
import pytest
from thucia.core.cases import read_db
from thucia.core.cases import write_db
from thucia.core.pipeline import cases_per_period
from thucia.core.pipeline import fit_model
from thucia.core.pipeline import merge_covariates
from thucia.core.pipeline import PipelineConfig
from thucia.core.pipeline import prepare_model_inputs
from thucia.core.pipeline import score_model
from thucia.core.quantiles import quantiles


def _raw_cases(n_months=40, seed=0):
    # line-list style input: string dates, integer per-case counts
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2017-01-31", periods=n_months, freq="ME").strftime(
        "%Y-%m-%d"
    )
    rows = []
    for g, base in [("G.1.1_2", 50.0), ("G.1.2_2", 20.0)]:
        for i, d in enumerate(dates):
            cases = int(
                max(
                    base * (1 + 0.5 * np.sin(2 * np.pi * i / 12)) + rng.uniform(-2, 2),
                    0.0,
                )
            )
            rows.append({"Date": d, "GID_1": "G.1_1", "GID_2": g, "Cases": cases})
    return pd.DataFrame(rows)


class FakeCovariateSource:
    name = "fake"
    ref = "fake"

    def merge(self, df, metrics=None, measures=None, use_cache=False):
        out = df.copy()
        out["tmin"] = 20.0
        out["tmax"] = 28.0
        out["prec"] = 100.0
        out["pop_count"] = 10000.0
        return out


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


def test_pipeline_end_to_end(tmp_path, monkeypatch, admin2_list):
    from thucia.core.registry import Registry

    fake_registry = Registry("covariate source")
    fake_registry.register()(FakeCovariateSource)
    monkeypatch.setattr("thucia.core.geo.source_registry", fake_registry)
    monkeypatch.setattr("thucia.core.geo.get_admin2_list", lambda iso3: admin2_list)

    raw = _raw_cases()
    cfg = PipelineConfig(
        path=tmp_path,
        start_date=pd.Period("2019-01", freq="M"),
        train_end_date=pd.Period("2018-12", freq="M"),
        horizons=[1],
        num_samples=50,
        source_specs=["fake.metric"],
        lag_spec=[
            {
                "name": "tmin_lag_1",
                "groupby": ["GID_2"],
                "column": "tmin",
                "pipeline": [{"op": "shift", "periods": 1}],
            },
            {
                "name": "prec_lag_1",
                "groupby": ["GID_2"],
                "column": "prec",
                "pipeline": [{"op": "shift", "periods": 1}],
            },
            {
                "name": "log_cases_lag_1",
                "groupby": ["GID_2"],
                "column": "Log_Cases",
                "pipeline": [{"op": "shift", "periods": 1}],
            },
        ],
    )

    # 1. aggregate to monthly + pad admin-2 + add future rows
    padded = cases_per_period(raw, cfg, freq="M")
    assert set(padded["GID_2"].unique()) == set(admin2_list["GID_2"])
    assert padded["future"].sum() == len(admin2_list) * cfg.future_months
    assert padded[~padded["future"]]["Cases"].sum() == pytest.approx(raw["Cases"].sum())

    # 2. merge covariate source + incidence rate
    merged = merge_covariates(padded, cfg)
    assert {"tmin", "tmax", "prec", "pop_count", "DIR"} <= set(merged.columns)

    # 3. prepare model inputs (lag features, sanitised) and persist/read back
    inputs, cov_cols = prepare_model_inputs(merged, cfg)
    assert {"Log_Cases", "tmin_lag_1", "prec_lag_1", "log_cases_lag_1"} <= set(
        inputs.columns
    )
    assert inputs[cov_cols].isna().sum().sum() == 0
    write_db(inputs, tmp_path / "model_input_data")
    restored = read_db(tmp_path / "model_input_data").df
    assert len(restored) == len(inputs)
    assert restored["Date"].dtype == "period[M]"

    # 4. fit a model (fast baseline) -> canonical quantile grid
    out = fit_model(inputs, "baseline", cfg, db_file=None)
    frame = out.df if hasattr(out, "df") else out
    assert sorted(frame["quantile"].unique()) == quantiles
    valid = frame["prediction"][frame["prediction"].notna()]
    assert (valid >= 0).all()

    # 5. score the forecasts (WIS is NaN for future/unobserved rows and the
    #    baseline warmup; assert the observed window scores finite)
    scored = score_model(frame, cfg)
    assert {"WIS", "R2", "horizon"} <= set(scored.columns)
    observed = scored["Cases"].notna()
    finite_wis = np.isfinite(scored.loc[observed, "WIS"])
    assert finite_wis.any() and finite_wis.sum() > 20
    finite_r2 = np.isfinite(scored.loc[observed, "R2"])
    assert finite_r2.any()
