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


def _raw_cases(n_periods=40, seed=0, freq="M", start="2017-01-31", season=12):
    # line-list style input: string dates, integer per-case counts
    rng = np.random.default_rng(seed)
    ts_freq = "ME" if freq == "M" else freq
    dates = pd.date_range(start, periods=n_periods, freq=ts_freq).strftime("%Y-%m-%d")
    rows = []
    for g, base in [("G.1.1_2", 50.0), ("G.1.2_2", 20.0)]:
        for i, d in enumerate(dates):
            cases = int(
                max(
                    base * (1 + 0.5 * np.sin(2 * np.pi * i / season))
                    + rng.uniform(-2, 2),
                    0.0,
                )
            )
            rows.append({"Date": d, "GID_1": "G.1_1", "GID_2": g, "Cases": cases})
    return pd.DataFrame(rows)


class FakeCovariateSource:
    name = "fake"
    ref = "fake"
    granularity = "M"

    def merge(self, df, metrics=None, measures=None, use_cache=False):
        # One covariate value per (GID_2, month), placed on the last period of
        # each month; other rows are NaN so the geo layer interpolates onto
        # finer (weekly/daily) grids.
        out = df.copy()
        end = pd.PeriodIndex(out["Date"]).to_timestamp(how="end")
        month = end.to_period("M")
        last_of_month = (
            out.groupby(["GID_2", month], observed=False)["Date"].transform("max")
            == out["Date"]
        )
        for metric, base in [
            ("tmin", 20.0),
            ("tmax", 28.0),
            ("prec", 100.0),
            ("pop_count", 10000.0),
        ]:
            out[metric] = np.nan
            out.loc[last_of_month, metric] = base
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
    assert padded["future"].sum() == len(admin2_list) * cfg.future_periods
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


def test_pipeline_end_to_end_weekly(tmp_path, monkeypatch, admin2_list):
    # The same full chain on a weekly (W-SAT) cadence: the Period anchor must
    # survive aggregation -> covariates -> inputs -> fit -> score.
    from thucia.core.registry import Registry

    fake_registry = Registry("covariate source")
    fake_registry.register()(FakeCovariateSource)
    monkeypatch.setattr("thucia.core.geo.source_registry", fake_registry)
    monkeypatch.setattr("thucia.core.geo.get_admin2_list", lambda iso3: admin2_list)

    raw = _raw_cases(n_periods=120, freq="W-SAT", start="2018-01-06", season=52)
    cfg = PipelineConfig(
        path=tmp_path,
        start_date=pd.Period("2019-12-07", freq="W-SAT"),
        train_end_date=pd.Period("2019-11-30", freq="W-SAT"),
        horizons=[1],
        num_samples=20,
        source_specs=["fake.metric"],
        lag_spec=[
            {
                "name": "log_cases_lag_1",
                "groupby": ["GID_2"],
                "column": "Log_Cases",
                "pipeline": [{"op": "shift", "periods": 1}],
            }
        ],
    )

    padded = cases_per_period(raw, cfg, freq="W-SAT")
    assert str(padded["Date"].dtype) == "period[W-SAT]"
    assert padded["future"].sum() == len(admin2_list) * cfg.future_periods

    # The fake source is month-granular: on a weekly grid the geo layer must
    # interpolate it onto every week and warn the user.
    with pytest.warns(UserWarning, match="interpolated"):
        merged = merge_covariates(padded, cfg)
    for col in ["tmin", "tmax", "prec", "pop_count"]:
        assert merged[col].notna().all()

    inputs, cov_cols = prepare_model_inputs(merged, cfg)
    assert inputs[cov_cols].isna().sum().sum() == 0

    frame = fit_model(inputs, "baseline", cfg, db_file=None)
    frame = frame.df if hasattr(frame, "df") else frame
    assert str(frame["Date"].dtype) == "period[W-SAT]"
    assert sorted(frame["quantile"].unique()) == quantiles
    valid = frame["prediction"][frame["prediction"].notna()]
    assert (valid >= 0).all()

    scored = score_model(frame, cfg)
    observed = scored["Cases"].notna()
    assert np.isfinite(scored.loc[observed, "WIS"]).any()
