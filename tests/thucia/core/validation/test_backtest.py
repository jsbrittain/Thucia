# Probing tests for thucia.core.validation backtesting: the cutoff sweep, the
# per-window fit/score loop, and the WIS skill aggregation must all behave
# correctly (not just run).
import numpy as np
import pandas as pd
import pytest
from thucia.core.pipeline import PipelineConfig
from thucia.core.quantiles import quantiles
from thucia.core.validation import BacktestConfig
from thucia.core.validation import expand_cutoffs
from thucia.core.validation import run_backtest


@pytest.fixture
def inputs():
    rng = np.random.default_rng(0)
    dates = pd.period_range("2016-01", periods=48, freq="M")
    rows = []
    for g, base in [("G.1.1_2", 50.0), ("G.1.2_2", 20.0)]:
        for i, d in enumerate(dates):
            cases = max(
                base * (1 + 0.5 * np.sin(2 * np.pi * i / 12)) + rng.uniform(-2, 2),
                0.0,
            )
            rows.append(
                {
                    "Date": d,
                    "GID_1": "G.1_1",
                    "GID_2": g,
                    "future": False,
                    "Cases": cases,
                }
            )
    df = pd.DataFrame(rows)
    df["GID_2"] = df["GID_2"].astype("category")
    df["Log_Cases"] = np.log1p(df["Cases"])
    df["tmin"] = 20.0
    return df


def _cfg(horizons=(1,), start="2016-01"):
    return PipelineConfig(
        horizons=list(horizons), start_date=pd.Period(start, freq="M")
    )


def _naive_walk_fit(slice_df, model_name, cfg, db_file=None):
    """Naive random walk: median = last level, intervals widen with horizon."""
    cutoff = cfg.train_end_date
    max_h = max(cfg.horizons)
    hist = slice_df[slice_df["Date"] <= cutoff]
    target = slice_df[
        (slice_df["Date"] > cutoff) & (slice_df["Date"] <= cutoff + max_h)
    ]
    rows = []
    for (gid2, d), g in target.groupby(["GID_2", "Date"], observed=False):
        h = (d - cutoff).n  # .n: number of periods (Period diff is a DateOffset)
        if h not in cfg.horizons:
            continue
        last = hist[hist["GID_2"] == gid2]["Cases"].iloc[-1]
        spread = 3.0 + 3.0 * h  # intervals widen strongly with horizon
        for q in quantiles:
            # case forecasts cannot go negative (models clip to >= 0)
            pred = max(last + spread * (q - 0.5), 0.0)
            rows.append(
                {
                    "Date": d,
                    "GID_2": gid2,
                    "quantile": q,
                    "prediction": pred,
                    "Cases": g["Cases"].iloc[0],
                    "horizon": h,
                }
            )
    return pd.DataFrame(rows)


def _cheating_fit(slice_df, model_name, cfg, db_file=None):
    """Perfect predictor: every quantile equals the observed Cases."""
    cutoff = cfg.train_end_date
    max_h = max(cfg.horizons)
    target = slice_df[
        (slice_df["Date"] > cutoff) & (slice_df["Date"] <= cutoff + max_h)
    ]
    rows = []
    for (gid2, d), g in target.groupby(["GID_2", "Date"], observed=False):
        h = (d - cutoff).n
        if h not in cfg.horizons:
            continue
        for q in quantiles:
            rows.append(
                {
                    "Date": d,
                    "GID_2": gid2,
                    "quantile": q,
                    "prediction": g["Cases"].iloc[0],
                    "Cases": g["Cases"].iloc[0],
                    "horizon": h,
                }
            )
    return pd.DataFrame(rows)


# --- expand_cutoffs ---


def test_expand_cutoffs_respects_min_history_and_max_horizon():
    dates = pd.period_range("2020-01", periods=40, freq="M")
    cutoffs = expand_cutoffs(dates, min_history=12, step=1, max_horizon=6)
    # first cutoff: 12 periods up to & including it (index 11)
    assert cutoffs[0] == dates[11]
    # last cutoff: leaves >= 6 periods after it
    assert cutoffs[-1] <= dates[33]
    assert all(dates[i] in cutoffs for i in range(11, 34, 1))


def test_expand_cutoffs_step_spacing():
    dates = pd.period_range("2020-01", periods=40, freq="M")
    cutoffs = expand_cutoffs(dates, min_history=10, step=5, max_horizon=2)
    assert [(c - dates[0]).n for c in cutoffs] == [9, 14, 19, 24, 29, 34]


def test_expand_cutoffs_empty_when_too_short():
    dates = pd.period_range("2020-01", periods=12, freq="M")
    assert expand_cutoffs(dates, min_history=20, max_horizon=1) == []
    assert expand_cutoffs(dates, min_history=6, max_horizon=10) == []


# --- run_backtest: real baseline ---


def test_run_backtest_schema_and_finite_scores(inputs):
    res = run_backtest(
        inputs,
        _cfg(horizons=(1,)),
        BacktestConfig(model_name="baseline", min_history=24, step=3),
    )
    assert set(res.scores.columns) >= {
        "cutoff",
        "GID_2",
        "Date",
        "horizon",
        "WIS",
        "RMSE",
        "R2",
    }
    assert (res.scores["WIS"] > 0).all()
    assert np.isfinite(res.scores["WIS"]).all()
    assert np.isfinite(res.scores["RMSE"]).all()
    # summary is per-horizon with a window count
    assert set(res.summary.columns) >= {"horizon", "WIS", "RMSE", "R2", "n"}
    assert (res.summary["n"] > 0).all()
    assert "skill" not in res.summary.columns  # no reference requested


def test_run_backtest_explicit_cutoffs(inputs):
    bt = BacktestConfig(
        model_name="baseline",
        cutoffs=[pd.Period("2018-01", freq="M"), pd.Period("2018-06", freq="M")],
    )
    res = run_backtest(inputs, _cfg(horizons=(1,)), bt)
    assert set(res.scores["cutoff"].unique()) == {
        pd.Period("2018-01", freq="M"),
        pd.Period("2018-06", freq="M"),
    }


def test_run_backtest_keep_forecasts(inputs):
    bt = BacktestConfig(
        model_name="baseline", min_history=24, step=6, keep_forecasts=True
    )
    res = run_backtest(inputs, _cfg(horizons=(1,)), bt)
    assert res.forecasts is not None
    for cutoff, frame in res.forecasts.items():
        assert (frame["Date"] > cutoff).all()  # held-out window only


def test_run_backtest_rolling_window_runs(inputs):
    expanding = run_backtest(
        inputs,
        _cfg(horizons=(1,)),
        BacktestConfig(model_name="baseline", min_history=24, step=4),
    )
    rolling = run_backtest(
        inputs,
        _cfg(horizons=(1,)),
        BacktestConfig(model_name="baseline", min_history=24, step=4, window=4),
    )
    assert not rolling.scores.empty
    assert not expanding.scores.empty


def test_run_backtest_does_not_litter_caller_directory(inputs, tmp_path):
    # Model fits persist quantile duckdb files internally; the backtest must
    # keep them in a scratch dir and never touch the caller's filesystem.
    cfg = PipelineConfig(
        path=tmp_path, horizons=[1], start_date=pd.Period("2016-01", freq="M")
    )
    run_backtest(
        inputs, cfg, BacktestConfig(model_name="baseline", min_history=24, step=6)
    )
    assert list(tmp_path.glob("*_cases_quantiles.duckdb")) == []


# --- run_backtest: guards ---


def test_fast_only_blocks_heavy_model(inputs):
    with pytest.raises(ValueError, match="not a fast backtest model"):
        run_backtest(inputs, _cfg(), BacktestConfig(model_name="sarima"))


def test_fast_only_blocks_heavy_reference(inputs):
    with pytest.raises(ValueError, match="reference model"):
        run_backtest(
            inputs,
            _cfg(horizons=(1,)),
            BacktestConfig(model_name="baseline", reference_model="sarima"),
        )


def test_fast_only_disabled_allows_custom_fit(inputs):
    res = run_backtest(
        inputs,
        _cfg(horizons=(1, 6, 12)),
        BacktestConfig(model_name="naive", min_history=24, step=6, fast_only=False),
        fit_fn=_naive_walk_fit,
    )
    assert set(res.summary["horizon"]) == {1, 6, 12}


# --- run_backtest: per-horizon scoring structure ---


def test_naive_backtest_wis_grows_with_horizon():
    # On a *flat* level (no seasonality) the naive random walk's median is a
    # good point forecast at every horizon; the only thing that changes with
    # horizon is the interval width, so WIS must grow monotonically. (RMSE of
    # the flat median is noise-driven and not expected to be monotonic.)
    rng = np.random.default_rng(1)
    dates = pd.period_range("2016-01", periods=48, freq="M")
    rows = []
    for g, base in [("G.1.1_2", 50.0), ("G.1.2_2", 20.0)]:
        for d in dates:
            rows.append(
                {
                    "Date": d,
                    "GID_1": "G.1_1",
                    "GID_2": g,
                    "future": False,
                    "Cases": max(base + rng.uniform(-0.5, 0.5), 0.0),
                }
            )
    flat = pd.DataFrame(rows)
    flat["GID_2"] = flat["GID_2"].astype("category")
    flat["Log_Cases"] = np.log1p(flat["Cases"])
    flat["tmin"] = 20.0

    res = run_backtest(
        flat,
        _cfg(horizons=(1, 6, 12)),
        BacktestConfig(model_name="naive", min_history=24, step=6, fast_only=False),
        fit_fn=_naive_walk_fit,
    )
    s = res.summary.set_index("horizon")["WIS"]
    assert s[1] < s[6] < s[12]
    assert np.isfinite(res.summary["RMSE"]).all()


def test_cheating_model_has_zero_wis_and_positive_skill(inputs):
    res = run_backtest(
        inputs,
        _cfg(horizons=(1, 6)),
        BacktestConfig(
            model_name="cheat",
            reference_model="baseline",
            min_history=24,
            step=6,
            fast_only=False,
        ),
        fit_fn=_cheating_fit,
    )
    # perfect predictor -> WIS ~ 0 on the raw scoring scale
    assert res.scores["WIS"].max() < 1e-6
    assert "skill" in res.summary.columns
    assert (res.summary["skill"].dropna() > 0.5).all()


def test_skill_zero_when_reference_equals_model(inputs):
    res = run_backtest(
        inputs,
        _cfg(horizons=(1,)),
        BacktestConfig(
            model_name="baseline", reference_model="baseline", min_history=24, step=6
        ),
    )
    assert (res.summary["skill"].abs() < 1e-6).all()


@pytest.mark.slow
def test_sarima_backtest_slow(inputs):
    # A real non-trivial model across a couple of cutoffs: exercises the
    # fit_model path (db_file=None, in-memory) rather than a custom fit_fn.
    bt = BacktestConfig(
        model_name="sarima",
        min_history=36,
        step=6,
        fast_only=False,
    )
    res = run_backtest(inputs, _cfg(horizons=(1,)), bt)
    assert not res.scores.empty
    assert np.isfinite(res.scores["WIS"]).all()
