import numpy as np
import pandas as pd
import pytest
from thucia.core.models.ensemble import apply_weights_to_forecasts
from thucia.core.models.ensemble import create_ensemble
from thucia.core.models.ensemble import fix_quantile_violations
from thucia.core.models.ensemble import merge_model_dfs
from thucia.core.models.ensemble import train_ensemble


def _model_frame(seed=0, n_dates=8, scale=1.0, q_levels=(0.1, 0.5, 0.9)):
    rng = np.random.default_rng(seed)
    dates = pd.period_range("2020-01", periods=n_dates, freq="M")
    rows = []
    for d in dates:
        for q in q_levels:
            rows.append(
                {
                    "Date": d,
                    "GID_2": "A",
                    "quantile": q,
                    "prediction": scale * float(rng.uniform(0, 10)),
                    "Cases": 5.0,
                }
            )
    return pd.DataFrame(rows)


def test_merge_model_dfs_tags_models():
    m1, m2 = _model_frame(), _model_frame(seed=1)
    merged = merge_model_dfs([m1, m2], ["m1", "m2"])
    assert set(merged["model"].unique()) == {"m1", "m2"}
    assert len(merged) == len(m1) + len(m2)


def test_merge_model_dfs_default_names():
    merged = merge_model_dfs([_model_frame()])
    assert merged["model"].iloc[0] == "model_0"


def test_train_ensemble_weights_sum_to_one():
    m1, m2 = _model_frame(), _model_frame(scale=2.0)
    merged = merge_model_dfs([m1, m2], ["m1", "m2"])
    weights = train_ensemble(merged, [0.1, 0.5, 0.9])
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(0.0 <= w <= 1.0 for w in weights.values())


def test_train_ensemble_degenerate_returns_equal_weights():
    # Inconsistent shapes should fall back to equal weights
    m1 = _model_frame(n_dates=8)
    m2 = _model_frame(n_dates=5)  # different dates -> shape mismatch
    merged = merge_model_dfs([m1, m2], ["m1", "m2"])
    weights = train_ensemble(merged, [0.1, 0.5, 0.9])
    assert sum(weights.values()) == pytest.approx(1.0)


def test_fix_quantile_violations_small_ones_only():
    df = pd.DataFrame(
        {
            "GID_2": ["A"] * 4,
            "Date": [pd.Period("2020-01", "M")] * 4,
            "model": ["x"] * 4,
            "quantile": [0.1, 0.5, 0.9, 0.95],
            "prediction": [5.0, 5.0 + 1e-12, 4.0, 9.0],
        }
    )
    fixed = fix_quantile_violations(df).sort_values("quantile")
    # 0.9 -> 4.0 is a large violation (vs 5.0) and must be left alone
    assert fixed.iloc[2]["prediction"] == pytest.approx(4.0)


def test_apply_weights_to_forecasts_uses_lagged_weights():
    m1, m2 = _model_frame(), _model_frame(scale=0.5)
    merged = merge_model_dfs([m1, m2], ["m1", "m2"])
    weights_df = pd.DataFrame(
        {"Date": sorted(merged["Date"].unique()), "m1": 0.5, "m2": 0.5}
    )
    out = apply_weights_to_forecasts(merged, weights_df, [0.1, 0.5, 0.9])
    assert "model" in out.columns
    assert set(out["model"].unique()) == {"ensemble"}


def test_create_ensemble_returns_weights_and_frame():
    m1, m2 = _model_frame(), _model_frame(seed=2)
    merged = [m1, m2]
    ens, weights = create_ensemble(merged, model_names=["m1", "m2"])
    assert "model" in ens.columns
    assert not weights.empty
    assert {"m1", "m2"} <= set(weights.columns) - {"Date"}
