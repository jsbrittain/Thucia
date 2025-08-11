import numpy as np
import pandas as pd
from thucia.core.models.timesfm import timesfm
from thucia.core.models.timesfm import TimeSFMSamples


def test_timesfm():
    N = 8
    df = pd.DataFrame(
        {
            "Date": pd.date_range(start="2023-01-01", periods=N, freq="ME"),
            "GID_1": ["A"] * N,
            "GID_2": ["X"] * N,
            "tmin": [1, 2, 1, 2, 1, 2, 1, 2] * (N // 8),
            "prec": [10, 15, 25, 15, 10, 15, 20, 25] * (N // 8),
            "Cases": list(range(1, N + 1)),
            "future": [False] * (N - 3) + [True] * 3,
        }
    )
    print(df)
    df_pred = timesfm(
        df,
        start_date=df["Date"].min(),
        end_date=df["Date"].max(),
        gid_1=["A"],
    )
    print(df_pred)
    assert False


def test_TimeSFMSamples_nocovar():
    df = pd.DataFrame(
        {
            "Date": pd.date_range(start="2023-01-01", periods=8, freq="ME"),
            "Covar": [1, 1, 1, 2, 2, 2, 3, 3],
            "GID_1": ["A", "A", "A", "A", "A", "A", "A", "A"],
            "GID_2": ["X", "X", "X", "X", "X", "X", "X", "X"],
            "tmin": [1, 2, 1, 2, 1, 2, 1, 2],
            "prec": [10, 15, 25, 15, 10, 15, 20, 25],
            "Cases": [10, 20, 30, 40, 50, 60, 70, 80],
            "future": [False, False, False, False, False, True, True, True],
        }
    )
    timesfm = TimeSFMSamples(
        timeseries_df=df,
        case_col="Cases",
        num_samples=1,
        horizon_len=3,
    )
    df_forecast = timesfm.predict(sigma=0.0)
    print(df_forecast)
    # Predictions should be NaN for (future = False)
    assert df_forecast["prediction"][~df_forecast["future"]].isna().all()
    # Predictions should not be NaN for (future = True)
    assert df_forecast["prediction"][df_forecast["future"]].notna().all()


def test_TimeSFMSamples_covar():
    df = pd.DataFrame(
        {
            "Date": pd.date_range(start="2023-01-01", periods=8, freq="ME"),
            "Covar": [1, 1, 1, 2, 2, 2, 3, 3],
            "GID_1": ["A", "A", "A", "A", "A", "A", "A", "A"],
            "GID_2": ["X", "X", "X", "X", "X", "X", "X", "X"],
            "tmin": [1, 2, 1, 2, 1, 2, 1, 2],
            "prec": [10, 15, 25, 15, 10, 15, 20, 25],
            "Cases": [10, 20, 30, 40, 50, 60, 70, 80],
            "future": [False, False, False, False, False, True, True, True],
        }
    )
    timesfm = TimeSFMSamples(
        timeseries_df=df,
        case_col="Cases",
        num_samples=1,
        horizon_len=3,
        dynamic_numerical_cols=["tmin", "prec"],
        dynamic_categorical_cols=["Covar"],
    )
    df_forecast = timesfm.predict(sigma=0.0)
    print(df_forecast)
    # Predictions should be NaN for [0:-horizon_len]
    assert df_forecast["prediction"][:-3].isna().all()
    # Predictions should not be NaN [-horizon_len:]
    assert df_forecast["prediction"][-3:].notna().all()


def test_TimeSFMSamples_long():
    N = 2000
    df = pd.DataFrame(
        {
            "Date": pd.date_range(start="2023-01-01", periods=N, freq="ME"),
            "Covar": np.random.randint(1, 4, N),
            "GID_1": ["A"] * N,
            "GID_2": ["X"] * N,
            "tmin": np.random.randint(1, 3, N),
            "prec": np.random.randint(10, 30, N),
            "Cases": np.random.randint(10, 100, N),
            "future": [False] * N,
        }
    )
    df["Log_Cases"] = df["Cases"].apply(lambda x: np.log1p(x))
    num_samples = 1000
    timesfm = TimeSFMSamples(
        timeseries_df=df,
        case_col="Log_Cases",
        # horizon_len = 1, by default
        num_samples=num_samples,
    )
    df_forecast = timesfm.predict()
    print(df_forecast)
    # Predictions should single row and NaN for all dates except the last one
    assert (
        df_forecast["prediction"][df_forecast["Date"] < df_forecast["Date"].max()]
        .isna()
        .all()
    )
    # Last date should have 'num_samples' predictions, which should not be NaN
    assert (
        df_forecast["prediction"][df_forecast["Date"] == df_forecast["Date"].max()]
        .notna()
        .all()
    )


def test_timesfm_core_model():
    from timesfm import TimesFm
    from timesfm import TimesFmCheckpoint
    from timesfm import TimesFmHparams

    hparams = TimesFmHparams(
        backend="cpu",
        context_len=32,  # still accepts and produces output for sizes down to n=1
        horizon_len=24,
        # input_patch_len=8,
        # output_patch_len=16,
        num_layers=50,
        # num_heads=4,
        # model_dims=128,
        per_core_batch_size=32,
        # # quantiles=None,
        use_positional_embedding=True,
        # point_forecast_mode="median"
    )
    checkpoint = TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
    )
    model = TimesFm(hparams=hparams, checkpoint=checkpoint)

    x = list(range(200))
    forecast, _ = model.forecast(
        inputs=[x],
        freq=[1],  # use 1 for monthly data
    )
    print(forecast)
    assert forecast.shape == (1, 24)  # Check the shape of the forecast
    # Check that forecasts are not NaN
    assert not np.isnan(forecast).any()
