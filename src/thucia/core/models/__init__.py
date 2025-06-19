from .baseline import baseline as baseline
from .climate import climate as climate
from .sarima import sarima as sarima
from .tcn import tcn as tcn
from .utils import samples_to_quantiles as samples_to_quantiles, quantiles as quantiles

import pandas as pd
from pathlib import Path
from thucia.core.geo import convert_to_incidence_rate
from thucia.core.cases import write_nc


def run_model(name: str, model_task, df: pd.DataFrame, path: Path, *args, **kwargs):
    # Cases
    df_cases_samples = model_task(df, *args, **kwargs)
    write_nc(df_cases_samples, path / f"{name}_cases_samples.nc")
    df_cases_quantiles = samples_to_quantiles(df_cases_samples)
    write_nc(df_cases_quantiles, path / f"{name}_cases_quantiles.nc")
    # Dengue incidence rate
    df_dir_samples = convert_to_incidence_rate(df_cases_samples, df)
    write_nc(df_dir_samples, path / f"{name}_dir_samples.nc")
    df_dir_quantiles = samples_to_quantiles(df_dir_samples)
    write_nc(df_dir_quantiles, path / f"{name}_dir_quantiles.nc")
