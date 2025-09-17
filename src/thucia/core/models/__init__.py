from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from thucia.core.cases import write_nc
from thucia.core.geo import convert_to_incidence_rate

from .utils import quantiles  # noqa: F401
from .utils import samples_to_quantiles

#
# Discover and lazy import model definitions
#

# discover candidate module -> symbol mappings once
_exports: dict[str, str] = {}
for _m in pkgutil.iter_modules(__path__):
    name = _m.name
    if not name.startswith("_"):
        _exports[name] = name

__all__ = sorted(_exports)  # advertise what the package exports


def __getattr__(name: str) -> Any:  # called on first access if not yet in globals
    mod_name = _exports.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(f".{mod_name}", __name__)
    obj = getattr(mod, name)  # expect symbol name == module name
    globals()[name] = obj  # cache for future lookups
    return obj


def run_model(
    name: str,
    model,
    df: pd.DataFrame,
    path: Path,
    save_samples=False,
    save_quantiles=True,
    *args,
    **kwargs,
):
    # Cases
    tic = pd.Timestamp.now()
    df_model = model(df, *args, **kwargs)
    toc = pd.Timestamp.now()

    if "Cases" not in df_model.columns and "Log_Cases" in df_model.columns:
        df_model["Cases"] = np.expm1(df_model["Log_Cases"]).clip(lower=0)
        df_model["prediction"] = np.expm1(df_model["prediction"]).clip(lower=0)

    logging.info(f"{name} model run time: {toc - tic}")

    # Save samples
    if save_samples:
        if "sample" not in df_model.columns:
            write_nc(df_model, path / f"{name}_cases_samples.nc")
        else:
            logging.warning(
                f"Model {name} did not produce samples, saving quantiles instead."
            )
            save_quantiles = True

    # Check if we need to convert samples to quantiles
    if save_quantiles and "quantile" not in df_model.columns:
        if "sample" in df_model.columns:
            df_model = samples_to_quantiles(df_model)
        else:
            logging.warning(
                f"Model {name} did not produce quantiles or samples, skipping save."
            )
            save_quantiles = False

    # Save quantiles
    if save_quantiles:
        if "quantile" not in df_model.columns:
            write_nc(df_model, path / f"{name}_cases_quantiles.nc")
        else:
            write_nc(df_model, path / f"{name}_cases_quantiles.nc")

    # # Dengue incidence rate
    # df_dir_samples = convert_to_incidence_rate(df_cases_samples, df)
    # if save_samples:
    #     write_nc(df_dir_samples, path / f"{name}_dir_samples.nc")
    # if save_quantiles:
    #     df_dir_quantiles = samples_to_quantiles(df_dir_samples)
    #     write_nc(df_dir_quantiles, path / f"{name}_dir_quantiles.nc")
