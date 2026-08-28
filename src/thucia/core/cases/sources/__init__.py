# Case-data source plugins.
#
# Case ingestion previously lived as ad-hoc, per-country scripts under
# `data/cases/<ISO3>/` with no shared abstraction. This subpackage introduces a
# small plugin registry (mirroring the covariate-source idiom in
# `thucia.core.geo`): a `CaseSource` base class, a shared `case_registry`, and a
# `load_sources()` that imports each driver module so it self-registers. Drivers
# are disease-generic -- the disease is a parameter passed at fetch time, never
# hard-coded in the driver name or body.
from __future__ import annotations

from typing import Optional

import pandas as pd
from thucia.core.registry import PluginNotFoundError
from thucia.core.registry import Registry

#: Registry of case-source plugins, keyed by their ``ref`` string. Drivers
#: self-register via ``@case_registry.register()`` at import time.
case_registry: Registry = Registry("case source")

#: Package (and module stem) that holds the case-source driver modules.
plugin_dir = __path__[0]
module_stem = __name__


class CaseSource:
    """Base class for case-data sources (e.g. an arboviral surveillance API).

    Subclasses set ``ref`` (the registry key) and ``name``, self-register via
    ``@case_registry.register()``, and implement ``fetch(**params)`` returning a
    pandas DataFrame of case data. Drivers must be disease-generic: the disease
    is a ``**params`` value, never a hard-coded constant or part of the name.
    """

    ref: Optional[str] = None
    name: str = "CaseSource"

    def fetch(self, **params) -> pd.DataFrame:
        raise NotImplementedError("Case-source plugins must implement fetch()")


def load_sources(
    plugin_dir: str = plugin_dir,
    module_stem: str = module_stem,
) -> dict[str, type]:
    """Import every case-source module so drivers self-register.

    A single broken module is logged and skipped rather than killing all
    plugins. Returns the current registry contents (ref -> class).
    """
    import importlib
    import logging
    import os

    for fname in sorted(os.listdir(plugin_dir)):
        if fname.endswith(".py") and not fname.startswith("__"):
            mod_name = fname[:-3]
            try:
                importlib.import_module(f"{module_stem}.{mod_name}")
            except Exception as exc:  # keep other plugins importable
                logging.error(f"Failed to load case source '{mod_name}': {exc}")
    return case_registry.all()


def load_case_source(ref: str, **params) -> CaseSource:
    """Instantiate a case-source driver by its ``ref``.

    Returned instance is configured with `params`; its ``fetch(**params)`` may be
    called with the same keys (later values win) to query one dataset.
    """
    _ensure_plugins_loaded()
    cls = case_registry.get(ref)
    return cls(**params)


def _ensure_plugins_loaded() -> None:
    if not case_registry.names():
        load_sources()


__all__ = [
    "CaseSource",
    "case_registry",
    "load_case_source",
    "load_sources",
    "PluginNotFoundError",
    "plugin_dir",
    "module_stem",
]
