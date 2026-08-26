from __future__ import annotations

from typing import Optional

import pandas as pd
from thucia.core.registry import Registry

#: Registry of covariate-source plugins, keyed by their ``ref`` string.
#: Sources self-register via ``@source_registry.register()`` at import time.
source_registry: Registry = Registry("covariate source")


class SourceBase:
    """Base class for covariate sources (e.g. WorldClim, EDO, NOAA, WorldPop).

    Subclasses set ``ref`` (the registry key) and ``name``, and implement
    ``merge(df, metrics, measures, use_cache)``.
    """

    ref: str | None = None
    name: str = "Base"

    def merge(
        self,
        df: pd.DataFrame,
        metrics: Optional[list[str]] = None,
        measures: Optional[list[str]] = None,
        use_cache: bool = False,
    ) -> pd.DataFrame:
        raise NotImplementedError("Source plugins must implement merge()")
