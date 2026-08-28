# Declarative metadata for forecast models.
#
# Each model module (e.g. `sarima.py`) declares a module-level `SPEC` describing
# which pipeline config knobs it consumes and how the pipeline should treat it.
# `thucia.core.pipeline.fit_model` and `thucia.core.validation.run_backtest`
# read the spec instead of hard-coding model identity/name branches, so adding
# a model with new knobs is a one-line spec, not an edit to the pipeline.
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

#: Config knobs a model may declare it supports (beyond the common kwargs that
#: every model receives). Each maps to a `PipelineConfig` attribute in
#: `thucia.core.pipeline.fit_model`.
SUPPORTED_KWARGS = frozenset(
    {"train_end_date", "retrain", "multivariate", "samples", "season_length"}
)

#: Well-known model families, for categorization and documentation.
FAMILIES = ("statistical", "darts", "container", "external")


@dataclass(frozen=True)
class ModelSpec:
    """Immutable metadata describing one forecast model."""

    name: str
    family: str = "statistical"
    #: Whether the model is fast enough to run in the fast backtest path
    #: (`thucia.core.validation.BacktestConfig.fast_only`). Replaces the old
    #: hard-coded `_FAST_MODELS` allowlist.
    fast: bool = False
    #: Subset of `SUPPORTED_KWARGS`: the config knobs, beyond the common ones,
    #: that this model's callable accepts.
    supports: frozenset[str] = field(default_factory=frozenset)
    #: How the model emits forecasts: "quantiles" directly, "samples" that
    #: `run_model` collapses to quantiles, or "none".
    sampling: str = "quantiles"
    #: Optional-dependency extras needed to run this model (for readable errors
    #: and documentation), e.g. ("torch",) or ("chronos",).
    extras: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unknown = self.supports - SUPPORTED_KWARGS
        if unknown:
            raise ValueError(
                f"{self.name}: unsupported spec kwargs {sorted(unknown)}; "
                f"allowed: {sorted(SUPPORTED_KWARGS)}"
            )
        if self.family not in FAMILIES:
            raise ValueError(
                f"{self.name}: unknown family {self.family!r}; "
                f"allowed: {FAMILIES}"
            )
        if self.sampling not in ("quantiles", "samples", "none"):
            raise ValueError(
                f"{self.name}: unknown sampling {self.sampling!r}"
            )
