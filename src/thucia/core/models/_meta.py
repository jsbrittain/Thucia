# Declarative metadata for forecast models.
#
# Each model module (e.g. `sarima.py`) declares a module-level `SPEC` describing
# which pipeline config knobs it consumes and how the pipeline should treat it.
# `thucia.core.pipeline.fit_model` and `thucia.core.validation.run_backtest`
# read the spec instead of hard-coding model identity/name branches, so adding
# a model with new knobs is a one-line spec, not an edit to the pipeline.
from __future__ import annotations

import inspect
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Mapping

#: Config knobs a model may declare it supports (beyond the common kwargs that
#: every model receives). Each maps to a `PipelineConfig` attribute in
#: `thucia.core.pipeline.fit_model`.
SUPPORTED_KWARGS = frozenset(
    {
        "train_end_date",
        "retrain",
        "multivariate",
        "samples",
        "season_length",
        "quantiles",
        "median_only",
        "past_covariate_cols",
        "future_covariate_cols",
    }
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
                f"{self.name}: unknown family {self.family!r}; allowed: {FAMILIES}"
            )
        if self.sampling not in ("quantiles", "samples", "none"):
            raise ValueError(f"{self.name}: unknown sampling {self.sampling!r}")


@dataclass(frozen=True)
class ModelSelection:
    """A configured model choice for one fit.

    Model *identity* is still a name resolved lazily through
    ``thucia.core.models.get_model``; this adds what a bare string cannot
    express: a label for the resulting artefacts and per-model configuration.

    Parameters
    ----------
    name:
        Advertised model name (e.g. ``"sarima"``).
    label:
        Name used for output files and score rows; defaults to ``name``. Set it
        to distinguish two runs of the same model with different settings.
    overrides:
        ``PipelineConfig`` fields applied for this model only (e.g.
        ``{"retrain": True, "season_length": 52}``). Validated against the
        config fields when the selection is fitted.
    params:
        Extra keyword arguments forwarded to the model callable (e.g.
        ``{"symmetrize": False}`` for ``baseline``). Validated against the
        callable's signature when the selection is fitted.
    """

    name: str
    label: str | None = None
    overrides: Mapping[str, Any] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ModelSelection.name must be a non-empty model name")
        if self.label is not None and not self.label:
            raise ValueError("ModelSelection.label must be non-empty when set")
        # Normalise to plain dicts so a selection can be serialised verbatim.
        object.__setattr__(self, "overrides", dict(self.overrides))
        object.__setattr__(self, "params", dict(self.params))

    @property
    def run_label(self) -> str:
        """Stable identifier for this run (``label`` or the model name)."""
        return self.label or self.name


def as_selection(model: str | ModelSelection) -> ModelSelection:
    """Normalise a model name or selection to a :class:`ModelSelection`."""
    if isinstance(model, ModelSelection):
        return model
    if isinstance(model, str):
        return ModelSelection(name=model)
    raise TypeError(
        f"model must be a model name or ModelSelection, got {type(model).__name__}"
    )


def validate_model_params(selection: ModelSelection, model_fn: Callable) -> None:
    """Reject ``selection.params`` the model callable does not accept.

    Raises rather than letting a typo be absorbed by a model's ``**kwargs`` and
    silently ignored. Names the model's declared ``ModelSpec`` knobs are also
    accepted, since the pipeline passes those through by name.
    """
    signature = inspect.signature(model_fn)
    explicit = {
        name
        for name, param in signature.parameters.items()
        if param.kind not in (param.VAR_KEYWORD, param.VAR_POSITIONAL)
    }
    allowed = explicit | set(SUPPORTED_KWARGS) | {"num_samples"}
    unknown = set(selection.params) - allowed
    if unknown:
        raise ValueError(
            f"{selection.run_label}: unknown model params {sorted(unknown)}; "
            f"{selection.name}() accepts: {sorted(allowed)}"
        )
