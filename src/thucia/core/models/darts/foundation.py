"""Shared scaffolding for ``darts`` ``FoundationModel`` wrappers.

Chronos-2, TimesFM 2.5 and TimesFM 3.0 are pre-trained models with a *fixed*
quantile head. ``darts`` slices the requested levels out of that head (the
likelihood's quantiles must be a subset of the grid the checkpoint was trained
on) and exposes them as ``<component>_q<level>`` components when
``predict_likelihood_parameters=True`` -- the naming :class:`DartsBase` already
parses back into ``quantile`` rows.

This module centralises what those wrappers share:

* validation of a requested quantile grid against the pre-trained one,
* the deterministic ``median_only`` mode (implemented as a one-level quantile
  likelihood, so the component naming stays uniform),
* covariate routing (each covariate goes to the past slot, the future slot, or
  both), and
* the zero-shot ``fit`` / ``historical_forecasts`` plumbing.

Subclasses set ``MODEL_QUANTILES`` (the pre-trained grid), ``SUPPORTS_COVARIATES``
and implement ``build_model(horizon)``.
"""
import logging
import warnings
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

from darts.utils.likelihood_models import QuantileRegression

from . import DartsBase

#: Chronos-2 pre-trained quantile grid, intersected with
#: :data:`thucia.core.quantiles.quantiles` (it lacks 0.025 / 0.975).
CHRONOS_QUANTILES = [
    0.01,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.99,
]

#: TimesFM 2.5 / 3.0 pre-trained quantile grid (both checkpoints share it).
TIMESFM_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

#: Tolerance used when matching requested levels to the pre-trained grid.
QUANTILE_TOL = 1e-9

#: Sentinel accepted by ``future_covariate_cols`` meaning "all covariates".
ALL_COVARIATES = "all"

CovariateCols = Union[Sequence[str], str, None]


class FoundationQuantileBase(DartsBase):
    """Base class for fixed-grid foundation-model wrappers."""

    #: Quantile levels the checkpoint was pre-trained on (set by the subclass).
    MODEL_QUANTILES: List[float] = []

    #: Whether the underlying darts model accepts covariates at all.
    SUPPORTS_COVARIATES: bool = True

    #: Foundation models identify variates implicitly; the numeric geo codes
    #: other darts models use as a covariate are meaningless (and TimesFM 2.5
    #: cannot accept covariates at all).
    add_geo_codes_covariate = False

    def __init__(
        self,
        *args,
        quantiles: Optional[Sequence[float]] = None,
        median_only: bool = False,
        past_covariate_cols: CovariateCols = None,
        future_covariate_cols: CovariateCols = None,
        **kwargs,
    ):
        resolved = self._resolve_quantiles(quantiles, median_only)
        self.median_only = bool(median_only)

        if not self.SUPPORTS_COVARIATES and (
            past_covariate_cols is not None or future_covariate_cols is not None
        ):
            raise ValueError(
                f"{type(self).__name__} does not support covariates; "
                "leave past_covariate_cols/future_covariate_cols unset."
            )

        super().__init__(*args, quantiles=resolved, **kwargs)

        # Resolve covariate routing against the covariates the pipeline supplied.
        available = list(self.covariate_cols)
        self.past_covariate_cols = self._select_cols(
            past_covariate_cols,
            available,
            default=available,
            name="past_covariate_cols",
        )
        self.future_covariate_cols = self._select_cols(
            future_covariate_cols,
            available,
            default=[],
            name="future_covariate_cols",
        )

        # darts-derived wrappers always emit quantiles, never samples.
        self.sampling_method = "quantiles"

    # ---- quantile grid ----

    def _resolve_quantiles(
        self, requested: Optional[Sequence[float]], median_only: bool
    ) -> List[float]:
        """Validate a requested grid against the model's pre-trained grid."""
        native = list(self.MODEL_QUANTILES)
        if not native:
            raise NotImplementedError(
                f"{type(self).__name__} must set MODEL_QUANTILES."
            )

        if median_only:
            if requested:
                raise ValueError(
                    "median_only=True and an explicit quantiles list are mutually "
                    "exclusive; pass one or the other."
                )
            return [0.5]

        if requested is None:
            return native

        requested = sorted({float(q) for q in requested})
        unsupported = [
            q
            for q in requested
            if not any(abs(q - native_q) <= QUANTILE_TOL for native_q in native)
        ]
        if unsupported:
            valid = [q for q in requested if q not in unsupported]
            raise ValueError(
                f"{type(self).__name__}: quantile(s) {unsupported} are not in the "
                f"pre-trained grid {native}. Use a subset of it"
                + (f", e.g. {valid}." if valid else ".")
            )

        if not any(abs(q - 0.5) <= QUANTILE_TOL for q in requested):
            warnings.warn(
                f"{type(self).__name__}: 0.5 is not among the requested quantiles "
                f"{requested}; WIS and the ensemble median will be interpolated "
                "from the neighbouring levels.",
                UserWarning,
                stacklevel=3,
            )
        if len(requested) < len(native):
            logging.info(
                f"{type(self).__name__}: using {len(requested)} of {len(native)} "
                f"pre-trained quantile levels."
            )
        return requested

    def check_quantile_components(self, forecasts) -> None:
        """Fail loudly if darts emitted a different grid than we requested."""
        first = forecasts[0] if isinstance(forecasts, (list, tuple)) else forecasts
        components = getattr(first, "components", None)
        if components is None:
            return
        levels = sorted(float(c.rsplit("_q", 1)[-1]) for c in components if "_q" in c)
        if levels != sorted(self.quantiles):
            raise RuntimeError(
                f"{type(self).__name__}: darts returned quantile levels {levels} "
                f"but {sorted(self.quantiles)} were requested."
            )

    def likelihood(self) -> QuantileRegression:
        return QuantileRegression(quantiles=self.quantiles)

    # ---- covariates ----

    @staticmethod
    def _select_cols(
        requested: CovariateCols,
        available: List[str],
        *,
        default: List[str],
        name: str,
    ) -> List[str]:
        if requested is None:
            return list(default)
        if isinstance(requested, str):
            if requested != ALL_COVARIATES:
                raise ValueError(
                    f"{name}={requested!r} is not valid; pass a list of column "
                    f"names or {ALL_COVARIATES!r}."
                )
            return list(available)
        unknown = [c for c in requested if c not in available]
        if unknown:
            raise ValueError(
                f"{name}: unknown covariate(s) {unknown}; available: {available}."
            )
        return list(dict.fromkeys(requested))

    def _select_covariates(self, cov, cols: List[str]):
        """Subset a covariate series (or a sequence of them) to ``cols``."""
        if cov is None or not cols:
            return None
        if isinstance(cov, (list, tuple)):
            selected = [self._select_covariates(c, cols) for c in cov]
            return None if all(c is None for c in selected) else selected
        # Series-like objects (darts TimeSeries, or duck-typed stand-ins) are
        # sliced; anything else (e.g. a sentinel) is passed through untouched.
        return cov[cols] if hasattr(cov, "components") else cov

    def _covariate_kwargs(self, cov) -> dict:
        """Build the ``past_covariates`` / ``future_covariates`` darts kwargs."""
        if not self.SUPPORTS_COVARIATES:
            return {}
        kwargs = {}
        past = self._select_covariates(cov, getattr(self, "past_covariate_cols", []))
        future = self._select_covariates(
            cov, getattr(self, "future_covariate_cols", [])
        )
        if past is not None:
            kwargs["past_covariates"] = past
        if future is not None:
            kwargs["future_covariates"] = future
        return kwargs

    # ---- zero-shot darts plumbing ----

    def build_model(self, horizon):
        raise NotImplementedError

    def pre_fit(self, target_gids=None, **kwargs):
        logging.info(
            f"Registering {type(self).__name__} on historical data "
            f"({self.train_start_date} to {self.train_end_date})..."
        )
        target_list, covar_list, _ = self.get_cases(
            future=False,
            target_gids=target_gids,
            start_date=self.train_start_date,
            end_date=self.train_end_date,
        )
        # Foundation models stay pre-trained; `fit` only records the series
        # (and, for covariate-aware checkpoints, their covariate slots).
        self.model.fit(
            series=target_list,
            verbose=True,
            **self._covariate_kwargs(covar_list),
        )

    def historical_forecasts(
        self,
        ts,
        cov,
        start_date=None,
        retrain=True,
        horizon=1,
        **kwargs,
    ):
        logging.info(
            f"Generating {type(self).__name__} historical forecasts "
            f"from {start_date} with retrain={retrain}..."
        )
        # `last_points_only=True` keeps the target step of each horizon vector
        # (Date = origin + horizon), matching the other darts models;
        # `predict_likelihood_parameters=True` returns the requested quantile
        # levels directly instead of drawing (and collapsing) samples.
        forecasts = self.model.historical_forecasts(
            series=ts,
            forecast_horizon=horizon,
            start=start_date,
            stride=1,
            retrain=retrain,
            last_points_only=True,
            predict_likelihood_parameters=True,
            verbose=False,
            **self._covariate_kwargs(cov),
        )
        self.check_quantile_components(forecasts)
        return forecasts
