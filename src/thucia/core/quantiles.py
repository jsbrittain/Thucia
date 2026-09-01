from __future__ import annotations

from typing import Final

#: Canonical quantile grid shared by all models and scoring utilities.
#: Keep this as the single source of truth; import it, don't redefine it.
quantiles: Final[list[float]] = [
    0.01,
    0.025,
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
    0.975,
    0.99,
]
