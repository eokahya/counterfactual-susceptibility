"""Validated configuration for backend-independent susceptibility math."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from cfsus.exceptions import NonFiniteInputError, ScientificInputError

DEFAULT_EPSILON = 1e-12
DEFAULT_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class SusceptibilityConfig:
    """Numerical constants for gate-crossing calculations.

    ``epsilon`` has the same units as a target preactivation or margin. It only
    regularizes the susceptibility denominator. ``tolerance`` is an absolute
    tolerance in those same units and is used for validity and crossing
    classification. These roles are deliberately separate.
    """

    epsilon: float = DEFAULT_EPSILON
    tolerance: float = DEFAULT_TOLERANCE

    def __post_init__(self) -> None:
        if not isfinite(self.epsilon):
            raise NonFiniteInputError("epsilon must be finite")
        if self.epsilon < 0.0:
            raise ScientificInputError("epsilon must be non-negative")
        if not isfinite(self.tolerance):
            raise NonFiniteInputError("tolerance must be finite")
        if self.tolerance < 0.0:
            raise ScientificInputError("tolerance must be non-negative")
