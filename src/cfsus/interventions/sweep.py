"""Deterministic construction of source-suppression sweeps."""

from __future__ import annotations

from itertools import pairwise
from math import isfinite

from cfsus.exceptions import NonFiniteInputError, ScientificInputError
from cfsus.types import FeatureRef, SourceSuppression


def regular_alpha_grid(steps: int) -> tuple[float, ...]:
    """Return ``steps + 1`` evenly spaced values including zero and one."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ScientificInputError("steps must be a positive integer")
    return tuple(index / steps for index in range(steps + 1))


def make_suppression_sweep(
    source: FeatureRef, alphas: tuple[float, ...]
) -> tuple[SourceSuppression, ...]:
    """Validate a strictly increasing alpha sequence and build interventions."""

    if not isinstance(source, FeatureRef):
        raise ScientificInputError("source must be a FeatureRef")
    if not alphas:
        raise ScientificInputError(
            "a suppression sweep must contain at least one alpha"
        )
    for alpha in alphas:
        if not isfinite(alpha):
            raise NonFiniteInputError(f"alpha must be finite, got {alpha!r}")
    if any(later <= earlier for earlier, later in pairwise(alphas)):
        raise ScientificInputError("sweep alpha values must be strictly increasing")
    return tuple(SourceSuppression(source=source, alpha=alpha) for alpha in alphas)
