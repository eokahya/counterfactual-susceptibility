"""Observed-crossing helpers using backend-reported gate activity."""

from __future__ import annotations

from itertools import pairwise

from cfsus.exceptions import ScientificInputError
from cfsus.types import ObservedInterventionPoint


def first_observed_crossing(
    points: tuple[ObservedInterventionPoint, ...],
) -> float | None:
    """Return the first active point's alpha after validating sweep order.

    Activity is supplied by the backend's exact feature nonlinearity rather than
    re-created from preactivation and threshold in backend-independent code.
    A returned grid alpha is a bracket endpoint, not a refined critical value.
    """

    if not points:
        raise ScientificInputError("an observed sweep must contain at least one point")
    if any(later.alpha <= earlier.alpha for earlier, later in pairwise(points)):
        raise ScientificInputError("sweep alpha values must be strictly increasing")
    for point in points:
        if point.target_active:
            return point.alpha
    return None
