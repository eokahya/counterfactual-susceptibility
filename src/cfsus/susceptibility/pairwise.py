"""Pure pairwise Counterfactual Susceptibility mathematics.

No function in this module applies a feature nonlinearity. The exact ``phi_i``
is a property of a loaded and audited backend. All values here are scalars:
``preactivation``, ``threshold``, ``margin``, and ``q`` share units;
``source_activation * local_response`` therefore also has preactivation units;
susceptibility, alpha, and the local response product's ratios are dimensionless.
"""

from __future__ import annotations

from math import isfinite

from cfsus.config import DEFAULT_EPSILON, DEFAULT_TOLERANCE
from cfsus.exceptions import (
    ActiveTargetError,
    NonFiniteInputError,
    ScientificInputError,
)
from cfsus.types import CrossingStatus, PredictedCrossing


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise NonFiniteInputError(f"{name} must be finite, got {value!r}")


def _validate_tolerance(tolerance: float) -> None:
    _require_finite("tolerance", tolerance)
    if tolerance < 0.0:
        raise ScientificInputError("tolerance must be non-negative")


def _validate_epsilon(epsilon: float) -> None:
    _require_finite("epsilon", epsilon)
    if epsilon < 0.0:
        raise ScientificInputError("epsilon must be non-negative")


def activation_margin(
    preactivation: float,
    threshold: float,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """Return the signed inactive-target margin ``threshold - preactivation``.

    This is the inactive-target API: a margin below ``-tolerance`` means the
    target is already beyond its gate and is rejected. A value within tolerance
    of zero remains signed so the classifier can mark it as ambiguous.
    """

    _require_finite("preactivation", preactivation)
    _require_finite("threshold", threshold)
    _validate_tolerance(tolerance)
    margin = threshold - preactivation
    _require_finite("margin", margin)
    if margin < -tolerance:
        raise ActiveTargetError(
            "target preactivation is above threshold beyond tolerance; "
            "inactive-target susceptibility is invalid"
        )
    return margin


def suppression_response(source_activation: float, local_response: float) -> float:
    """Return ``q = -source_activation * local_response`` without sign folding.

    ``local_response`` is ``d z_i / d a_j`` and has units of target
    preactivation per unit of source activation. Positive ``q`` moves the target
    toward its threshold under source suppression.
    """

    _require_finite("source_activation", source_activation)
    _require_finite("local_response", local_response)
    q = -source_activation * local_response
    _require_finite("q", q)
    return q


def pairwise_susceptibility(
    q: float,
    margin: float,
    *,
    epsilon: float = DEFAULT_EPSILON,
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """Return signed susceptibility ``q / (margin + epsilon)``.

    ``epsilon`` has margin units and must be non-negative. The function
    preserves the sign of ``q``. It does not reinterpret a boundary target as a
    valid definitely-inactive target; callers should also inspect the crossing
    classification.
    """

    _require_finite("q", q)
    _require_finite("margin", margin)
    _validate_tolerance(tolerance)
    _validate_epsilon(epsilon)
    if margin < -tolerance:
        raise ActiveTargetError(
            "negative margin beyond tolerance is invalid for inactive-target analysis"
        )
    denominator = margin + epsilon
    _require_finite("margin + epsilon", denominator)
    if denominator <= 0.0:
        raise ScientificInputError(
            "margin + epsilon must be positive; boundary classification is required"
        )
    value = q / denominator
    _require_finite("susceptibility", value)
    return value


def critical_suppression_fraction(
    margin: float,
    q: float,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> float | None:
    """Return the signed fixed formula ``margin / q`` for every ``q > 0``.

    Returns ``None`` for ``q <= 0`` instead of manufacturing an infinite value.
    Tolerances affect crossing classification, not this scientific quantity: a
    tiny positive ``q`` therefore yields a large finite alpha, and a signed
    near-zero margin is not snapped to zero.
    """

    _require_finite("margin", margin)
    _require_finite("q", q)
    _validate_tolerance(tolerance)
    if margin < -tolerance:
        raise ActiveTargetError(
            "negative margin beyond tolerance is invalid for inactive-target analysis"
        )
    if q <= 0.0:
        return None
    value = margin / q
    _require_finite("predicted critical alpha", value)
    return value


def classify_predicted_crossing(
    margin: float,
    q: float,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> CrossingStatus:
    """Classify whether full suppression predicts a gate crossing.

    The full-suppression boundary is ``q == margin``. Equality and a band of
    width ``tolerance`` around it are ambiguous because gate strictness and
    floating-point error matter. Non-finite inputs are explicitly classified so
    diagnostic records can preserve the reason; scalar math functions reject
    the same inputs with ``NonFiniteInputError``.
    """

    _validate_tolerance(tolerance)
    if not isfinite(margin) or not isfinite(q):
        return CrossingStatus.NON_FINITE_INPUT
    if margin < -tolerance:
        return CrossingStatus.INVALID_TARGET
    if abs(margin) <= tolerance:
        return CrossingStatus.BOUNDARY_AMBIGUOUS
    if q <= 0.0:
        return CrossingStatus.NOT_CROSSING

    full_suppression_excess = q - margin
    if full_suppression_excess > tolerance:
        return CrossingStatus.DEFINITELY_CROSSING
    if abs(full_suppression_excess) <= tolerance:
        return CrossingStatus.BOUNDARY_AMBIGUOUS
    return CrossingStatus.NOT_CROSSING


def predicted_target_preactivation(
    baseline_preactivation: float,
    alpha: float,
    q: float,
) -> float:
    """Return ``z_hat(alpha) = baseline_preactivation + alpha * q``.

    This is equivalent to ``z_i - alpha * a_j * J_ij`` under the declared local
    linear approximation. ``alpha`` follows the closed interval ``[0, 1]``.
    """

    _require_finite("baseline_preactivation", baseline_preactivation)
    _require_finite("alpha", alpha)
    _require_finite("q", q)
    if not 0.0 <= alpha <= 1.0:
        raise ScientificInputError("alpha must lie in the closed interval [0, 1]")
    value = baseline_preactivation + alpha * q
    _require_finite("predicted target preactivation", value)
    return value


def predict_crossing(
    margin: float,
    q: float,
    *,
    epsilon: float = DEFAULT_EPSILON,
    tolerance: float = DEFAULT_TOLERANCE,
) -> PredictedCrossing:
    """Build a complete diagnostic record from already-computed ``margin``/``q``."""

    _validate_tolerance(tolerance)
    _validate_epsilon(epsilon)
    status = classify_predicted_crossing(margin, q, tolerance=tolerance)
    if status is CrossingStatus.NON_FINITE_INPUT:
        return PredictedCrossing(
            margin=margin,
            q=q,
            susceptibility=None,
            predicted_critical_alpha=None,
            status=status,
            reason="margin or q is non-finite",
        )
    if status is CrossingStatus.INVALID_TARGET:
        return PredictedCrossing(
            margin=margin,
            q=q,
            susceptibility=None,
            predicted_critical_alpha=None,
            status=status,
            reason="target is already active beyond tolerance",
        )

    susceptibility: float | None = None
    if margin + epsilon > 0.0:
        susceptibility = pairwise_susceptibility(
            q, margin, epsilon=epsilon, tolerance=tolerance
        )
    critical_alpha = critical_suppression_fraction(margin, q, tolerance=tolerance)
    if status is CrossingStatus.DEFINITELY_CROSSING:
        reason = "positive q exceeds the target margin beyond tolerance"
    elif status is CrossingStatus.BOUNDARY_AMBIGUOUS:
        reason = "target or full-suppression prediction lies on a tolerance boundary"
    elif q <= 0.0:
        reason = "source suppression does not move the target toward its gate"
    else:
        reason = "full source suppression is insufficient to reach the gate"
    return PredictedCrossing(
        margin=margin,
        q=q,
        susceptibility=susceptibility,
        predicted_critical_alpha=critical_alpha,
        status=status,
        reason=reason,
    )
