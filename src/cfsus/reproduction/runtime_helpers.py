"""Dependency-light helpers for Stage 1A runtime semantics checks.

The functions in this module operate on Python scalars and sequences so that
the core package stays import-safe when the optional model runtime (including
PyTorch and ``circuit-tracer``) is absent. Runtime scripts should convert only
the small tensor slices they need to plain numeric sequences before calling
these helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TypeAlias

from cfsus.exceptions import NonFiniteInputError, ScientificInputError

NumericSequence: TypeAlias = tuple[float, ...] | list[float]


def _finite_float(name: str, value: float | int) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ScientificInputError(f"{name} must be a real scalar")
    converted = float(value)
    if not isfinite(converted):
        raise NonFiniteInputError(f"{name} must be finite, got {value!r}")
    return converted


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ScientificInputError(f"{name} must be a positive integer")
    return value


def _finite_vector(name: str, values: NumericSequence) -> tuple[float, ...]:
    if not isinstance(values, (tuple, list)):
        raise ScientificInputError(f"{name} must be a tuple or list of real scalars")
    if not values:
        raise ScientificInputError(f"{name} must not be empty")
    return tuple(
        _finite_float(f"{name}[{index}]", value) for index, value in enumerate(values)
    )


def _aligned_vectors(
    names_and_values: tuple[tuple[str, NumericSequence], ...],
) -> tuple[tuple[float, ...], ...]:
    if not names_and_values:
        raise ScientificInputError("at least one numeric vector is required")
    converted = tuple(_finite_vector(name, values) for name, values in names_and_values)
    expected_length = len(converted[0])
    if any(len(values) != expected_length for values in converted[1:]):
        shapes = ", ".join(
            f"{name}={len(values)}"
            for (name, _), values in zip(names_and_values, converted, strict=True)
        )
        raise ScientificInputError(f"numeric vectors must have equal length; {shapes}")
    return converted


def desired_activation(baseline_activation: float | int, alpha: float | int) -> float:
    """Map project suppression ``alpha`` to an absolute upstream value.

    The fixed project convention is ``desired = (1 - alpha) * baseline``.
    This helper intentionally returns the absolute desired post-gate activation,
    never a multiplier, preactivation, threshold displacement, or decoder delta.
    """

    baseline = _finite_float("baseline_activation", baseline_activation)
    suppression = _finite_float("alpha", alpha)
    if not 0.0 <= suppression <= 1.0:
        raise ScientificInputError("alpha must lie in the closed interval [0, 1]")
    desired = (1.0 - suppression) * baseline
    if not isfinite(desired):
        raise NonFiniteInputError("desired activation must be finite")
    return desired


def strict_jumprelu(preactivation: float | int, threshold: float | int) -> float:
    """Apply the pinned upstream scalar rule ``z if z > tau else 0``.

    Equality is deliberately inactive. This pure helper is for checking values
    observed from the actually loaded activation function, not for replacing
    backend-reported activity in scientific code.
    """

    z = _finite_float("preactivation", preactivation)
    tau = _finite_float("threshold", threshold)
    return z if z > tau else 0.0


def apply_strict_jumprelu(
    preactivations: NumericSequence,
    thresholds: NumericSequence,
) -> tuple[float, ...]:
    """Apply :func:`strict_jumprelu` to two aligned finite vectors."""

    z_values, tau_values = _aligned_vectors(
        (("preactivations", preactivations), ("thresholds", thresholds))
    )
    return tuple(
        strict_jumprelu(z, tau) for z, tau in zip(z_values, tau_values, strict=True)
    )


def resolve_position_selector(requested_position: int, token_count: int) -> int:
    """Resolve an upstream token selector after tokenization.

    Non-negative absolute positions are accepted. The only supported negative
    selector is the official demo's ``-1``, resolved after the final token count
    is known. Supporting arbitrary Python negative indexing here would obscure
    the exact position recorded in a Stage 1A artifact.
    """

    _positive_integer("token_count", token_count)
    if isinstance(requested_position, bool) or not isinstance(requested_position, int):
        raise ScientificInputError("requested_position must be an integer")
    if requested_position == -1:
        return token_count - 1
    if requested_position < 0:
        raise ScientificInputError(
            "the only supported negative position selector is -1"
        )
    if requested_position >= token_count:
        raise ScientificInputError(
            f"requested_position {requested_position} is outside token count "
            f"{token_count}"
        )
    return requested_position


@dataclass(frozen=True, slots=True)
class GateSampleSelection:
    """Deterministic feature IDs for a bounded runtime gate check."""

    active_feature_id: int
    inactive_feature_id: int
    closest_margin_feature_id: int


def select_gate_samples(
    preactivations: NumericSequence,
    thresholds: NumericSequence,
) -> GateSampleSelection:
    """Select one active, inactive, and closest-margin feature deterministically.

    Active and inactive IDs are the lowest IDs in their respective classes.
    The closest-margin ID minimizes ``abs(threshold - preactivation)`` and uses
    feature ID as a stable tie-breaker. Equality is in the inactive class.
    Callers should pass one already-selected layer/position vector; this helper
    is not a cross-layer candidate scanner.
    """

    z_values, tau_values = _aligned_vectors(
        (("preactivations", preactivations), ("thresholds", thresholds))
    )
    active_ids = tuple(
        index
        for index, (z, tau) in enumerate(zip(z_values, tau_values, strict=True))
        if z > tau
    )
    inactive_ids = tuple(
        index
        for index, (z, tau) in enumerate(zip(z_values, tau_values, strict=True))
        if z <= tau
    )
    if not active_ids:
        raise ScientificInputError("gate sample requires at least one active feature")
    if not inactive_ids:
        raise ScientificInputError("gate sample requires at least one inactive feature")

    margins: list[tuple[float, int]] = []
    for feature_id, (z, tau) in enumerate(zip(z_values, tau_values, strict=True)):
        distance = abs(tau - z)
        if not isfinite(distance):
            raise NonFiniteInputError(
                f"absolute gate margin overflowed for feature {feature_id}"
            )
        margins.append((distance, feature_id))

    return GateSampleSelection(
        active_feature_id=active_ids[0],
        inactive_feature_id=inactive_ids[0],
        closest_margin_feature_id=min(margins)[1],
    )


def deterministic_top_k(values: NumericSequence, k: int) -> tuple[int, ...]:
    """Return top-``k`` indices ordered by value descending, then ID ascending."""

    numeric_values = _finite_vector("values", values)
    _positive_integer("k", k)
    if k > len(numeric_values):
        raise ScientificInputError(
            f"k={k} exceeds the vector length {len(numeric_values)}"
        )
    return tuple(
        sorted(
            range(len(numeric_values)),
            key=lambda index: (-numeric_values[index], index),
        )[:k]
    )


def fixed_top_k_union(
    value_sets: tuple[NumericSequence, ...] | list[NumericSequence],
    k: int,
) -> tuple[int, ...]:
    """Return a sorted fixed union of deterministic top-``k`` token IDs.

    All run vectors must have the same vocabulary length. Sorting the final set
    by token ID makes later numeric serialization independent of run order.
    """

    if not isinstance(value_sets, (tuple, list)) or not value_sets:
        raise ScientificInputError(
            "value_sets must contain at least one numeric vector"
        )
    named = tuple(
        (f"value_sets[{index}]", values) for index, values in enumerate(value_sets)
    )
    converted = _aligned_vectors(named)
    _positive_integer("k", k)
    if k > len(converted[0]):
        raise ScientificInputError(
            f"k={k} exceeds the vector length {len(converted[0])}"
        )
    selected: set[int] = set()
    for values in converted:
        selected.update(deterministic_top_k(values, k))
    return tuple(sorted(selected))


@dataclass(frozen=True, slots=True)
class NumericComparison:
    """Finite allclose-style comparison over a declared fixed index set."""

    indices: tuple[int, ...]
    maximum_absolute_error: float
    mismatched_count: int
    within_tolerance: bool
    absolute_tolerance: float
    relative_tolerance: float


def compare_numeric_sequences(
    reference: NumericSequence,
    candidate: NumericSequence,
    *,
    indices: tuple[int, ...] | list[int] | None = None,
    absolute_tolerance: float | int = 0.0,
    relative_tolerance: float | int = 0.0,
) -> NumericComparison:
    """Compare aligned vectors globally or over a fixed sorted index union.

    Closeness follows ``abs(candidate-reference) <= atol + rtol*abs(reference)``.
    Passing indices from :func:`fixed_top_k_union` compares the same token set
    across every intervention run instead of independently changing top-k sets.
    """

    reference_values, candidate_values = _aligned_vectors(
        (("reference", reference), ("candidate", candidate))
    )
    atol = _finite_float("absolute_tolerance", absolute_tolerance)
    rtol = _finite_float("relative_tolerance", relative_tolerance)
    if atol < 0.0 or rtol < 0.0:
        raise ScientificInputError("comparison tolerances must be non-negative")

    if indices is None:
        selected_indices = tuple(range(len(reference_values)))
    else:
        if not isinstance(indices, (tuple, list)) or not indices:
            raise ScientificInputError("indices must be a non-empty tuple or list")
        selected_indices = tuple(indices)
        if any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in selected_indices
        ):
            raise ScientificInputError("comparison indices must be integers")
        if tuple(sorted(set(selected_indices))) != selected_indices:
            raise ScientificInputError(
                "comparison indices must be unique and strictly increasing"
            )
        if selected_indices[-1] >= len(reference_values) or selected_indices[0] < 0:
            raise ScientificInputError("comparison index is outside the vector shape")

    maximum_error = 0.0
    mismatched_count = 0
    for index in selected_indices:
        error = abs(candidate_values[index] - reference_values[index])
        if not isfinite(error):
            raise NonFiniteInputError(
                f"absolute comparison error overflowed at index {index}"
            )
        maximum_error = max(maximum_error, error)
        permitted_error = atol + rtol * abs(reference_values[index])
        if not isfinite(permitted_error):
            raise NonFiniteInputError(
                f"comparison tolerance overflowed at index {index}"
            )
        if error > permitted_error:
            mismatched_count += 1

    return NumericComparison(
        indices=selected_indices,
        maximum_absolute_error=maximum_error,
        mismatched_count=mismatched_count,
        within_tolerance=mismatched_count == 0,
        absolute_tolerance=atol,
        relative_tolerance=rtol,
    )


__all__ = [
    "GateSampleSelection",
    "NumericComparison",
    "apply_strict_jumprelu",
    "compare_numeric_sequences",
    "desired_activation",
    "deterministic_top_k",
    "fixed_top_k_union",
    "resolve_position_selector",
    "select_gate_samples",
    "strict_jumprelu",
]
