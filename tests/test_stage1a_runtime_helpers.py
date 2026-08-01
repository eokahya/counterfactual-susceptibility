from __future__ import annotations

import math

import pytest

from cfsus.exceptions import NonFiniteInputError, ScientificInputError
from cfsus.reproduction.runtime_helpers import (
    apply_strict_jumprelu,
    compare_numeric_sequences,
    desired_activation,
    deterministic_top_k,
    fixed_top_k_union,
    resolve_position_selector,
    select_gate_samples,
    strict_jumprelu,
)


@pytest.mark.parametrize(
    ("alpha", "expected"),
    [(0.0, 4.0), (0.5, 2.0), (1.0, 0.0)],
)
def test_desired_activation_uses_absolute_suppression_mapping(
    alpha: float, expected: float
) -> None:
    assert desired_activation(4.0, alpha) == expected


def test_desired_activation_preserves_signed_baseline() -> None:
    assert desired_activation(-4.0, 0.25) == -3.0


@pytest.mark.parametrize("alpha", [-0.01, 1.01, True])
def test_desired_activation_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ScientificInputError):
        desired_activation(4.0, alpha)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_runtime_scalar_helpers_reject_nonfinite_values(value: float) -> None:
    with pytest.raises(NonFiniteInputError):
        desired_activation(value, 0.5)
    with pytest.raises(NonFiniteInputError):
        strict_jumprelu(value, 0.0)


def test_strict_jumprelu_makes_threshold_equality_inactive() -> None:
    assert strict_jumprelu(0.5, 0.5) == 0.0
    assert strict_jumprelu(math.nextafter(0.5, math.inf), 0.5) > 0.5
    assert strict_jumprelu(math.nextafter(0.5, -math.inf), 0.5) == 0.0


def test_vector_jumprelu_validates_shape_and_applies_strict_gate() -> None:
    assert apply_strict_jumprelu([0.2, 0.5, 0.7], [0.5, 0.5, 0.5]) == (
        0.0,
        0.0,
        0.7,
    )
    with pytest.raises(ScientificInputError, match="equal length"):
        apply_strict_jumprelu([0.2], [0.5, 0.6])


def test_position_selector_resolves_only_official_negative_one() -> None:
    assert resolve_position_selector(-1, token_count=7) == 6
    assert resolve_position_selector(0, token_count=7) == 0
    assert resolve_position_selector(6, token_count=7) == 6

    with pytest.raises(ScientificInputError, match="only supported negative"):
        resolve_position_selector(-2, token_count=7)
    with pytest.raises(ScientificInputError, match="outside token count"):
        resolve_position_selector(7, token_count=7)
    with pytest.raises(ScientificInputError, match="positive integer"):
        resolve_position_selector(-1, token_count=0)


def test_gate_sample_selection_is_deterministic_and_equality_is_inactive() -> None:
    selection = select_gate_samples(
        preactivations=[0.8, 0.5, 0.49, 0.9],
        thresholds=[0.5, 0.5, 0.5, 1.0],
    )

    assert selection.active_feature_id == 0
    assert selection.inactive_feature_id == 1
    assert selection.closest_margin_feature_id == 1


def test_gate_sample_selection_uses_feature_id_to_break_margin_ties() -> None:
    selection = select_gate_samples(
        preactivations=[0.4, 0.6],
        thresholds=[0.5, 0.5],
    )

    assert selection.closest_margin_feature_id == 0


@pytest.mark.parametrize(
    ("preactivations", "thresholds", "message"),
    [
        ([0.1, 0.2], [0.5, 0.5], "active"),
        ([0.6, 0.7], [0.5, 0.5], "inactive"),
    ],
)
def test_gate_sample_selection_requires_both_activity_classes(
    preactivations: list[float], thresholds: list[float], message: str
) -> None:
    with pytest.raises(ScientificInputError, match=message):
        select_gate_samples(preactivations, thresholds)


def test_deterministic_top_k_breaks_ties_by_token_id() -> None:
    assert deterministic_top_k([1.0, 3.0, 3.0, 2.0], k=3) == (1, 2, 3)


def test_fixed_top_k_union_is_sorted_and_independent_of_run_order() -> None:
    baseline = [9.0, 8.0, 0.0, 0.0]
    intervened = [0.0, 8.0, 10.0, 0.0]

    expected = (0, 1, 2)
    assert fixed_top_k_union([baseline, intervened], k=2) == expected
    assert fixed_top_k_union([intervened, baseline], k=2) == expected


def test_fixed_top_k_union_requires_aligned_finite_vectors() -> None:
    with pytest.raises(ScientificInputError, match="equal length"):
        fixed_top_k_union([[1.0, 2.0], [3.0]], k=1)
    with pytest.raises(NonFiniteInputError):
        fixed_top_k_union([[1.0, math.nan]], k=1)
    with pytest.raises(ScientificInputError, match="exceeds"):
        fixed_top_k_union([[1.0, 2.0]], k=3)


def test_numeric_comparison_supports_noop_and_fixed_union_checks() -> None:
    reference = [1.0, 10.0, -2.0, 5.0]
    candidate = [1.5, 10.00001, -2.0, 5.00001]

    full = compare_numeric_sequences(
        reference,
        candidate,
        absolute_tolerance=2e-5,
    )
    assert full.within_tolerance is False
    assert full.mismatched_count == 1
    assert full.maximum_absolute_error == pytest.approx(0.5)

    fixed_union = compare_numeric_sequences(
        reference,
        candidate,
        indices=[1, 3],
        absolute_tolerance=2e-5,
    )
    assert fixed_union.within_tolerance is True
    assert fixed_union.indices == (1, 3)
    assert fixed_union.maximum_absolute_error == pytest.approx(1e-5)


def test_numeric_comparison_uses_allclose_style_relative_tolerance() -> None:
    result = compare_numeric_sequences(
        [1000.0],
        [1000.5],
        absolute_tolerance=0.0,
        relative_tolerance=5e-4,
    )

    assert result.within_tolerance is True


@pytest.mark.parametrize("indices", [[], [1, 1], [1, 0], [-1], [2]])
def test_numeric_comparison_rejects_invalid_fixed_indices(indices: list[int]) -> None:
    with pytest.raises(ScientificInputError):
        compare_numeric_sequences([1.0, 2.0], [1.0, 2.0], indices=indices)


def test_numeric_comparison_rejects_shape_and_tolerance_errors() -> None:
    with pytest.raises(ScientificInputError, match="equal length"):
        compare_numeric_sequences([1.0], [1.0, 2.0])
    with pytest.raises(ScientificInputError, match="non-negative"):
        compare_numeric_sequences([1.0], [1.0], absolute_tolerance=-1.0)
