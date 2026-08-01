from __future__ import annotations

import math

import pytest

from cfsus.config import SusceptibilityConfig
from cfsus.exceptions import (
    ActiveTargetError,
    NonFiniteInputError,
    ScientificInputError,
)
from cfsus.susceptibility.pairwise import (
    activation_margin,
    classify_predicted_crossing,
    critical_suppression_fraction,
    pairwise_susceptibility,
    predict_crossing,
    predicted_target_preactivation,
    suppression_response,
)
from cfsus.types import CrossingStatus


def test_inhibitory_source_predicts_crossing() -> None:
    margin = activation_margin(0.2, 0.5)
    q = suppression_response(source_activation=2.0, local_response=-0.25)

    assert q == 0.5
    assert classify_predicted_crossing(margin, q) is CrossingStatus.DEFINITELY_CROSSING
    assert critical_suppression_fraction(margin, q) == pytest.approx(0.6)
    assert predicted_target_preactivation(0.2, 1.0, q) == pytest.approx(0.7)


def test_inhibitory_source_can_be_insufficient_to_cross() -> None:
    margin = activation_margin(0.2, 0.5)
    q = suppression_response(source_activation=2.0, local_response=-0.1)

    assert classify_predicted_crossing(margin, q) is CrossingStatus.NOT_CROSSING
    assert critical_suppression_fraction(margin, q) == pytest.approx(1.5)


def test_suppressing_excitatory_source_moves_target_away_from_gate() -> None:
    margin = activation_margin(0.2, 0.5)
    q = suppression_response(source_activation=2.0, local_response=0.25)

    assert q == -0.5
    assert pairwise_susceptibility(q, margin) < 0.0
    assert classify_predicted_crossing(margin, q) is CrossingStatus.NOT_CROSSING
    assert critical_suppression_fraction(margin, q) is None
    assert predicted_target_preactivation(0.2, 1.0, q) < 0.2


def test_target_already_active_is_invalid_for_inactive_target_analysis() -> None:
    with pytest.raises(ActiveTargetError, match="above threshold"):
        activation_margin(0.6, 0.5, tolerance=1e-9)

    assert (
        classify_predicted_crossing(-0.1, 0.5, tolerance=1e-9)
        is CrossingStatus.INVALID_TARGET
    )
    prediction = predict_crossing(-0.1, 0.5)
    assert prediction.status is CrossingStatus.INVALID_TARGET
    assert prediction.predicted_critical_alpha is None


def test_exact_zero_margin_is_boundary_ambiguous() -> None:
    margin = activation_margin(0.5, 0.5)

    assert margin == 0.0
    assert classify_predicted_crossing(margin, 0.2) is CrossingStatus.BOUNDARY_AMBIGUOUS
    assert critical_suppression_fraction(margin, 0.2) == 0.0


@pytest.mark.parametrize("margin", [-5e-10, 5e-10])
def test_near_zero_margin_is_boundary_ambiguous(margin: float) -> None:
    assert (
        classify_predicted_crossing(margin, 0.2, tolerance=1e-9)
        is CrossingStatus.BOUNDARY_AMBIGUOUS
    )


@pytest.mark.parametrize("q", [0.0, -5e-10, -1e-4])
def test_zero_and_negative_q_do_not_create_infinite_alpha(q: float) -> None:
    assert critical_suppression_fraction(0.3, q, tolerance=1e-9) is None
    assert (
        classify_predicted_crossing(0.3, q, tolerance=1e-9)
        is CrossingStatus.NOT_CROSSING
    )


def test_tiny_positive_q_preserves_large_finite_critical_alpha() -> None:
    assert critical_suppression_fraction(0.3, 5e-10) == pytest.approx(6e8)
    assert (
        classify_predicted_crossing(0.3, 5e-10, tolerance=1e-9)
        is CrossingStatus.NOT_CROSSING
    )


@pytest.mark.parametrize("margin", [-5e-10, 5e-10])
def test_near_zero_margin_remains_signed_in_critical_alpha(margin: float) -> None:
    assert critical_suppression_fraction(margin, 0.2) == pytest.approx(margin / 0.2)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_scientific_inputs_are_rejected(value: float) -> None:
    with pytest.raises(NonFiniteInputError):
        activation_margin(value, 0.5)
    with pytest.raises(NonFiniteInputError):
        suppression_response(1.0, value)
    with pytest.raises(NonFiniteInputError):
        pairwise_susceptibility(value, 0.3)
    with pytest.raises(NonFiniteInputError):
        critical_suppression_fraction(0.3, value)
    with pytest.raises(NonFiniteInputError):
        predicted_target_preactivation(0.2, 0.5, value)
    assert classify_predicted_crossing(0.3, value) is CrossingStatus.NON_FINITE_INPUT


def test_overflow_in_derived_scientific_values_is_rejected() -> None:
    with pytest.raises(NonFiniteInputError):
        activation_margin(-1e308, 1e308)
    with pytest.raises(NonFiniteInputError):
        pairwise_susceptibility(1.0, 1e308, epsilon=1e308)
    with pytest.raises(NonFiniteInputError):
        critical_suppression_fraction(1.0, math.nextafter(0.0, 1.0))


@pytest.mark.parametrize("alpha", [-0.01, 1.01, math.nan])
def test_predicted_preactivation_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ScientificInputError):
        predicted_target_preactivation(0.2, alpha, 0.5)


def test_predicted_critical_alpha_decreases_with_inhibitory_strength() -> None:
    strengths = (0.4, 0.6, 1.2)
    alphas = tuple(critical_suppression_fraction(0.3, q) for q in strengths)

    resolved = tuple(alpha for alpha in alphas if alpha is not None)
    assert len(resolved) == len(strengths)
    assert resolved[0] > resolved[1] > resolved[2]


@pytest.mark.parametrize(
    ("margin", "q"),
    [(0.4, 0.8), (0.4, 0.2), (2.0, 3.0), (2.0, 1.0)],
)
def test_s_greater_than_one_matches_critical_alpha_below_one_away_from_boundaries(
    margin: float, q: float
) -> None:
    susceptibility = pairwise_susceptibility(q, margin, epsilon=0.0, tolerance=1e-9)
    critical_alpha = critical_suppression_fraction(margin, q, tolerance=1e-9)

    assert critical_alpha is not None
    assert (susceptibility > 1.0) is (critical_alpha < 1.0)


def test_full_suppression_equality_is_boundary_ambiguous() -> None:
    assert (
        classify_predicted_crossing(0.3, 0.3, tolerance=1e-9)
        is CrossingStatus.BOUNDARY_AMBIGUOUS
    )
    assert critical_suppression_fraction(0.3, 0.3) == pytest.approx(1.0)


def test_documented_numerical_worked_example() -> None:
    """Canonical example added to RESEARCH_SPEC during Stage 0 integration."""

    preactivation = 0.2
    threshold = 0.5
    source_activation = 2.0
    local_response = -0.25
    margin = activation_margin(preactivation, threshold)
    q = suppression_response(source_activation, local_response)

    assert margin == pytest.approx(0.3)
    assert q == pytest.approx(0.5)
    assert pairwise_susceptibility(q, margin, epsilon=0.01) == pytest.approx(0.5 / 0.31)
    assert critical_suppression_fraction(margin, q) == pytest.approx(0.6)
    assert predicted_target_preactivation(preactivation, 1.0, q) == pytest.approx(0.7)


def test_prediction_record_preserves_diagnostics_without_infinity() -> None:
    prediction = predict_crossing(0.3, 0.0)

    assert prediction.status is CrossingStatus.NOT_CROSSING
    assert prediction.predicted_critical_alpha is None
    assert prediction.susceptibility == 0.0
    assert prediction.reason


@pytest.mark.parametrize("epsilon", [math.nan, math.inf, -1.0, -1e-12])
def test_prediction_record_rejects_invalid_epsilon(epsilon: float) -> None:
    with pytest.raises(ScientificInputError):
        predict_crossing(0.3, 0.5, epsilon=epsilon)


def test_zero_epsilon_is_valid_for_positive_margin() -> None:
    prediction = predict_crossing(0.3, 0.5, epsilon=0.0)
    assert prediction.susceptibility == pytest.approx(0.5 / 0.3)
    assert SusceptibilityConfig(epsilon=0.0).epsilon == 0.0


def test_config_rejects_negative_epsilon() -> None:
    with pytest.raises(ScientificInputError):
        SusceptibilityConfig(epsilon=-1e-12)
