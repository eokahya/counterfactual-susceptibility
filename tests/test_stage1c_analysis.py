from __future__ import annotations

import pytest

from cfsus.stage1c.analysis import (
    ScientificOutcome,
    aggregate_analyses,
    analyze_pair,
    classify_outcome,
)

ANALYSIS = {
    "minimum_nonzero_points": 3,
    "movement_sign_agreement_minimum": 0.80,
    "median_movement_sne_maximum": 0.50,
    "p95_movement_sne_maximum": 1.00,
    "critical_bracket_distance_maximum": 0.125,
}


def _pair(
    *,
    group: str = "primary",
    q: float = 2.0,
    alpha: float | None = 0.5,
) -> dict[str, object]:
    return {
        "pair_id": f"{group}-pair",
        "group": group,
        "target_preactivation": 0.0,
        "target_threshold": 1.0,
        "q": q,
        "predicted_alpha_star": alpha,
        "predicted_status": (
            "definitely_crossing"
            if q > 0.0 and alpha is not None and alpha <= 1.0
            else "not_crossing"
        ),
    }


def _points(
    *, crossing: bool = True, directional: bool = False
) -> list[dict[str, object]]:
    # z=threshold is deliberately represented as inactive at realized alpha=.5.
    return [
        {
            "realized_suppression": 0.0,
            "target_preactivation": 0.0,
            "target_active": False,
        },
        {
            "realized_suppression": 0.25,
            "target_preactivation": 0.5 * (1 if not directional else -1),
            "target_active": False,
        },
        {
            "realized_suppression": 0.5,
            "target_preactivation": 1.0 * (1 if not directional else -1),
            "target_active": False,
        },
        {
            "realized_suppression": 0.75,
            "target_preactivation": 1.5 * (1 if not directional else -1),
            "target_active": crossing,
        },
        {
            "realized_suppression": 1.0,
            "target_preactivation": 2.0 * (1 if not directional else -1),
            "target_active": crossing,
        },
    ]


def test_analysis_uses_realized_suppression_and_constructs_crossing_bracket() -> None:
    result = analyze_pair(_pair(), _points(), ANALYSIS)
    assert result["observed_critical_bracket"] == {
        "lower_realized_suppression": 0.5,
        "upper_realized_suppression": 0.75,
    }
    assert result["observed_full_ablation_crossing"] is True
    assert result["movement_sign_agreement"] == pytest.approx(1.0)
    assert result["critical_bracket_distance"] == 0.0
    assert result["local_calibration_passed"] is True
    assert result["supporting_primary"] is True


def test_analysis_censors_no_crossing_and_directional_controls() -> None:
    no_crossing = analyze_pair(_pair(), _points(crossing=False), ANALYSIS)
    assert no_crossing["observed_critical_bracket"] is None
    assert no_crossing["observed_full_ablation_crossing"] is False
    assert no_crossing["supporting_primary"] is False

    directional = analyze_pair(
        _pair(group="directional", q=-2.0, alpha=None),
        _points(crossing=False, directional=True),
        ANALYSIS,
    )
    assert directional["directional_control_violation"] is False
    assert directional["observed_critical_bracket"] is None


def test_threshold_equality_is_inactive_in_crossing_sequence() -> None:
    pair = _pair()
    points = _points()
    equality = next(item for item in points if item["realized_suppression"] == 0.5)
    assert equality["target_preactivation"] == 1.0
    assert equality["target_active"] is False
    result = analyze_pair(pair, points, ANALYSIS)
    assert result["observed_critical_bracket"]["lower_realized_suppression"] == 0.5


def test_outcome_precedence_distinguishes_supported_mixed_not_supported_and_empty() -> (
    None
):
    supporting = analyze_pair(_pair(), _points(), ANALYSIS)
    assert classify_outcome([supporting]) is ScientificOutcome.SUPPORTED

    near = analyze_pair(
        _pair(group="near_boundary", q=0.5, alpha=2.0),
        _points(),
        ANALYSIS,
    )
    assert classify_outcome([supporting, near]) is ScientificOutcome.MIXED

    unsupported = analyze_pair(_pair(), _points(crossing=False), ANALYSIS)
    assert classify_outcome([unsupported]) is ScientificOutcome.NOT_SUPPORTED
    assert classify_outcome([]) is ScientificOutcome.NO_ELIGIBLE_PAIRS


def test_aggregate_metrics_keep_undefined_values_null_with_reasons() -> None:
    empty = aggregate_analyses([])
    assert empty["primary_pair_count"] == 0
    assert empty["primary_full_ablation_crossing_precision"] is None
    assert empty["primary_precision_undefined_reason"] == "no_primary_pairs"
    assert empty["near_boundary_crossing_fraction"] is None
    assert empty["directional_violation_fraction"] is None
    assert empty["critical_suppression_spearman"] is None
    assert empty["critical_suppression_spearman_undefined_reason"]


def test_aggregate_spearman_is_undefined_for_one_observed_crossing() -> None:
    one = analyze_pair(_pair(), _points(), ANALYSIS)
    aggregate = aggregate_analyses([one])
    assert aggregate["critical_suppression_spearman"] is None
    assert aggregate["critical_suppression_spearman_pair_count"] == 1
