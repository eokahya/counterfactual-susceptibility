from __future__ import annotations

from typing import Any

from cfsus.stage1c_v3.analysis import aggregate_analyses, analyze_pair, classify_outcome


def _pair(group: str = "primary", *, crossing: bool = True) -> dict[str, Any]:
    baseline = {
        "realized_suppression": 0.0,
        "target_preactivation": 0.0,
        "target_active": False,
    }
    full = {
        "realized_suppression": 1.0,
        "target_preactivation": 2.0 if crossing else 0.5,
        "target_active": crossing,
    }
    return {
        "pair_id": f"{group}-pair",
        "group": group,
        "target_preactivation": 0.0,
        "target_threshold": 1.0,
        "q": 2.0,
        "predicted_alpha_star": 0.5,
        "predicted_status": "definitely_crossing",
        "points": [baseline, full],
    }


def test_v3_analysis_classification_and_empty_terminal() -> None:
    pair = _pair()
    result = analyze_pair(
        pair,
        pair["points"],
        {
            "minimum_nonzero_points": 1,
            "movement_sign_agreement_minimum": 0.8,
            "median_movement_sne_maximum": 0.5,
            "p95_movement_sne_maximum": 1.0,
            "critical_bracket_distance_maximum": 0.125,
        },
    )
    assert result["supporting_primary"] is True
    assert classify_outcome([]).value == "no_eligible_pairs"
    assert aggregate_analyses([])["primary_pair_count"] == 0


def test_v3_analysis_requires_exact_full_ablation() -> None:
    pair = _pair()
    points = pair["points"][:1]
    try:
        analyze_pair(
            pair,
            points,
            {
                "minimum_nonzero_points": 1,
                "movement_sign_agreement_minimum": 0.8,
                "median_movement_sne_maximum": 0.5,
                "p95_movement_sne_maximum": 1.0,
                "critical_bracket_distance_maximum": 0.125,
            },
        )
    except ValueError as error:
        assert "full ablation" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("analysis accepted a sweep without full ablation")
