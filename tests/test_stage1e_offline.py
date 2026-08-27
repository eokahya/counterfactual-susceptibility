from __future__ import annotations

from pathlib import Path

import pytest

from cfsus.stage1d.validation import validate_bundle as validate_stage1d_bundle
from cfsus.stage1e.offline import (
    OUTPUT_DIRECTORY,
    STAGE1D_DIRECTORY,
    compute_offline_analysis,
    estimate_e0,
    estimate_e1,
    estimate_e2,
    select_serialized_probe,
)
from cfsus.stage1e.validation import publish_offline_bundle, validate_offline_bundle


def _probe(alpha: float, z: float) -> dict[str, float]:
    return {
        "nominal_requested_alpha": alpha,
        "desired_source_activation": 1.0 - alpha,
        "applied_bf16_source_activation": 1.0 - alpha,
        "realized_alpha": alpha,
        "target_preactivation": z,
    }


def test_frozen_estimator_formulas_and_abstentions() -> None:
    assert estimate_e0(margin=0.25, q=0.5)["predicted_alpha"] == 0.5
    assert estimate_e0(margin=0.25, q=0.0)["status"] == "abstained"

    e1 = estimate_e1(margin=0.25, baseline_z=1.0, probe=_probe(0.25, 1.125))
    assert e1["predicted_alpha"] == 0.5
    assert (
        estimate_e1(margin=0.25, baseline_z=1.0, probe=_probe(0.25, 0.875))["status"]
        == "abstained"
    )

    e2 = estimate_e2(
        margin=0.75,
        baseline_z=1.0,
        first_probe=_probe(0.25, 1.3125),
        second_probe=_probe(0.5, 1.75),
    )
    assert e2["predicted_alpha"] == pytest.approx(0.5)
    no_root = estimate_e2(
        margin=0.75,
        baseline_z=1.0,
        first_probe=_probe(0.25, 0.9),
        second_probe=_probe(0.5, 0.7),
    )
    assert no_root["status"] == "abstained"


def test_quantization_aware_probe_uses_first_serialized_applied_value() -> None:
    pair = {"source_activation": 1.0}
    points = [
        {
            "actual_bf16_value_passed": 0.75,
            "realized_suppression": 0.25,
            "target_preactivation": 1.5,
        }
    ]
    selected = select_serialized_probe(pair, points, (0.125, 0.1875, 0.25))
    assert selected is not None
    assert selected["nominal_requested_alpha"] == 0.25
    assert selected["realized_alpha"] == 0.25


def test_real_stage1d_offline_gate_and_standalone_validation(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    stage1d = repository / STAGE1D_DIRECTORY
    stage1d_validation = validate_stage1d_bundle(repository, stage1d)
    analysis = compute_offline_analysis(stage1d, stage1d_validation=stage1d_validation)
    assert analysis["terminal_status"] == "completed_stage1e_offline_negative"
    assert analysis["selected_estimator"] is None
    assert analysis["estimator_metrics"]["E0"]["eligible_pair_count"] == 12
    assert analysis["estimator_metrics"]["E1"]["eligible_pair_count"] == 9
    assert analysis["estimator_metrics"]["E2"]["eligible_pair_count"] == 7
    assert analysis["offline_gates"]["E1"]["passed"] is False
    assert analysis["offline_gates"]["E2"]["passed"] is False
    output = tmp_path / OUTPUT_DIRECTORY.name
    publish_offline_bundle(output, analysis)
    result = validate_offline_bundle(repository, stage1d, output)
    assert result["status"] == "passed"
    assert result["phase_a_model_calls"] == 0
    assert result["phase_b_model_calls"] == 0
