"""Offline synthetic and hostile-input tests for the Stage 1C-v2 artifact layer."""

from __future__ import annotations

import hashlib
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/stage1c_v2"))

import assemble_stage1c_artifacts as assembler  # noqa: E402
import validate_stage1c_v2_artifacts as validator  # noqa: E402

from cfsus.stage1c_v2.worker_result import build_detached_worker_result  # noqa: E402


def _protocol() -> dict[str, Any]:
    return {
        "scanner": {
            "selected_layers": list(range(18)),
            "selected_positions": [1, 2, 3, 4, 5],
            "feature_width": 16_384,
            "dense_oracle_chunk_size": 16_384,
            "canonical_chunk_size": 1_024,
            "top_k_per_group": 8,
            "global_top_k": 128,
        },
        "source_pool": {
            "selection": "all_exact_loaded_active_features_with_causal_target",
            "ordering": ["layer", "position", "feature_id"],
            "require_positive_activation": True,
            "require_strictly_earlier_layer": True,
            "require_causal_position": True,
            "raw_graph_input": "forbidden",
            "maximum_active_sources": 10_000,
        },
        "responses": {
            "method": "target_encoder_reverse_vjp_many_source_contraction",
            "convention": "attribution_matched_target_preactivation_pre_gate",
            "graph_edge_input": "forbidden",
            "target_batch_size": 8,
            "maximum_eligible_pairs": 500_000,
        },
        "engineering_calibration": {
            "endpoint_class": (
                "baseline_active_source_active_target_disjoint_from_inactive_pool"
            ),
            "pair_count": 4,
            "reference_method": "stage1b_independent_pairwise_targeted_vjp",
            "comparison": "exact_bf16_identity",
            "inactive_target_intervention_calls": 0,
        },
        "scoring": {
            "epsilon": 1.0e-12,
            "crossing_tolerance": 1.0e-9,
            "pair_seed": validator.PAIR_ID_SEED,
        },
        "selection": {
            "primary_maximum": 12,
            "near_boundary_maximum": 8,
            "directional_maximum": 8,
            "maximum_per_target": 1,
            "maximum_primary_per_source": 2,
            "primary_order": [
                "susceptibility_desc",
                "alpha_hat_asc",
                "target",
                "source",
            ],
            "near_order": ["distance_above_one_asc", "target", "source"],
            "directional_order": ["movement_over_margin_desc", "target", "source"],
            "prefer_unused_control_targets": True,
            "control_overlap_fallback": "deterministic_after_unique_exhausted",
        },
        "schedule": {
            "coarse_alphas": [0.0, 0.25, 0.5, 0.75, 1.0],
            "alpha_hat_offset": 0.015625,
            "maximum_bisection_steps": 8,
            "deduplicate_applied_bf16": True,
        },
        "intervention_regime": {
            "source_count": 1,
            "mapping": "desired=(1-alpha)*baseline",
            "freeze_attention": True,
            "constrained_layers": None,
            "target_clamp_allowed": False,
            "canonical_attempts": 1,
        },
        "analysis": {
            "minimum_nonzero_points": 3,
            "movement_sign_agreement_minimum": 0.80,
            "median_movement_sne_maximum": 0.50,
            "p95_movement_sne_maximum": 1.00,
            "critical_bracket_distance_maximum": 0.125,
            "undefined_metric_policy": "null_with_reason",
        },
    }


def _row(
    group: str,
    source: tuple[int, int, int],
    target: tuple[int, int, int],
    response: float,
    alpha: float | None,
) -> dict[str, Any]:
    source_ref = dict(zip(("layer", "position", "feature_id"), source, strict=True))
    target_ref = dict(zip(("layer", "position", "feature_id"), target, strict=True))
    q = -2.0 * response
    margin = 1.0
    tolerance = 1.0e-9
    if abs(margin) <= tolerance:
        status = "boundary_ambiguous"
    elif q <= 0.0:
        status = "not_crossing"
    elif q - margin > tolerance:
        status = "definitely_crossing"
    else:
        status = "not_crossing"
    requested = [0.0, 0.25, 0.5, 0.75, 1.0]
    if alpha is not None and 0.0 <= alpha <= 1.0:
        requested.extend((alpha - 0.015625, alpha, alpha + 0.015625))
    requested = sorted({max(0.0, min(1.0, value)) for value in requested})
    record = {
        "pair_id": validator.canonical_pair_id(source=source, target=target),
        "group": group,
        "source": source_ref,
        "target": target_ref,
        "source_activation": 2.0,
        "target_preactivation": 0.0,
        "target_threshold": 1.0,
        "margin": margin,
        "targeted_response": response,
        "q": q,
        "susceptibility": q / (margin + 1.0e-12),
        "predicted_alpha_star": alpha,
        "predicted_status": status,
        "requested_alphas": requested,
    }
    return record


def _point(pair: dict[str, Any], requested_alpha: float) -> dict[str, Any]:
    baseline = float(pair["source_activation"])
    desired = (1.0 - requested_alpha) * baseline
    applied = validator.bf16_round(desired)
    realized = 1.0 - applied / baseline
    z = float(pair["target_preactivation"]) + realized * float(pair["q"])
    active = z > float(pair["target_threshold"])
    mapping = {
        "requested_alpha": requested_alpha,
        "desired_high_precision": desired,
        "actual_bf16_value_passed": applied,
        "realized_suppression": realized,
    }
    return {
        "pair_id": pair["pair_id"],
        "group": pair["group"],
        "source": pair["source"],
        "target": pair["target"],
        "representative_requested_alpha": requested_alpha,
        "requested_alpha": requested_alpha,
        "desired_high_precision": desired,
        "requested_mappings": [mapping],
        "actual_bf16_value_passed": applied,
        "realized_suppression": realized,
        "collapsed_request_count": 1,
        "source_value_device": "mps:0",
        "source_value_dtype": "torch.bfloat16",
        "stage": "coarse_and_alpha_hat_schedule",
        "bisection_step": None,
        "target_preactivation": z,
        "target_threshold": pair["target_threshold"],
        "target_activation": z if active else 0.0,
        "target_active": active,
        "loaded_gate": "a=z*1[z>tau]",
        "threshold_equality_activity": "inactive",
        "source_activation_observation": (
            "actual_bf16_value_passed_to_absolute_intervention_tuple"
        ),
        "freeze_attention": True,
        "constrained_layers": None,
        "target_clamped": False,
        "logits_finite": True,
        "logits_shape": [1, 1],
        "predicted_target_preactivation": z,
        "predicted_target_activation": z if active else 0.0,
        "predicted_target_active": active,
        "target_preactivation_absolute_error": 0.0,
        "target_preactivation_symmetric_normalized_error": 0.0,
    }


def _fixture() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    groups = {
        "primary": [_row("primary", (0, 1, 1), (1, 1, 2), -1.0, 0.5)],
        "near_boundary": [_row("near_boundary", (0, 1, 3), (1, 1, 4), -0.4, 1.25)],
        "directional": [_row("directional", (0, 1, 5), (1, 2, 6), 0.0, None)],
    }
    prediction: dict[str, Any] = {
        "schema_version": 2,
        "artifact_type": validator.PREDICTION_TYPE,
        "status": "prediction_frozen_ready_for_commit",
        "experiment_class": validator.EXPERIMENT_CLASS,
        "base_commit": validator.BASE_COMMIT,
        "branch": validator.BRANCH,
        "pair_id_domain": validator.PAIR_ID_DOMAIN,
        "runtime_identity": {
            "backend": "nnsight",
            "device": "mps:0",
            "dtype": "torch.bfloat16",
            "model_identifier": "google/gemma-3-270m",
            "model_revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
            "transcoder_identifier": "mwhanna/gemma-scope-2-270m-pt",
            "transcoder_revision": "fada11860ac1d337c1e41e9da308798405b94c8e",
            "transcoder_subfolder": "transcoder_all/width_16k_l0_small",
            "layer_count": 18,
            "feature_width": 16_384,
            "upstream_revision": "8f1e2438df612464e229e44c4a00ff637bf9379b",
        },
        "prompt": {
            "id": validator.PROMPT_ID,
            "text": validator.PROMPT_TEXT,
            "token_ids": [2, 818, 5279, 529, 9405, 563],
        },
        "protocol": _protocol(),
        "baseline_pools": {"eligible_pair_count": 3},
        "selected_groups": groups,
        "selection_audit": {
            "primary_count": 1,
            "near_boundary_count": 1,
            "directional_count": 1,
            "near_overlap_fallback_count": 0,
            "directional_overlap_fallback_count": 0,
            "groups_disjoint": True,
            "primary_target_unique": True,
            "primary_source_cap": 2,
        },
        "prediction_only_guards": {
            "source_suppression_api_calls": 0,
            "prior_inactive_target_outcome_read": False,
            "intervention_worker_imported": False,
            "raw_graph_read": False,
            "raw_adjacency_read": False,
        },
        "protocol_file_sha256": {"synthetic_protocol.json": "a" * 64},
        "config_sha256": "b" * 64,
        "artifact_schema_sha256": "c" * 64,
        "claim_boundary": dict(assembler.CLAIM_BOUNDARY),
    }
    sweeps = []
    for row in (*groups["primary"], *groups["near_boundary"], *groups["directional"]):
        points = [_point(row, alpha) for alpha in row["requested_alphas"]]
        sweeps.append(
            {
                "pair_id": row["pair_id"],
                "group": row["group"],
                "source": row["source"],
                "target": row["target"],
                "target_preactivation": row["target_preactivation"],
                "target_threshold": row["target_threshold"],
                "q": row["q"],
                "predicted_alpha_star": row["predicted_alpha_star"],
                "predicted_status": row["predicted_status"],
                "baseline_source_activation": row["source_activation"],
                "point_count": len(points),
                "bisection_step_count": 0,
                "points": points,
            }
        )
    analyses = [
        validator._analysis(row, sweep["points"], prediction["protocol"]["analysis"])
        for row, sweep in zip(
            (*groups["primary"], *groups["near_boundary"], *groups["directional"]),
            sweeps,
            strict=True,
        )
    ]
    outcome = validator.scientific_outcome(analyses)
    aggregate = validator._aggregate(analyses)
    prediction_worker = {
        "schema_version": 2,
        "artifact_type": "stage1c_v2_prediction_worker",
        "status": "passed",
        "prediction_manifest": prediction,
        "asset_manifest": {"status": "verified"},
        "environment": {"machine": "synthetic"},
        "telemetry": {},
    }
    intervention_worker = build_detached_worker_result(
        sweeps,
        intervention_artifacts={
            "intervention_sweeps": {"pairs": sweeps},
            "crossing_summary": {
                "pairs": analyses,
                "aggregate_metrics": aggregate,
                "scientific_outcome": outcome,
            },
            "memory_timing_summary": {"telemetry": {}},
            "attempts": {
                "attempt_count": 1,
                "scientific_retry_count": 0,
                "intervention_required": True,
            },
        },
        canonical_source_suppression_api_calls=sum(
            len(item["points"]) for item in sweeps
        ),
        schema_version=2,
        artifact_type="stage1c_v2_intervention_worker",
        status="passed",
        prediction_manifest_sha256=hashlib.sha256(
            assembler._canonical_bytes(prediction)
        ).hexdigest(),
        scientific_outcome=outcome,
        attempt_count=1,
        asset_manifest={"status": "verified"},
        environment={"machine": "synthetic"},
        telemetry={},
    )
    return prediction, prediction_worker, intervention_worker


def _assembled_bundle(tmp_path: Path) -> Path:
    prediction, prediction_worker, intervention_worker = _fixture()
    result = assembler.records(
        prediction,
        prediction_worker,
        intervention_worker,
        execution=validator.BASE_COMMIT,
    )
    bundle = tmp_path.resolve() / "bundle"
    assembler.write_bundle(result, bundle)
    return bundle


def test_synthetic_nonempty_assembly_passes_standalone_validator(
    tmp_path: Path,
) -> None:
    bundle = _assembled_bundle(tmp_path)
    result = validator.validate_bundle(bundle, validator.BASE_COMMIT)
    assert result["status"] == "passed"
    assert result["point_count"] == 17
    assert result["api_call_count"] == 17


def test_worker_detachment_survives_cleanup_and_preserves_equal_pair_lists() -> None:
    _, _, worker = _fixture()
    top = worker["sweeps"]
    nested = worker["intervention_artifacts"]["intervention_sweeps"]["pairs"]
    assert top == nested and top is not nested
    top[0]["points"].clear()
    assert nested[0]["points"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda result: result["run_manifest.json"].update(
            canonical_source_suppression_api_calls=0
        ),
        lambda result: result["intervention_sweeps.json"]["pairs"].reverse(),
        lambda result: result["intervention_sweeps.json"]["pairs"][0].update(
            point_count=1
        ),
    ],
)
def test_core_invariants_reject_tampered_assembly(
    tmp_path: Path, mutation: Any
) -> None:
    prediction, prediction_worker, intervention_worker = _fixture()
    result = assembler.records(prediction, prediction_worker, intervention_worker)
    mutation(result)
    with pytest.raises((ValueError, validator.ValidationError)):
        assembler.write_bundle(result, tmp_path / "bad")


def test_no_eligible_terminal_result_is_zero_call_zero_sweep(tmp_path: Path) -> None:
    prediction, prediction_worker, _intervention_worker = _fixture()
    for group in prediction["selected_groups"].values():
        group.clear()
    prediction["selection_audit"].update(
        primary_count=0, near_boundary_count=0, directional_count=0
    )
    prediction_worker["prediction_manifest"] = prediction
    empty = build_detached_worker_result(
        [],
        intervention_artifacts={
            "intervention_sweeps": {"pairs": []},
            "crossing_summary": {
                "pairs": [],
                "aggregate_metrics": validator._aggregate([]),
                "scientific_outcome": "no_eligible_pairs",
            },
            "attempts": {
                "attempt_count": 0,
                "scientific_retry_count": 0,
                "intervention_required": False,
            },
        },
        canonical_source_suppression_api_calls=0,
        schema_version=2,
        artifact_type="stage1c_v2_intervention_worker",
        status="passed",
        scientific_outcome="no_eligible_pairs",
        attempt_count=0,
        asset_manifest={"status": "verified"},
        environment={"machine": "synthetic"},
    )
    result = assembler.records(prediction, prediction_worker, empty)
    assembler.write_bundle(result, tmp_path / "empty")
    assert validator.validate_bundle(tmp_path / "empty")["api_call_count"] == 0


def test_symlink_and_hardlink_bundle_members_are_rejected(tmp_path: Path) -> None:
    bundle = _assembled_bundle(tmp_path)
    target = bundle / "run_manifest.json"
    contents = target.read_bytes()
    target.unlink()
    target.symlink_to(bundle / "attempts.json")
    with pytest.raises(validator.ValidationError):
        validator.validate_bundle(bundle)
    target.unlink()
    target.write_bytes(contents)
    hardlink = bundle / "hardlink-source"
    os.link(target, hardlink)
    with pytest.raises(validator.ValidationError):
        validator.validate_bundle(bundle)


def test_zip_allowlist_comments_extras_and_traversal_are_rejected(
    tmp_path: Path,
) -> None:
    bundle = _assembled_bundle(tmp_path)
    valid_zip = tmp_path / "valid.zip"
    with zipfile.ZipFile(valid_zip, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in validator.ALLOWLIST:
            archive.writestr(name, (bundle / name).read_bytes())
    validator.validate_zip(valid_zip)

    commented = tmp_path / "comment.zip"
    with zipfile.ZipFile(commented, "w") as archive:
        archive.comment = b"forbidden"
        for name in validator.ALLOWLIST:
            archive.writestr(name, (bundle / name).read_bytes())
    with pytest.raises(validator.ValidationError):
        validator.validate_zip(commented)

    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as archive:
        for name in validator.ALLOWLIST[:-1]:
            archive.writestr(name, (bundle / name).read_bytes())
        archive.writestr(
            "../checksums.sha256", (bundle / "checksums.sha256").read_bytes()
        )
    with pytest.raises(validator.ValidationError):
        validator.validate_zip(traversal)


@pytest.mark.parametrize(
    "value",
    [
        {"api_key": "secret"},
        {"note": "Bearer abcdefghijklmnop"},
        # commit-safety: allow-test-fixture
        {"note": "/Users/example/private/file.json"},
        {"note": "line\nfeed"},
    ],
)
def test_secret_private_path_and_control_payloads_are_rejected(
    value: dict[str, str],
) -> None:
    with pytest.raises(validator.ValidationError):
        validator.scan_value(value)
