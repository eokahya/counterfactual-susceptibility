from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import zipfile
from pathlib import Path

import pytest

from cfsus.stage1c.analysis import aggregate_analyses

SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "stage1c" / "validate_stage1c_artifacts.py"
)
SPEC = importlib.util.spec_from_file_location("stage1c_validator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_strict_json_rejects_duplicate_nonfinite_and_forbidden() -> None:
    with pytest.raises(MODULE.ValidationError, match="duplicate"):
        MODULE.strict_json(b'{"x": 1, "x": 2}', "x.json")
    with pytest.raises(MODULE.ValidationError, match="non-finite"):
        MODULE.strict_json(b'{"x": NaN}', "x.json")
    with pytest.raises(MODULE.ValidationError, match="forbidden payload"):
        MODULE.strict_json(b'{"raw_graph": true}', "x.json")


def _minimal_bundle(path: Path) -> None:
    path.mkdir()
    for name in MODULE.JSON_NAMES:
        (path / name).write_text("{}\n", encoding="utf-8")
    lines = [
        f"{hashlib.sha256((path / name).read_bytes()).hexdigest()}  {name}"
        for name in sorted(MODULE.JSON_NAMES)
    ]
    (path / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_bundle_rejects_extra_symlink_and_hardlink(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _minimal_bundle(bundle)
    (bundle / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(MODULE.ValidationError, match="allowlist"):
        MODULE.load_bundle(bundle)
    (bundle / "extra.json").unlink()
    source = bundle / "run_manifest.json"
    linked = bundle / "asset_manifest.json"
    linked.unlink()
    linked.symlink_to(source)
    with pytest.raises(MODULE.ValidationError, match="regular"):
        MODULE.load_bundle(bundle)


def test_bundle_rejects_hardlink_and_private_text(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _minimal_bundle(bundle)
    source = bundle / "run_manifest.json"
    target = bundle / "asset_manifest.json"
    target.unlink()
    os.link(source, target)
    with pytest.raises(MODULE.ValidationError, match="regular"):
        MODULE.load_bundle(bundle)
    target.unlink()
    target.write_text('{"note":"/Users/example/private"}\n', encoding="utf-8")
    (bundle / "checksums.sha256").write_text(
        "\n".join(
            f"{hashlib.sha256((bundle / name).read_bytes()).hexdigest()}  {name}"
            for name in sorted(MODULE.JSON_NAMES)
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(MODULE.ValidationError, match="secret or private"):
        MODULE.load_bundle(bundle)


def test_prediction_manifest_forbids_intervention_fields() -> None:
    with pytest.raises(MODULE.ValidationError, match="intervention field"):
        MODULE.scan_prediction(
            {
                "status": "prediction_frozen_ready_for_commit",
                "base_commit": "efbf70a7e462e640a0e1819a93f3b92727bbd193",
                "observed_crossing": False,
            }
        )


def test_hostile_zip_paths_and_links_are_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "hostile.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../escape.json", "x")
    with pytest.raises(MODULE.ValidationError, match="traversal"):
        MODULE.validate_zip(archive)


def test_standalone_validator_accepts_exact_no_primary_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "valid"
    bundle.mkdir()
    execution = "1" * 40
    hashes = {f"protocol/{index}.py": "a" * 64 for index in range(16)}
    selection = {
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
        "directional_order": [
            "movement_over_margin_desc",
            "target",
            "source",
        ],
        "prefer_unused_control_targets": True,
        "control_overlap_fallback": "deterministic_after_unique_exhausted",
    }
    prediction = {
        "schema_version": 1,
        "artifact_type": "stage1c_prediction_manifest",
        "status": "prediction_frozen_ready_for_commit",
        "experiment_class": "stage1c_first_prospective_prediction",
        "base_commit": "efbf70a7e462e640a0e1819a93f3b92727bbd193",
        "branch": "stage-1c-first-prospective-prediction",
        "runtime_identity": {
            "backend": "nnsight",
            "device": "mps:0",
            "dtype": "torch.bfloat16",
            "model_revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
            "transcoder_revision": "fada11860ac1d337c1e41e9da308798405b94c8e",
            "transcoder_subfolder": "transcoder_all/width_16k_l0_small",
            "upstream_revision": "8f1e2438df612464e229e44c4a00ff637bf9379b",
        },
        "prompt": {
            "id": "pilot",
            "text": "The capital of France is",
            "token_ids": [2, 818, 5279, 529, 7001, 563],
        },
        "protocol": {
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
                "require_positive_activation": True,
                "require_strictly_earlier_layer": True,
                "require_causal_position": True,
                "raw_graph_input": "forbidden",
                "maximum_active_sources": 10_000,
            },
            "responses": {
                "method": "target_encoder_reverse_vjp_many_source_contraction",
                "graph_edge_input": "forbidden",
                "target_batch_size": 8,
                "maximum_eligible_pairs": 500_000,
            },
            "scoring": {
                "epsilon": 1.0e-12,
                "crossing_tolerance": 1.0e-9,
                "pair_seed": "stage1c-first-prospective-v1",
            },
            "selection": selection,
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
                "movement_sign_agreement_minimum": 0.8,
                "median_movement_sne_maximum": 0.5,
                "p95_movement_sne_maximum": 1.0,
                "critical_bracket_distance_maximum": 0.125,
                "undefined_metric_policy": "null_with_reason",
            },
        },
        "baseline_pools": {
            "many_source_vjp_engineering_calibration": {
                "pair_count": 4,
                "comparison": "exact_bf16_identity",
                "passed": True,
                "inactive_target_intervention_calls": 0,
            }
        },
        "selected_groups": {
            "primary": [],
            "near_boundary": [],
            "directional": [],
        },
        "selection_audit": {
            "primary_count": 0,
            "near_boundary_count": 0,
            "directional_count": 0,
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
        "protocol_file_sha256": hashes,
    }
    aggregate = aggregate_analyses([])
    prediction_bytes = (
        json.dumps(prediction, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    readiness = {
        "stage1b_measurement_primitives": "completed",
        "stage1c_first_prediction": "completed",
        "stage1c_scientific_outcome": "no_eligible_pairs",
        "counterfactual_susceptibility_result": "none",
        "gate_crossing_result": "none",
        "behavioral_importance_result": "none",
        "mediation_result": "none",
        "official_bf16_reproduction": "pending",
        "reference_clt_reproduction": "pending",
        "paper_results_readiness": False,
    }
    claim = {
        key: readiness[key]
        for key in (
            "behavioral_importance_result",
            "mediation_result",
            "official_bf16_reproduction",
            "reference_clt_reproduction",
            "paper_results_readiness",
        )
    }
    supervisor = {
        "returncode": 0,
        "timed_out": False,
        "safety_terminated": False,
        "telemetry_failures": 0,
    }
    artifacts = {
        "run_manifest.json": {
            "status": "completed_stage1c_first_prospective_prediction",
            "verdict": "completed_stage1c_first_prospective_prediction",
            "scientific_outcome": "no_eligible_pairs",
            "branch": "stage-1c-first-prospective-prediction",
            "base_commit": "efbf70a7e462e640a0e1819a93f3b92727bbd193",
            "execution_commit": execution,
            "pre_intervention_commit": execution,
            "fresh_canonical_run": False,
            "intervention_required": False,
            "canonical_source_suppression_api_calls": 0,
            "scientific_retry_count": 0,
            "prediction_manifest_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
            "claim_boundary": claim,
            "readiness": readiness,
            "secondary_regime": {"status": "not_implemented", "result": None},
        },
        "asset_manifest.json": {
            "status": "verified",
            "download_performed": False,
            "network_accessed": False,
            "authentication_used": False,
            "authentication_value_recorded": False,
            "exact_allowlist_hashes_verified": True,
            "actual_total_bytes": 2_087_816_677,
        },
        "environment_manifest.json": {
            "status": "passed",
            "execution_commit": execution,
            "accelerator": {
                "device": "mps:0",
                "dtype": "torch.bfloat16",
                "fallback_variable_present": False,
                "outer_autocast_enabled": False,
                "scientific_tensor_device": "mps",
            },
        },
        "prediction_manifest.json": prediction,
        "intervention_sweeps.json": {"pairs": []},
        "crossing_summary.json": {
            "scientific_outcome": "no_eligible_pairs",
            "pairs": [],
            "aggregate": aggregate,
        },
        "local_linearity_summary.json": {
            "point_count": 0,
            "median_symmetric_normalized_error": None,
            "p95_symmetric_normalized_error": None,
        },
        "memory_timing_summary.json": {
            "prediction_supervisor": supervisor,
            "intervention_supervisor": supervisor,
        },
        "attempts.json": {
            "prediction_attempts": 1,
            "intervention_attempts": 0,
            "scientific_retry_count": 0,
            "intervention_required": False,
        },
    }
    for name, value in artifacts.items():
        data = (
            prediction_bytes
            if name == "prediction_manifest.json"
            else (
                json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
            ).encode()
        )
        (bundle / name).write_bytes(data)
    checksum_lines = [
        f"{hashlib.sha256((bundle / name).read_bytes()).hexdigest()}  {name}"
        for name in sorted(MODULE.JSON_NAMES)
    ]
    (bundle / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    assert MODULE.validate_bundle(bundle, execution)["status"] == "passed"
