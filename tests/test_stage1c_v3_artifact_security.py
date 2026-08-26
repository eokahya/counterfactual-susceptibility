"""Offline synthetic and hostile-input tests for the Stage 1C-v3 artifact layer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/stage1c_v3"))


def _load_script(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts/stage1c_v3" / filename
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = _load_script(
    "stage1c_v3_artifact_security_validator", "validate_stage1c_v3_artifacts.py"
)
assembler = _load_script(
    "stage1c_v3_artifact_security_assembler", "assemble_stage1c_artifacts.py"
)
worker = _load_script(
    "stage1c_v4_production_worker", "run_stage1c_v3_intervention_worker.py"
)

from cfsus.stage1c_v3.execution_journal import CanonicalExecutionJournal  # noqa: E402
from cfsus.stage1c_v3.intervention import applied_plan_record  # noqa: E402
from cfsus.stage1c_v3.worker_result import build_detached_worker_result  # noqa: E402
from cfsus.types import (  # noqa: E402
    FeatureActivity,
    FeatureRef,
    MeasuredFeatureState,
)


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
            "tie_break": ["margin", "layer", "position", "feature_id"],
            "dense_oracle_lifecycle": "one_group_ephemeral",
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
            "scientific_retries": 0,
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
    historical_pairs, historical_endpoints = validator._historical_metadata()
    pair_key = source, target
    record["exact_pair_key"] = validator._pair_record(pair_key)
    record["endpoint_overlap_category"] = validator._overlap_category(
        source, target, historical_pairs, historical_endpoints
    )
    return record


def _point(
    pair: dict[str, Any], requested_alpha: float, *, call_index: int
) -> dict[str, Any]:
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
        "exact_pair_key": pair["exact_pair_key"],
        "endpoint_overlap_category": pair["endpoint_overlap_category"],
        "baseline_source_activation": pair["source_activation"],
        "baseline_target_preactivation": pair["target_preactivation"],
        "targeted_response": pair["targeted_response"],
        "q": pair["q"],
        "predicted_alpha_star": pair["predicted_alpha_star"],
        "predicted_status": pair["predicted_status"],
        "representative_requested_alpha": requested_alpha,
        "requested_alpha": requested_alpha,
        "desired_high_precision": desired,
        "requested_mappings": [mapping],
        "actual_bf16_value_passed": applied,
        "realized_suppression": realized,
        "collapsed_request_count": 1,
        "source_value_device": "mps:0",
        "source_value_dtype": "torch.bfloat16",
        "target_value_device": "mps:0",
        "target_value_dtype": "torch.bfloat16",
        "source_suppression_api_call_index": call_index,
        "point_elapsed_seconds": 0.01,
        "memory_stage_identity": f"synthetic_point_{call_index}",
        "finite_value_checks_passed": True,
        "finite_checks": {
            "applied_source": True,
            "logits": True,
            "preactivation_cache": True,
            "target_preactivation": True,
            "target_threshold": True,
            "target_activation": True,
        },
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


def _protocol_hashes() -> dict[str, str]:
    return json.loads(
        (
            ROOT
            / "results/stage1c_v3_preregistered_prospective_prediction"
            / "prediction_manifest.json"
        ).read_text(encoding="utf-8")
    )["protocol_file_sha256"]


def _asset_evidence() -> dict[str, Any]:
    return {
        "status": "verified",
        "download_performed": False,
        "network_accessed": False,
        "authentication_used": False,
        "authentication_value_recorded": False,
        "actual_total_bytes": 2_087_816_677,
        "model_total_bytes": 575_454_257,
        "transcoder_total_bytes": 1_512_362_420,
        "model_revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
        "transcoder_revision": "fada11860ac1d337c1e41e9da308798405b94c8e",
        "exact_allowlist_hashes_verified": True,
    }


def _environment_evidence() -> dict[str, Any]:
    return {
        "system": "Darwin",
        "machine": "arm64",
        "python": "3.11.13",
        "circuit-tracer": "0.5.2",
        "torch": "2.6.0",
        "nnsight": "0.6.1",
        "transformers": "4.57.3",
        "mps_built": True,
        "mps_available": True,
        "fallback_variable_present": False,
        "outer_autocast_enabled": False,
    }


def _peaks() -> dict[str, int]:
    return {
        "mps_current_bytes": 1,
        "mps_driver_bytes": 2,
        "process_rss_bytes": 3,
        "swap_used_bytes": 0,
        "swap_growth_bytes": 0,
        "minimum_available_memory_bytes": 8_589_934_592,
    }


def _telemetry(stages: list[str]) -> dict[str, Any]:
    return {
        "started_at_unix": 1.0,
        "finished_at_unix": 2.0,
        "sample_count": max(1, len(stages)),
        "sampling_interval_seconds": 1.0,
        "attempt_peaks": _peaks(),
        "stage_peaks": {stage: _peaks() for stage in stages or ["worker_start"]},
        "thermal_states": ["nominal"],
        "violations": [],
        "telemetry_failures": 0,
    }


def _supervisor() -> dict[str, Any]:
    return {
        "returncode": 0,
        "timed_out": False,
        "safety_terminated": False,
        "termination_signal": None,
        "telemetry_failures": 0,
        "sample_count": 1,
        "peak_process_group_rss_bytes": 3,
        "minimum_available_memory_bytes": 8_589_934_592,
        "peak_swap_growth_bytes": 0,
        "thermal_states": ["nominal"],
        "started_at_unix": 1.0,
        "finished_at_unix": 2.0,
    }


def _fixture() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    groups = {
        "primary": [_row("primary", (0, 1, 1), (1, 1, 2), -1.0, 0.5)],
        "near_boundary": [_row("near_boundary", (0, 1, 3), (1, 1, 4), -0.4, 1.25)],
        "directional": [_row("directional", (0, 1, 5), (1, 2, 6), 0.0, None)],
    }
    protocol_hashes = _protocol_hashes()
    overlap_counts = {name: 0 for name in validator.OVERLAP_CATEGORIES}
    for row in (*groups["primary"], *groups["near_boundary"], *groups["directional"]):
        overlap_counts[row["endpoint_overlap_category"]] += 1
    historical_pairs, historical_endpoints = validator._historical_metadata()
    del historical_pairs
    prediction: dict[str, Any] = {
        "schema_version": 3,
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
            "token_ids": [2, 818, 5279, 529, 32649, 563],
        },
        "prompt_derivation": {
            "algorithm": "sha256_prefix16_mod_pool_length",
            "base_commit": validator.BASE_COMMIT,
            "salt": validator.PROMPT_SALT,
            "message": f"{validator.BASE_COMMIT}|{validator.PROMPT_SALT}",
            "sha256_hex": validator.PROMPT_SHA256,
            "index": validator.PROMPT_INDEX,
            "prompt": validator.PROMPT_TEXT,
            "prompt_id": validator.PROMPT_ID,
            "pool": validator.PROMPT_POOL,
        },
        "historical_independence": {
            "source_manifest_path": validator.HISTORICAL_SOURCE.as_posix(),
            "source_manifest_sha256": validator.HISTORICAL_SOURCE_SHA256,
            "source_manifest_git_blob_sha1": validator.HISTORICAL_SOURCE_BLOB_SHA1,
            "source_manifest_freeze_commit": validator.HISTORICAL_FREEZE_COMMIT,
            "denylist_path": validator.DENYLIST_PATH.as_posix(),
            "denylist_sha256": validator.DENYLIST_SHA256,
            "exact_pair_count": 28,
            "historical_endpoint_count": len(historical_endpoints),
            "mask_applied_before_ranking": True,
            "endpoint_overlap_policy": "audit_only",
            "historical_intervention_outcome_read": False,
            "v2_temporary_baseline_artifact_read": False,
        },
        "protocol": _protocol(),
        "baseline_pools": {
            "scanner_candidate_count": 3,
            "eligible_target_count": 3,
            "excluded_no_causal_source_target_count": 0,
            "target_pool_sha256": "1" * 64,
            "raw_active_source_count": 3,
            "eligible_source_count": 3,
            "source_pool_sha256": "2" * 64,
            "eligible_pair_count_before_historical_mask": 3,
            "excluded_exact_historical_pair_count": 0,
            "eligible_pair_count_after_historical_mask": 3,
            "pair_score_sha256_before_historical_mask": "3" * 64,
            "pair_score_sha256_after_historical_mask": "3" * 64,
            "predicted_status_counts": {
                "boundary_ambiguous": 0,
                "definitely_crossing": 1,
                "not_crossing": 2,
            },
            "q_sign_counts": {"positive": 2, "zero": 1, "negative": 0},
            "complete_derivative_matrix_persisted": False,
            "dense_scanner_arrays_persisted": False,
            "many_source_vjp_engineering_calibration": {
                **_protocol()["engineering_calibration"],
                "passed": True,
            },
            "scanner_dense_oracle_validation": {
                "group_count": 90,
                "exact_dense_oracle_identity_and_order": True,
                "bounded_oracle_recall": 1.0,
                "dense_oracle_persisted": False,
            },
        },
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
            "exact_pair_mask_applied_before_ranking": True,
            "selected_exact_pair_overlap_count": 0,
            "endpoint_overlap_category_counts": overlap_counts,
            "endpoint_overlap_used_for_ranking_or_quota": False,
        },
        "prediction_only_guards": {
            "source_suppression_api_calls": 0,
            "prior_inactive_target_outcome_read": False,
            "historical_intervention_outcome_read": False,
            "v2_temporary_baseline_artifact_read": False,
            "intervention_worker_imported": False,
            "raw_graph_read": False,
            "raw_adjacency_read": False,
        },
        "protocol_file_sha256": protocol_hashes,
        "config_sha256": protocol_hashes[
            "configs/stage1c_v3_preregistered_prospective_prediction.yaml"
        ],
        "artifact_schema_sha256": protocol_hashes[
            "configs/stage1c_v3_preregistered_prospective_prediction_artifact_schema.json"
        ],
        "claim_boundary": dict(assembler.CLAIM_BOUNDARY),
    }
    sweeps = []
    call_index = 0
    for row in (*groups["primary"], *groups["near_boundary"], *groups["directional"]):
        points = []
        for alpha in row["requested_alphas"]:
            call_index += 1
            points.append(_point(row, alpha, call_index=call_index))
        sweeps.append(
            {
                "pair_id": row["pair_id"],
                "group": row["group"],
                "source": row["source"],
                "target": row["target"],
                "exact_pair_key": row["exact_pair_key"],
                "endpoint_overlap_category": row["endpoint_overlap_category"],
                "source_activation": row["source_activation"],
                "targeted_response": row["targeted_response"],
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
        "schema_version": 3,
        "artifact_type": "stage1c_v3_prediction_worker",
        "status": "passed",
        "prediction_manifest": prediction,
        "asset_manifest": _asset_evidence(),
        "environment": _environment_evidence(),
        "telemetry": _telemetry(["prediction_dense_oracle"]),
    }
    point_stages = [
        point["memory_stage_identity"] for item in sweeps for point in item["points"]
    ]
    intervention_telemetry = _telemetry(point_stages)
    intervention_worker = build_detached_worker_result(
        sweeps,
        intervention_artifacts={
            "intervention_sweeps": {"pairs": sweeps},
            "crossing_summary": {
                "pairs": analyses,
                "aggregate_metrics": aggregate,
                "scientific_outcome": outcome,
            },
            "memory_timing_summary": {"telemetry": intervention_telemetry},
            "attempts": {
                "attempt_count": 1,
                "scientific_retry_count": 0,
                "intervention_required": True,
            },
        },
        canonical_source_suppression_api_calls=sum(
            len(item["points"]) for item in sweeps
        ),
        instrumented_source_suppression_api_calls=sum(
            len(item["points"]) for item in sweeps
        ),
        schema_version=3,
        artifact_type="stage1c_v3_intervention_worker",
        status="passed",
        prediction_manifest_sha256=hashlib.sha256(
            assembler._canonical_bytes(prediction)
        ).hexdigest(),
        scientific_outcome=outcome,
        attempt_count=1,
        asset_manifest=_asset_evidence(),
        environment=_environment_evidence(),
        telemetry=intervention_telemetry,
    )
    return prediction, prediction_worker, intervention_worker


def _assembled_bundle(tmp_path: Path) -> Path:
    prediction, prediction_worker, intervention_worker = _fixture()
    result = assembler.records(
        prediction,
        prediction_worker,
        intervention_worker,
        prediction_supervisor=_supervisor(),
        intervention_supervisor=_supervisor(),
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


class _FakeDevice:
    type = "mps"


class _FakeTensor:
    def __init__(self, value: float, dtype: object) -> None:
        self._value = validator.bf16_round(float(value))
        self.device = _FakeDevice()
        self.dtype = dtype

    def reshape(self, *_: object) -> _FakeTensor:
        return self

    def numel(self) -> int:
        return 1

    def item(self) -> float:
        return self._value


class _FakeTorch:
    bfloat16 = object()
    Tensor = _FakeTensor

    @classmethod
    def tensor(cls, value: float, *, device: str, dtype: object) -> _FakeTensor:
        assert device == "mps" and dtype is cls.bfloat16
        return _FakeTensor(value, dtype)


class _SyntheticBackend:
    def __init__(
        self,
        groups: dict[str, list[dict[str, Any]]],
        recorder: Any,
    ) -> None:
        self.groups = groups
        self.recorder = recorder
        self.source_suppression_api_calls = 0

    def _row_for(self, feature: FeatureRef) -> tuple[dict[str, Any], bool]:
        for rows in self.groups.values():
            for row in rows:
                if feature == FeatureRef(**row["source"]):
                    return row, True
                if feature == FeatureRef(**row["target"]):
                    return row, False
        raise AssertionError("unexpected synthetic feature")

    def measure_states(
        self, features: tuple[FeatureRef, ...]
    ) -> dict[FeatureRef, MeasuredFeatureState]:
        result: dict[FeatureRef, MeasuredFeatureState] = {}
        for feature in features:
            row, is_source = self._row_for(feature)
            if is_source:
                activation = float(row["source_activation"])
                result[feature] = MeasuredFeatureState(
                    feature=feature,
                    preactivation=activation,
                    activation=activation,
                    threshold=0.0,
                    activity=FeatureActivity.ACTIVE,
                    device="mps:0",
                    dtype="torch.bfloat16",
                )
            else:
                result[feature] = MeasuredFeatureState(
                    feature=feature,
                    preactivation=float(row["target_preactivation"]),
                    activation=0.0,
                    threshold=float(row["target_threshold"]),
                    activity=FeatureActivity.INACTIVE,
                    device="mps:0",
                    dtype="torch.bfloat16",
                )
        return result

    def measure_point(
        self,
        pair: dict[str, Any],
        plan: Any,
        *,
        freeze_attention: bool,
        constrained_layers: None,
        stage: str,
    ) -> dict[str, Any]:
        assert freeze_attention is True and constrained_layers is None
        next_index = self.source_suppression_api_calls + 1
        self.recorder(pair, next_index)
        self.source_suppression_api_calls = next_index
        z = float(pair["target_preactivation"]) + float(
            plan.realized_suppression
        ) * float(pair["q"])
        tau = float(pair["target_threshold"])
        active = z > tau
        result = applied_plan_record(plan)
        result.update(
            {
                "source_suppression_api_call_index": next_index,
                "point_elapsed_seconds": 0.01,
                "stage": stage,
                "target_preactivation": z,
                "target_threshold": tau,
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
                "source_value_device": "mps:0",
                "source_value_dtype": "torch.bfloat16",
                "target_value_device": "mps:0",
                "target_value_dtype": "torch.bfloat16",
                "finite_checks": {
                    "applied_source": True,
                    "logits": True,
                    "preactivation_cache": True,
                    "target_preactivation": True,
                    "target_threshold": True,
                    "target_activation": True,
                },
                "logits_finite": True,
                "logits_shape": [1, 6, 262_144],
            }
        )
        return result


class _SyntheticSampler:
    def __init__(self) -> None:
        self.stages: list[str] = []

    @contextmanager
    def stage(self, name: str) -> Any:
        self.stages.append(name)
        yield


def test_actual_production_sweep_path_reaches_standalone_validator(
    tmp_path: Path,
) -> None:
    prediction, prediction_worker, _ = _fixture()
    groups = prediction["selected_groups"]
    pair_ids = tuple(
        row["pair_id"]
        for group in ("primary", "near_boundary", "directional")
        for row in groups[group]
    )
    journal_path = tmp_path / "point_journal.jsonl"
    lock_path = tmp_path / "canonical_attempt_v1.lock"
    journal = CanonicalExecutionJournal(
        journal_path,
        lock_path,
        frozen_pair_ids=pair_ids,
        pre_intervention_commit=validator.EXECUTION_START_COMMIT,
        prediction_manifest_sha256="f" * 64,
    )
    sampler = _SyntheticSampler()
    backend = _SyntheticBackend(groups, journal.before_source_suppression)
    try:
        sweeps, states = worker._execute_production_sweeps(
            backend,
            groups,
            token_count=6,
            config={"schedule": prediction["protocol"]["schedule"]},
            torch=_FakeTorch,
            sampler=sampler,
            journal=journal,
        )
        point_count = sum(item["point_count"] for item in sweeps)
        journal.verify_complete(expected_point_count=point_count)
    finally:
        journal.close()
    assert len(states) == 6
    assert {item["group"] for item in sweeps} == {
        "primary",
        "near_boundary",
        "directional",
    }
    journal_rows = [json.loads(line) for line in journal_path.read_text().splitlines()]
    assert sum(row["record_type"] == "point_completed" for row in journal_rows) == (
        point_count
    )
    assert backend.source_suppression_api_calls == point_count

    telemetry = _telemetry(sampler.stages)
    artifacts, outcome = worker._artifact_bundle(
        sweeps, {"analysis": prediction["protocol"]["analysis"]}, telemetry
    )
    intervention_worker = build_detached_worker_result(
        sweeps,
        intervention_artifacts=artifacts,
        canonical_source_suppression_api_calls=point_count,
        instrumented_source_suppression_api_calls=point_count,
        schema_version=3,
        artifact_type="stage1c_v3_intervention_worker",
        status="passed",
        prediction_manifest_sha256=hashlib.sha256(
            assembler._canonical_bytes(prediction)
        ).hexdigest(),
        scientific_outcome=outcome,
        attempt_count=1,
        asset_manifest=_asset_evidence(),
        environment=_environment_evidence(),
        telemetry=telemetry,
    )
    records = assembler.records(
        prediction,
        prediction_worker,
        intervention_worker,
        prediction_supervisor=_supervisor(),
        intervention_supervisor=_supervisor(),
        execution=validator.EXECUTION_START_COMMIT,
    )
    bundle = tmp_path / "bundle-production-path"
    assembler.write_bundle(records, bundle)
    result = validator.validate_bundle(bundle, validator.EXECUTION_START_COMMIT)
    assert result == {
        "status": "passed",
        "artifact_count": 10,
        "verdict": validator.COMPLETED_STATUS,
        "point_count": point_count,
        "api_call_count": point_count,
    }


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
    result = assembler.records(
        prediction,
        prediction_worker,
        intervention_worker,
        prediction_supervisor=_supervisor(),
        intervention_supervisor=_supervisor(),
    )
    mutation(result)
    with pytest.raises((ValueError, validator.ValidationError)):
        assembler.write_bundle(result, tmp_path / "bad")


def test_no_eligible_terminal_result_is_zero_call_zero_sweep(tmp_path: Path) -> None:
    prediction, prediction_worker, _intervention_worker = _fixture()
    for group in prediction["selected_groups"].values():
        group.clear()
    prediction["selection_audit"].update(
        primary_count=0,
        near_boundary_count=0,
        directional_count=0,
        endpoint_overlap_category_counts={
            name: 0 for name in validator.OVERLAP_CATEGORIES
        },
    )
    prediction["baseline_pools"]["predicted_status_counts"] = {
        "boundary_ambiguous": 0,
        "definitely_crossing": 0,
        "not_crossing": 3,
    }
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
        instrumented_source_suppression_api_calls=0,
        schema_version=3,
        artifact_type="stage1c_v3_intervention_worker",
        status="passed",
        scientific_outcome="no_eligible_pairs",
        attempt_count=0,
        asset_manifest=_asset_evidence(),
        environment=_environment_evidence(),
    )
    result = assembler.records(
        prediction,
        prediction_worker,
        empty,
        prediction_supervisor=_supervisor(),
        intervention_supervisor=_supervisor(),
    )
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


def test_prediction_validator_rejects_exact_historical_pair_and_outcome_alias() -> None:
    prediction, _, _ = _fixture()
    historical_pairs, _ = validator._historical_metadata()
    source, target = next(iter(historical_pairs))
    row = prediction["selected_groups"]["primary"][0]
    row["source"] = validator._feature_record(source)
    row["target"] = validator._feature_record(target)
    row["exact_pair_key"] = validator._pair_record((source, target))
    with pytest.raises(validator.ValidationError, match="historical exact pair"):
        validator.scan_prediction(prediction)

    prediction, _, _ = _fixture()
    prediction["selected_groups"]["primary"][0]["Observed-Outcome"] = "forbidden"
    with pytest.raises(validator.ValidationError, match="outcome field"):
        validator.scan_prediction(prediction)


def test_standalone_prediction_mode_validates_canonical_manifest(
    tmp_path: Path,
) -> None:
    prediction, _, _ = _fixture()
    path = tmp_path.resolve() / "prediction_manifest.json"
    path.write_bytes(assembler._canonical_bytes(prediction))
    completed = validator.main(["--prediction", str(path)])
    assert completed == 0
