from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from cfsus.reproduction.artifacts import (
    ArtifactValidationError,
    write_checksum_manifest_atomic,
    write_json_atomic,
)
from cfsus.stage1b_artifacts import (
    ARTIFACT_ALLOWLIST,
    load_bundle,
    strict_json_bytes,
    validate_bundle_structure,
    validate_stage1b_artifacts,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STAGE1B_SCRIPTS = REPOSITORY_ROOT / "scripts/stage1b"
if str(STAGE1B_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STAGE1B_SCRIPTS))

from assemble_stage1b_artifacts import _records  # noqa: E402


def _minimal_bundle(directory: Path) -> None:
    directory.mkdir()
    json_names = sorted(ARTIFACT_ALLOWLIST - {"checksums.sha256"})
    entries: list[str] = []
    for name in json_names:
        data = b'{"schema_version":1}\n'
        (directory / name).write_bytes(data)
        entries.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
    (directory / "checksums.sha256").write_text(
        "\n".join(entries) + "\n", encoding="utf-8"
    )


def test_strict_json_rejects_duplicate_keys_nonfinite_and_forbidden_payloads() -> None:
    with pytest.raises(ArtifactValidationError, match="duplicate JSON key"):
        strict_json_bytes(b'{"x":1,"x":2}', label="duplicate.json")
    with pytest.raises(ArtifactValidationError, match="non-finite"):
        strict_json_bytes(b'{"x":NaN}', label="nonfinite.json")
    with pytest.raises(ArtifactValidationError, match="forbidden payload key"):
        strict_json_bytes(b'{"raw_graph":false}', label="graph.json")
    with pytest.raises(ArtifactValidationError, match="private home"):
        strict_json_bytes(b'{"path":"/Users/example/private/cache"}', label="path.json")


def test_exact_allowlist_and_snapshot_checksum_validation(tmp_path: Path) -> None:
    directory = tmp_path / "bundle"
    _minimal_bundle(directory)
    artifacts = load_bundle(directory)
    assert set(artifacts) == ARTIFACT_ALLOWLIST - {"checksums.sha256"}
    (directory / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="exact allowlist"):
        validate_bundle_structure(directory)


def test_checksum_mutation_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "bundle"
    _minimal_bundle(directory)
    (directory / "run_manifest.json").write_text(
        '{"schema_version":2}\n', encoding="utf-8"
    )
    with pytest.raises(ArtifactValidationError, match="checksum mismatch"):
        load_bundle(directory)


def test_hardlinks_and_symlinks_are_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "bundle"
    _minimal_bundle(directory)
    target = directory / "run_manifest.json"
    linked = tmp_path / "linked.json"
    os.link(target, linked)
    with pytest.raises(ArtifactValidationError, match="single-link regular"):
        validate_bundle_structure(directory)
    linked.unlink()
    target.unlink()
    target.symlink_to(directory / "asset_manifest.json")
    with pytest.raises(ArtifactValidationError, match="single-link regular"):
        validate_bundle_structure(directory)


def test_secret_and_private_path_text_scan_is_independent_of_json_keys(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "bundle"
    _minimal_bundle(directory)
    target = directory / "attempts.json"
    target.write_text(
        json.dumps({"note": "Bearer abcdefghijklmnopqrstuvwxyz"}),
        encoding="utf-8",
    )
    with pytest.raises(ArtifactValidationError, match="secret or private path"):
        validate_bundle_structure(directory)


def _synthetic_complete_inputs() -> tuple[
    dict[str, object], dict[str, object], dict[str, object]
]:
    canonical_ids = [
        hashlib.sha256(f"canonical-{index}".encode()).hexdigest() for index in range(64)
    ]
    calibration_ids = [
        hashlib.sha256(f"calibration-{index}".encode()).hexdigest()
        for index in range(16)
    ]
    endpoints: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    for index, pair_id in enumerate(canonical_ids):
        source = {
            "layer": index % 3,
            "position": 1,
            "feature_id": 100 + index,
        }
        target = {
            "layer": 6 + index % 6,
            "position": 1 + index % 3,
            "feature_id": 500 + index,
        }
        raw_edge = (-1.0 if index % 2 else 1.0) * (0.02 + index * 0.01)
        response = raw_edge / 2.0
        endpoints.append({"pair_id": pair_id, "source": source, "target": target})
        pairs.append(
            {
                "pair_id": pair_id,
                "source": source,
                "target": target,
                "source_activation": 2.0,
                "target_preactivation": 1.0,
                "raw_edge": raw_edge,
                "targeted_response": response,
                "reconstructed_edge": 2.0 * response,
                "symmetric_normalized_error": 0.0,
                "device": "mps:0",
                "dtype": "torch.bfloat16",
                "method": "target_encoder_reverse_vjp_source_decoder_contraction",
                "convention": "attribution_matched_target_preactivation_pre_gate",
                "graph_edge_used": False,
            }
        )
    endpoint_digest = hashlib.sha256(
        json.dumps(endpoints, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    limits = {
        "maximum_mps_driver_bytes": 25_769_803_776,
        "maximum_process_rss_bytes": 25_769_803_776,
        "maximum_swap_growth_bytes": 4_294_967_296,
        "minimum_available_memory_bytes": 4_294_967_296,
        "maximum_graph_buffer_bytes": 6_442_450_944,
        "maximum_artifact_bundle_bytes": 5_242_880,
        "accepted_thermal_states": ["nominal", "fair"],
        "sample_interval_seconds": 1.0,
        "telemetry_failure_limit": 3,
        "terminate_grace_seconds": 15.0,
        "kill_grace_seconds": 5.0,
        "calibration_timeout_seconds": 3600,
        "canonical_timeout_seconds": 7200,
    }
    config: dict[str, object] = {
        "phase": "canonical_frozen",
        "scanner": {
            "selected_layers": list(range(18)),
            "selected_positions": [1, 2, 3, 4, 5],
            "feature_width": 16_384,
            "global_top_k": 128,
        },
        "responses": {
            "calibration_pair_ids": calibration_ids,
            "canonical_pair_ids": canonical_ids,
            "canonical_endpoint_manifest_sha256": endpoint_digest,
            "edge_floor": 0.015625,
            "method": "target_encoder_reverse_vjp_source_decoder_contraction",
            "convention": "attribution_matched_target_preactivation_pre_gate",
        },
        "tolerances": {
            "spearman_minimum": 0.98,
            "sign_agreement_minimum": 0.95,
            "median_symmetric_normalized_error_maximum": 0.05,
            "p95_symmetric_normalized_error_maximum": 0.20,
        },
        "safety_limits": limits,
        "readiness_on_success": {
            "stage1b_measurement_primitives": "completed",
            "stage1c_first_prediction_readiness": True,
            "stage1b_empirical_claim_readiness": False,
            "counterfactual_susceptibility_result": "none",
            "gate_crossing_result": "none",
            "behavioral_importance_result": "none",
            "mediation_result": "none",
            "official_bf16_reproduction": "pending",
            "reference_clt_reproduction": "pending",
            "paper_results_readiness": False,
        },
    }
    stage_names = {
        "worker_start",
        "replacement_runtime_loading",
        "scanner_dense_oracle",
        "scanner_chunk_257",
        "scanner_chunk_1024",
        "scanner_chunk_4096",
        "ephemeral_graph_reference",
        "targeted_vjp_canonical",
    }
    peaks = {
        "mps_current_bytes": 100,
        "mps_driver_bytes": 200,
        "process_rss_bytes": 300,
        "swap_used_bytes": 400,
        "swap_growth_bytes": 0,
        "minimum_available_memory_bytes": 8_000_000_000,
    }
    worker: dict[str, object] = {
        "asset_manifest": {
            "status": "verified",
            "download_performed": False,
            "network_accessed": False,
            "authentication_used": False,
            "authentication_value_recorded": False,
            "full_repository_downloaded": False,
            "other_widths_consumed": False,
            "feature_visualization_consumed": False,
            "actual_total_bytes": 2_087_816_677,
            "model": {
                "identifier": "google/gemma-3-270m",
                "revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
                "total_bytes": 575_454_257,
            },
            "transcoder": {
                "identifier": "mwhanna/gemma-scope-2-270m-pt",
                "revision": "fada11860ac1d337c1e41e9da308798405b94c8e",
                "subfolder": "transcoder_all/width_16k_l0_small",
                "total_bytes": 1_512_362_420,
            },
        },
        "environment": {
            "machine": "arm64",
            "python": "3.11.13",
            "torch": "2.6.0",
            "nnsight": "0.6.1",
            "transformers": "4.57.3",
            "mps_built": True,
            "mps_available": True,
            "fallback_variable_present": False,
            "outer_autocast_enabled": False,
        },
        "scanner": {
            "group_count": 90,
            "chunk_sizes": [257, 1024, 4096],
            "dense_oracle_chunk_size": 16_384,
            "canonical_chunk_size": 1024,
            "top_k_per_group": 8,
            "global_top_k": 128,
            "exact_candidate_identity_and_order": True,
            "bounded_oracle_recall": 1.0,
            "candidate_count": 1,
            "candidates": [
                {
                    "feature": {"layer": 0, "position": 1, "feature_id": 1},
                    "preactivation": 1.0,
                    "activation": 0.0,
                    "threshold": 1.0,
                    "margin": 0.0,
                    "activity": "inactive",
                    "device": "mps:0",
                    "dtype": "torch.bfloat16",
                }
            ],
            "maximum_retained_candidates": 4104,
            "persisted_dense_arrays": False,
            "loaded_gate": "a=z*1[z>tau]",
            "threshold_equality_activity": "inactive",
            "device": "mps:0",
            "dtype": "torch.bfloat16",
        },
        "local_response_validation": {
            "metrics": {
                "pair_count": 64,
                "above_edge_floor_count": 64,
                "spearman": 1.0,
                "sign_agreement": 1.0,
                "median_symmetric_normalized_error": 0.0,
                "p95_symmetric_normalized_error": 0.0,
            },
            "pairs": pairs,
        },
        "telemetry": {
            "started_at_unix": 2.0,
            "finished_at_unix": 3.0,
            "sample_count": 2,
            "sampling_interval_seconds": 1.0,
            "attempt_peaks": peaks,
            "stage_peaks": {name: peaks for name in stage_names},
            "thermal_states": ["nominal"],
            "violations": [],
            "telemetry_failures": 0,
        },
    }
    supervisor: dict[str, object] = {
        "returncode": 0,
        "timed_out": False,
        "safety_terminated": False,
        "termination_signal": None,
        "telemetry_failures": 0,
        "sample_count": 2,
        "peak_process_group_rss_bytes": 300,
        "minimum_available_memory_bytes": 8_000_000_000,
        "peak_swap_growth_bytes": 0,
        "thermal_states": ["nominal"],
        "started_at_unix": 1.0,
        "finished_at_unix": 4.0,
        "stdout_tail": "",
        "stderr_tail": "",
    }
    return config, worker, supervisor


def test_complete_bundle_is_independently_recomputed_and_validated(
    tmp_path: Path,
) -> None:
    config, worker, supervisor = _synthetic_complete_inputs()
    execution_commit = "a" * 40
    bundle = tmp_path / "complete"
    bundle.mkdir()
    records = _records(worker, supervisor, config, execution_commit)
    for name, record in records.items():
        write_json_atomic(bundle / name, record)
    write_checksum_manifest_atomic(
        bundle / "checksums.sha256",
        [bundle / name for name in sorted(records)],
        root=bundle,
    )
    result = validate_stage1b_artifacts(
        bundle, config=config, execution_commit=execution_commit
    )
    assert result == {
        "status": "passed",
        "artifact_count": 10,
        "pair_count": 64,
        "candidate_count": 1,
        "verdict": "completed_stage1b_measurement_primitives",
    }
    summary_path = bundle / "local_response_validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["spearman"] = 0.99
    write_json_atomic(summary_path, summary)
    write_checksum_manifest_atomic(
        bundle / "checksums.sha256",
        [bundle / name for name in sorted(records)],
        root=bundle,
    )
    with pytest.raises(ArtifactValidationError, match="independently reproduced"):
        validate_stage1b_artifacts(
            bundle, config=config, execution_commit=execution_commit
        )
