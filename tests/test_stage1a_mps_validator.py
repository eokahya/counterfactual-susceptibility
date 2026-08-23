from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from cfsus.reproduction.artifacts import ArtifactValidationError, make_artifact_envelope

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/stage1a/validate_mps_fp16_artifacts.py"
SPEC = importlib.util.spec_from_file_location("validate_mps_fp16_artifacts", SCRIPT)
assert SPEC and SPEC.loader
mps = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mps)

RUN_ID = "test-stage1a-mps-run"
COMMIT = "1" * 40
SHA = "a" * 64
TELEMETRY_METHOD = "sampled torch.mps counters plus RSS pressure swap"


def _provenance() -> dict[str, Any]:
    return {
        "project_commit": COMMIT,
        "base_commit": mps.PROJECT_BASE_COMMIT,
        "upstream_repository": mps.OFFICIAL_UPSTREAM_REPOSITORY,
        "upstream_revision": mps.OFFICIAL_UPSTREAM_REVISION,
        "model_identifier": mps.OFFICIAL_MODEL_ID,
        "model_revision": mps.OFFICIAL_MODEL_REVISION,
        "transcoder_identifier": mps.OFFICIAL_TRANSCODER_ID,
        "transcoder_revision": mps.OFFICIAL_TRANSCODER_REVISION,
        "backend": "transformerlens",
        "accelerator_backend": "mps",
        "device": "mps",
        "dtype": "float16",
        "execution_dtype": "float16",
        "reference_dtype": "bfloat16",
        "reproduction_class": mps.REPRODUCTION_CLASS,
        "execution_class": mps.EXECUTION_CLASS,
        "architecture": "arm64",
        "hardware_family": "Apple M2 Max",
        "offload": "disk",
        "fallback_enabled": False,
        "fallback_used": False,
        "official_bf16_reproduction": False,
        "t4_fp16_reproduction": False,
    }


def _envelope(
    kind: str,
    payload: dict[str, Any],
    *,
    status: str,
    deviations: tuple[str, ...] = tuple(mps.CANONICAL_DEVIATIONS),
) -> dict[str, Any]:
    return make_artifact_envelope(
        artifact_type=kind,
        run_id=RUN_ID,
        status=status,
        provenance=_provenance(),
        payload=payload,
        deviations=deviations,
    )


def _timing(
    start: float,
    finish: float,
    *,
    current: int,
    driver: int,
    rss: int,
    swap: int,
    samples: int = 4,
) -> dict[str, Any]:
    return {
        "started_at_unix": start,
        "finished_at_unix": finish,
        "wall_seconds": finish - start,
        "sampling_method": TELEMETRY_METHOD,
        "sampling_interval_seconds": 0.1,
        "sample_count": samples,
        "mps_current_allocated_peak_bytes": current,
        "mps_driver_allocated_peak_bytes": driver,
        "mps_recommended_max_bytes": 1_000,
        "process_rss_peak_bytes": rss,
        "memory_pressure_states": ["normal"],
        "swap_used_peak_bytes": swap,
    }


def _peaks(timing: dict[str, Any]) -> dict[str, int]:
    return {key: int(timing[key]) for key in mps.PEAK_KEYS}


def _asset(
    identifier: str,
    revision: str,
    pins: dict[str, tuple[int, str]],
) -> dict[str, Any]:
    ordered = sorted(pins)
    files = [
        {
            "path": name,
            "size_bytes": pins[name][0],
            "sha256": pins[name][1],
        }
        for name in ordered
    ]
    return {
        "identifier": identifier,
        "revision": revision,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(size for size, _digest in pins.values()),
        "complete": True,
        "snapshot_containment_verified": True,
        "offline_ready": True,
    }


def _preflight_payload() -> dict[str, Any]:
    primitives = {
        name: {
            "attempted": True,
            "passed": True,
            "device": "mps",
            "dtype": "float16",
            "cpu_reference_passed": True,
        }
        for name in mps.EXPECTED_OPERATOR_NAMES
    }
    operations: dict[str, dict[str, Any]] = {
        name: {
            "attempted": True,
            "passed": True,
            "device": "mps",
            "dtype": "float16",
            "cpu_reference_passed": True,
        }
        for name in mps.EXPECTED_PREFLIGHT_OPERATIONS
    }
    operations["operators"]["operations"] = primitives
    operations["strict_jumprelu"].update(strict_gate_equal=True, equality_inactive=True)
    operations["sparse_metadata_boundary"].update(
        cpu_metadata_explicit=True,
        replacement_boundary_passed=True,
        dense_scientific_device="mps",
    )
    operations["disk_offload_safetensors"].update(
        upstream_disk_offload_helper_tested=True
    )
    return {
        "probe_status": "passed",
        "environment": {
            "python": "3.11.13",
            "system": "Darwin",
            "architecture": "arm64",
            "torch_version": "2.6.0",
            "mps_built": True,
            "mps_available": True,
            "fallback_enabled": False,
            "fallback_used": False,
            "fallback_env_value_present": False,
            "high_watermark_override": None,
        },
        "checks": {name: True for name in mps.EXPECTED_PREFLIGHT_CHECKS},
        "operations": operations,
        "tolerances": {
            "absolute": 0.005,
            "relative": 0.002,
            "finite_required": True,
        },
        "large_assets_downloaded": False,
        "scientific_model_result": False,
    }


def _fixture(tmp_path: Path) -> Path:
    directory = tmp_path / "results/stage1a_mps_fp16"
    (directory / "preflight").mkdir(parents=True)

    semantics_timing = _timing(1.0, 2.0, current=20, driver=40, rss=100, swap=5)
    intervention_timing = _timing(2.0, 3.0, current=30, driver=50, rss=110, swap=6)
    attribution_timing = _timing(3.0, 4.0, current=40, driver=80, rss=120, swap=7)
    memory_timing = _timing(
        1.0, 4.0, current=40, driver=80, rss=120, swap=7, samples=12
    )
    attempt = {
        "batch_size": 256,
        "outcome": "completed",
        "category": "completed",
        "failure_stage": None,
        "fresh_process": True,
        "cleanup_succeeded": True,
        "process_exit_code": 0,
        "sample_count": 12,
        "oom_classifier_match": False,
        "exception_type": None,
        "diagnostic_redacted": True,
        "stage_peaks": {
            "runtime_loading": {
                "mps_current_allocated_peak_bytes": 10,
                "mps_driver_allocated_peak_bytes": 30,
                "process_rss_peak_bytes": 90,
                "swap_used_peak_bytes": 4,
            },
            "model_only_forward": {
                "mps_current_allocated_peak_bytes": 15,
                "mps_driver_allocated_peak_bytes": 35,
                "process_rss_peak_bytes": 95,
                "swap_used_peak_bytes": 4,
            },
            "semantics": _peaks(semantics_timing),
            "intervention": _peaks(intervention_timing),
            "attribution": _peaks(attribution_timing),
            "cleanup": {
                "mps_current_allocated_peak_bytes": 0,
                "mps_driver_allocated_peak_bytes": 10,
                "process_rss_peak_bytes": 80,
                "swap_used_peak_bytes": 7,
            },
        },
        "attempt_peaks": _peaks(memory_timing),
    }

    records: dict[str, dict[str, Any]] = {
        "feasibility_report.json": _envelope(
            "feasibility_report",
            {
                "status": "feasible_with_explicit_execution_deviation",
                "passed": True,
                "downloads_authorized": True,
                "physical_memory_bytes": 32 * 1024**3,
                "system_memory_pressure": "normal",
                "swap_used_bytes": 512 * 1024**2,
                "snapshot_sizes": {
                    "model_bytes": mps.MODEL_REQUIRED_BYTES,
                    "transcoder_bytes": mps.TRANSCODER_REQUIRED_BYTES,
                },
                "estimate_components": {
                    "model_resident_estimate_bytes": int(
                        mps.MODEL_REQUIRED_BYTES * 0.60
                    ),
                    "transcoder_resident_estimate_bytes": mps.TRANSCODER_REQUIRED_BYTES,
                    "temporary_and_system_headroom_bytes": 6 * 1024**3,
                },
                "conservative_budget_bytes": 26 * 1024**3,
                "estimated_peak_bytes": (
                    int(mps.MODEL_REQUIRED_BYTES * 0.60)
                    + mps.TRANSCODER_REQUIRED_BYTES
                    + 6 * 1024**3
                ),
                "safety_reserve_bytes": 6 * 1024**3,
                "scientific_parameters_changed": False,
            },
            status="resolved",
        ),
        "environment_manifest.json": _envelope(
            "environment_manifest",
            {
                "platform": {
                    "system": "Darwin",
                    "machine": "arm64",
                    "hardware_family": "Apple M2 Max",
                    "physical_memory_bytes": 32 * 1024**3,
                },
                "python": {"version": "3.11.13", "architecture": "arm64"},
                "packages": {
                    "torch": "2.6.0",
                    "transformer_lens": "3.2.1",
                    "circuit_tracer": "0.5.2",
                    "circuit_tracer_revision": mps.OFFICIAL_UPSTREAM_REVISION,
                    "circuit_tracer_vcs_url": mps.EXPECTED_CIRCUIT_TRACER_VCS_URL,
                    "circuit_tracer_record_hashes_verified": 42,
                },
                "runtime": {
                    "backend": "transformerlens",
                    "accelerator_backend": "mps",
                    "device": "mps",
                    "dtype": "float16",
                    "execution_class": mps.EXECUTION_CLASS,
                    "offload": "disk",
                },
                "mps": {
                    "built": True,
                    "available": True,
                    "allocation_probe": {
                        "success": True,
                        "device": "mps",
                        "dtype": "float16",
                        "finite": True,
                    },
                },
                "fallback_enabled": False,
                "fallback_used": False,
                "fallback_env_value_present": False,
                "high_watermark_override": None,
                "memory_guardrails_preserved": True,
                "pip_check": "passed",
                "lock_path": "environments/stage1a_mps/requirements-lock.txt",
                "lock_sha256": mps.EXPECTED_LOCK_SHA256,
                "lock_match": "exact",
            },
            status="observed",
        ),
        "asset_manifest.json": _envelope(
            "asset_manifest",
            {
                "verification": "exact_file_content_hashes_matched",
                "immutable_revisions_only": True,
                "project_external_cache": True,
                "unmanifested_file_count": 0,
                "assets": {
                    "model": _asset(
                        mps.OFFICIAL_MODEL_ID,
                        mps.OFFICIAL_MODEL_REVISION,
                        mps.MODEL_FILE_PINS,
                    ),
                    "transcoder": _asset(
                        mps.OFFICIAL_TRANSCODER_ID,
                        mps.OFFICIAL_TRANSCODER_REVISION,
                        mps.TRANSCODER_FILE_PINS,
                    ),
                },
            },
            status="resolved",
        ),
        "attribution_summary.json": _envelope(
            "attribution_summary",
            {
                "parameters": {
                    "prompt": "The capital of state containing Dallas is",
                    "max_n_logits": 10,
                    "desired_logit_probability": 0.95,
                    "max_feature_nodes": 8192,
                    "offload": "disk",
                },
                "accepted_batch_size": 256,
                "graph": {
                    "finite": True,
                    "node_count": 18,
                    "selected_feature_count": 3,
                    "active_feature_count": 6,
                    "edge_count": 5,
                    "logit_node_count": 10,
                    "input_node_count": 4,
                    "error_node_count": 1,
                    "adjacency_shape": [18, 18],
                },
                "raw_validation": {
                    "passed": True,
                    "raw_graph_committed": False,
                },
                "nonfinite_count": 0,
                "timing": attribution_timing,
            },
            status="completed",
        ),
        "intervention_summary.json": _envelope(
            "intervention_summary",
            {
                "parameters": {
                    "prompt": "Hecho: Michael Jordan juega al",
                    "feature": {"layer": 20, "position": -1, "feature_id": 341},
                    "alphas": [0.0, 0.5, 1.0],
                    "freeze_attention": True,
                    "constrained_layers": None,
                },
                "baseline_activation_captured": True,
                "baseline_activation": 10.0,
                "baseline_repeat_error": 0.03,
                "baseline_repeat_max_combined_tolerance_ratio": 0.75,
                "baseline_noop_comparison": {
                    "within_tolerance": True,
                    "max_abs_error": 0.03,
                    "max_rel_error": 0.003,
                    "max_combined_tolerance_ratio": 0.75,
                    "absolute_tolerance": 0.02,
                    "relative_tolerance": 0.002,
                },
                "desired_values": [
                    {
                        "alpha": alpha,
                        "expected_activation": expected,
                        "observed_activation": expected,
                        "absolute_error": 0.0,
                        "within_tolerance": True,
                        "output_finite": True,
                    }
                    for alpha, expected in ((0.0, 10.0), (0.5, 5.0), (1.0, 0.0))
                ],
                "outputs_finite": True,
                "same_assets_and_runtime": True,
                "nonfinite_count": 0,
                "timing": intervention_timing,
            },
            status="completed",
        ),
        "semantics_summary.json": _envelope(
            "semantics_summary",
            {
                "loaded_runtime": {
                    "passed": True,
                    "model_loaded": True,
                    "transcoder_loaded": True,
                    "model_device": "mps",
                    "transcoder_device": "mps",
                    "model_dtype": "float16",
                    "transcoder_dtype": "float16",
                    "model_only_forward_passed": True,
                    "model_only_forward": {
                        "passed": True,
                        "prompt": "The capital of state containing Dallas is",
                        "token_count": 7,
                        "logits_shape": [1, 7, 256_000],
                        "tokenizer_revision": mps.OFFICIAL_MODEL_REVISION,
                        "device": "mps",
                        "dtype": "float16",
                        "finite": True,
                        "completed_before_transcoder_load": True,
                    },
                    "output_finite": True,
                    "fallback_used": False,
                },
                "preactivation": {
                    "verified": True,
                    "threshold_retrieved": True,
                    "definition": "F.linear(feature_input, W_enc, b_enc)",
                    "bias_convention": "b_enc is included; b_dec is excluded",
                    "cache_shape": [26, 6, 16_384],
                    "projection_absolute_tolerance": 0.005,
                },
                "gate_check": {
                    "rule": "z if z > threshold else 0",
                    "strict_greater_than": True,
                    "equality_inactive": True,
                    "equality_probe_maximum_absolute_output": 0.0,
                    "absolute_tolerance": 0.005,
                    "active_example": {
                        "layer": 20,
                        "position": -1,
                        "feature_id": 5,
                        "preactivation": 1.1,
                        "threshold": 1.0,
                        "post_gate_activation": 1.1,
                        "active": True,
                        "signed_margin": 0.1,
                    },
                    "inactive_example": {
                        "layer": 20,
                        "position": -1,
                        "feature_id": 6,
                        "preactivation": 1.0,
                        "threshold": 1.0,
                        "post_gate_activation": 0.0,
                        "active": False,
                        "signed_margin": 0.0,
                    },
                    "official_intervention_source": {
                        "layer": 20,
                        "position": -1,
                        "feature_id": 341,
                        "preactivation": 10.0,
                        "threshold": 1.0,
                        "post_gate_activation": 10.0,
                        "active": True,
                        "signed_margin": 9.0,
                    },
                },
                "intervention_value_check": {
                    "passed": True,
                    "formula": "(1-alpha)*baseline_activation",
                    "alphas": [0.0, 0.5, 1.0],
                },
                "feature": {
                    "layer": 20,
                    "position": -1,
                    "feature_id": 341,
                    "baseline_activation": 10.0,
                },
                "baseline_repeat_error": 0.001,
                "projection_discrepancy": 0.001,
                "gate_discrepancy": 0.0,
                "nonfinite_count": 0,
                "timing": semantics_timing,
            },
            status="completed",
        ),
        "memory_summary.json": _envelope(
            "memory_summary",
            {
                "nonfinite_count": 0,
                "timing": memory_timing,
                "telemetry_method": TELEMETRY_METHOD,
                "sampling_interval_seconds": 0.1,
                "accepted_attempt_index": 0,
                "accepted_batch_size": 256,
                "attempts": [attempt],
            },
            status="completed",
        ),
    }
    for name, value in records.items():
        (directory / name).write_text(
            json.dumps(value, sort_keys=True), encoding="utf-8"
        )
    preflight = _envelope("preflight_summary", _preflight_payload(), status="observed")
    (directory / "preflight/preflight_summary.json").write_text(
        json.dumps(preflight, sort_keys=True), encoding="utf-8"
    )

    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": mps.COMPLETED_STATUS,
        "reproduction_class": mps.REPRODUCTION_CLASS,
        "claim_boundary": mps.CLAIM_BOUNDARY,
        "project": {
            "base_commit": mps.PROJECT_BASE_COMMIT,
            "execution_commit": COMMIT,
            "source_clean_excluding_preserved_t4": True,
            "preserved_t4_untracked": True,
        },
        "upstream": {
            "identifier": mps.OFFICIAL_UPSTREAM_REPOSITORY,
            "revision": mps.OFFICIAL_UPSTREAM_REVISION,
        },
        "model": {
            "identifier": mps.OFFICIAL_MODEL_ID,
            "revision": mps.OFFICIAL_MODEL_REVISION,
        },
        "transcoder": {
            "identifier": mps.OFFICIAL_TRANSCODER_ID,
            "revision": mps.OFFICIAL_TRANSCODER_REVISION,
        },
        "runtime": {
            "backend": "transformerlens",
            "accelerator_backend": "mps",
            "device": "mps",
            "architecture": "arm64",
            "hardware_family": "Apple M2 Max",
            "reference_dtype": "bfloat16",
            "execution_dtype": "float16",
            "execution_class": mps.EXECUTION_CLASS,
            "official_bf16_reproduction": False,
            "t4_fp16_reproduction": False,
            "fallback_enabled": False,
            "fallback_used": False,
            "offload": "disk",
            "execution_deviations": mps.CANONICAL_DEVIATIONS,
            "accepted_batch_size": 256,
            "retry_occurred": False,
        },
        "timings": {
            "attribution": attribution_timing,
            "intervention": intervention_timing,
            "semantics": semantics_timing,
            "memory": memory_timing,
        },
        "retry_history": [attempt],
        "checks": {
            "preflight_passed": True,
            "feasibility_passed": True,
            "assets_verified": True,
            "model_only_forward_passed": True,
            "loaded_runtime_semantics_passed": True,
            "attribution_passed": True,
            "intervention_passed": True,
            "telemetry_passed": True,
            "no_hidden_fallback": True,
            "nonfinite_count": 0,
        },
        "artifacts": {
            "preflight": "preflight/preflight_summary.json",
            "feasibility": "feasibility_report.json",
            "environment": "environment_manifest.json",
            "assets": "asset_manifest.json",
            "attribution": "attribution_summary.json",
            "intervention": "intervention_summary.json",
            "semantics": "semantics_summary.json",
            "memory": "memory_summary.json",
            "checksums": "checksums.sha256",
        },
        "readiness": {
            "stage1b_engineering_readiness": True,
            "stage1b_empirical_claim_readiness": False,
        },
    }
    (directory / mps.RUN_MANIFEST_NAME).write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    mps.write_mps_checksums(directory)
    return directory


def _edit(
    directory: Path, relative: str, mutator: Callable[[dict[str, Any]], None]
) -> None:
    path = directory / relative
    value = json.loads(path.read_text(encoding="utf-8"))
    mutator(value)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    # Model an attacker supplying a checksum-valid malicious bundle without
    # weakening the production checksum writer's pre-hash secret scan.
    mps.write_checksum_manifest_atomic(
        directory / mps.CHECKSUM_NAME,
        mps.checksum_targets(directory),
        root=directory,
    )


def test_valid_canonical_fixture_and_identity_split(tmp_path: Path) -> None:
    directory = _fixture(tmp_path)
    names = mps.validate_mps_artifact_directory(directory)
    attribution = json.loads(
        (directory / "attribution_summary.json").read_text(encoding="utf-8")
    )["payload"]["graph"]
    comparison = json.loads(
        (directory / "intervention_summary.json").read_text(encoding="utf-8")
    )["payload"]["baseline_noop_comparison"]
    assert "preflight/preflight_summary.json" in names
    assert mps.REPRODUCTION_CLASS == "hardware_adapted_mps_fp16"
    assert mps.COMPLETED_STATUS == "completed_hardware_adapted_mps_fp16"
    assert attribution["selected_feature_count"] < attribution["active_feature_count"]
    assert comparison["max_abs_error"] > comparison["absolute_tolerance"]
    assert comparison["max_rel_error"] > comparison["relative_tolerance"]
    assert comparison["max_combined_tolerance_ratio"] <= 1.0
    assert (directory / "checksums.sha256").is_file()


@pytest.mark.parametrize(
    ("relative", "mutator"),
    [
        (
            "feasibility_report.json",
            lambda value: value["payload"].update(status="blocked"),
        ),
        (
            "feasibility_report.json",
            lambda value: value["payload"].update(passed=False),
        ),
        (
            "feasibility_report.json",
            lambda value: value["payload"].update(
                conservative_budget_bytes=100 * 1024**3,
                estimated_peak_bytes=99 * 1024**3,
            ),
        ),
        (
            "feasibility_report.json",
            lambda value: value.update(deviations=["Changed attribution threshold."]),
        ),
        (
            "asset_manifest.json",
            lambda value: value["payload"]["assets"]["model"].update(files=[]),
        ),
        (
            "environment_manifest.json",
            lambda value: value["payload"].update(runtime={}),
        ),
        (
            "environment_manifest.json",
            lambda value: value["payload"].update(fallback_used=True),
        ),
        (
            "environment_manifest.json",
            lambda value: value["payload"]["packages"].update(
                nvidia_cuda_runtime="12.4"
            ),
        ),
        (
            "environment_manifest.json",
            lambda value: value["payload"].update(lock_sha256="b" * 64),
        ),
        (
            "environment_manifest.json",
            lambda value: value["payload"].update(lock_match="digest_only"),
        ),
        (
            "environment_manifest.json",
            lambda value: value["payload"]["packages"].update(
                circuit_tracer_vcs_url="https://example.invalid/circuit-tracer.git"
            ),
        ),
        (
            "environment_manifest.json",
            lambda value: value["payload"]["packages"].update(
                circuit_tracer_record_hashes_verified=0
            ),
        ),
        (
            "preflight/preflight_summary.json",
            lambda value: value["payload"]["checks"].update(matmul=False),
        ),
        (
            "preflight/preflight_summary.json",
            lambda value: value["payload"]["operations"]["operators"]["operations"][
                "matmul"
            ].update(dtype="float32"),
        ),
        (
            "semantics_summary.json",
            lambda value: value["payload"].pop("loaded_runtime"),
        ),
        (
            "semantics_summary.json",
            lambda value: value["payload"]["preactivation"].update(
                definition="some projection", bias_convention="unknown"
            ),
        ),
        (
            "semantics_summary.json",
            lambda value: value["payload"]["loaded_runtime"][
                "model_only_forward"
            ].update(completed_before_transcoder_load=False),
        ),
        (
            "attribution_summary.json",
            lambda value: value["payload"]["parameters"].update(prompt="wrong"),
        ),
        (
            "attribution_summary.json",
            lambda value: value["payload"]["graph"].update(edge_count=0),
        ),
        (
            "attribution_summary.json",
            lambda value: value["payload"]["graph"].update(active_feature_count=2),
        ),
        (
            "attribution_summary.json",
            lambda value: value["payload"]["graph"].update(
                node_count=1,
                edge_count=999,
                input_node_count=100,
                error_node_count=100,
                adjacency_shape=[1, 1],
            ),
        ),
        (
            "intervention_summary.json",
            lambda value: value["payload"].pop("baseline_activation"),
        ),
        (
            "intervention_summary.json",
            lambda value: value["payload"].update(
                baseline_repeat_max_combined_tolerance_ratio=1.0001
            ),
        ),
        (
            "intervention_summary.json",
            lambda value: value["payload"]["baseline_noop_comparison"].update(
                max_combined_tolerance_ratio=1.0001
            ),
        ),
        (
            "intervention_summary.json",
            lambda value: value["payload"]["desired_values"][1].update(
                observed_activation=9.0, absolute_error=4.0
            ),
        ),
        (
            "intervention_summary.json",
            lambda value: (
                value["payload"].update(baseline_activation=10_000.0),
                value["payload"]["desired_values"][0].update(
                    expected_activation=10_019.0,
                    observed_activation=10_019.0,
                    absolute_error=0.0,
                ),
                value["payload"]["desired_values"][1].update(
                    expected_activation=5_009.5,
                    observed_activation=5_009.5,
                    absolute_error=0.0,
                ),
            ),
        ),
        (
            mps.RUN_MANIFEST_NAME,
            lambda value: value.update(
                claim_boundary=(
                    "native-BF16 pending; T4 numerical equivalence is complete, "
                    "not provisional."
                )
            ),
        ),
        (
            mps.RUN_MANIFEST_NAME,
            lambda value: value.update(reproduction_class=mps.EXECUTION_CLASS),
        ),
        (
            mps.RUN_MANIFEST_NAME,
            lambda value: value["runtime"].update(device="cpu"),
        ),
        (
            mps.RUN_MANIFEST_NAME,
            lambda value: value["runtime"].update(official_bf16_reproduction=True),
        ),
        (
            mps.RUN_MANIFEST_NAME,
            lambda value: value.update(timings={}),
        ),
        (
            "attribution_summary.json",
            lambda value: value["payload"]["timing"].update(
                sampling_interval_seconds=100.0
            ),
        ),
        (
            "memory_summary.json",
            lambda value: value["payload"]["attempts"][0]["attempt_peaks"].update(
                mps_driver_allocated_peak_bytes=1,
                process_rss_peak_bytes=10_000,
            ),
        ),
        (
            "memory_summary.json",
            lambda value: value["payload"]["attempts"][0].update(
                category="failed_runtime"
            ),
        ),
        (
            "memory_summary.json",
            lambda value: value["payload"]["attempts"][0].update(sample_count=11),
        ),
    ],
)
def test_completion_false_positives_are_rejected(
    tmp_path: Path,
    relative: str,
    mutator: Callable[[dict[str, Any]], None],
) -> None:
    directory = _fixture(tmp_path)
    _edit(directory, relative, mutator)
    with pytest.raises(ArtifactValidationError):
        mps.validate_mps_artifact_directory(directory)


@pytest.mark.parametrize(
    ("relative", "mutator"),
    [
        (
            "asset_manifest.json",
            lambda value: value["payload"].update(api_key="[REDACTED]"),
        ),
        (
            "attribution_summary.json",
            lambda value: value["payload"].update(
                runtime_observation={"actual_device": "cpu"}
            ),
        ),
        (
            "semantics_summary.json",
            lambda value: value["payload"].update(raw_graph=[1, 2, 3]),
        ),
    ],
)
def test_security_boundaries_reject_redacted_keys_cpu_and_raw_data(
    tmp_path: Path,
    relative: str,
    mutator: Callable[[dict[str, Any]], None],
) -> None:
    directory = _fixture(tmp_path)
    _edit(directory, relative, mutator)
    with pytest.raises(ArtifactValidationError):
        mps.validate_mps_artifact_directory(directory)


@pytest.mark.parametrize(
    "value",
    [
        {"accessToken": "[REDACTED]"},
        {"authTokenRedacted": "[REDACTED]"},
        {"accessTokenHash": "0" * 64},
        {"outer": {"authTokenRedacted": "[REDACTED]"}},
        {"outerModelAuthTokenRedacted": "[REDACTED]"},
        {"hfToken": "[REDACTED]"},
        {"rawGraph": []},
        {"modelWeights": []},
        {"fallbackEnabled": True},
        {"runtimeFallbackUsed": "unknown"},
        {"PYTORCH_ENABLE_MPS_FALLBACK": "1"},
        {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0"},
        {"officialBF16Reproduction": True},
        {"scientificBackend": "cpu"},
        {"scientificBackend": "CPU:0"},
        {"activation_values": [1.0]},
        {"raw_graph_data": {"nodes": []}},
        {"bf16_equivalence": "established"},
        {"warnings": ["Official BF16 reproduction completed and T4 equivalent."]},
        {"note": "BFloat16 output parity confirmed."},
        {"note": "BF16 remains pending; CUDA equivalence succeeded."},
        {"note": "BF16 pending and T4 tests passed."},
    ],
)
def test_global_security_scanner_rejects_camel_raw_and_overclaim(
    value: dict[str, Any],
) -> None:
    with pytest.raises(ArtifactValidationError):
        mps._walk_safety(value)


def test_global_security_scanner_allows_benign_tokenization_evidence() -> None:
    mps._walk_safety(
        {
            "token_count": 7,
            "token_ids_hash": "not-a-credential",
            "tokenizer_revision": mps.OFFICIAL_MODEL_REVISION,
            "claim_boundary": mps.CLAIM_BOUNDARY,
            "reference_dtype": "bfloat16",
        }
    )


@pytest.mark.parametrize(
    "value",
    [
        "x" * (mps.MAX_JSON_STRING_LENGTH + 1),
        {"x" * (mps.MAX_JSON_KEY_LENGTH + 1): True},
    ],
)
def test_json_scalar_bounds_reject_oversized_strings_and_keys(value: Any) -> None:
    with pytest.raises(ArtifactValidationError, match="exceeds the artifact limit"):
        mps._bounded_json(value)


def test_checksum_writer_never_hashes_secret_like_content(tmp_path: Path) -> None:
    directory = _fixture(tmp_path)
    checksum_path = directory / mps.CHECKSUM_NAME
    original_checksum = checksum_path.read_bytes()
    path = directory / "environment_manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["payload"]["note"] = "hf_" + "A" * 24
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")

    with pytest.raises(ArtifactValidationError):
        mps.write_mps_checksums(directory)
    assert checksum_path.read_bytes() == original_checksum


def test_cross_file_run_id_timing_and_commit_must_match(tmp_path: Path) -> None:
    for suffix, relative, mutator in (
        (
            "run-id",
            "asset_manifest.json",
            lambda value: value.update(run_id="different-run"),
        ),
        (
            "commit",
            "semantics_summary.json",
            lambda value: value["provenance"].update(project_commit="2" * 40),
        ),
        (
            "timing",
            mps.RUN_MANIFEST_NAME,
            lambda value: value["timings"]["attribution"].update(wall_seconds=9.0),
        ),
    ):
        directory = _fixture(tmp_path / suffix)
        _edit(directory, relative, mutator)
        with pytest.raises(ArtifactValidationError):
            mps.validate_mps_artifact_directory(directory)


def test_semantics_and_intervention_baselines_must_match(tmp_path: Path) -> None:
    directory = _fixture(tmp_path)

    def mutate(value: dict[str, Any]) -> None:
        payload = value["payload"]
        payload["feature"]["baseline_activation"] = 999.0
        source = payload["gate_check"]["official_intervention_source"]
        source.update(
            preactivation=999.0,
            threshold=1.0,
            post_gate_activation=999.0,
            active=True,
            signed_margin=998.0,
        )

    _edit(directory, "semantics_summary.json", mutate)
    with pytest.raises(ArtifactValidationError, match="baselines"):
        mps.validate_mps_artifact_directory(directory)


def test_incomplete_status_never_satisfies_full_bundle_contract(tmp_path: Path) -> None:
    directory = _fixture(tmp_path)
    _edit(
        directory,
        mps.RUN_MANIFEST_NAME,
        lambda value: value.update(status="blocked"),
    )
    with pytest.raises(ArtifactValidationError, match="only accepts a completed run"):
        mps.validate_mps_artifact_directory(directory, require_complete=False)


def test_asset_path_hash_size_and_revision_are_strict(tmp_path: Path) -> None:
    mutations = (
        lambda value: value["payload"]["assets"]["model"]["files"][0].update(
            path="../config.json"
        ),
        lambda value: value["payload"]["assets"]["model"]["files"][0].update(
            sha256="not-a-hash"
        ),
        lambda value: value["payload"]["assets"]["model"]["files"][0].update(
            sha256="b" * 64
        ),
        lambda value: value["payload"]["assets"]["model"].update(revision="main"),
        lambda value: value["payload"]["assets"]["model"].update(total_bytes=1),
    )
    for index, mutator in enumerate(mutations):
        directory = _fixture(tmp_path / str(index))
        _edit(directory, "asset_manifest.json", mutator)
        with pytest.raises(ArtifactValidationError):
            mps.validate_mps_artifact_directory(directory)


def test_allowlist_symlink_duplicate_keys_and_checksum_mismatch(tmp_path: Path) -> None:
    directory = _fixture(tmp_path / "extra")
    (directory / "weights.safetensors").write_bytes(b"not weights")
    with pytest.raises(ArtifactValidationError):
        mps.validate_mps_artifact_directory(directory)

    directory = _fixture(tmp_path / "symlink")
    (directory / "extra.json").symlink_to(directory / "asset_manifest.json")
    with pytest.raises(ArtifactValidationError):
        mps.validate_mps_artifact_directory(directory)

    directory = _fixture(tmp_path / "duplicate")
    path = directory / "feasibility_report.json"
    path.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="duplicate"):
        mps.validate_mps_artifact_directory(directory)

    directory = _fixture(tmp_path / "checksum")
    path = directory / "memory_summary.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="checksum"):
        mps.validate_mps_artifact_directory(directory)


def test_cli_does_not_resolve_away_artifact_root_symlink(tmp_path: Path) -> None:
    directory = _fixture(tmp_path / "real")
    link = tmp_path / "linked-results"
    link.symlink_to(directory, target_is_directory=True)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--artifact-dir", str(link)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "unsafe" in result.stdout
