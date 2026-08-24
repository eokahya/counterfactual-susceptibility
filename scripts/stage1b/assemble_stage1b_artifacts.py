#!/usr/bin/env python3
"""Assemble and independently validate the compact Stage 1B canonical bundle."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cfsus.reproduction.artifacts import (  # noqa: E402
    sha256_file,
    write_checksum_manifest_atomic,
    write_json_atomic,
)
from cfsus.stage1b import (  # noqa: E402
    BASE_COMMIT,
    BRANCH,
    COMPLETED_STATUS,
    CONFIG_PATH,
    MODEL_REVISION,
    TRANSCODER_REVISION,
    TRANSCODER_SUBFOLDER,
    UPSTREAM_REVISION,
    load_stage1b_config,
)
from cfsus.stage1b_artifacts import (  # noqa: E402
    ARTIFACT_ALLOWLIST,
    strict_json_bytes,
    validate_stage1b_artifacts,
)

SCHEMA_PATH = Path("configs/stage1b_measurement_primitives_artifact_schema.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / CONFIG_PATH)
    parser.add_argument("--canonical-worker", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--supervisor", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results/stage1b_measurement_primitives",
    )
    return parser


def _load(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise RuntimeError(f"{label} must be a single-link regular file")
    if path.stat().st_size > 2 * 1024 * 1024:
        raise RuntimeError(f"{label} exceeds the bounded input cap")
    return strict_json_bytes(path.read_bytes(), label=label)


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.stdout.strip()


def _compact_supervisor(value: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "returncode",
        "timed_out",
        "safety_terminated",
        "termination_signal",
        "telemetry_failures",
        "sample_count",
        "peak_process_group_rss_bytes",
        "minimum_available_memory_bytes",
        "peak_swap_growth_bytes",
        "thermal_states",
        "started_at_unix",
        "finished_at_unix",
    }
    if not keys.issubset(value):
        raise RuntimeError("supervisor record is incomplete")
    return {key: value[key] for key in sorted(keys)}


def _validate_inputs(
    worker: dict[str, Any],
    preflight: dict[str, Any],
    supervisor: dict[str, Any],
    config: dict[str, Any],
    execution_commit: str,
) -> None:
    if config["phase"] != "canonical_frozen":
        raise RuntimeError("artifact assembly requires canonical-frozen config")
    if worker.get("artifact_type") != (
        "stage1b_measurement_primitives_canonical_worker"
    ) or any(
        worker.get(key) != expected
        for key, expected in (
            ("status", "passed"),
            ("fresh_canonical_run", True),
            ("scientific_retry_count", 0),
            ("calibration_artifact_read", False),
        )
    ):
        raise RuntimeError("canonical worker identity is invalid")
    worker_git = worker.get("git", {})
    if worker_git != {
        "head": execution_commit,
        "branch": BRANCH,
        "working_tree_clean": True,
    }:
        raise RuntimeError("canonical worker Git identity is invalid")
    preflight_git = preflight.get("git", {})
    if (
        preflight.get("status") != "passed"
        or preflight.get("mode") != "canonical"
        or preflight_git.get("head") != execution_commit
        or preflight_git.get("branch") != BRANCH
        or preflight_git.get("working_tree_clean") is not True
    ):
        raise RuntimeError("canonical preflight identity is invalid")
    selection = worker.get("pair_selection", {})
    responses = config["responses"]
    for key in (
        "calibration_pair_ids",
        "canonical_pair_ids",
        "canonical_endpoint_manifest_sha256",
        "edge_floor",
    ):
        if selection.get(key) != responses[key]:
            raise RuntimeError(f"canonical pair selection changed: {key}")
    if selection.get("disjoint") is not True:
        raise RuntimeError("calibration and canonical pair sets overlap")
    if any(
        supervisor.get(key) != expected
        for key, expected in (
            ("returncode", 0),
            ("timed_out", False),
            ("safety_terminated", False),
            ("telemetry_failures", 0),
        )
    ):
        raise RuntimeError("canonical supervisor did not pass")


def _records(
    worker: dict[str, Any],
    supervisor: dict[str, Any],
    config: dict[str, Any],
    execution_commit: str,
) -> dict[str, dict[str, Any]]:
    assets = worker["asset_manifest"]
    scanner = worker["scanner"]
    validation = worker["local_response_validation"]
    metrics = validation["metrics"]
    compact_supervisor = _compact_supervisor(supervisor)
    records: dict[str, dict[str, Any]] = {
        "run_manifest.json": {
            "schema_version": 1,
            "artifact_type": "stage1b_measurement_primitives_run_manifest",
            "status": COMPLETED_STATUS,
            "branch": BRANCH,
            "base_commit": BASE_COMMIT,
            "execution_commit": execution_commit,
            "fresh_canonical_run": True,
            "scientific_retry_count": 0,
            "config_sha256": sha256_file(REPOSITORY_ROOT / CONFIG_PATH),
            "artifact_schema_sha256": sha256_file(REPOSITORY_ROOT / SCHEMA_PATH),
            "runtime_identity": {
                "backend": "nnsight",
                "device": "mps:0",
                "dtype": "torch.bfloat16",
                "upstream_revision": UPSTREAM_REVISION,
                "model_revision": MODEL_REVISION,
                "transcoder_revision": TRANSCODER_REVISION,
                "transcoder_subfolder": TRANSCODER_SUBFOLDER,
                "prompt_id": "pilot",
            },
            "claim_boundary": config["readiness_on_success"],
        },
        "asset_manifest.json": {
            "schema_version": 1,
            "artifact_type": "stage1b_measurement_primitives_asset_manifest",
            "status": assets["status"],
            "download_performed": assets["download_performed"],
            "network_accessed": assets["network_accessed"],
            "authentication_used": assets["authentication_used"],
            "authentication_value_recorded": assets["authentication_value_recorded"],
            "exact_allowlist_hashes_verified": True,
            "full_repository_downloaded": assets["full_repository_downloaded"],
            "other_widths_consumed": assets["other_widths_consumed"],
            "feature_visualization_consumed": assets["feature_visualization_consumed"],
            "actual_total_bytes": assets["actual_total_bytes"],
            "model": {
                "identifier": assets["model"]["identifier"],
                "revision": assets["model"]["revision"],
                "total_bytes": assets["model"]["total_bytes"],
            },
            "transcoder": {
                "identifier": assets["transcoder"]["identifier"],
                "revision": assets["transcoder"]["revision"],
                "subfolder": assets["transcoder"]["subfolder"],
                "layer_count": 18,
                "total_bytes": assets["transcoder"]["total_bytes"],
            },
        },
        "environment_manifest.json": {
            "schema_version": 1,
            "artifact_type": "stage1b_measurement_primitives_environment",
            "status": "passed",
            "execution_commit": execution_commit,
            "platform": {
                "system": "Darwin",
                "machine": worker["environment"]["machine"],
                "python": worker["environment"]["python"],
                "host_class": "Apple M2 Max 32 GiB unified memory",
            },
            "packages": {
                "circuit-tracer": "0.5.2",
                "nnsight": worker["environment"]["nnsight"],
                "torch": worker["environment"]["torch"],
                "transformers": worker["environment"]["transformers"],
            },
            "accelerator": {
                "device": "mps:0",
                "dtype": "torch.bfloat16",
                "mps_built": worker["environment"]["mps_built"],
                "mps_available": worker["environment"]["mps_available"],
                "mps_bfloat16_probe": "passed",
                "fallback_variable_present": worker["environment"][
                    "fallback_variable_present"
                ],
                "outer_autocast_enabled": worker["environment"][
                    "outer_autocast_enabled"
                ],
                "scientific_tensor_device": "mps",
                "graph_metadata_device": "cpu",
            },
            "privacy": {
                "network_accessed": False,
                "credential_values_read": False,
                "secret_values_recorded": False,
                "private_paths_recorded": False,
            },
        },
        "scanner_oracle_summary.json": {
            "schema_version": 1,
            "artifact_type": "stage1b_measurement_primitives_scanner_oracle_summary",
            "status": "passed",
            "group_count": scanner["group_count"],
            "selected_layers": config["scanner"]["selected_layers"],
            "selected_positions": config["scanner"]["selected_positions"],
            "feature_width": config["scanner"]["feature_width"],
            "chunk_sizes": scanner["chunk_sizes"],
            "dense_oracle_chunk_size": scanner["dense_oracle_chunk_size"],
            "canonical_chunk_size": scanner["canonical_chunk_size"],
            "top_k_per_group": scanner["top_k_per_group"],
            "global_top_k": scanner["global_top_k"],
            "exact_candidate_identity_and_order": scanner[
                "exact_candidate_identity_and_order"
            ],
            "bounded_oracle_recall": scanner["bounded_oracle_recall"],
            "candidate_count": scanner["candidate_count"],
            "maximum_retained_candidates": scanner["maximum_retained_candidates"],
            "persisted_dense_arrays": scanner["persisted_dense_arrays"],
            "loaded_gate": scanner["loaded_gate"],
            "threshold_equality_activity": scanner["threshold_equality_activity"],
            "device": scanner["device"],
            "dtype": scanner["dtype"],
        },
        "near_threshold_candidates.json": {
            "schema_version": 1,
            "artifact_type": (
                "stage1b_measurement_primitives_near_threshold_candidates"
            ),
            "candidate_count": scanner["candidate_count"],
            "candidates": scanner["candidates"],
        },
        "local_response_validation_summary.json": {
            "schema_version": 1,
            "artifact_type": (
                "stage1b_measurement_primitives_local_response_validation_summary"
            ),
            "status": "passed",
            "edge_floor": config["responses"]["edge_floor"],
            **metrics,
            "method": config["responses"]["method"],
            "convention": config["responses"]["convention"],
            "graph_edge_orientation": "target_row_source_column",
            "graph_edge_used_by_targeted_path": False,
            "calibration_pair_ids_disjoint": True,
        },
        "local_response_validation_pairs.json": {
            "schema_version": 1,
            "artifact_type": (
                "stage1b_measurement_primitives_local_response_validation_pairs"
            ),
            "pair_count": len(validation["pairs"]),
            "pairs": validation["pairs"],
        },
        "memory_timing_summary.json": {
            "schema_version": 1,
            "artifact_type": "stage1b_measurement_primitives_memory_timing",
            "status": "passed",
            "safety_limits": config["safety_limits"],
            "worker": worker["telemetry"],
            "supervisor": compact_supervisor,
        },
        "attempts.json": {
            "schema_version": 1,
            "artifact_type": "stage1b_measurement_primitives_attempts",
            "status": "passed",
            "scientific_retry_count": 0,
            "attempts": [
                {
                    "mode": "canonical",
                    "fresh_process": True,
                    "worker_status": "passed",
                    "supervisor_returncode": supervisor["returncode"],
                    "timed_out": supervisor["timed_out"],
                    "safety_terminated": supervisor["safety_terminated"],
                    "telemetry_failures": supervisor["telemetry_failures"],
                    "calibration_artifact_read": False,
                }
            ],
        },
    }
    if set(records) != ARTIFACT_ALLOWLIST - {"checksums.sha256"}:
        raise RuntimeError("assembler record set differs from the frozen allowlist")
    return records


def main() -> int:
    args = _parser().parse_args()
    config = load_stage1b_config(args.config)
    output = args.output_dir.resolve(strict=False)
    expected_output = (
        REPOSITORY_ROOT / config["artifacts"]["result_directory"]
    ).resolve(strict=False)
    if output != expected_output or output.exists():
        raise RuntimeError(
            "final output must be the new exact Stage 1B result directory"
        )
    if (
        _git("rev-parse", "HEAD") != args.execution_commit
        or _git("branch", "--show-current") != BRANCH
        or _git("status", "--porcelain")
    ):
        raise RuntimeError("artifact assembly requires the clean execution commit")
    worker = _load(args.canonical_worker, "canonical worker")
    preflight = _load(args.preflight, "canonical preflight")
    supervisor = _load(args.supervisor, "canonical supervisor")
    _validate_inputs(worker, preflight, supervisor, config, args.execution_commit)
    records = _records(worker, supervisor, config, args.execution_commit)

    generated_root = REPOSITORY_ROOT / config["artifacts"]["generated_directory"]
    generated_root.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix="bundle-stage-", dir=generated_root)
    ).resolve(strict=True)
    for name, record in records.items():
        write_json_atomic(staging / name, record)
    targets = [staging / name for name in sorted(records)]
    write_checksum_manifest_atomic(staging / "checksums.sha256", targets, root=staging)
    result = validate_stage1b_artifacts(
        staging, config=config, execution_commit=args.execution_commit
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, output)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
