"""Compact Stage 1D artifact assembly from the durable point journal."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from cfsus.stage1c_v3.serialization import read_json_strict, write_json_new
from cfsus.stage1d.config import COMPLETED_STATUS, EXPERIMENT_CLASS
from cfsus.stage1d.metrics import compute_benchmark_summary

JSON_ARTIFACTS = (
    "protocol_manifest.json",
    "prediction_manifest.json",
    "panel_membership.json",
    "quantization_audit.json",
    "full_ablation_points.json",
    "calibration_sweeps.json",
    "benchmark_summary.json",
    "run_manifest.json",
    "environment_manifest.json",
)


def read_completed_journal(path: Path) -> list[dict[str, Any]]:
    """Read exact alternating call-start/completed records from JSONL."""

    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("point journal is missing, unsafe, or oversized")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or len(lines) % 2:
        raise ValueError("point journal does not contain complete record pairs")
    points: list[dict[str, Any]] = []
    for expected, (left, right) in enumerate(
        zip(lines[::2], lines[1::2], strict=True), start=1
    ):
        started = json.loads(left)
        completed = json.loads(right)
        if (
            not isinstance(started, dict)
            or not isinstance(completed, dict)
            or started
            != {
                "call_index": expected,
                "pair_id": started.get("pair_id"),
                "record_type": "source_suppression_call_started",
            }
            or completed.get("record_type") != "point_completed"
            or completed.get("call_index") != expected
            or completed.get("pair_id") != started["pair_id"]
            or set(completed) != {"record_type", "call_index", "pair_id", "point"}
            or not isinstance(completed.get("point"), dict)
            or completed["point"].get("pair_id") != started["pair_id"]
            or completed["point"].get("source_suppression_api_call_index") != expected
        ):
            raise ValueError("point journal ordering or identity differs")
        points.append(completed["point"])
    return points


def _prediction_pairs(prediction: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {
        pair["pair_id"]: pair
        for prompt in prediction["prompts"]
        for pair in prompt["execution_pairs"]
    }
    expected = sum(len(prompt["execution_pairs"]) for prompt in prediction["prompts"])
    if len(result) != expected:
        raise ValueError("prediction execution pair IDs are not globally unique")
    return result


def _group_points(
    prediction: dict[str, Any], points: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    pairs = _prediction_pairs(prediction)
    grouped: dict[str, list[dict[str, Any]]] = {pair_id: [] for pair_id in pairs}
    for point in points:
        pair_id = point.get("pair_id")
        if pair_id not in grouped:
            raise ValueError("journal contains a non-frozen pair")
        grouped[pair_id].append(point)
    if any(not rows for rows in grouped.values()):
        raise ValueError("a frozen execution pair has no completed point")
    for pair_id, rows in grouped.items():
        rows.sort(key=lambda item: float(item["realized_suppression"]))
        if len({float(item["actual_bf16_value_passed"]) for item in rows}) != len(rows):
            raise ValueError(f"pair {pair_id} repeats an applied BF16 value")
    return grouped


def build_records(
    *,
    protocol: dict[str, Any],
    prediction: dict[str, Any],
    worker: dict[str, Any],
    points: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Rebuild every scientific artifact from journal points in a fresh process."""

    grouped = _group_points(prediction, points)
    pairs = _prediction_pairs(prediction)
    reported_calls = worker.get("instrumented_evaluation_api_calls")
    if reported_calls != len(points):
        raise ValueError("worker call count differs from completed journal points")
    full_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    detailed_full_references: list[str] = []
    for pair_id, pair in pairs.items():
        rows = grouped[pair_id]
        if pair["detailed_role"] is not None:
            calibration_rows.append(
                {
                    "prompt_id": pair["prompt_id"],
                    "pair_id": pair_id,
                    "detailed_role": pair["detailed_role"],
                    "point_count": len(rows),
                    "points": rows,
                }
            )
            if pair["full_ablation_selected"]:
                detailed_full_references.append(pair_id)
        else:
            if len(rows) != 1:
                raise ValueError("non-detailed pair must have one full-ablation point")
            if not any(
                mapping.get("requested_alpha") == 1.0
                for mapping in rows[0]["requested_mappings"]
            ):
                raise ValueError("non-detailed point is not full ablation")
            full_rows.append(rows[0])
    panel = {
        "schema_version": 1,
        "artifact_type": "stage1d_panel_membership",
        "status": "passed",
        "experiment_class": EXPERIMENT_CLASS,
        "prompts": [
            {
                "id": prompt["id"],
                "method_pair_ids": prompt["method_pair_ids"],
                "directional_pair_ids": prompt["directional_pair_ids"],
                "detailed_pair_ids": prompt["detailed_pair_ids"],
                "missing_strata": prompt["missing_strata"],
            }
            for prompt in prediction["prompts"]
        ],
    }
    quantization = {
        "schema_version": 1,
        "artifact_type": "stage1d_quantization_audit",
        "status": "passed",
        "experiment_class": EXPERIMENT_CLASS,
        "full_candidate_rows_persisted": False,
        "prompts": [
            {"id": prompt["id"], **prompt["quantization_audit"]}
            for prompt in prediction["prompts"]
        ],
    }
    full = {
        "schema_version": 1,
        "artifact_type": "stage1d_full_ablation_points",
        "status": "passed",
        "experiment_class": EXPERIMENT_CLASS,
        "points": full_rows,
        "detailed_full_point_references": detailed_full_references,
    }
    calibration = {
        "schema_version": 1,
        "artifact_type": "stage1d_calibration_sweeps",
        "status": "passed",
        "experiment_class": EXPERIMENT_CLASS,
        "sweeps": calibration_rows,
    }
    summary = compute_benchmark_summary(prediction, grouped, config)
    summary["experiment_class"] = EXPERIMENT_CLASS
    telemetry = worker["telemetry"]
    environment = {
        "schema_version": 1,
        "artifact_type": "stage1d_environment_manifest",
        "status": "passed",
        "experiment_class": EXPERIMENT_CLASS,
        "runtime": worker["environment"],
        "telemetry": {
            "started_at_unix": telemetry["started_at_unix"],
            "finished_at_unix": telemetry["finished_at_unix"],
            "sample_count": telemetry["sample_count"],
            "sampling_interval_seconds": telemetry["sampling_interval_seconds"],
            "attempt_peaks": telemetry["attempt_peaks"],
            "thermal_states": telemetry["thermal_states"],
            "violations": telemetry["violations"],
            "telemetry_failures": telemetry["telemetry_failures"],
        },
        "privacy": {
            "network_accessed": False,
            "credential_values_read": False,
            "secret_values_recorded": False,
            "private_paths_recorded": False,
        },
    }
    run = {
        "schema_version": 1,
        "artifact_type": "stage1d_run_manifest",
        "status": COMPLETED_STATUS,
        "experiment_class": EXPERIMENT_CLASS,
        "base_commit": config["base_commit"],
        "protocol_freeze_commit": protocol["protocol_commit"],
        "prediction_freeze_commit": worker["prediction_freeze_commit"],
        "pre_run_commit": worker["pre_run_commit"],
        "canonical_attempt_count": 1,
        "scientific_retry_count": 0,
        "instrumented_evaluation_api_calls": len(points),
        "completed_journal_points": len(points),
        "serialized_unique_point_rows": len(points),
        "final_artifacts_rebuilt_from_journal_in_fresh_process": True,
        "standalone_recomputation_required": True,
        "project_decision": summary["project_decision"],
        "claim_boundary": dict(config["claim_boundary"]),
    }
    return {
        "protocol_manifest.json": protocol,
        "prediction_manifest.json": prediction,
        "panel_membership.json": panel,
        "quantization_audit.json": quantization,
        "full_ablation_points.json": full,
        "calibration_sweeps.json": calibration,
        "benchmark_summary.json": summary,
        "run_manifest.json": run,
        "environment_manifest.json": environment,
    }


def publish_records(output: Path, records: dict[str, dict[str, Any]]) -> None:
    """Publish every absent allowlisted JSON and then its checksum sidecar."""

    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise ValueError("artifact output directory is a symlink")
    for name in JSON_ARTIFACTS:
        value = records[name]
        path = output / name
        if path.exists():
            existing = read_json_strict(path)
            if existing != value:
                raise ValueError(f"existing frozen artifact differs: {name}")
        else:
            write_json_new(path, value)
    lines = [
        f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}"
        for name in JSON_ARTIFACTS
    ]
    checksum = output / "checksums.sha256"
    if checksum.exists():
        raise ValueError("checksum sidecar already exists")
    encoded = ("\n".join(lines) + "\n").encode("ascii")
    descriptor = os.open(
        checksum,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        if os.write(descriptor, encoded) != len(encoded):
            raise ValueError("short checksum-sidecar write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "JSON_ARTIFACTS",
    "build_records",
    "publish_records",
    "read_completed_journal",
]
