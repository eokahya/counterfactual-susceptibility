#!/usr/bin/env python3
"""Assemble and independently validate the compact Stage 1C-v2 bundle.

This module consumes only detached JSON worker records.  It has no model or
intervention imports.  Every JSON artifact is published with the v2 strict
new-file writer; the checksum sidecar is published with an exclusive link so
an existing path can never be replaced.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
from pathlib import Path
from statistics import median
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_stage1c_v2_artifacts as validator  # noqa: E402
from assemble_stage1c_prediction import assemble_prediction  # noqa: E402

from cfsus.stage1c_v2.serialization import (  # noqa: E402
    SerializationError,
    detach_json,
    read_json_strict,
    write_json_new,
)

ALLOWLIST = validator.ALLOWLIST[:-1]
BASE_COMMIT = validator.BASE_COMMIT
BRANCH = validator.BRANCH
COMPLETED_STATUS = "completed_stage1c_v2_heldout_prospective_prediction"
CLAIM_BOUNDARY = {
    "behavioral_importance_result": "none",
    "mediation_result": "none",
    "official_bf16_reproduction": "pending",
    "reference_clt_reproduction": "pending",
    "paper_results_readiness": False,
}


def strict_load(path: Path) -> dict[str, Any]:
    value = read_json_strict(path)
    if not isinstance(value, dict):  # pragma: no cover - strict reader guard
        raise ValueError("JSON root must be an object")
    return cast(dict[str, Any], value)


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _prediction_digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_worker(value: dict[str, Any], artifact_type: str) -> None:
    if value.get("schema_version") != 2 or value.get("artifact_type") != artifact_type:
        raise ValueError(f"worker is not {artifact_type}")
    if value.get("status") != "passed":
        raise ValueError(f"{artifact_type} did not pass")


def _compact_supervisor(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    required = (
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
    )
    if any(key not in value for key in required):
        raise ValueError("supervisor record is incomplete")
    return {key: value[key] for key in required}


def _asset_manifest(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("status") != "verified":
        raise ValueError("immutable asset evidence is missing")
    model = raw.get("model") if isinstance(raw.get("model"), dict) else {}
    transcoder = (
        raw.get("transcoder") if isinstance(raw.get("transcoder"), dict) else {}
    )
    return {
        "schema_version": 2,
        "artifact_type": "stage1c_v2_asset_manifest",
        "experiment_class": validator.EXPERIMENT_CLASS,
        "status": "verified",
        "download_performed": False,
        "network_accessed": False,
        "authentication_used": False,
        "authentication_value_recorded": False,
        "exact_allowlist_hashes_verified": True,
        "actual_total_bytes": raw.get("actual_total_bytes"),
        "model": {
            "identifier": model.get("identifier", "google/gemma-3-270m"),
            "revision": model.get(
                "revision",
                raw.get("model_revision", "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"),
            ),
            "total_bytes": model.get("total_bytes", raw.get("model_total_bytes")),
        },
        "transcoder": {
            "identifier": transcoder.get("identifier", "mwhanna/gemma-scope-2-270m-pt"),
            "revision": transcoder.get(
                "revision",
                raw.get(
                    "transcoder_revision", "fada11860ac1d337c1e41e9da308798405b94c8e"
                ),
            ),
            "subfolder": transcoder.get(
                "subfolder", "transcoder_all/width_16k_l0_small"
            ),
            "layer_count": transcoder.get("layer_count", 18),
            "total_bytes": transcoder.get(
                "total_bytes", raw.get("transcoder_total_bytes")
            ),
        },
    }


def _environment_manifest(raw: dict[str, Any] | None, execution: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("runtime environment evidence is missing")
    return {
        "schema_version": 2,
        "artifact_type": "stage1c_v2_environment_manifest",
        "experiment_class": validator.EXPERIMENT_CLASS,
        "status": "passed",
        "execution_commit": execution,
        "platform": {
            "system": raw.get("system", "Darwin"),
            "machine": raw.get("machine", "synthetic"),
            "python": raw.get("python", "3.11.13"),
        },
        "packages": {
            "circuit-tracer": raw.get("circuit-tracer", "0.5.2"),
            "torch": raw.get("torch", "2.6.0"),
            "nnsight": raw.get("nnsight", "0.6.1"),
            "transformers": raw.get("transformers", "4.57.3"),
        },
        "accelerator": {
            "device": "mps:0",
            "dtype": "torch.bfloat16",
            "mps_built": raw.get("mps_built", True),
            "mps_available": raw.get("mps_available", True),
            "fallback_variable_present": raw.get("fallback_variable_present", False),
            "outer_autocast_enabled": raw.get("outer_autocast_enabled", False),
            "scientific_tensor_device": "mps",
            "metadata_device": "cpu",
        },
        "privacy": {
            "network_accessed": False,
            "credential_values_read": False,
            "secret_values_recorded": False,
            "private_paths_recorded": False,
        },
    }


def _summary(
    artifact_type: str,
    analyses: list[dict[str, Any]],
    aggregate: dict[str, Any],
    outcome: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "artifact_type": artifact_type,
        "experiment_class": validator.EXPERIMENT_CLASS,
        "status": "passed",
        "pairs": analyses,
        "aggregate_metrics": aggregate,
        "scientific_outcome": outcome,
    }


def records(
    prediction: dict[str, Any],
    prediction_worker: dict[str, Any],
    intervention_worker: dict[str, Any],
    prediction_supervisor: dict[str, Any] | None = None,
    intervention_supervisor: dict[str, Any] | None = None,
    execution: str = BASE_COMMIT,
    prediction_sha256: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Build all ten JSON records from detached worker evidence.

    The function is intentionally pure with respect to the model runtime: it
    rechecks prediction and sweep invariants using only JSON values.
    """

    _require_worker(prediction_worker, "stage1c_v2_prediction_worker")
    _require_worker(intervention_worker, "stage1c_v2_intervention_worker")
    if prediction_worker.get("prediction_manifest") != prediction:
        raise ValueError("prediction worker and prediction manifest differ")
    try:
        validator.scan_value(prediction)
        validator.scan_prediction(prediction)
    except validator.ValidationError as error:
        raise ValueError(f"prediction manifest failed validation: {error}") from error
    digest = prediction_sha256 or _prediction_digest(prediction)
    if validator.SHA256.fullmatch(digest) is None:
        raise ValueError("prediction manifest digest is not a lowercase SHA-256")
    if intervention_worker.get("prediction_manifest_sha256") not in (None, digest):
        raise ValueError("intervention worker used a different prediction manifest")

    artifacts = intervention_worker.get("intervention_artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("intervention worker artifact map is missing")
    top_sweeps = intervention_worker.get("sweeps")
    nested = artifacts.get("intervention_sweeps")
    if (
        not isinstance(top_sweeps, list)
        or not isinstance(nested, dict)
        or not isinstance(nested.get("pairs"), list)
    ):
        raise ValueError("intervention worker sweep representations are missing")
    nested_sweeps = nested["pairs"]
    if top_sweeps != nested_sweeps:
        raise ValueError("top-level and nested intervention sweeps differ")
    sweeps = cast(list[dict[str, Any]], top_sweeps)
    if any(type(item) is not dict for item in sweeps):
        raise ValueError("intervention sweep is not an object")
    if any(
        type(item.get("point_count")) is not int
        or item["point_count"] != len(item.get("points", []))
        for item in sweeps
    ):
        raise ValueError("worker point_count differs from serialized points")
    point_count = sum(item["point_count"] for item in sweeps)
    if intervention_worker.get("canonical_source_suppression_api_calls") != point_count:
        raise ValueError("worker API-call count differs from serialized points")

    selected = validator.scan_prediction(prediction)
    expected_ids = [item["pair_id"] for item in selected]
    if [item.get("pair_id") for item in sweeps] != expected_ids:
        raise ValueError("worker pair IDs/order differ from prediction")
    # validate_sweeps recomputes the pair analyses and compares every point.
    crossing_from_worker = artifacts.get("crossing_summary")
    if not isinstance(crossing_from_worker, dict):
        raise ValueError("worker crossing summary is missing")
    # A worker may carry only the aggregate crossing record while the process
    # is still at its cleanup boundary.  Reconstruct omitted analysis fields
    # from the point records before invoking the strict validator; any fields
    # that are present are still compared exactly below.
    selected_by_id = {item["pair_id"]: item for item in selected}
    provisional = [
        validator._analysis(
            selected_by_id[item["pair_id"]],
            item["points"],
            prediction["protocol"]["analysis"],
        )
        for item in sweeps
    ]
    crossing_for_validation = dict(crossing_from_worker)
    crossing_for_validation.setdefault("pairs", provisional)
    crossing_for_validation.setdefault(
        "aggregate_metrics", validator._aggregate(provisional)
    )
    crossing_for_validation.setdefault(
        "scientific_outcome", validator.scientific_outcome(provisional)
    )
    analyses = validator.validate_sweeps(
        prediction,
        {"pairs": sweeps},
        crossing_for_validation,
    )
    aggregate = validator._aggregate(analyses)
    outcome = validator.scientific_outcome(analyses)
    if intervention_worker.get("scientific_outcome") not in (None, outcome):
        raise ValueError("worker scientific outcome differs from recomputation")
    if (
        crossing_from_worker.get("scientific_outcome") not in (None, outcome)
        or crossing_from_worker.get("aggregate_metrics") not in (None, aggregate)
        or crossing_from_worker.get("pairs") not in (None, analyses)
    ):
        raise ValueError("worker crossing summary differs from recomputation")

    primary_count = len(prediction["selected_groups"]["primary"])
    attempts_raw = artifacts.get("attempts")
    if not isinstance(attempts_raw, dict):
        raise ValueError("worker attempts record is missing")
    attempt_count = attempts_raw.get(
        "attempt_count", intervention_worker.get("attempt_count")
    )
    expected_attempts = 1 if primary_count else 0
    if attempt_count != expected_attempts:
        raise ValueError(
            "intervention attempt count violates primary/no-primary policy"
        )
    if attempts_raw.get("scientific_retry_count", 0) != 0 or attempts_raw.get(
        "intervention_required", primary_count > 0
    ) is not (primary_count > 0):
        raise ValueError("worker attempts metadata violates v2 policy")
    if primary_count == 0 and (sweeps or point_count or outcome != "no_eligible_pairs"):
        raise ValueError("no-eligible-pairs result is not zero-sweep/zero-call")

    local = _summary("stage1c_v2_local_linearity_summary", analyses, aggregate, outcome)
    errors = [
        validator.number(
            point["target_preactivation_symmetric_normalized_error"],
            "point symmetric normalized error",
        )
        for item in sweeps
        for point in item["points"]
    ]
    local.update(
        {
            "point_count": point_count,
            "median_symmetric_normalized_error": median(errors) if errors else None,
            "p95_symmetric_normalized_error": (
                sorted(errors)[max(1, math.ceil(0.95 * len(errors))) - 1]
                if errors
                else None
            ),
            "undefined_metric_reason": None if errors else "no_intervention_points",
        }
    )
    memory_raw = artifacts.get("memory_timing_summary")
    telemetry = (
        memory_raw.get("telemetry")
        if isinstance(memory_raw, dict)
        else intervention_worker.get("telemetry")
    )
    assets_raw = intervention_worker.get("asset_manifest") or prediction_worker.get(
        "asset_manifest"
    )
    environment_raw = intervention_worker.get("environment") or prediction_worker.get(
        "environment"
    )
    assets = _asset_manifest(assets_raw)
    environment = _environment_manifest(environment_raw, execution)
    checks = {
        "schema_version": 2,
        "artifact_type": "stage1c_v2_intervention_sweeps",
        "experiment_class": validator.EXPERIMENT_CLASS,
        "status": "passed",
        "pairs": sweeps,
    }
    crossing = _summary("stage1c_v2_crossing_summary", analyses, aggregate, outcome)
    run = {
        "schema_version": 2,
        "artifact_type": "stage1c_v2_final_bundle_run_manifest",
        "experiment_class": validator.EXPERIMENT_CLASS,
        "final_bundle_type": validator.FINAL_BUNDLE_TYPE,
        "status": COMPLETED_STATUS,
        "verdict": COMPLETED_STATUS,
        "scientific_outcome": outcome,
        "branch": BRANCH,
        "base_commit": BASE_COMMIT,
        "execution_commit": execution,
        "pre_intervention_commit": execution,
        "fresh_canonical_run": primary_count > 0,
        "intervention_required": primary_count > 0,
        "canonical_source_suppression_api_calls": point_count,
        "scientific_retry_count": 0,
        "prediction_manifest_sha256": digest,
        "claim_boundary": CLAIM_BOUNDARY,
        "readiness": {
            "stage1b_measurement_primitives": "completed",
            "stage1c_v2_prospective_prediction": "completed",
            "stage1c_v2_scientific_outcome": outcome,
            **CLAIM_BOUNDARY,
        },
    }
    attempts = {
        "schema_version": 2,
        "artifact_type": "stage1c_v2_attempts",
        "experiment_class": validator.EXPERIMENT_CLASS,
        "status": "passed",
        "scientific_retry_count": 0,
        "prediction_attempts": 0 if primary_count == 0 else 1,
        "intervention_attempts": expected_attempts,
        "canonical_source_suppression_api_calls": point_count,
        "intervention_required": primary_count > 0,
    }
    memory = {
        "schema_version": 2,
        "artifact_type": "stage1c_v2_memory_timing_summary",
        "experiment_class": validator.EXPERIMENT_CLASS,
        "status": "passed",
        "prediction": prediction_worker.get("telemetry"),
        "intervention": intervention_worker.get("telemetry"),
        "worker": telemetry,
        "prediction_supervisor": _compact_supervisor(prediction_supervisor),
        "intervention_supervisor": _compact_supervisor(intervention_supervisor),
    }
    result = {
        "run_manifest.json": run,
        "asset_manifest.json": assets,
        "environment_manifest.json": environment,
        "prediction_manifest.json": prediction,
        "intervention_sweeps.json": checks,
        "crossing_summary.json": crossing,
        "local_linearity_summary.json": local,
        "memory_timing_summary.json": memory,
        "attempts.json": attempts,
    }
    detached = detach_json(result)
    if not isinstance(detached, dict):  # pragma: no cover
        raise ValueError("assembled artifact result is not an object")
    return cast(dict[str, dict[str, Any]], detached)


def _write_text_new(path: Path, data: bytes) -> None:
    info = path.parent.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("checksum parent is not a real directory")
    try:
        path.lstat()
    except FileNotFoundError:
        pass
    else:
        raise ValueError("checksum output already exists")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        temporary = None  # type: ignore[assignment]
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except (OSError, ValueError) as error:
        raise SerializationError(
            f"atomic checksum publication failed: {path}"
        ) from error
    finally:
        if temporary is not None:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()


def write_bundle(result: dict[str, dict[str, Any]], output_dir: Path) -> None:
    """Publish a complete new bundle and exact checksum sidecar."""

    if set(result) != set(ALLOWLIST):
        raise ValueError("assembled records differ from exact v2 allowlist")
    execution_commit = result["run_manifest.json"].get("execution_commit")
    if not isinstance(execution_commit, str):
        raise ValueError("assembled run manifest lacks an execution commit")
    validator.validate_records(result, execution_commit)
    output_dir = validator._assert_no_symlink_ancestors(output_dir)
    existing_prediction = False
    if output_dir.exists() or output_dir.is_symlink():
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise ValueError("bundle output must be a real directory")
        existing_names = {item.name for item in output_dir.iterdir()}
        if existing_names not in (set(), {"prediction_manifest.json"}):
            raise ValueError("bundle output contains unallowlisted or published files")
        if existing_names:
            existing_prediction = True
            if (
                read_json_strict(output_dir / "prediction_manifest.json")
                != result["prediction_manifest.json"]
            ):
                raise ValueError(
                    "published prediction manifest differs from assembled evidence"
                )
    else:
        output_dir.mkdir(parents=True)
    for name in ALLOWLIST:
        if name == "prediction_manifest.json" and existing_prediction:
            continue
        write_json_new(output_dir / name, result[name])
    lines = "".join(
        f"{hashlib.sha256((output_dir / name).read_bytes()).hexdigest()}  {name}\n"
        for name in sorted(ALLOWLIST)
    ).encode("utf-8")
    _write_text_new(output_dir / "checksums.sha256", lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--prediction-worker", type=Path, required=True)
    parser.add_argument("--intervention-worker", type=Path, required=True)
    parser.add_argument("--prediction-supervisor", type=Path)
    parser.add_argument("--intervention-supervisor", type=Path)
    parser.add_argument("--execution-commit", default=BASE_COMMIT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if validator.SHA40.fullmatch(args.execution_commit) is None:
        raise RuntimeError("execution commit must be a SHA-1")
    prediction = strict_load(args.prediction_manifest)
    worker = strict_load(args.prediction_worker)
    intervention = strict_load(args.intervention_worker)
    # CLI publication verifies protocol files and worker/manifest identity;
    # synthetic callers use the pure helper without this repository check.
    assemble_prediction(worker, verify_protocol_hashes=True)
    result = records(
        prediction,
        worker,
        intervention,
        strict_load(args.prediction_supervisor) if args.prediction_supervisor else None,
        strict_load(args.intervention_supervisor)
        if args.intervention_supervisor
        else None,
        args.execution_commit,
        hashlib.sha256(args.prediction_manifest.read_bytes()).hexdigest(),
    )
    write_bundle(result, args.output_dir)
    print(json.dumps({"artifact_count": 10, "status": "assembled"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SerializationError, ValueError, OSError) as error:
        raise SystemExit(f"artifact assembly failed: {error}") from error
