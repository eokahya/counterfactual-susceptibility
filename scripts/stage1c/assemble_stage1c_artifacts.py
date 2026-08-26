#!/usr/bin/env python3
"""Assemble and validate the compact Stage 1C result directory.

Only JSON records emitted by the bounded workers cross this boundary.  No
model/runtime object is imported here, and the final directory is validated
through the independent stdlib validator before it is materialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import median
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cfsus.stage1c.analysis import aggregate_analyses, analyze_pair, classify_outcome

ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = (
    "run_manifest.json",
    "asset_manifest.json",
    "environment_manifest.json",
    "prediction_manifest.json",
    "intervention_sweeps.json",
    "crossing_summary.json",
    "local_linearity_summary.json",
    "memory_timing_summary.json",
    "attempts.json",
)


def _compact_supervisor(value: dict[str, Any]) -> dict[str, Any]:
    keys = (
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
    if any(key not in value for key in keys):
        raise RuntimeError("supervisor record is incomplete")
    return {key: value[key] for key in keys}


def _readiness(outcome: str, primary_crossings: int) -> dict[str, Any]:
    return {
        "stage1b_measurement_primitives": "completed",
        "stage1c_first_prediction": "completed",
        "stage1c_scientific_outcome": outcome,
        "counterfactual_susceptibility_result": (
            "preliminary_single_prompt"
            if outcome in {"supported", "mixed"}
            else "negative_single_prompt"
            if outcome == "not_supported"
            else "none"
        ),
        "gate_crossing_result": (
            "prospective_single_prompt" if primary_crossings > 0 else "none"
        ),
        "behavioral_importance_result": "none",
        "mediation_result": "none",
        "official_bf16_reproduction": "pending",
        "reference_clt_reproduction": "pending",
        "paper_results_readiness": False,
    }


def load(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise RuntimeError(f"unsafe input: {path}")
    if path.stat().st_size > 2 * 1024 * 1024:
        raise RuntimeError(f"input exceeds cap: {path}")

    def constant(value: str) -> None:
        raise RuntimeError(f"non-finite JSON constant: {value}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise RuntimeError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    value = json.loads(
        raw.decode("utf-8"), parse_constant=constant, object_pairs_hook=unique
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be object: {path}")
    return value


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
        env={
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        },
    )
    return result.stdout.strip()


def records(
    prediction: dict[str, Any],
    prediction_worker: dict[str, Any],
    intervention_worker: dict[str, Any],
    prediction_supervisor: dict[str, Any],
    intervention_supervisor: dict[str, Any],
    execution: str,
    prediction_sha256: str,
) -> dict[str, dict[str, Any]]:
    if prediction_worker.get("status") != "passed":
        raise RuntimeError("prediction worker did not pass")
    if intervention_worker.get("status") != "passed":
        raise RuntimeError("intervention worker did not pass")
    if prediction_worker.get("prediction_manifest") != prediction:
        raise RuntimeError("prediction worker and tracked prediction differ")
    if intervention_worker.get("prediction_manifest_sha256") != prediction_sha256:
        raise RuntimeError("intervention worker used a different prediction manifest")
    for label, supervisor in (
        ("prediction", prediction_supervisor),
        ("intervention", intervention_supervisor),
    ):
        if any(
            supervisor.get(key) != expected
            for key, expected in (
                ("returncode", 0),
                ("timed_out", False),
                ("safety_terminated", False),
                ("telemetry_failures", 0),
            )
        ):
            raise RuntimeError(f"{label} supervisor did not pass")
    sweeps = intervention_worker.get("sweeps", [])
    if not isinstance(sweeps, list):
        raise RuntimeError("intervention worker sweeps are invalid")
    rows_by_id = {
        row["pair_id"]: row
        for group in prediction.get("selected_groups", {}).values()
        for row in group
    }
    pair_analyses = [
        analyze_pair(
            rows_by_id[item["pair_id"]],
            item["points"],
            prediction["protocol"]["analysis"],
        )
        for item in sweeps
    ]
    outcome = classify_outcome(pair_analyses).value
    aggregate = aggregate_analyses(pair_analyses)
    selected = [
        row for group in prediction.get("selected_groups", {}).values() for row in group
    ]
    primary_count = len(prediction["selected_groups"]["primary"])
    selected_ids = {row["pair_id"] for row in selected}
    sweep_ids = {item["pair_id"] for item in sweeps}
    if primary_count == 0:
        if sweeps or intervention_worker.get("attempt_count") != 0:
            raise RuntimeError("no-primary result contains an intervention attempt")
    elif sweep_ids != selected_ids or intervention_worker.get("attempt_count") != 1:
        raise RuntimeError("canonical sweeps differ from the frozen selected groups")
    if intervention_worker.get("scientific_outcome") != outcome:
        raise RuntimeError("worker and frozen analysis scientific outcomes differ")
    crossing = {
        "schema_version": 1,
        "artifact_type": "stage1c_crossing_summary",
        "status": "passed",
        "scientific_outcome": outcome,
        "pairs": pair_analyses,
        "aggregate": aggregate,
    }
    local_errors = [
        float(point["target_preactivation_symmetric_normalized_error"])
        for item in sweeps
        for point in item["points"]
    ]
    local_linearity = {
        "schema_version": 1,
        "artifact_type": "stage1c_local_linearity_summary",
        "status": "passed",
        "point_count": len(local_errors),
        "median_symmetric_normalized_error": (
            median(local_errors) if local_errors else None
        ),
        "p95_symmetric_normalized_error": (
            sorted(local_errors)[max(1, math.ceil(0.95 * len(local_errors))) - 1]
            if local_errors
            else None
        ),
        "undefined_metric_reason": None if local_errors else "no_intervention_points",
    }
    primary_crossings = int(aggregate["primary_full_ablation_crossing_count"])
    readiness = _readiness(outcome, primary_crossings)
    run = {
        "schema_version": 1,
        "artifact_type": "stage1c_first_prospective_prediction_run_manifest",
        "status": intervention_worker.get(
            "run_status", "completed_stage1c_first_prospective_prediction"
        ),
        "verdict": intervention_worker.get(
            "verdict", "completed_stage1c_first_prospective_prediction"
        ),
        "scientific_outcome": outcome,
        "branch": "stage-1c-first-prospective-prediction",
        "base_commit": "efbf70a7e462e640a0e1819a93f3b92727bbd193",
        "execution_commit": execution,
        "pre_intervention_commit": execution,
        "fresh_canonical_run": primary_count > 0,
        "intervention_required": primary_count > 0,
        "canonical_source_suppression_api_calls": int(
            intervention_worker.get("canonical_source_suppression_api_calls", 0)
        ),
        "scientific_retry_count": int(
            intervention_worker.get("scientific_retry_count", 0)
        ),
        "prediction_manifest_sha256": prediction_sha256,
        "claim_boundary": {
            "behavioral_importance_result": "none",
            "mediation_result": "none",
            "official_bf16_reproduction": "pending",
            "reference_clt_reproduction": "pending",
            "paper_results_readiness": False,
        },
        "readiness": readiness,
        "secondary_regime": {
            "status": "not_implemented",
            "result": None,
        },
    }
    raw_assets = intervention_worker.get("asset_manifest") or prediction_worker.get(
        "asset_manifest"
    )
    if not isinstance(raw_assets, dict) or raw_assets.get("status") != "verified":
        raise RuntimeError("immutable asset evidence is missing")
    assets = {
        "schema_version": 1,
        "artifact_type": "stage1c_asset_manifest",
        "status": "verified",
        "download_performed": False,
        "network_accessed": False,
        "authentication_used": False,
        "authentication_value_recorded": False,
        "exact_allowlist_hashes_verified": True,
        "actual_total_bytes": raw_assets["actual_total_bytes"],
        "model": {
            "identifier": raw_assets["model"]["identifier"],
            "revision": raw_assets["model"]["revision"],
            "total_bytes": raw_assets["model"]["total_bytes"],
        },
        "transcoder": {
            "identifier": raw_assets["transcoder"]["identifier"],
            "revision": raw_assets["transcoder"]["revision"],
            "subfolder": raw_assets["transcoder"]["subfolder"],
            "layer_count": 18,
            "total_bytes": raw_assets["transcoder"]["total_bytes"],
        },
    }
    raw_environment = intervention_worker.get("environment") or prediction_worker.get(
        "environment"
    )
    if not isinstance(raw_environment, dict):
        raise RuntimeError("runtime environment evidence is missing")
    environment = {
        "schema_version": 1,
        "artifact_type": "stage1c_environment_manifest",
        "status": "passed",
        "execution_commit": execution,
        "platform": {
            "system": "Darwin",
            "machine": raw_environment["machine"],
            "python": raw_environment["python"],
            "host_class": "Apple M2 Max 32 GiB unified memory",
        },
        "packages": {
            "circuit-tracer": "0.5.2",
            "torch": raw_environment["torch"],
            "nnsight": raw_environment["nnsight"],
            "transformers": raw_environment["transformers"],
        },
        "accelerator": {
            "device": "mps:0",
            "dtype": "torch.bfloat16",
            "mps_built": raw_environment["mps_built"],
            "mps_available": raw_environment["mps_available"],
            "fallback_variable_present": raw_environment["fallback_variable_present"],
            "outer_autocast_enabled": raw_environment["outer_autocast_enabled"],
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
    result: dict[str, dict[str, Any]] = {
        "run_manifest.json": run,
        "asset_manifest.json": assets,
        "environment_manifest.json": environment,
        "prediction_manifest.json": prediction,
        "intervention_sweeps.json": {
            "schema_version": 1,
            "artifact_type": "stage1c_intervention_sweeps",
            "status": "passed",
            "pairs": sweeps,
        },
        "crossing_summary.json": crossing,
        "local_linearity_summary.json": local_linearity,
        "memory_timing_summary.json": {
            "schema_version": 1,
            "artifact_type": "stage1c_memory_timing_summary",
            "status": "passed",
            "prediction": prediction_worker.get("telemetry"),
            "intervention": intervention_worker.get("telemetry"),
            "prediction_supervisor": _compact_supervisor(prediction_supervisor),
            "intervention_supervisor": _compact_supervisor(intervention_supervisor),
        },
        "attempts.json": {
            "schema_version": 1,
            "artifact_type": "stage1c_attempts",
            "status": "passed",
            "scientific_retry_count": int(
                intervention_worker.get("scientific_retry_count", 0)
            ),
            "prediction_attempts": 1,
            "intervention_attempts": int(intervention_worker.get("attempt_count", 0)),
            "canonical_source_suppression_api_calls": int(
                intervention_worker.get("canonical_source_suppression_api_calls", 0)
            ),
            "intervention_required": primary_count > 0,
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--prediction-worker", type=Path, required=True)
    parser.add_argument("--prediction-supervisor", type=Path, required=True)
    parser.add_argument("--intervention-worker", type=Path, required=True)
    parser.add_argument("--intervention-supervisor", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (
        git("rev-parse", "HEAD") != args.execution_commit
        or git("branch", "--show-current") != "stage-1c-first-prospective-prediction"
    ):
        raise RuntimeError("artifact assembly Git identity mismatch")
    if git("status", "--porcelain", "--untracked-files=all") != "":
        raise RuntimeError("artifact assembly requires a clean execution commit")
    expected_output = ROOT / "results/stage1c_first_prospective_prediction"
    if args.output_dir.absolute() != expected_output:
        raise RuntimeError("final artifact directory differs from the frozen path")
    prediction = load(args.prediction_manifest)
    prediction_sha256 = hashlib.sha256(
        args.prediction_manifest.read_bytes()
    ).hexdigest()
    if (
        args.prediction_manifest.absolute()
        != expected_output / "prediction_manifest.json"
    ):
        raise RuntimeError("artifact assembly requires the tracked prediction manifest")
    if not args.output_dir.is_dir() or args.output_dir.is_symlink():
        raise RuntimeError("final result directory must contain the frozen prediction")
    if {item.name for item in args.output_dir.iterdir()} != {
        "prediction_manifest.json"
    }:
        raise RuntimeError("pre-intervention result directory contains extra files")
    prediction_worker = load(args.prediction_worker)
    intervention_worker = load(args.intervention_worker)
    prediction_supervisor = load(args.prediction_supervisor)
    intervention_supervisor = load(args.intervention_supervisor)
    result = records(
        prediction,
        prediction_worker,
        intervention_worker,
        prediction_supervisor,
        intervention_supervisor,
        args.execution_commit,
        prediction_sha256,
    )
    staging = Path(tempfile.mkdtemp(prefix="stage1c-bundle-", dir="/private/tmp"))
    try:
        for name, value in result.items():
            (staging / name).write_text(
                json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        lines = []
        for name in sorted(result):
            digest = hashlib.sha256((staging / name).read_bytes()).hexdigest()
            lines.append(f"{digest}  {name}")
        (staging / "checksums.sha256").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        validator = ROOT / "scripts/stage1c/validate_stage1c_artifacts.py"
        subprocess.run(
            [
                sys.executable,
                str(validator),
                "--bundle",
                str(staging),
                "--execution-commit",
                args.execution_commit,
            ],
            check=True,
            timeout=60,
        )
        if (staging / "prediction_manifest.json").read_bytes() != (
            args.output_dir / "prediction_manifest.json"
        ).read_bytes():
            raise RuntimeError("staged prediction manifest changed")
        for name in (*ALLOWLIST, "checksums.sha256"):
            if name != "prediction_manifest.json":
                os.replace(staging / name, args.output_dir / name)
    except BaseException:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": "assembled", "artifact_count": 10}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
