#!/usr/bin/env python3
"""Assemble only the allowlisted accepted Stage 1A-S-BF16 artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cfsus.reproduction.artifacts import (  # noqa: E402
    sha256_file,
    write_json_atomic,
)
from cfsus.reproduction.small_model_mps_bf16 import (  # noqa: E402
    ARTIFACT_ALLOWLIST,
    BACKEND,
    COMPLETED_STATUS,
    EXECUTION_BASE_COMMIT,
    EXPERIMENT_CLASS,
    MODEL_IDENTIFIER,
    MODEL_REVISION,
    TRANSCODER_IDENTIFIER,
    TRANSCODER_REVISION,
    TRANSCODER_SUBFOLDER,
    UPSTREAM_REVISION,
    assert_fallback_disabled,
    load_bf16_config,
)
from cfsus.reproduction.small_model_mps_bf16_artifacts import (  # noqa: E402
    BRANCH,
    derive_memory_entries,
    strict_json_load,
    validate_bundle,
)

GENERATED_ROOT = Path("results/generated/stage1a_small_model_mps_bf16")
ATTEMPT_SET_PATHS = (
    "model_gate_verified_cache/model_gate_attempts.json",
    "loaded_semantics/loaded_semantics_attempts.json",
    "loaded_semantics_inference_fix/loaded_semantics_attempts.json",
    "full_plt/full_plt_attempts.json",
    "replacement_runtime/replacement_runtime_attempts.json",
    "smoke/smoke_attempts.json",
    "smoke_layerwise_threshold_fix/smoke_attempts.json",
    "smoke_selection_audit/smoke_attempts.json",
    "accepted/accepted_attempts.json",
)
ORPHAN_ATTEMPT_PATHS = ("model_gate/model_forward_attempt.json",)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _git(command: list[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *command],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env={
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        },
    )
    return result.stdout.strip()


def _read(relative: str) -> dict[str, Any]:
    path = (REPOSITORY_ROOT / GENERATED_ROOT / relative).resolve(strict=True)
    generated = (REPOSITORY_ROOT / GENERATED_ROOT).resolve(strict=True)
    if path.is_symlink() or not path.is_file() or not path.is_relative_to(generated):
        raise RuntimeError(f"generated source is missing or unsafe: {relative}")
    return strict_json_load(path)


def _accepted_worker(attempts: dict[str, Any]) -> dict[str, Any]:
    passed: list[dict[str, Any]] = []
    for attempt in attempts.get("attempts", []):
        worker = attempt.get("worker", {})
        if attempt.get("stage") == "accepted" and worker.get("status") == "passed":
            passed.append(worker)
    if len(passed) != 1:
        raise RuntimeError("exactly one accepted worker must have passed")
    return passed[0]


def _write_checksums(output: Path) -> None:
    names = sorted(ARTIFACT_ALLOWLIST - {"checksums.sha256"})
    text = "".join(f"{sha256_file(output / name)}  {name}\n" for name in names)
    temporary = output / ".checksums.sha256.tmp"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, output / "checksums.sha256")


def main() -> int:
    arguments = _parser().parse_args()
    assert_fallback_disabled()
    config = load_bf16_config(arguments.config)
    head = _git(["rev-parse", "HEAD"])
    branch = _git(["branch", "--show-current"])
    status = _git(["status", "--porcelain"])
    if head != arguments.execution_commit or branch != BRANCH or status:
        raise RuntimeError("artifact assembly requires the clean execution commit")

    output = arguments.output_dir
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    expected_output = (
        REPOSITORY_ROOT / config["artifacts"]["result_directory"]
    ).resolve()
    if output.resolve() != expected_output or output.is_symlink():
        raise RuntimeError("artifact output directory identity is invalid")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("refusing to overwrite an existing artifact bundle")
    output.mkdir(parents=True, exist_ok=True)

    operator = _read("preflight/operator_probe_summary.json")
    assets = _read("preflight/asset_manifest_corrected_cache.json")
    model = _read("model_gate_verified_cache/model_forward_summary.json")
    fp32 = _read("model_gate_verified_cache/fp32_reference_summary.json")
    one_layer = _read("loaded_semantics_inference_fix/loaded_semantics_summary.json")
    full_plt = _read("full_plt/full_plt_summary.json")
    replacement = _read("replacement_runtime/replacement_runtime_summary.json")
    accepted = _read("accepted/accepted_summary.json")
    accepted_attempts = _read("accepted/accepted_attempts.json")
    worker = _accepted_worker(accepted_attempts)
    if worker.get("git", {}).get("execution_commit") != arguments.execution_commit:
        raise RuntimeError("accepted worker did not consume the execution commit")
    if accepted.get("profile") != "accepted" or accepted.get("status") != "passed":
        raise RuntimeError("accepted empirical summary is not successful")

    attempt_sets = [
        {"source": relative, "record": _read(relative)}
        for relative in ATTEMPT_SET_PATHS
    ]
    attempts_record = {
        "schema_version": 1,
        "artifact_type": "stage1a_small_model_mps_bf16_attempts",
        "status": "complete",
        "attempt_sets": attempt_sets,
        "orphan_worker_attempts": [
            {"source": relative, "record": _read(relative)}
            for relative in ORPHAN_ATTEMPT_PATHS
        ],
        "all_empirical_attempts_preserved": True,
    }
    memory_entries = derive_memory_entries(attempts_record)
    accepted_telemetry = worker["telemetry"]

    records: dict[str, dict[str, Any]] = {
        "environment_manifest.json": {
            "schema_version": 1,
            "artifact_type": "stage1a_small_model_mps_bf16_environment",
            "status": "passed",
            "branch": BRANCH,
            "execution_commit": arguments.execution_commit,
            "environment_path_class": (".venv-stage1a-small-model-mps-bf16/bin/python"),
            "environment": operator["environment"],
            "fallback_enabled": False,
            "secret_values_recorded": False,
        },
        "asset_manifest.json": assets,
        "preflight_summary.json": {
            "schema_version": 1,
            "artifact_type": "stage1a_small_model_mps_bf16_preflight",
            "status": "passed",
            "execution_commit": arguments.execution_commit,
            "operator_probe_status": operator["status"],
            "asset_status": assets["status"],
            "memory_projection": operator["memory_projection"],
            "projected_download_bytes": operator["projected_download_bytes"],
            "fallback_enabled": False,
            "network_used_for_execution": False,
            "paid_compute_used": False,
        },
        "operator_probe_summary.json": operator,
        "model_forward_summary.json": model,
        "fp32_reference_summary.json": fp32,
        "loaded_semantics_summary.json": {
            "schema_version": 1,
            "artifact_type": "stage1a_small_model_mps_bf16_loaded_semantics",
            "status": "passed",
            "one_layer": one_layer,
            "accepted": accepted["loaded_semantics"],
            "full_plt": full_plt,
            "replacement_runtime": replacement,
        },
        "attribution_summary.json": {
            "schema_version": 1,
            "artifact_type": "stage1a_small_model_mps_bf16_attribution",
            "status": accepted["status"],
            "profile": accepted["profile"],
            "batch_size": accepted["batch_size"],
            "prompt": accepted["prompt"],
            "token_ids": accepted["token_ids"],
            "token_count": accepted["token_count"],
            "module_guard": accepted["module_guard"],
            "attribution": accepted["attribution"],
            "selection": accepted["selection"],
            "selection_audit": accepted["selection_audit"],
        },
        "intervention_summary.json": {
            "schema_version": 1,
            "artifact_type": "stage1a_small_model_mps_bf16_intervention",
            "status": accepted["status"],
            "profile": accepted["profile"],
            "prompt": accepted["prompt"],
            "token_ids": accepted["token_ids"],
            "selection": accepted["selection"],
            "loaded_semantics": accepted["loaded_semantics"],
            "intervention": accepted["intervention"],
        },
        "memory_timing_summary.json": {
            "schema_version": 1,
            "artifact_type": "stage1a_small_model_mps_bf16_memory_timing",
            "status": "passed",
            "safety_limits": config["safety_limits"],
            "attempts": memory_entries,
            "accepted_started_at_unix": accepted_telemetry["started_at_unix"],
            "accepted_finished_at_unix": accepted_telemetry["finished_at_unix"],
            "accepted_attempt_peaks": accepted_telemetry["attempt_peaks"],
            "accepted_stage_peaks": accepted_telemetry["stage_peaks"],
        },
        "attempts.json": attempts_record,
        "run_manifest.json": {
            "schema_version": 1,
            "artifact_type": "stage1a_small_model_mps_bf16_run_manifest",
            "verdict": COMPLETED_STATUS,
            "experiment_class": EXPERIMENT_CLASS,
            "branch": BRANCH,
            "execution_commit": arguments.execution_commit,
            "provenance": {
                "execution_base_commit": EXECUTION_BASE_COMMIT,
                "upstream_revision": UPSTREAM_REVISION,
                "model_identifier": MODEL_IDENTIFIER,
                "model_revision": MODEL_REVISION,
                "transcoder_identifier": TRANSCODER_IDENTIFIER,
                "transcoder_revision": TRANSCODER_REVISION,
                "transcoder_subfolder": TRANSCODER_SUBFOLDER,
                "backend": BACKEND,
                "device": "mps",
                "dtype": "bfloat16",
            },
            "accepted_batch_size": accepted["batch_size"],
            "stage1b_engineering_readiness": True,
            "stage1b_empirical_claim_readiness": False,
            "official_bf16_reproduction": "pending",
            "reference_clt_reproduction": "pending",
            "counterfactual_susceptibility_result": "none",
            "paper_results_readiness": False,
            "raw_graph_persisted": False,
            "weights_or_cache_committed": False,
            "paper_results_changed": False,
            "claim_class": "local_small_model_runtime_validation_only",
            "artifact_files": sorted(ARTIFACT_ALLOWLIST),
        },
    }
    if set(records) != ARTIFACT_ALLOWLIST - {"checksums.sha256"}:
        raise RuntimeError("assembler record set differs from the frozen allowlist")
    for name, record in records.items():
        write_json_atomic(output / name, record)
    _write_checksums(output)
    result = validate_bundle(
        output,
        repository_root=REPOSITORY_ROOT,
        execution_commit=arguments.execution_commit,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
