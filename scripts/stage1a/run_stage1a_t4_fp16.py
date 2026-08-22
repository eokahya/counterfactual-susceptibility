#!/usr/bin/env python3
"""Orchestrate isolated Stage 1A T4/FP16 attempts with CUDA-OOM-only retry."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from reproduce_attribution import load_yaml, repository_root  # noqa: E402
from run_stage1a import _write_metadata_artifacts  # noqa: E402
from validate_t4_fp16_artifacts import (  # noqa: E402
    RUN_MANIFEST_NAME,
    validate_t4_artifact_directory,
    write_t4_checksums,
)

WORKER = SCRIPT_DIRECTORY / "run_stage1a_t4_fp16_worker.py"


def _git_output(*arguments: str) -> str:
    try:
        return subprocess.run(
            ("git", *arguments),
            cwd=repository_root(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("project Git provenance check failed") from exc


def _validate_source_checkout() -> tuple[str, bool]:
    from cfsus.reproduction.t4_fp16 import PROJECT_BASE_COMMIT

    commit = _git_output("rev-parse", "HEAD")
    if len(commit) != 40:
        raise RuntimeError("project HEAD is not an immutable commit")
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", PROJECT_BASE_COMMIT, commit),
        cwd=repository_root(),
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("project checkout does not contain the required base commit")
    dirty = bool(_git_output("status", "--porcelain"))
    if dirty:
        raise RuntimeError("project source checkout must be clean before execution")
    return commit, dirty


def _load_json(path: Path) -> dict[str, Any]:
    from cfsus.reproduction.artifacts import (
        assert_publication_safe,
        validate_json_value,
    )

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "isolated worker did not produce a valid attempt report"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError("isolated worker attempt report is not an object")
    validate_json_value(value)
    assert_publication_safe(value)
    return value


def _runtime_record(provenance: dict[str, Any]) -> dict[str, Any]:
    gpu = provenance.get("gpu")
    if not isinstance(gpu, dict):
        try:
            import torch  # type: ignore[import-not-found]

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            properties = torch.cuda.get_device_properties(0)
            gpu = {
                "name": str(properties.name),
                "compute_capability": [
                    int(properties.major),
                    int(properties.minor),
                ],
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                "torch_version": str(torch.__version__),
                "torch_cuda_version": str(torch.version.cuda),
            }
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError("CUDA provenance could not be observed") from exc
    return {
        "backend": "transformerlens",
        "device": "cuda",
        "gpu_name": gpu.get("name"),
        "compute_capability": gpu.get("compute_capability"),
        "torch_version": gpu.get("torch_version"),
        "torch_cuda_version": gpu.get("torch_cuda_version"),
        "reference_dtype": "bfloat16",
        "execution_dtype": "float16",
        "bf16_supported": gpu.get("bf16_supported"),
    }


def _summary_payload(directory: Path, name: str) -> dict[str, Any] | None:
    path = directory / name
    if not path.is_file() or path.is_symlink():
        return None
    value = _load_json(path)
    payload = value.get("payload")
    return payload if isinstance(payload, dict) else None


def _artifact_records(directory: Path) -> dict[str, dict[str, int | str]]:
    from cfsus.reproduction.artifacts import sha256_file
    from cfsus.reproduction.t4_fp16 import T4_SMALL_FILES

    records: dict[str, dict[str, int | str]] = {}
    for name in sorted(T4_SMALL_FILES - {RUN_MANIFEST_NAME}):
        path = directory / name
        if path.is_file() and not path.is_symlink():
            records[name] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return records


def _successful_checks(directory: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    attribution = _summary_payload(directory, "attribution_summary.json") or {}
    intervention = _summary_payload(directory, "intervention_summary.json") or {}
    semantics = _summary_payload(directory, "semantics_summary.json") or {}
    graph = attribution.get("graph")
    comparison = intervention.get("baseline_noop_comparison")
    determinism = intervention.get("determinism")
    gate = semantics.get("gate_check")
    value_check = semantics.get("intervention_value_check")
    parameter_check = provenance.get("parameter_finiteness_sample")
    threshold_check = provenance.get("threshold_finiteness")
    asset_integrity = provenance.get("asset_integrity")
    return {
        "immutable_assets_loaded": isinstance(asset_integrity, dict)
        and asset_integrity.get("verification") == "exact_file_content_hashes_matched",
        "model_parameter_samples_finite": isinstance(parameter_check, dict)
        and parameter_check.get("passed") is True,
        "thresholds_finite": isinstance(threshold_check, dict)
        and threshold_check.get("passed") is True,
        "baseline_logits_finite": intervention.get("nonfinite_count") == 0,
        "cached_values_finite": semantics.get("nonfinite_count") == 0,
        "attribution_values_finite": attribution.get("nonfinite_count") == 0,
        "intervention_values_finite": intervention.get("nonfinite_count") == 0,
        "nonfinite_count": sum(
            int(payload.get("nonfinite_count", 1))
            for payload in (attribution, intervention, semantics)
        ),
        "baseline_repeat_within_tolerance": isinstance(determinism, dict)
        and determinism.get("within_tolerance") is True,
        "noop_within_tolerance": isinstance(comparison, dict)
        and comparison.get("within_tolerance") is True,
        "jumprelu_semantics_passed": isinstance(gate, dict)
        and gate.get("strict_greater_than") is True
        and gate.get("equality_inactive") is True,
        "desired_value_mapping_passed": isinstance(value_check, dict)
        and len(value_check.get("desired_values", [])) == 3,
        "artifact_validation_passed": True,
        "attribution_graph_nonempty": isinstance(graph, dict)
        and int(graph.get("selected_feature_count", 0)) > 0,
        "intervention_completed": bool(intervention),
        "semantics_completed": bool(semantics),
    }


def _failed_checks(
    directory: Path,
    provenance: dict[str, Any],
    *,
    nonfinite_detected: bool,
) -> dict[str, Any]:
    attribution = _summary_payload(directory, "attribution_summary.json") or {}
    intervention = _summary_payload(directory, "intervention_summary.json") or {}
    semantics = _summary_payload(directory, "semantics_summary.json") or {}
    graph = attribution.get("graph")
    comparison = intervention.get("baseline_noop_comparison")
    determinism = intervention.get("determinism")
    gate = semantics.get("gate_check")
    value_check = semantics.get("intervention_value_check")
    parameter_check = provenance.get("parameter_finiteness_sample")
    threshold_check = provenance.get("threshold_finiteness")
    asset_integrity = provenance.get("asset_integrity")
    return {
        "immutable_assets_loaded": isinstance(asset_integrity, dict)
        and asset_integrity.get("verification") == "exact_file_content_hashes_matched",
        "model_parameter_samples_finite": isinstance(parameter_check, dict)
        and parameter_check.get("passed") is True,
        "thresholds_finite": isinstance(threshold_check, dict)
        and threshold_check.get("passed") is True,
        "baseline_logits_finite": intervention.get("nonfinite_count") == 0
        and bool(intervention),
        "cached_values_finite": semantics.get("nonfinite_count") == 0
        and bool(semantics),
        "attribution_values_finite": attribution.get("nonfinite_count") == 0
        and bool(attribution),
        "intervention_values_finite": intervention.get("nonfinite_count") == 0
        and bool(intervention),
        "nonfinite_count": max(
            int(nonfinite_detected),
            sum(
                int(payload.get("nonfinite_count", 0))
                for payload in (attribution, intervention, semantics)
            ),
        ),
        "baseline_repeat_within_tolerance": isinstance(determinism, dict)
        and determinism.get("within_tolerance") is True,
        "noop_within_tolerance": isinstance(comparison, dict)
        and comparison.get("within_tolerance") is True,
        "jumprelu_semantics_passed": isinstance(gate, dict)
        and gate.get("strict_greater_than") is True
        and gate.get("equality_inactive") is True,
        "desired_value_mapping_passed": isinstance(value_check, dict)
        and len(value_check.get("desired_values", [])) == 3,
        "artifact_validation_passed": False,
        "attribution_graph_nonempty": isinstance(graph, dict)
        and int(graph.get("selected_feature_count", 0)) > 0,
        "intervention_completed": bool(intervention),
        "semantics_completed": bool(semantics),
    }


def _timings(directory: Path, history: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "attempt_wall_seconds": [float(item["wall_seconds"]) for item in history],
        "attempt_peak_memory_bytes": [
            item.get("peak_memory_bytes") for item in history
        ],
    }
    for name, key in (
        ("attribution_summary.json", "attribution"),
        ("intervention_summary.json", "intervention"),
        ("semantics_summary.json", "semantics"),
    ):
        payload = _summary_payload(directory, name)
        timing = payload.get("timing") if payload else None
        if isinstance(timing, dict):
            result[key] = timing
    return result


def _write_manifest(
    *,
    directory: Path,
    project_commit: str,
    project_dirty: bool,
    status: str,
    selected_batch: int | None,
    history: list[dict[str, Any]],
    provenance: dict[str, Any],
    success: bool,
) -> dict[str, Any]:
    from cfsus.reproduction.artifacts import write_json_atomic
    from cfsus.reproduction.config import (
        OFFICIAL_MODEL_ID,
        OFFICIAL_MODEL_REVISION,
        OFFICIAL_TRANSCODER_ID,
        OFFICIAL_TRANSCODER_REVISION,
        OFFICIAL_UPSTREAM_REPOSITORY,
        OFFICIAL_UPSTREAM_REVISION,
    )
    from cfsus.reproduction.t4_fp16 import (
        PROJECT_BASE_COMMIT,
        REPRODUCTION_CLASS,
        T4_CLAIM_BOUNDARY,
        batch_deviation,
        validate_t4_run_manifest,
    )

    attempted = [int(item["batch_size"]) for item in history]
    history_text = " ".join(str(item.get("message", "")) for item in history).casefold()
    checks = (
        _successful_checks(directory, provenance)
        if success
        else _failed_checks(
            directory,
            provenance,
            nonfinite_detected=(
                status == "failed_precision"
                and any(
                    marker in history_text for marker in ("non-finite", "nan", "inf")
                )
            ),
        )
    )
    engineering_ready = (
        success
        and all(
            value is True for key, value in checks.items() if key != "nonfinite_count"
        )
        and checks["nonfinite_count"] == 0
    )
    manifest = {
        "schema_version": 1,
        "status": status,
        "reproduction_class": REPRODUCTION_CLASS,
        "claim_boundary": T4_CLAIM_BOUNDARY,
        "project": {
            "base_commit": PROJECT_BASE_COMMIT,
            "execution_commit": project_commit,
            "dirty": project_dirty,
        },
        "upstream": {
            "repository": OFFICIAL_UPSTREAM_REPOSITORY,
            "revision": OFFICIAL_UPSTREAM_REVISION,
        },
        "model": {
            "identifier": OFFICIAL_MODEL_ID,
            "revision": OFFICIAL_MODEL_REVISION,
        },
        "transcoder": {
            "identifier": OFFICIAL_TRANSCODER_ID,
            "revision": OFFICIAL_TRANSCODER_REVISION,
        },
        "runtime": _runtime_record(provenance),
        "attribution": {
            "attempted_batch_sizes": attempted,
            "selected_batch_size": selected_batch,
            "batch_deviation": (
                batch_deviation(selected_batch) if selected_batch is not None else None
            ),
        },
        "retry_history": history,
        "timings": _timings(directory, history),
        "checks": checks,
        "artifacts": _artifact_records(directory),
        "readiness": {
            "stage1b_engineering_readiness": engineering_ready,
            "stage1b_empirical_claim_readiness": False,
        },
        "bf16_reference": {
            "dtype": "bfloat16",
            "status": "pending",
            "statement": "Native-BF16 reference reproduction remains pending.",
        },
    }
    validate_t4_run_manifest(manifest)
    write_json_atomic(directory / RUN_MANIFEST_NAME, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=repository_root()
        / "configs/stage1a_gemma2_2b_t4_fp16_reproduction.yaml",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--transcoder-snapshot", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    from cfsus.reproduction.t4_fp16 import (
        T4RunStatus,
        classify_t4_failure,
        should_retry_attempt,
        validate_t4_fp16_mapping,
    )

    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    config = load_yaml(args.config.resolve())
    validated = validate_t4_fp16_mapping(config)
    project_commit, project_dirty = _validate_source_checkout()
    if (args.model_snapshot is None) != (args.transcoder_snapshot is None):
        raise RuntimeError("model and transcoder snapshot overrides must be paired")
    if args.allow_download and args.model_snapshot is not None:
        raise RuntimeError("download mode cannot be combined with snapshot overrides")
    result_directory = repository_root() / "results/stage1a_t4_fp16"
    generated_directory = repository_root() / "results/generated/stage1a_t4_fp16"
    result_directory.mkdir(parents=True, exist_ok=True)
    generated_directory.mkdir(parents=True, exist_ok=True)
    try:
        print("Resolving and verifying the pinned Stage 1A assets", flush=True)
        _write_metadata_artifacts(
            config,
            allow_download=args.allow_download,
            model_snapshot=args.model_snapshot,
            transcoder_snapshot=args.transcoder_snapshot,
        )
        print("Pinned Stage 1A assets are ready", flush=True)
    except Exception as error:
        terminal_status = classify_t4_failure(error)
        manifest = _write_manifest(
            directory=result_directory,
            project_commit=project_commit,
            project_dirty=project_dirty,
            status=terminal_status.value,
            selected_batch=None,
            history=[],
            provenance={},
            success=False,
        )
        print(f"Stage 1A T4/FP16 status={manifest['status']} before execution")
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
        return 2

    history: list[dict[str, Any]] = []
    selected_batch: int | None = None
    terminal_status = T4RunStatus.FAILED_RUNTIME
    runtime_provenance: dict[str, Any] = {}
    child_environment = os.environ.copy()
    child_environment.update(
        {
            "CFSUS_PROJECT_COMMIT": project_commit,
            "CFSUS_PROJECT_DIRTY_BEFORE_RUN": "1" if project_dirty else "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    child_environment.pop("HF_TOKEN", None)
    child_environment.pop("HUGGING_FACE_HUB_TOKEN", None)

    for batch_size in validated.oom_retry.batch_sizes:
        print(f"Starting isolated T4 worker at batch {batch_size}", flush=True)
        report_path = generated_directory / f"attempt-{batch_size}.json"
        report_path.unlink(missing_ok=True)
        command = [
            sys.executable,
            str(WORKER),
            "--config",
            str(args.config.resolve()),
            "--batch-size",
            str(batch_size),
            "--attempt-report",
            str(report_path),
        ]
        if args.model_snapshot is not None and args.transcoder_snapshot is not None:
            command.extend(
                [
                    "--model-snapshot",
                    str(args.model_snapshot.resolve()),
                    "--transcoder-snapshot",
                    str(args.transcoder_snapshot.resolve()),
                ]
            )
        completed = subprocess.run(
            command,
            cwd=repository_root(),
            env=child_environment,
            check=False,
        )
        attempt = _load_json(report_path)
        if attempt.get("batch_size") != batch_size:
            raise RuntimeError("worker attempt report has the wrong batch size")
        history.append(
            {
                "batch_size": batch_size,
                "outcome": attempt.get("outcome"),
                "category": attempt.get("category"),
                "exception_type": attempt.get("exception_type"),
                "message": attempt.get("message"),
                "failure_stage": attempt.get("failure_stage"),
                "peak_memory_bytes": attempt.get("peak_memory_bytes"),
                "wall_seconds": attempt.get("wall_seconds"),
                "cleanup_succeeded": attempt.get("cleanup_succeeded"),
            }
        )
        observed_provenance = attempt.get("runtime_provenance")
        if isinstance(observed_provenance, dict) and observed_provenance:
            runtime_provenance = observed_provenance
        if completed.returncode == 0 and attempt.get("outcome") == "completed":
            selected_batch = batch_size
            terminal_status = T4RunStatus.COMPLETED
            break
        if should_retry_attempt(
            batch_size=batch_size,
            category=str(attempt.get("category")),
            failure_stage=str(attempt.get("failure_stage")),
        ):
            continue
        if attempt.get("category") == "cuda_out_of_memory":
            terminal_status = T4RunStatus.BLOCKED_RESOURCE
            break
        try:
            terminal_status = T4RunStatus(str(attempt.get("category")))
        except ValueError:
            terminal_status = T4RunStatus.FAILED_RUNTIME
        break

    success = terminal_status is T4RunStatus.COMPLETED
    if success:
        required_before_manifest = {
            "environment_manifest.json",
            "asset_manifest.json",
            "attribution_summary.json",
            "intervention_summary.json",
            "semantics_summary.json",
        }
        present = {path.name for path in result_directory.iterdir() if path.is_file()}
        if not required_before_manifest.issubset(present):
            raise RuntimeError("successful worker omitted required summaries")
        write_t4_checksums(result_directory)
        validate_t4_artifact_directory(
            result_directory,
            require_run_manifest=False,
            require_complete=False,
        )
    manifest = _write_manifest(
        directory=result_directory,
        project_commit=project_commit,
        project_dirty=project_dirty,
        status=terminal_status.value,
        selected_batch=selected_batch,
        history=history,
        provenance=runtime_provenance,
        success=success,
    )
    if success:
        validate_t4_artifact_directory(result_directory)
    elapsed = time.perf_counter() - started
    print(
        f"Stage 1A T4/FP16 status={manifest['status']} "
        f"selected_batch={selected_batch} wall_seconds={elapsed:.3f}"
    )
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
