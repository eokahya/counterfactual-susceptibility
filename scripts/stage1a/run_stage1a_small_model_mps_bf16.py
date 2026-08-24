#!/usr/bin/env python3
"""Fail-closed supervisor for progressive Stage 1A-S-BF16 stages."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cfsus.reproduction.artifacts import write_json_atomic  # noqa: E402
from cfsus.reproduction.small_model_mps_bf16 import (  # noqa: E402
    CONFIG_PATH,
    assert_fallback_disabled,
    load_bf16_config,
    supervise_process_group,
)


class WorkerStageFailure(RuntimeError):
    """Preserve a failed worker's bounded attempt and supervisor evidence."""

    def __init__(
        self,
        stage: str,
        attempt: dict[str, Any] | None,
        supervisor: dict[str, Any],
    ) -> None:
        super().__init__(f"{stage} worker failed under supervisor")
        self.stage = stage
        self.attempt = attempt
        self.supervisor = supervisor


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / CONFIG_PATH)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "model_gate",
            "loaded_semantics",
            "full_plt",
            "replacement_runtime",
            "smoke",
            "accepted",
        ),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _run_text(command: list[str]) -> str:
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
        env={
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        },
    )
    return result.stdout.strip()


def _swap_used_bytes() -> int:
    output = _run_text(["sysctl", "vm.swapusage"])
    match = re.search(r"used = ([0-9.]+)M", output)
    if match is None:
        raise RuntimeError("swap telemetry is unavailable")
    return round(float(match.group(1)) * 1024**2)


def _thermal_state() -> str:
    output = _run_text(["pmset", "-g", "therm"]).casefold()
    if "serious" in output or "critical" in output:
        return "serious_or_critical"
    if "no thermal warning level has been recorded" in output:
        return "nominal"
    if "fair" in output:
        return "fair"
    return "unknown"


def _host_sampler(limits: dict[str, Any], swap_start: int) -> Any:
    import psutil

    def sample(pid: int) -> dict[str, Any]:
        process = psutil.Process(pid)
        processes = [process, *process.children(recursive=True)]
        rss = sum(item.memory_info().rss for item in processes if item.is_running())
        available = int(psutil.virtual_memory().available)
        swap = _swap_used_bytes()
        thermal = _thermal_state()
        violations: list[str] = []
        if rss > int(limits["maximum_process_rss_bytes"]):
            violations.append("maximum_process_rss_bytes")
        if swap - swap_start > int(limits["maximum_swap_growth_bytes"]):
            violations.append("maximum_swap_growth_bytes")
        if available < int(limits["minimum_available_memory_bytes"]):
            violations.append("minimum_available_memory_bytes")
        if thermal not in set(limits["accepted_thermal_states"]):
            violations.append("accepted_thermal_states")
        return {
            "sampled_at_unix": time.time(),
            "process_group_rss_bytes": int(rss),
            "available_memory_bytes": available,
            "swap_used_bytes": swap,
            "swap_growth_bytes": max(0, swap - swap_start),
            "thermal_state": thermal,
            "violations": violations,
        }

    return sample


def _safe_worker_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
        and key not in {"PYTORCH_ENABLE_MPS_FALLBACK", "HF_TOKEN"}
    }
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": str(SOURCE_ROOT),
        }
    )
    return environment


def _summarize_supervisor(outcome: Any) -> dict[str, Any]:
    samples = list(outcome.samples)
    return {
        "returncode": outcome.returncode,
        "timed_out": outcome.timed_out,
        "safety_terminated": outcome.safety_terminated,
        "termination_signal": outcome.termination_signal,
        "telemetry_failures": outcome.telemetry_failures,
        "sample_count": len(samples),
        "peak_process_group_rss_bytes": max(
            (int(item.get("process_group_rss_bytes", 0)) for item in samples),
            default=0,
        ),
        "minimum_available_memory_bytes": min(
            (
                int(item["available_memory_bytes"])
                for item in samples
                if "available_memory_bytes" in item
            ),
            default=0,
        ),
        "peak_swap_growth_bytes": max(
            (int(item.get("swap_growth_bytes", 0)) for item in samples), default=0
        ),
        "thermal_states": sorted(
            {str(item["thermal_state"]) for item in samples if "thermal_state" in item}
        ),
        "started_at_unix": outcome.started_at_unix,
        "finished_at_unix": outcome.finished_at_unix,
        "stdout_tail": outcome.stdout,
        "stderr_tail": outcome.stderr,
    }


def _read_attempt(path: Path, stage: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{stage} worker did not produce a safe attempt record")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("stage") != stage or value.get("status") not in {
        "passed",
        "failed",
        "failed_safety",
    }:
        raise RuntimeError(f"{stage} attempt record is invalid")
    return value


def _run_worker(
    *,
    config: dict[str, Any],
    stage: str,
    cache: Path,
    output: Path,
    comparison: Path,
    batch_size: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    limits = config["safety_limits"]
    swap_start = _swap_used_bytes()
    command = [
        sys.executable,
        str(
            REPOSITORY_ROOT
            / "scripts/stage1a/run_stage1a_small_model_mps_bf16_worker.py"
        ),
        "--config",
        str(REPOSITORY_ROOT / CONFIG_PATH),
        "--hf-cache",
        str(cache),
        "--stage",
        stage,
        "--output",
        str(output),
        "--comparison-file",
        str(comparison),
    ]
    if batch_size is not None:
        command.extend(("--batch-size", str(batch_size)))
    timeout = float(limits["stage_timeout_seconds"][stage])
    supervised = supervise_process_group(
        command,
        timeout_seconds=timeout,
        sample_interval_seconds=float(limits["sample_interval_seconds"]),
        sample_host=_host_sampler(limits, swap_start),
        telemetry_failure_limit=int(limits["telemetry_failure_limit"]),
        terminate_grace_seconds=float(limits["terminate_grace_seconds"]),
        kill_grace_seconds=float(limits["kill_grace_seconds"]),
        environment=_safe_worker_environment(),
    )
    supervisor_record = _summarize_supervisor(supervised)
    attempt = _read_attempt(output, stage) if output.is_file() else None
    if (
        supervised.returncode != 0
        or supervised.timed_out
        or supervised.safety_terminated
        or attempt is None
        or attempt.get("status") != "passed"
        or attempt.get("telemetry", {}).get("violations")
    ):
        raise WorkerStageFailure(stage, attempt, supervisor_record)
    return attempt, supervisor_record


def _attempt_entry(
    stage: str,
    worker: dict[str, Any] | None,
    supervisor: dict[str, Any],
    *,
    batch_size: int | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "batch_size": batch_size,
        "worker": worker,
        "supervisor": supervisor,
    }


def _is_verified_attribution_oom(failure: WorkerStageFailure) -> bool:
    attempt = failure.attempt or {}
    message = str(attempt.get("error", {}).get("message", "")).casefold()
    stage_peaks = attempt.get("telemetry", {}).get("stage_peaks", {})
    return (
        "out of memory" in message
        and "attribution" in stage_peaks
        and not attempt.get("telemetry", {}).get("violations")
        and not failure.supervisor.get("safety_terminated")
        and not failure.supervisor.get("timed_out")
    )


def _safe_cleanup_temp(directory: Path) -> None:
    resolved = directory.resolve(strict=True)
    root = Path("/private/tmp").resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.name.startswith(
        "stage1a-bf16-"
    ):
        raise RuntimeError("refusing unsafe temporary cleanup")
    shutil.rmtree(resolved)


def _write_attempts(
    output_dir: Path,
    stage: str,
    status: str,
    attempts: list[dict[str, Any]],
) -> None:
    write_json_atomic(
        output_dir / f"{stage}_attempts.json",
        {
            "schema_version": 1,
            "artifact_type": "stage1a_small_model_mps_bf16_stage_attempts",
            "status": status,
            "stage": stage,
            "attempts": attempts,
        },
    )


def _run_model_gate(
    config: dict[str, Any],
    cache: Path,
    output_dir: Path,
    comparison: Path,
    attempts: list[dict[str, Any]],
) -> None:
    try:
        model_attempt, model_supervisor = _run_worker(
            config=config,
            stage="model_forward",
            cache=cache,
            output=output_dir / "model_forward_attempt.json",
            comparison=comparison,
        )
    except WorkerStageFailure as failure:
        attempts.append(
            _attempt_entry(failure.stage, failure.attempt, failure.supervisor)
        )
        _write_attempts(output_dir, "model_gate", "failed", attempts)
        raise
    attempts.append(_attempt_entry("model_forward", model_attempt, model_supervisor))
    if not comparison.is_file():
        raise RuntimeError("model worker did not leave comparison vectors")
    try:
        fp32_attempt, fp32_supervisor = _run_worker(
            config=config,
            stage="fp32_reference",
            cache=cache,
            output=output_dir / "fp32_reference_attempt.json",
            comparison=comparison,
        )
    except WorkerStageFailure as failure:
        attempts.append(
            _attempt_entry(failure.stage, failure.attempt, failure.supervisor)
        )
        _write_attempts(output_dir, "model_gate", "failed", attempts)
        raise
    attempts.append(_attempt_entry("fp32_reference", fp32_attempt, fp32_supervisor))
    model_summary = model_attempt["outcome"]
    fp32_summary = fp32_attempt["outcome"]
    model_summary["fp32_diagnostic_process_overlap"] = False
    fp32_summary["temporary_raw_vectors_deleted_after_validation"] = True
    write_json_atomic(output_dir / "model_forward_summary.json", model_summary)
    write_json_atomic(output_dir / "fp32_reference_summary.json", fp32_summary)
    _write_attempts(output_dir, "model_gate", "passed", attempts)


def _run_progressive_stage(
    config: dict[str, Any],
    cache: Path,
    stage: str,
    output_dir: Path,
    comparison: Path,
    attempts: list[dict[str, Any]],
) -> None:
    batches: list[int | None]
    if stage in {"smoke", "accepted"}:
        batches = [int(value) for value in config[stage]["attribution_batch_sizes"]]
    else:
        batches = [None]
    summary_names = {
        "loaded_semantics": "loaded_semantics_summary.json",
        "full_plt": "full_plt_summary.json",
        "replacement_runtime": "replacement_runtime_summary.json",
        "smoke": "smoke_summary.json",
        "accepted": "accepted_summary.json",
    }
    for index, batch_size in enumerate(batches):
        suffix = "" if batch_size is None else f"_batch_{batch_size}"
        try:
            attempt, supervisor = _run_worker(
                config=config,
                stage=stage,
                cache=cache,
                output=output_dir / f"{stage}{suffix}_attempt.json",
                comparison=comparison,
                batch_size=batch_size,
            )
        except WorkerStageFailure as failure:
            attempts.append(
                _attempt_entry(
                    stage,
                    failure.attempt,
                    failure.supervisor,
                    batch_size=batch_size,
                )
            )
            may_retry = (
                stage in {"smoke", "accepted"}
                and index + 1 < len(batches)
                and _is_verified_attribution_oom(failure)
            )
            if may_retry:
                continue
            _write_attempts(output_dir, stage, "failed", attempts)
            raise
        attempts.append(
            _attempt_entry(stage, attempt, supervisor, batch_size=batch_size)
        )
        write_json_atomic(output_dir / summary_names[stage], attempt["outcome"])
        _write_attempts(output_dir, stage, "passed", attempts)
        return
    raise RuntimeError("frozen attribution retry sequence was exhausted")


def main() -> int:
    arguments = _parser().parse_args()
    assert_fallback_disabled()
    config = load_bf16_config(arguments.config)
    output_dir = arguments.output_dir
    if not output_dir.is_absolute():
        output_dir = REPOSITORY_ROOT / output_dir
    generated = (REPOSITORY_ROOT / config["artifacts"]["generated_directory"]).resolve()
    if not output_dir.resolve().is_relative_to(generated):
        raise RuntimeError("runner output must stay under ignored generated directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_directory = Path(tempfile.mkdtemp(prefix="stage1a-bf16-", dir="/private/tmp"))
    comparison = temp_directory / "mps_reference_vectors.npz"
    attempts: list[dict[str, Any]] = []
    try:
        if arguments.stage == "model_gate":
            _run_model_gate(
                config,
                arguments.hf_cache,
                output_dir,
                comparison,
                attempts,
            )
        else:
            _run_progressive_stage(
                config,
                arguments.hf_cache,
                arguments.stage,
                output_dir,
                comparison,
                attempts,
            )
    finally:
        _safe_cleanup_temp(temp_directory)
    print(
        json.dumps(
            {
                "status": "passed",
                "stage": arguments.stage,
                "attempt_count": len(attempts),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
