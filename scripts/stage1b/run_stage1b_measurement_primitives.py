#!/usr/bin/env python3
"""Bounded process-group supervisor for Stage 1B calibration/canonical runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cfsus.mps_telemetry import swap_used_bytes, thermal_state  # noqa: E402
from cfsus.reproduction.artifacts import (  # noqa: E402
    REDACTED,
    redact_sensitive,
    write_json_atomic,
)
from cfsus.reproduction.small_model_mps_bf16 import (  # noqa: E402
    assert_fallback_disabled,
    supervise_process_group,
)
from cfsus.stage1b import CONFIG_PATH, load_stage1b_config  # noqa: E402

CREDENTIAL_VARIABLES = frozenset(
    {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"}
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / CONFIG_PATH)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--mode", choices=("calibration", "canonical"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def safe_worker_environment(source_root: Path = SOURCE_ROOT) -> dict[str, str]:
    """Return an offline child environment with auth and fallback removed."""

    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
        and key not in CREDENTIAL_VARIABLES
        and key != "PYTORCH_ENABLE_MPS_FALLBACK"
    }
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": str(source_root),
        }
    )
    return environment


def _host_sampler(limits: dict[str, Any], swap_start: int) -> Any:
    import psutil  # type: ignore[import-untyped]

    def sample(pid: int) -> dict[str, Any]:
        process = psutil.Process(pid)
        processes = [process, *process.children(recursive=True)]
        rss = sum(item.memory_info().rss for item in processes if item.is_running())
        available = int(psutil.virtual_memory().available)
        swap = swap_used_bytes()
        thermal = thermal_state()
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


def _summarize(outcome: Any) -> dict[str, Any]:
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
        "stdout_tail": _safe_process_tail(outcome.stdout),
        "stderr_tail": _safe_process_tail(outcome.stderr),
    }


def _safe_process_tail(value: str) -> str:
    """Keep bounded diagnostic text while removing host paths and credentials."""

    normalized = "\n".join(line.rstrip() for line in value.splitlines()[-80:])
    normalized = re.sub(
        r"/(?:Users|home)/[^/\s]+(?:/[^\s:\"]+)*",
        "<LOCAL_PATH>",
        normalized,
    )
    normalized = re.sub(
        r"/private/(?:var|tmp)(?:/[^\s:\"]+)*",
        "<LOCAL_PATH>",
        normalized,
    )
    redacted = redact_sensitive({"text": normalized})["text"]
    return "<REDACTED>" if redacted == REDACTED else str(redacted)[-20_000:]


def _validate_output_directory(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise RuntimeError(
            "output directory must be an existing absolute real directory"
        )
    if any(path.iterdir()):
        raise RuntimeError("output directory must be empty")
    resolved = path.resolve(strict=True)
    repository = REPOSITORY_ROOT.resolve(strict=True)
    generated = (
        repository / "results/generated/stage1b_measurement_primitives"
    ).resolve()
    temporary = Path("/private/tmp").resolve(strict=True)
    if not (
        resolved.is_relative_to(generated)
        or (
            resolved.is_relative_to(temporary)
            and resolved.name.startswith("stage1b-measurement-")
        )
    ):
        raise RuntimeError("output directory is outside the frozen safe roots")
    return resolved


def main() -> int:
    args = _parser().parse_args()
    assert_fallback_disabled()
    output_dir = _validate_output_directory(args.output_dir)
    config = load_stage1b_config(args.config)
    expected_phase = "calibration" if args.mode == "calibration" else "canonical_frozen"
    if config["phase"] != expected_phase:
        raise RuntimeError("run mode and frozen config phase differ")
    environment = safe_worker_environment()
    preflight_path = output_dir / "preflight.json"
    preflight = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/stage1b/preflight_stage1b.py"),
            "--config",
            str(args.config),
            "--hf-cache",
            str(args.hf_cache),
            "--mode",
            args.mode,
            "--output",
            str(preflight_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
        env=environment,
    )
    if preflight.returncode != 0 or not preflight_path.is_file():
        raise RuntimeError("Stage 1B preflight failed")
    preflight_record = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight_record.get("status") != "passed":
        raise RuntimeError("Stage 1B preflight did not pass")

    worker_output = output_dir / f"{args.mode}_worker.json"
    emergency_output = output_dir / f"{args.mode}_emergency.json"
    limits = config["safety_limits"]
    timeout = float(limits[f"{args.mode}_timeout_seconds"])
    swap_start = swap_used_bytes()
    outcome = supervise_process_group(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/stage1b/run_stage1b_measurement_worker.py"),
            "--config",
            str(args.config),
            "--hf-cache",
            str(args.hf_cache),
            "--mode",
            args.mode,
            "--output",
            str(worker_output),
            "--emergency-output",
            str(emergency_output),
        ],
        timeout_seconds=timeout,
        sample_interval_seconds=float(limits["sample_interval_seconds"]),
        sample_host=_host_sampler(limits, swap_start),
        telemetry_failure_limit=int(limits["telemetry_failure_limit"]),
        terminate_grace_seconds=float(limits["terminate_grace_seconds"]),
        kill_grace_seconds=float(limits["kill_grace_seconds"]),
        environment=environment,
    )
    supervisor = _summarize(outcome)
    write_json_atomic(
        output_dir / "supervisor.json",
        {
            "schema_version": 1,
            "artifact_type": "stage1b_measurement_primitives_supervisor",
            "mode": args.mode,
            **supervisor,
        },
    )
    passed = (
        outcome.returncode == 0
        and not outcome.timed_out
        and not outcome.safety_terminated
        and outcome.telemetry_failures == 0
        and worker_output.is_file()
        and not emergency_output.exists()
    )
    if not passed:
        raise RuntimeError("Stage 1B worker failed under the bounded supervisor")
    result = json.loads(worker_output.read_text(encoding="utf-8"))
    if result.get("status") != "passed":
        raise RuntimeError("Stage 1B worker result did not pass")
    print(json.dumps({"status": "passed", "mode": args.mode}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
