#!/usr/bin/env python3
"""Bounded process-group supervisor for the two Stage 1C phases."""

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
SCRIPT_ROOT = REPOSITORY_ROOT / "scripts" / "stage1c"
for root in (SOURCE_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from preflight_stage1c import (  # noqa: E402
    CREDENTIAL_VARIABLES,
)

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
from cfsus.stage1c.config import CONFIG_PATH, load_stage1c_config  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / CONFIG_PATH)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("prediction", "intervention"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pre-intervention-commit", default=None)
    parser.add_argument("--prediction-manifest", type=Path, default=None)
    parser.add_argument("--prediction-manifest-sha256", default=None)
    return parser


def safe_worker_environment(source_root: Path = SOURCE_ROOT) -> dict[str, str]:
    """Return an offline child environment without credentials or fallback."""

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


def _safe_process_tail(value: str) -> str:
    """Keep bounded diagnostics while removing paths and credentials."""

    normalized = "\n".join(line.rstrip() for line in value.splitlines()[-80:])
    normalized = re.sub(
        r"/(?:Users|home)/[^/\s]+(?:/[^\s:\"]+)*", "<LOCAL_PATH>", normalized
    )
    normalized = re.sub(
        r"/private/(?:var|tmp)(?:/[^\s:\"]+)*", "<LOCAL_PATH>", normalized
    )
    redacted = redact_sensitive({"text": normalized})["text"]
    return "<REDACTED>" if redacted == REDACTED else str(redacted)[-20_000:]


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


def _validate_output_directory(path: Path, config: dict[str, Any]) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise RuntimeError(
            "output directory must be an existing absolute real directory"
        )
    if any(path.iterdir()):
        raise RuntimeError("output directory must be empty")
    resolved = path.resolve(strict=True)
    repository = REPOSITORY_ROOT.resolve(strict=True)
    generated = (repository / config["artifacts"]["generated_directory"]).resolve()
    temporary = Path("/private/tmp").resolve(strict=True)
    if not (
        resolved.is_relative_to(generated)
        or (resolved.is_relative_to(temporary) and resolved.name.startswith("stage1c-"))
    ):
        raise RuntimeError("output directory is outside the frozen Stage 1C roots")
    return resolved


def _worker_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    worker = SCRIPT_ROOT / (
        "run_stage1c_prediction_worker.py"
        if args.phase == "prediction"
        else "run_stage1c_intervention_worker.py"
    )
    if not worker.is_file() or worker.is_symlink():
        raise RuntimeError(f"Stage 1C {args.phase} worker is missing")
    command = [
        sys.executable,
        str(worker),
        "--config",
        str(args.config),
        "--hf-cache",
        str(args.hf_cache),
        "--output",
        str(output_dir / f"{args.phase}_worker.json"),
        "--emergency-output",
        str(output_dir / f"{args.phase}_emergency.json"),
    ]
    if args.phase == "intervention":
        if (
            args.pre_intervention_commit is None
            or args.prediction_manifest is None
            or args.prediction_manifest_sha256 is None
        ):
            raise RuntimeError(
                "intervention requires the pre-intervention commit and prediction "
                "manifest identity"
            )
        command.extend(
            [
                "--pre-intervention-commit",
                args.pre_intervention_commit,
                "--prediction-manifest",
                str(args.prediction_manifest),
                "--prediction-manifest-sha256",
                args.prediction_manifest_sha256,
            ]
        )
    elif (
        args.prediction_manifest is not None
        or args.prediction_manifest_sha256 is not None
    ):
        raise RuntimeError("prediction phase must not receive an intervention manifest")
    return command


def main() -> int:
    args = _parser().parse_args()
    assert_fallback_disabled()
    config = load_stage1c_config(args.config)
    output_dir = _validate_output_directory(args.output_dir, config)
    environment = safe_worker_environment()
    preflight_path = output_dir / "preflight.json"
    preflight = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_ROOT / "preflight_stage1c.py"),
            "--config",
            str(args.config),
            "--hf-cache",
            str(args.hf_cache),
            "--phase",
            args.phase,
            "--output",
            str(preflight_path),
            *(
                (["--pre-intervention-commit", args.pre_intervention_commit])
                if args.pre_intervention_commit is not None
                else []
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
        env=environment,
    )
    if preflight.returncode != 0 or not preflight_path.is_file():
        raise RuntimeError("Stage 1C preflight failed")
    preflight_record = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight_record.get("status") != "passed":
        raise RuntimeError("Stage 1C preflight did not pass")

    limits = config["safety_limits"]
    timeout_key = (
        "prediction_timeout_seconds"
        if args.phase == "prediction"
        else "canonical_timeout_seconds"
    )
    swap_start = swap_used_bytes()
    outcome = supervise_process_group(
        _worker_command(args, output_dir),
        timeout_seconds=float(limits[timeout_key]),
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
            "artifact_type": "stage1c_first_prospective_prediction_supervisor",
            "phase": args.phase,
            **supervisor,
        },
    )
    worker_output = output_dir / f"{args.phase}_worker.json"
    emergency_output = output_dir / f"{args.phase}_emergency.json"
    passed = (
        outcome.returncode == 0
        and not outcome.timed_out
        and not outcome.safety_terminated
        and outcome.telemetry_failures == 0
        and worker_output.is_file()
        and not emergency_output.exists()
    )
    if not passed:
        raise RuntimeError("Stage 1C worker failed under the bounded supervisor")
    result = json.loads(worker_output.read_text(encoding="utf-8"))
    if result.get("status") != "passed":
        raise RuntimeError("Stage 1C worker result did not pass")
    print(json.dumps({"status": "passed", "phase": args.phase}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
