#!/usr/bin/env python3
"""Bounded offline supervisor for the Stage 1C-v3 workers."""

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
SCRIPT_ROOT = REPOSITORY_ROOT / "scripts" / "stage1c_v3"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cfsus.mps_telemetry import swap_used_bytes, thermal_state  # noqa: E402
from cfsus.reproduction.small_model_mps_bf16 import (  # noqa: E402
    assert_fallback_disabled,
    supervise_process_group,
)
from cfsus.stage1c_v3.config import CONFIG_PATH, load_stage1c_v3_config  # noqa: E402
from cfsus.stage1c_v3.serialization import write_json_new  # noqa: E402

try:
    from preflight_stage1c_v3 import CREDENTIAL_VARIABLES
except ImportError:  # pragma: no cover - direct package import fallback
    CREDENTIAL_VARIABLES = frozenset(
        {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"}
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / CONFIG_PATH)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("prediction", "intervention"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pre-intervention-commit")
    parser.add_argument("--prediction-manifest", type=Path)
    parser.add_argument("--prediction-manifest-sha256")
    return parser


def safe_worker_environment(source_root: Path = SOURCE_ROOT) -> dict[str, str]:
    """Return a child environment without credentials, Git overrides, or fallback."""

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
    normalized = " | ".join(line.rstrip() for line in value.splitlines()[-80:])
    normalized = re.sub(r"[\x00-\x1f\x7f-\x9f]+", " ", normalized)
    normalized = re.sub(
        r"/(?:Users|home)/[^/\s]+(?:/[^\s:\"]+)*", "<LOCAL_PATH>", normalized
    )
    normalized = re.sub(
        r"/private/(?:var|tmp)(?:/[^\s:\"]+)*", "<LOCAL_PATH>", normalized
    )
    normalized = re.sub(
        r"(?i)\b(?:hf|gh)[_-][A-Za-z0-9_-]{12,}\b", "<REDACTED>", normalized
    )
    return normalized[-20_000:]


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
            (int(item.get("process_group_rss_bytes", 0)) for item in samples), default=0
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
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_dir()
        or any(path.iterdir())
    ):
        raise RuntimeError(
            "v3 output directory must be an empty absolute real directory"
        )
    resolved = path.resolve(strict=True)
    repository = REPOSITORY_ROOT.resolve(strict=True)
    generated = (repository / config["artifacts"]["generated_directory"]).resolve()
    temporary = Path("/private/tmp").resolve(strict=True)
    if not (
        resolved.is_relative_to(generated)
        or (
            resolved.is_relative_to(temporary)
            and resolved.name.startswith("stage1c-v3-")
        )
    ):
        raise RuntimeError("v3 output directory is outside the frozen generated roots")
    return resolved


def _attempt_lock_path(config: dict[str, Any]) -> Path:
    generated = (
        REPOSITORY_ROOT / str(config["artifacts"]["generated_directory"])
    ).resolve(strict=False)
    lock = (
        REPOSITORY_ROOT / str(config["artifacts"]["canonical_attempt_lock"])
    ).resolve(strict=False)
    if (
        lock.parent != generated
        or lock.name != "canonical_attempt_v1.lock"
        or generated.is_symlink()
    ):
        raise RuntimeError("v4 canonical-attempt path differs from frozen config")
    generated.mkdir(parents=True, exist_ok=True)
    if generated.is_symlink() or not generated.is_dir():
        raise RuntimeError("v4 canonical-attempt directory is unsafe")
    return lock


def _worker_command(
    args: argparse.Namespace, output_dir: Path, config: dict[str, Any]
) -> list[str]:
    worker_name = (
        "run_stage1c_v3_prediction_worker.py"
        if args.phase == "prediction"
        else "run_stage1c_v3_intervention_worker.py"
    )
    worker = SCRIPT_ROOT / worker_name
    if worker.is_symlink() or not worker.is_file():
        raise RuntimeError("v3 worker is missing or unsafe")
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
            not args.pre_intervention_commit
            or not args.prediction_manifest
            or not args.prediction_manifest_sha256
        ):
            raise RuntimeError(
                "intervention requires commit, manifest, and manifest digest"
            )
        command.extend(
            [
                "--pre-intervention-commit",
                args.pre_intervention_commit,
                "--prediction-manifest",
                str(args.prediction_manifest),
                "--prediction-manifest-sha256",
                args.prediction_manifest_sha256,
                "--attempt-lock",
                str(_attempt_lock_path(config)),
                "--point-journal",
                str(output_dir / "point_journal.jsonl"),
            ]
        )
    elif (
        args.pre_intervention_commit
        or args.prediction_manifest
        or args.prediction_manifest_sha256
    ):
        raise RuntimeError("prediction phase must not receive intervention identity")
    return command


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    assert_fallback_disabled()
    config = load_stage1c_v3_config(args.config)
    output_dir = _validate_output_directory(args.output_dir, config)
    environment = safe_worker_environment()
    preflight_path = output_dir / "preflight.json"
    preflight_command = [
        sys.executable,
        str(SCRIPT_ROOT / "preflight_stage1c_v3.py"),
        "--config",
        str(args.config),
        "--hf-cache",
        str(args.hf_cache),
        "--phase",
        args.phase,
        "--output",
        str(preflight_path),
    ]
    if args.pre_intervention_commit:
        preflight_command.extend(
            ["--pre-intervention-commit", args.pre_intervention_commit]
        )
    if args.prediction_manifest:
        preflight_command.extend(
            ["--prediction-manifest", str(args.prediction_manifest)]
        )
    if args.prediction_manifest_sha256:
        preflight_command.extend(
            ["--prediction-manifest-sha256", args.prediction_manifest_sha256]
        )
    preflight = subprocess.run(
        preflight_command,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
        env=environment,
    )
    if preflight.returncode != 0 or not preflight_path.is_file():
        raise RuntimeError("v3 preflight failed")
    preflight_record = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight_record.get("status") != "passed":
        raise RuntimeError("v3 preflight did not pass")
    limits = config["safety_limits"]
    timeout_key = (
        "prediction_timeout_seconds"
        if args.phase == "prediction"
        else "canonical_timeout_seconds"
    )
    outcome = supervise_process_group(
        _worker_command(args, output_dir, config),
        timeout_seconds=float(limits[timeout_key]),
        sample_interval_seconds=float(limits["sample_interval_seconds"]),
        sample_host=_host_sampler(limits, swap_used_bytes()),
        telemetry_failure_limit=int(limits["telemetry_failure_limit"]),
        terminate_grace_seconds=float(limits["terminate_grace_seconds"]),
        kill_grace_seconds=float(limits["kill_grace_seconds"]),
        environment=environment,
    )
    supervisor = _summarize(outcome)
    write_json_new(
        output_dir / "supervisor.json",
        {
            "schema_version": 3,
            "artifact_type": "stage1c_v3_supervisor",
            "phase": args.phase,
            **supervisor,
        },
    )
    worker_output = output_dir / f"{args.phase}_worker.json"
    emergency_output = output_dir / f"{args.phase}_emergency.json"
    if (
        outcome.returncode != 0
        or outcome.timed_out
        or outcome.safety_terminated
        or outcome.telemetry_failures != 0
        or not worker_output.is_file()
        or emergency_output.exists()
    ):
        raise RuntimeError("v3 worker failed under bounded supervisor")
    result = json.loads(worker_output.read_text(encoding="utf-8"))
    if result.get("status") != "passed":
        raise RuntimeError("v3 worker result did not pass")
    print(json.dumps({"status": "passed", "phase": args.phase}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
