#!/usr/bin/env python3
"""Report local environment metadata without network access or downloads.

This command is deliberately diagnostic only.  It does not import model tooling,
load model/transcoder weights, install packages, or contact remote services.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_PACKAGES = (
    "cfsus",
    "circuit-tracer",
    "torch",
    "transformers",
    "pydantic",
    "PyYAML",
)
OPTIONAL_COMMANDS = (
    "git",
    "uv",
    "latexmk",
    "pdflatex",
    "nvidia-smi",
    "system_profiler",
)


def _package_version(distribution: str) -> str | None:
    """Return an installed distribution version without importing the package."""

    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_available(module: str) -> bool:
    """Check whether a module can be resolved without importing it."""

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _run_local(command: Sequence[str]) -> str | None:
    """Run a bounded, read-only local command and return stripped stdout."""

    try:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_metadata() -> dict[str, Any]:
    if shutil.which("git") is None:
        return {"available": False, "commit": None, "dirty": None}

    commit = _run_local(("git", "rev-parse", "HEAD"))
    status = _run_local(("git", "status", "--porcelain"))
    return {
        "available": True,
        "commit": commit,
        "dirty": None if status is None else bool(status),
    }


def _accelerator_metadata() -> dict[str, Any]:
    """Probe display/GPU metadata using local OS commands only."""

    report: dict[str, Any] = {
        "probe": "local_commands_only",
        "apple_display_devices": [],
        "nvidia_gpus": [],
    }

    if platform.system() == "Darwin" and shutil.which("system_profiler"):
        raw_displays = _run_local(
            ("system_profiler", "SPDisplaysDataType", "-json", "-detailLevel", "mini")
        )
        if raw_displays:
            try:
                parsed = json.loads(raw_displays)
                devices = parsed.get("SPDisplaysDataType", [])
                report["apple_display_devices"] = [
                    {
                        "name": device.get("sppci_model") or device.get("_name"),
                        "metal_support": device.get("spdisplays_metal"),
                    }
                    for device in devices
                    if isinstance(device, dict)
                ]
            except (json.JSONDecodeError, AttributeError):
                report["apple_probe_error"] = "unparseable system_profiler output"

    if shutil.which("nvidia-smi"):
        raw_nvidia = _run_local(
            (
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            )
        )
        if raw_nvidia:
            report["nvidia_gpus"] = [
                {
                    "name": fields[0],
                    "memory_total_mib": fields[1],
                    "driver_version": fields[2],
                }
                for line in raw_nvidia.splitlines()
                if len(fields := [field.strip() for field in line.split(",")]) == 3
            ]

    return report


def collect_report() -> dict[str, Any]:
    """Collect JSON-serializable, local-only environment information."""

    disk = shutil.disk_usage(REPOSITORY_ROOT)
    package_versions = {
        distribution: _package_version(distribution)
        for distribution in OPTIONAL_PACKAGES
    }
    module_availability = {
        module: _module_available(module)
        for module in ("cfsus", "circuit_tracer", "torch", "transformers", "yaml")
    }
    command_paths = {command: shutil.which(command) for command in OPTIONAL_COMMANDS}

    return {
        "schema_version": 1,
        "offline_only": True,
        "repository": {
            "root": str(REPOSITORY_ROOT),
            "source_package_present": (REPOSITORY_ROOT / "src" / "cfsus").is_dir(),
            "research_spec_present": (
                REPOSITORY_ROOT / "docs" / "RESEARCH_SPEC.md"
            ).is_file(),
            "git": _git_metadata(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "prefix": sys.prefix,
            "virtual_environment": (
                sys.prefix != getattr(sys, "base_prefix", sys.prefix)
            ),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "cpu_count": os.cpu_count(),
        },
        "disk": {
            "repository_filesystem_free_bytes": disk.free,
            "repository_filesystem_total_bytes": disk.total,
        },
        "capabilities": {
            "installed_distributions": package_versions,
            "importable_modules": module_availability,
            "commands": command_paths,
            "accelerators": _accelerator_metadata(),
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the JSON report; parent directories are created.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = collect_report()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser()
        if not output.is_absolute():
            output = REPOSITORY_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
