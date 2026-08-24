#!/usr/bin/env python3
"""Offline fail-closed preflight for Stage 1B measurement execution."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cfsus.mps_telemetry import swap_used_bytes, thermal_state  # noqa: E402
from cfsus.reproduction.artifacts import write_json_atomic  # noqa: E402
from cfsus.reproduction.small_model_mps_bf16 import (  # noqa: E402
    assert_fallback_disabled,
)
from cfsus.stage1b import (  # noqa: E402
    BASE_COMMIT,
    BRANCH,
    CONFIG_PATH,
    MODEL_REVISION,
    STAGE1A_EXECUTION_COMMIT,
    TRANSCODER_REVISION,
    UPSTREAM_REVISION,
    load_stage1b_config,
)
from cfsus.stage1b_runtime import resolve_offline_snapshots  # noqa: E402

EXPECTED_VERSIONS = {
    "circuit-tracer": "0.5.2",
    "nnsight": "0.6.1",
    "torch": "2.6.0",
    "transformers": "4.57.3",
}
FORBIDDEN_CREDENTIAL_VARIABLES = frozenset(
    {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"}
)
SHA40_RE = re.compile(r"\A[0-9a-f]{40}\Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / CONFIG_PATH)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--mode", choices=("calibration", "canonical"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
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


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _direct_url_identity() -> dict[str, str | None]:
    distribution = importlib.metadata.distribution("circuit-tracer")
    raw = distribution.read_text("direct_url.json")
    if raw is None:
        raise RuntimeError("circuit-tracer direct-url provenance is absent")
    parsed = json.loads(raw)
    vcs = parsed.get("vcs_info")
    if not isinstance(vcs, dict):
        raise RuntimeError("circuit-tracer VCS provenance is absent")
    return {
        "url": parsed.get("url"),
        "vcs": vcs.get("vcs"),
        "commit_id": vcs.get("commit_id"),
        "requested_revision": vcs.get("requested_revision"),
    }


def _verify_git(mode: str) -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    clean = _git("status", "--porcelain", "--untracked-files=normal") == ""
    if SHA40_RE.fullmatch(head) is None or branch != BRANCH:
        raise RuntimeError("Stage 1B branch identity is invalid")
    if _git("merge-base", "--is-ancestor", BASE_COMMIT, head) != "":
        raise RuntimeError("unexpected merge-base output")
    if _git("rev-parse", "origin/stage-1a-small-model-mps-bf16") != BASE_COMMIT:
        raise RuntimeError("protected Stage 1A base ref changed")
    if _git("merge-base", "--is-ancestor", STAGE1A_EXECUTION_COMMIT, BASE_COMMIT) != "":
        raise RuntimeError("unexpected accepted-run ancestry output")
    if mode == "canonical" and (not clean or head == BASE_COMMIT):
        raise RuntimeError("canonical execution requires a clean pre-run commit")
    return {
        "branch": branch,
        "head": head,
        "working_tree_clean": clean,
        "base_commit": BASE_COMMIT,
        "accepted_stage1a_execution_commit": STAGE1A_EXECUTION_COMMIT,
        "base_ancestry_verified": True,
        "protected_origin_base_verified": True,
    }


def _verify_assets(cache: Path) -> dict[str, Any]:
    model, transcoder = resolve_offline_snapshots(cache, REPOSITORY_ROOT)
    result = subprocess.run(
        [
            sys.executable,
            str(
                REPOSITORY_ROOT
                / "scripts/stage1a/verify_small_model_mps_bf16_assets.py"
            ),
            "--hf-cache",
            str(cache),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
        env={
            **{
                key: value
                for key, value in os.environ.items()
                if key not in FORBIDDEN_CREDENTIAL_VARIABLES
                and key != "PYTORCH_ENABLE_MPS_FALLBACK"
            },
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )
    record = json.loads(result.stdout)
    if record.get("status") != "verified":
        raise RuntimeError("immutable offline asset validation failed")
    if model.name != MODEL_REVISION or transcoder.name != TRANSCODER_REVISION:
        raise RuntimeError("resolved asset revisions changed")
    return {
        "status": "verified",
        "download_performed": False,
        "network_accessed": False,
        "authentication_used": False,
        "actual_total_bytes": record.get("actual_total_bytes"),
        "model_revision": MODEL_REVISION,
        "transcoder_revision": TRANSCODER_REVISION,
        "exact_allowlist_hashes_verified": True,
    }


def collect_preflight(config_path: Path, cache: Path, mode: str) -> dict[str, Any]:
    """Collect a compact, path-free report and raise on any hard gate."""

    assert_fallback_disabled()
    if any(name in os.environ for name in FORBIDDEN_CREDENTIAL_VARIABLES):
        raise RuntimeError("credential-bearing environment variable is present")
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
    ):
        raise RuntimeError("preflight requires enforced offline mode")
    config = load_stage1b_config(config_path)
    expected_phase = "calibration" if mode == "calibration" else "canonical_frozen"
    if config["phase"] != expected_phase:
        raise RuntimeError("config phase does not match the requested run mode")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("native macOS arm64 runtime is required")
    if platform.python_version() != "3.11.13":
        raise RuntimeError("exact Python 3.11.13 runtime is required")
    versions = {name: _distribution_version(name) for name in EXPECTED_VERSIONS}
    if versions != EXPECTED_VERSIONS:
        raise RuntimeError("runtime package versions differ from the frozen lock")
    direct_url = _direct_url_identity()
    if direct_url != {
        "url": "https://github.com/decoderesearch/circuit-tracer.git",
        "vcs": "git",
        "commit_id": UPSTREAM_REVISION,
        "requested_revision": UPSTREAM_REVISION,
    }:
        raise RuntimeError("circuit-tracer immutable VCS identity changed")

    import psutil  # type: ignore[import-untyped]
    import torch  # type: ignore[import-not-found]

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("native MPS is unavailable")
    probe = torch.ones((2, 2), device="mps", dtype=torch.bfloat16)
    observed = torch.matmul(probe, probe)
    if observed.device.type != "mps" or observed.dtype != torch.bfloat16:
        raise RuntimeError("MPS/BF16 operator probe changed device or dtype")
    del observed, probe
    torch.mps.empty_cache()
    available = int(psutil.virtual_memory().available)
    swap = swap_used_bytes()
    thermal = thermal_state()
    limits = config["safety_limits"]
    if available < int(limits["minimum_available_memory_bytes"]):
        raise RuntimeError("available memory is below the frozen preflight floor")
    if thermal not in set(limits["accepted_thermal_states"]):
        raise RuntimeError("thermal state is outside the frozen safe set")
    return {
        "schema_version": 1,
        "artifact_type": "stage1b_measurement_primitives_preflight",
        "status": "passed",
        "mode": mode,
        "git": _verify_git(mode),
        "runtime": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "versions": versions,
            "upstream_direct_url": direct_url,
            "mps_built": True,
            "mps_available": True,
            "mps_bfloat16_probe": "passed",
            "fallback_variable_present": False,
            "outer_autocast_enabled": torch.is_autocast_enabled(),
        },
        "assets": _verify_assets(cache),
        "host_safety": {
            "available_memory_bytes": available,
            "swap_used_bytes": swap,
            "thermal_state": thermal,
            "minimum_available_memory_bytes": int(
                limits["minimum_available_memory_bytes"]
            ),
        },
        "privacy": {
            "network_accessed": False,
            "credential_values_read": False,
            "private_paths_recorded": False,
        },
    }


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        raise RuntimeError(
            "preflight output must be a new file in an existing directory"
        )
    record = collect_preflight(args.config, args.hf_cache, args.mode)
    write_json_atomic(args.output, record)
    print(json.dumps({"status": "passed", "mode": args.mode}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
