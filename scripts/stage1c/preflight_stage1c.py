#!/usr/bin/env python3
"""Fail-closed, offline preflight for the Stage 1C pilot.

Prediction preflight intentionally runs at the exact Stage 1B base commit.  The
new Stage 1C protocol is therefore allowed to be present as untracked files,
but tracked changes and unrecognised untracked files are rejected.  The
intervention preflight is stricter: it requires a clean worktree and the exact
pre-intervention commit that was frozen and pushed by the orchestrator.
"""

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
from cfsus.stage1b_runtime import resolve_offline_snapshots  # noqa: E402
from cfsus.stage1c.config import (  # noqa: E402
    BASE_COMMIT,
    BRANCH,
    CONFIG_PATH,
    MODEL_REVISION,
    TRANSCODER_REVISION,
    UPSTREAM_REVISION,
    load_stage1c_config,
)

EXPECTED_VERSIONS = {
    "circuit-tracer": "0.5.2",
    "nnsight": "0.6.1",
    "torch": "2.6.0",
    "transformers": "4.57.3",
}
CREDENTIAL_VARIABLES = frozenset(
    {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"}
)
SHA40_RE = re.compile(r"\A[0-9a-f]{40}\Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / CONFIG_PATH)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("prediction", "intervention"), required=True
    )
    parser.add_argument("--pre-intervention-commit", default=None)
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
    return result.stdout.rstrip("\n")


def _git_status_paths() -> tuple[tuple[str, ...], tuple[str, ...]]:
    tracked: list[str] = []
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    untracked: list[str] = []
    for line in status.splitlines():
        if len(line) < 3:
            raise RuntimeError("malformed git status output")
        code, path = line[:2], line[3:]
        if code == "??":
            untracked.append(path)
        elif code.strip():
            tracked.append(path)
        else:
            raise RuntimeError("unexpected git status entry")
    return tuple(sorted(tracked)), tuple(sorted(untracked))


def _stage1c_protocol_path(path: str) -> bool:
    return (
        path.startswith("src/cfsus/stage1c/")
        or path.startswith("scripts/stage1c/")
        or path
        in {
            "configs/stage1c_first_prospective_prediction.yaml",
            "configs/stage1c_first_prospective_prediction_artifact_schema.json",
        }
        or (path.startswith("tests/test_stage1c_") and path.endswith(".py"))
    )


def _verify_git(phase: str, pre_intervention_commit: str | None) -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    tracked, untracked = _git_status_paths()
    has_changes = bool(tracked or untracked)
    if SHA40_RE.fullmatch(head) is None or branch != BRANCH:
        raise RuntimeError("Stage 1C branch identity is invalid")
    if phase == "prediction":
        if head != BASE_COMMIT:
            raise RuntimeError("prediction must run at the exact Stage 1B base")
        if any(path != "docs/DECISIONS.md" for path in tracked):
            raise RuntimeError("prediction has an unrecognised tracked change")
        if any(not _stage1c_protocol_path(path) for path in untracked):
            raise RuntimeError("prediction has an unrecognised untracked file")
        if pre_intervention_commit is not None:
            raise RuntimeError("prediction must not receive an intervention commit")
        return {
            "phase": phase,
            "branch": branch,
            "head": head,
            "working_tree_clean_except_protocol": True,
            "protocol_tracked_paths": list(tracked),
            "protocol_untracked_paths": list(untracked),
            "base_commit": BASE_COMMIT,
        }

    if (
        pre_intervention_commit is None
        or SHA40_RE.fullmatch(pre_intervention_commit) is None
    ):
        raise RuntimeError("intervention requires a valid pre-intervention commit")
    if head != pre_intervention_commit or has_changes:
        raise RuntimeError(
            "intervention requires a clean exact pre-intervention commit"
        )
    origin_head = _git("rev-parse", f"origin/{BRANCH}")
    if origin_head != pre_intervention_commit:
        raise RuntimeError(
            "origin Stage 1C branch is not at the pre-intervention commit"
        )
    return {
        "phase": phase,
        "branch": branch,
        "head": head,
        "working_tree_clean": True,
        "pre_intervention_commit": pre_intervention_commit,
        "origin_branch_head": origin_head,
        "origin_branch_verified": True,
    }


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _direct_url_identity() -> dict[str, str | None]:
    distribution = importlib.metadata.distribution("circuit-tracer")
    raw = distribution.read_text("direct_url.json")
    if raw is None:
        raise RuntimeError("circuit-tracer VCS provenance is absent")
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
                if key not in CREDENTIAL_VARIABLES
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


def collect_preflight(
    config_path: Path,
    cache: Path,
    phase: str,
    pre_intervention_commit: str | None = None,
) -> dict[str, Any]:
    """Collect a compact path-free report and raise on every hard gate."""

    assert_fallback_disabled()
    if CREDENTIAL_VARIABLES.intersection(os.environ):
        raise RuntimeError("credential-bearing environment variable is present")
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
    ):
        raise RuntimeError("preflight requires enforced offline mode")
    config = load_stage1c_config(config_path)
    git = _verify_git(phase, pre_intervention_commit)
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("native macOS arm64 runtime is required")
    if platform.python_version() != "3.11.13":
        raise RuntimeError("exact Python 3.11.13 runtime is required")
    versions = {name: _distribution_version(name) for name in EXPECTED_VERSIONS}
    if versions != EXPECTED_VERSIONS:
        raise RuntimeError("runtime package versions differ from the frozen lock")
    direct_url = _direct_url_identity()
    expected_url = {
        "url": "https://github.com/decoderesearch/circuit-tracer.git",
        "vcs": "git",
        "commit_id": UPSTREAM_REVISION,
        "requested_revision": UPSTREAM_REVISION,
    }
    if direct_url != expected_url:
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
        "artifact_type": "stage1c_first_prospective_prediction_preflight",
        "status": "passed",
        "phase": phase,
        "config_phase": config["phase"],
        "git": git,
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
    record = collect_preflight(
        args.config,
        args.hf_cache,
        args.phase,
        args.pre_intervention_commit,
    )
    write_json_atomic(args.output, record)
    print(json.dumps({"status": "passed", "phase": args.phase}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
