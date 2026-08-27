"""Offline Git, runtime, and artifact preflight for Stage 1D."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from cfsus.mps_telemetry import swap_used_bytes, thermal_state
from cfsus.reproduction.small_model_mps_bf16 import assert_fallback_disabled
from cfsus.stage1b_runtime import resolve_offline_snapshots
from cfsus.stage1d.config import BASE_COMMIT, BRANCH, load_stage1d_config

PROTECTED_ORIGIN_REFS = {
    "main": "7aacf30d888f96a29a1cfc82d035fca489ed0c17",
    "stage-1a-small-model-mps-bf16": "fb2fc158b45c842743804040e4e273776e666a48",
    "stage-1b-measurement-primitives": "efbf70a7e462e640a0e1819a93f3b92727bbd193",
    "stage-1c-first-prospective-prediction": "cc47cb604fc2422deb50aacbc7fde77499b532c5",
    "stage-1c-v2-heldout-prospective-prediction": (
        "ee9cc944fbdabaa6437b7be3c997725fce5de0a6"
    ),
    "stage-1c-v3-preregistered-prospective-prediction": (
        "92ba35cde279c46e1907f0a48ccb56ad378ccbd5"
    ),
    "stage-1c-v4-protocol-preserving-execution": BASE_COMMIT,
}
EXPECTED_PACKAGES = {
    "circuit-tracer": "0.5.2",
    "nnsight": "0.6.1",
    "torch": "2.6.0",
    "transformers": "4.57.3",
}


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
        env={
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        },
    ).stdout.rstrip("\n")


def verify_git(repository: Path, *, expected_head: str | None = None) -> dict[str, Any]:
    """Fail closed unless the isolated branch is clean and exactly pushed."""

    head = _git(repository, "rev-parse", "HEAD")
    branch = _git(repository, "branch", "--show-current")
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    if branch != BRANCH or status:
        raise RuntimeError("Stage 1D requires its clean isolated worktree")
    if expected_head is not None and head != expected_head:
        raise RuntimeError("Stage 1D HEAD differs from the frozen execution commit")
    _git(repository, "merge-base", "--is-ancestor", BASE_COMMIT, head)
    protected = {
        name: _git(repository, "rev-parse", f"refs/remotes/origin/{name}")
        for name in PROTECTED_ORIGIN_REFS
    }
    if protected != PROTECTED_ORIGIN_REFS:
        raise RuntimeError("a protected origin ref differs from its frozen SHA")
    origin_head = _git(repository, "rev-parse", f"refs/remotes/origin/{BRANCH}")
    upstream = _git(
        repository, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"
    )
    if origin_head != head or upstream != f"origin/{BRANCH}":
        raise RuntimeError("Stage 1D local/origin branch identity differs")
    return {
        "branch": branch,
        "head": head,
        "origin_branch_head": origin_head,
        "upstream": upstream,
        "working_tree_clean": True,
        "base_commit": BASE_COMMIT,
        "base_ancestry_verified": True,
        "protected_origin_refs": protected,
    }


def runtime_identity(cache: Path, repository: Path) -> dict[str, Any]:
    """Verify native-arm64 versions and already-pinned offline snapshots."""

    config = load_stage1d_config(
        repository / "configs/stage1d_multiprompt_gate_benchmark.yaml"
    )
    assert_fallback_disabled()
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
    ):
        raise RuntimeError("Stage 1D workers require enforced offline mode")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Stage 1D requires native macOS arm64")
    observed = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES}
    if observed != EXPECTED_PACKAGES or platform.python_version() != "3.11.13":
        raise RuntimeError("Stage 1D runtime versions differ")
    cpu = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    memory = int(
        subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    )
    if cpu != "Apple M2 Max" or memory != 32 * 1024**3:
        raise RuntimeError("Stage 1D host identity differs")
    direct_url_raw = importlib.metadata.distribution("circuit-tracer").read_text(
        "direct_url.json"
    )
    if direct_url_raw is None:
        raise RuntimeError("circuit-tracer immutable install provenance is missing")
    direct_url = json.loads(direct_url_raw)
    if (
        direct_url.get("vcs_info", {}).get("commit_id")
        != config["runtime"]["upstream_revision"]
    ):
        raise RuntimeError("circuit-tracer install revision differs")
    model, transcoder = resolve_offline_snapshots(cache, repository)
    if (
        model.name != config["runtime"]["model_revision"]
        or transcoder.name != config["runtime"]["transcoder_revision"]
    ):
        raise RuntimeError("Stage 1D immutable snapshot revisions differ")
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": observed,
        "mps_built": None,
        "mps_available": None,
        "fallback_variable_present": "PYTORCH_ENABLE_MPS_FALLBACK" in os.environ,
        "network_accessed": False,
        "model_revision": model.name,
        "transcoder_revision": transcoder.name,
        "swap_used_bytes": swap_used_bytes(),
        "thermal_state": thermal_state(),
        "host_cpu": cpu,
        "physical_memory_bytes": memory,
        "upstream_revision_verified": True,
    }


__all__ = [
    "EXPECTED_PACKAGES",
    "PROTECTED_ORIGIN_REFS",
    "runtime_identity",
    "verify_git",
]
