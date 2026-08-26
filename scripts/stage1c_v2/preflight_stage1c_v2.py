#!/usr/bin/env python3
"""Fail-closed, offline preflight for Stage 1C-v2.

Preflight verifies the frozen runtime, immutable local assets, exact held-out
tokenization, native MPS/BF16 support, and host safety without loading model
weights or calling an intervention API.
"""

from __future__ import annotations

import argparse
import hashlib
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
from cfsus.stage1b_runtime import resolve_offline_snapshots  # noqa: E402
from cfsus.stage1c_v2.serialization import (  # noqa: E402
    SerializationError,
    read_json_strict,
    write_json_new,
)

V2_EXPERIMENT_CLASS = "stage1c_v2_heldout_prospective_prediction"
V2_BRANCH = "stage-1c-v2-heldout-prospective-prediction"
V2_BASE_COMMIT = "cc47cb604fc2422deb50aacbc7fde77499b532c5"
V2_PROMPT_ID = "capital_germany_heldout_v2"
V2_PROMPT_TEXT = "The capital of Germany is"
V2_CONFIG_RELATIVE = "configs/stage1c_v2_heldout_prospective_prediction.yaml"
V2_SCHEMA_RELATIVE = (
    "configs/stage1c_v2_heldout_prospective_prediction_artifact_schema.json"
)
SHA40_RE = re.compile(r"\A[0-9a-f]{40}\Z")

EXPECTED_VERSIONS = {
    "circuit-tracer": "0.5.2",
    "nnsight": "0.6.1",
    "torch": "2.6.0",
    "transformers": "4.57.3",
}
EXPECTED_DIRECT_URL = {
    "url": "https://github.com/decoderesearch/circuit-tracer.git",
    "vcs": "git",
    "commit_id": "8f1e2438df612464e229e44c4a00ff637bf9379b",
    "requested_revision": "8f1e2438df612464e229e44c4a00ff637bf9379b",
}
EXPECTED_TOKEN_IDS = [2, 818, 5279, 529, 9405, 563]
EXPECTED_SELECTED_POSITIONS = [1, 2, 3, 4, 5]
EXPECTED_HOST = "Apple M2 Max"
EXPECTED_PHYSICAL_MEMORY_BYTES = 32 * 1024**3
CREDENTIAL_VARIABLES = frozenset(
    {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"}
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / V2_CONFIG_RELATIVE
    )
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("prediction", "intervention"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pre-intervention-commit")
    parser.add_argument("--prediction-manifest", type=Path)
    parser.add_argument("--prediction-manifest-sha256")
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


def _status_paths() -> tuple[tuple[str, ...], tuple[str, ...]]:
    tracked: list[str] = []
    untracked: list[str] = []
    for line in _git("status", "--porcelain=v1", "--untracked-files=all").splitlines():
        if len(line) < 3:
            raise RuntimeError("malformed Git status output")
        code, path = line[:2], line[3:]
        if code == "??":
            untracked.append(path)
        elif code.strip():
            tracked.append(path)
        else:
            raise RuntimeError("unexpected Git status entry")
    return tuple(sorted(tracked)), tuple(sorted(untracked))


def verify_git(
    phase: str, pre_intervention_commit: str | None = None
) -> dict[str, Any]:
    """Verify v2 branch/base/cleanliness and origin identity."""

    head = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    tracked, untracked = _status_paths()
    if SHA40_RE.fullmatch(head) is None or branch != V2_BRANCH:
        raise RuntimeError("Stage 1C-v2 branch identity is invalid")

    _git("merge-base", "--is-ancestor", V2_BASE_COMMIT, head)
    if _git("rev-parse", "origin/stage-1c-first-prospective-prediction") != (
        V2_BASE_COMMIT
    ):
        raise RuntimeError("protected Stage 1C-v1 origin base changed")

    if phase == "prediction":
        if head == V2_BASE_COMMIT or tracked or untracked:
            raise RuntimeError(
                "prediction requires a clean committed v2 protocol descendant"
            )
        if pre_intervention_commit is not None:
            raise RuntimeError("prediction must not receive an intervention commit")
        return {
            "phase": phase,
            "branch": branch,
            "head": head,
            "working_tree_clean": True,
            "protocol_commit": head,
            "base_commit": V2_BASE_COMMIT,
            "base_ancestry_verified": True,
            "protected_origin_base_verified": True,
        }

    if (
        pre_intervention_commit is None
        or SHA40_RE.fullmatch(pre_intervention_commit) is None
    ):
        raise RuntimeError("intervention requires a valid v2 pre-intervention commit")
    if head != pre_intervention_commit or tracked or untracked:
        raise RuntimeError(
            "intervention requires a clean exact v2 pre-intervention commit"
        )
    origin_head = _git("rev-parse", f"refs/remotes/origin/{V2_BRANCH}")
    if origin_head != pre_intervention_commit:
        raise RuntimeError("origin v2 branch is not at the pre-intervention commit")
    upstream = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    if upstream != f"origin/{V2_BRANCH}":
        raise RuntimeError("v2 branch does not track its origin branch")
    return {
        "phase": phase,
        "branch": branch,
        "head": head,
        "working_tree_clean": True,
        "pre_intervention_commit": pre_intervention_commit,
        "origin_branch_head": origin_head,
        "upstream": upstream,
        "base_commit": V2_BASE_COMMIT,
        "base_ancestry_verified": True,
        "protected_origin_base_verified": True,
    }


def _load_config(path: Path) -> dict[str, Any]:
    """Load the v2 config through its typed loader when available."""

    try:
        from cfsus.stage1c_v2.config import load_stage1c_v2_config
    except ImportError:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as error:  # pragma: no cover - dependency gate
            raise RuntimeError("PyYAML or the v2 config loader is required") from error
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError("v2 config root must be an object") from None
        return value
    return load_stage1c_v2_config(path)


def _config_identity(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("experiment_class") != V2_EXPERIMENT_CLASS:
        raise RuntimeError("config is not the v2 experiment class")
    if config.get("branch") != V2_BRANCH or config.get("base_commit") != V2_BASE_COMMIT:
        raise RuntimeError("v2 config Git identity differs")
    prompt = config.get("prompt")
    if not isinstance(prompt, dict):
        raise RuntimeError("v2 prompt config is missing")
    if prompt.get("id") != V2_PROMPT_ID or prompt.get("text") != V2_PROMPT_TEXT:
        raise RuntimeError("v2 held-out prompt identity differs")
    return {
        "experiment_class": V2_EXPERIMENT_CLASS,
        "branch": V2_BRANCH,
        "base_commit": V2_BASE_COMMIT,
        "prompt_id": V2_PROMPT_ID,
        "prompt_text": V2_PROMPT_TEXT,
    }


def _scan_no_legacy_identity(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if "stage1c_first" in key.casefold() or "france" in key.casefold():
                raise RuntimeError(f"legacy v1 identity appears at {path}.{key}")
            _scan_no_legacy_identity(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_no_legacy_identity(item, f"{path}[{index}]")


def verify_prediction_manifest(
    path: Path, digest: str | None, config: dict[str, Any]
) -> dict[str, Any]:
    """Verify a frozen v2 manifest without importing runtime/model packages."""

    try:
        value = read_json_strict(path)
    except SerializationError as error:
        raise RuntimeError("prediction manifest is not strict finite JSON") from error
    if not isinstance(value, dict):  # pragma: no cover - read_json_strict contract
        raise RuntimeError("prediction manifest must be an object")
    _scan_no_legacy_identity(value)
    _config_identity(config)
    if value.get("experiment_class") != V2_EXPERIMENT_CLASS:
        raise RuntimeError("prediction manifest experiment class differs")
    if value.get("status") != "prediction_frozen_ready_for_commit":
        raise RuntimeError("prediction manifest is not freeze-ready")
    if value.get("base_commit") != V2_BASE_COMMIT or value.get("branch") != V2_BRANCH:
        raise RuntimeError("prediction manifest Git identity differs")
    prompt = value.get("prompt")
    if not isinstance(prompt, dict):
        raise RuntimeError("prediction manifest prompt is missing")
    expected_prompt = config["prompt"]
    if prompt.get("id") != expected_prompt.get("id") or prompt.get(
        "text"
    ) != expected_prompt.get("text"):
        raise RuntimeError("prediction manifest prompt identity differs")
    token_ids = prompt.get("token_ids")
    if token_ids != EXPECTED_TOKEN_IDS:
        raise RuntimeError("prediction manifest token identity differs")
    if digest is not None:
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != digest:
            raise RuntimeError("prediction manifest digest differs")
    return value


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in EXPECTED_VERSIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


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


def _sysctl_text(name: str) -> str:
    return subprocess.run(
        ["sysctl", "-n", name],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
        env={
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        },
    ).stdout.strip()


def _verify_assets_and_tokenizer(
    cache: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    model, transcoder = resolve_offline_snapshots(cache, REPOSITORY_ROOT)
    if (
        model.name != config["model"]["revision"]
        or transcoder.name != config["transcoder"]["revision"]
    ):
        raise RuntimeError("resolved immutable asset revision differs")
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
    verification = json.loads(result.stdout)
    if verification.get("status") != "verified":
        raise RuntimeError("immutable offline asset validation failed")

    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    tokenizer = AutoTokenizer.from_pretrained(
        model,
        local_files_only=True,
        trust_remote_code=False,
    )
    token_ids = [
        int(item)
        for item in tokenizer.encode(
            str(config["prompt"]["text"]), add_special_tokens=True
        )
    ]
    selected_positions = list(range(1, len(token_ids)))
    if (
        token_ids != EXPECTED_TOKEN_IDS
        or token_ids != config["prompt"]["expected_token_ids"]
        or selected_positions != EXPECTED_SELECTED_POSITIONS
        or selected_positions != config["scanner"]["selected_positions"]
    ):
        raise RuntimeError("immutable held-out tokenizer identity changed")
    return (
        {
            "status": "verified",
            "download_performed": False,
            "network_accessed": False,
            "authentication_used": False,
            "actual_total_bytes": verification.get("actual_total_bytes"),
            "model_revision": model.name,
            "transcoder_revision": transcoder.name,
            "exact_allowlist_hashes_verified": True,
        },
        {
            "status": "verified",
            "prompt_id": V2_PROMPT_ID,
            "token_ids": token_ids,
            "selected_positions": selected_positions,
            "tokenizer_payload_persisted": False,
        },
    )


def collect_preflight(
    config_path: Path,
    cache: Path,
    phase: str,
    *,
    pre_intervention_commit: str | None = None,
    prediction_manifest: Path | None = None,
    prediction_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Collect path-free identity evidence and enforce hard process gates."""

    if CREDENTIAL_VARIABLES.intersection(os.environ):
        raise RuntimeError("credential-bearing environment variable is present")
    if "PYTORCH_ENABLE_MPS_FALLBACK" in os.environ:
        raise RuntimeError("MPS fallback variable must be absent")
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
    ):
        raise RuntimeError("v2 preflight requires enforced offline mode")
    config = _load_config(config_path)
    identity = _config_identity(config)
    git = verify_git(phase, pre_intervention_commit)
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("native macOS arm64 runtime is required")
    if platform.python_version() != "3.11.13":
        raise RuntimeError("exact Python 3.11.13 runtime is required")
    versions = _package_versions()
    if versions != EXPECTED_VERSIONS:
        raise RuntimeError("runtime package versions differ from the frozen lock")
    direct_url = _direct_url_identity()
    if direct_url != EXPECTED_DIRECT_URL:
        raise RuntimeError("circuit-tracer immutable VCS identity changed")

    import psutil  # type: ignore[import-untyped]
    import torch  # type: ignore[import-not-found]

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("native MPS is unavailable")
    if torch.is_autocast_enabled():
        raise RuntimeError("outer autocast must be disabled")
    probe = torch.ones((2, 2), device="mps", dtype=torch.bfloat16)
    observed = torch.matmul(probe, probe)
    if observed.device.type != "mps" or observed.dtype != torch.bfloat16:
        raise RuntimeError("MPS/BF16 operator probe changed device or dtype")
    del observed, probe
    torch.mps.empty_cache()

    host = _sysctl_text("machdep.cpu.brand_string")
    physical_memory = int(_sysctl_text("hw.memsize"))
    if host != EXPECTED_HOST or physical_memory != EXPECTED_PHYSICAL_MEMORY_BYTES:
        raise RuntimeError("host hardware identity differs from the frozen runtime")
    available = int(psutil.virtual_memory().available)
    swap = swap_used_bytes()
    thermal = thermal_state()
    limits = config["safety_limits"]
    if available < int(limits["minimum_available_memory_bytes"]):
        raise RuntimeError("available memory is below the frozen preflight floor")
    if thermal not in set(limits["accepted_thermal_states"]):
        raise RuntimeError("thermal state is outside the frozen safe set")
    assets, tokenizer = _verify_assets_and_tokenizer(cache, config)
    manifest_identity: dict[str, Any] | None = None
    if phase == "intervention":
        if prediction_manifest is None or prediction_manifest_sha256 is None:
            raise RuntimeError(
                "intervention requires the v2 prediction manifest digest"
            )
        manifest = verify_prediction_manifest(
            prediction_manifest, prediction_manifest_sha256, config
        )
        manifest_identity = {
            "status": "passed",
            "experiment_class": manifest.get("experiment_class"),
            "prompt_id": manifest.get("prompt", {}).get("id"),
            "token_count": len(manifest["prompt"]["token_ids"]),
        }
    return {
        "schema_version": 2,
        "artifact_type": f"{V2_EXPERIMENT_CLASS}_preflight",
        "status": "passed",
        "phase": phase,
        "identity": identity,
        "git": git,
        "runtime": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "package_versions": versions,
            "upstream_direct_url": direct_url,
            "mps_built": True,
            "mps_available": True,
            "mps_bfloat16_probe": "passed",
            "outer_autocast_enabled": False,
            "offline": True,
            "fallback_variable_present": False,
        },
        "host": {
            "host_class": "Apple M2 Max, 32 GiB unified memory",
            "physical_memory_bytes": physical_memory,
            "available_memory_bytes": available,
            "swap_used_bytes": swap,
            "thermal_state": thermal,
        },
        "assets": assets,
        "tokenizer": tokenizer,
        "prediction_manifest": manifest_identity,
        "privacy": {
            "network_accessed": False,
            "credential_values_read": False,
            "private_paths_recorded": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (
        args.output.exists()
        or args.output.is_symlink()
        or not args.output.parent.is_dir()
    ):
        raise RuntimeError(
            "preflight output must be a new file in an existing directory"
        )
    record = collect_preflight(
        args.config,
        args.hf_cache,
        args.phase,
        pre_intervention_commit=args.pre_intervention_commit,
        prediction_manifest=args.prediction_manifest,
        prediction_manifest_sha256=args.prediction_manifest_sha256,
    )
    write_json_new(args.output, record)
    print(json.dumps({"status": "passed", "phase": args.phase}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
