#!/usr/bin/env python3
"""Emit a local-only, privacy-safe Stage 1A environment preflight report."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_UPSTREAM_URL = "https://github.com/decoderesearch/circuit-tracer.git"
EXPECTED_UPSTREAM_COMMIT = "8f1e2438df612464e229e44c4a00ff637bf9379b"
MODEL_REPOSITORY = "google/gemma-2-2b"
MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
TRANSCODER_REPOSITORY = "mwhanna/gemma-scope-transcoders"
TRANSCODER_REVISION = "bd5773156dea09893636c801df1237d0410307d2"
TRANSCODER_METADATA_BYTES = 7_855_395_600
MODEL_SNAPSHOT = REPOSITORY_ROOT / "results/generated/stage1a/assets/google-gemma-2-2b"
TRANSCODER_SNAPSHOT = (
    REPOSITORY_ROOT / "results/generated/stage1a/assets/mwhanna-gemma-scope-transcoders"
)
EXPECTED_VERSIONS = {
    "circuit-tracer": "0.5.2",
    "nnsight": "0.6.1",
    "torch": "2.13.0",
    "transformer-lens": "3.2.1",
    "transformers": "4.57.3",
}
REPORTED_DISTRIBUTIONS = (
    "circuit-tracer",
    "huggingface-hub",
    "nnsight",
    "numpy",
    "safetensors",
    "tokenizers",
    "torch",
    "transformer-lens",
    "transformers",
)
TRUTHY_ENV_VALUES = frozenset({"1", "on", "true", "yes"})


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_local(command: Sequence[str]) -> str | None:
    """Run a fixed read-only local command without forwarding its raw output."""

    try:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _apple_chip() -> str | None:
    if platform.system() != "Darwin" or shutil.which("system_profiler") is None:
        return platform.processor() or None

    hardware = _run_local(("system_profiler", "SPHardwareDataType"))
    if hardware is None:
        return platform.processor() or None
    match = re.search(r"^\s*Chip:\s*(.+?)\s*$", hardware, flags=re.MULTILINE)
    return match.group(1) if match else platform.processor() or None


def _physical_memory_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (OSError, TypeError, ValueError):
        return None
    total = pages * page_size
    return total if total > 0 else None


def _snapshot_present(path: Path, *, config_filename: str) -> bool:
    """Require both configuration and weight content, not merely a directory."""

    return (
        path.is_dir()
        and (path / config_filename).is_file()
        and any(path.rglob("*.safetensors"))
    )


def _installed_distribution_inventory() -> list[dict[str, str]]:
    """Return a path-free exact inventory of the active Python environment."""

    inventory: dict[str, tuple[str, str]] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if not name:
            continue
        inventory[name.casefold()] = (name, distribution.version)
    return [
        {"name": name, "version": version}
        for name, version in sorted(
            inventory.values(), key=lambda item: item[0].casefold()
        )
    ]


def _safe_error(error: BaseException) -> str:
    """Redact likely private paths or credential-shaped fragments."""

    rendered = " ".join(str(error).split())
    rendered = rendered.replace(str(Path.home()), "<HOME>")
    rendered = re.sub(r"/(?:Users|home)/[^/\s]+", "<HOME>", rendered)
    rendered = re.sub(
        r"(?i)\b(token|authorization|password)(\s*[:=]\s*)\S+",
        r"\1\2<REDACTED>",
        rendered,
    )
    return rendered[:512]


def _circuit_tracer_metadata() -> dict[str, Any]:
    empty_direct_url = {
        "commit_id": None,
        "requested_revision": None,
        "url": None,
        "vcs": None,
    }
    try:
        distribution = importlib.metadata.distribution("circuit-tracer")
    except importlib.metadata.PackageNotFoundError:
        return {"direct_url": empty_direct_url, "version": None}

    raw = distribution.read_text("direct_url.json")
    if raw is None:
        return {"direct_url": empty_direct_url, "version": distribution.version}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"direct_url": empty_direct_url, "version": distribution.version}

    url = parsed.get("url")
    vcs_info = parsed.get("vcs_info")
    if url != EXPECTED_UPSTREAM_URL or not isinstance(vcs_info, dict):
        return {"direct_url": empty_direct_url, "version": distribution.version}

    commit_id = vcs_info.get("commit_id")
    requested_revision = vcs_info.get("requested_revision")
    vcs = vcs_info.get("vcs")
    return {
        "direct_url": {
            "commit_id": commit_id if isinstance(commit_id, str) else None,
            "requested_revision": (
                requested_revision if isinstance(requested_revision, str) else None
            ),
            "url": EXPECTED_UPSTREAM_URL,
            "vcs": vcs if isinstance(vcs, str) else None,
        },
        "version": distribution.version,
    }


def _dtype_probe(torch: Any, *, device: str, dtype: Any) -> dict[str, Any]:
    probe: dict[str, Any] = {
        "attempted": True,
        "error": None,
        "error_type": None,
        "success": False,
    }
    try:
        values = torch.ones((2, 2), device=device, dtype=dtype)
        torch.matmul(values, values)
    except Exception as error:  # Runtime backend failures are version-specific.
        probe["error"] = _safe_error(error)
        probe["error_type"] = type(error).__name__
    else:
        probe["success"] = True
    return probe


def _accelerators() -> tuple[dict[str, Any], str, list[str]]:
    warnings: list[str] = []
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        warnings.append("torch_not_importable")
        return (
            {
                "cuda": {
                    "available": False,
                    "compiled_version": None,
                    "device_count": 0,
                },
                "mps": {
                    "allocation_probe": {
                        "attempted": False,
                        "dtype": "float32",
                        "error": None,
                        "error_type": None,
                        "success": None,
                    },
                    "available": False,
                    "built": False,
                },
                "dtype_support": {
                    "cpu_bfloat16": {"attempted": False, "success": None},
                    "cuda_bfloat16": {"attempted": False, "success": None},
                    "mps_bfloat16": {"attempted": False, "success": None},
                },
                "torch_importable": False,
            },
            "cpu",
            warnings,
        )

    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_built = bool(mps_backend is not None and mps_backend.is_built())
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    mps_probe: dict[str, Any] = {
        "attempted": mps_built,
        "dtype": "float32",
        "error": None,
        "error_type": None,
        "success": None,
    }
    if mps_built:
        try:
            torch.ones(1, device="mps", dtype=torch.float32)
        except Exception as error:  # Runtime backend failures are version-specific.
            mps_probe["error"] = _safe_error(error)
            mps_probe["error_type"] = type(error).__name__
            mps_probe["success"] = False
        else:
            mps_probe["success"] = True

    cuda_available = bool(torch.cuda.is_available())
    cuda_devices = int(torch.cuda.device_count()) if cuda_available else 0
    cuda_version = torch.version.cuda
    cpu_bfloat16 = _dtype_probe(torch, device="cpu", dtype=torch.bfloat16)
    mps_bfloat16 = (
        _dtype_probe(torch, device="mps", dtype=torch.bfloat16)
        if mps_available
        else {"attempted": False, "success": None}
    )
    cuda_bfloat16 = (
        _dtype_probe(torch, device="cuda", dtype=torch.bfloat16)
        if cuda_available and bool(torch.cuda.is_bf16_supported())
        else {"attempted": False, "success": None}
    )

    if mps_built and not mps_available:
        warnings.append("mps_built_but_unavailable")
    if mps_probe["success"] is False:
        warnings.append("mps_allocation_probe_failed")

    selected_device = "mps" if mps_available else "cuda" if cuda_available else "cpu"
    return (
        {
            "cuda": {
                "available": cuda_available,
                "compiled_version": (
                    str(cuda_version) if cuda_version is not None else None
                ),
                "device_count": cuda_devices,
            },
            "mps": {
                "allocation_probe": mps_probe,
                "available": mps_available,
                "built": mps_built,
            },
            "dtype_support": {
                "cpu_bfloat16": cpu_bfloat16,
                "cuda_bfloat16": cuda_bfloat16,
                "mps_bfloat16": mps_bfloat16,
            },
            "torch_importable": True,
        },
        selected_device,
        warnings,
    )


def _git_head() -> str | None:
    raw = _run_local(("git", "rev-parse", "HEAD"))
    if raw is None:
        return None
    candidate = raw.strip()
    return candidate if re.fullmatch(r"[0-9a-f]{40}", candidate) else None


def _git_dirty() -> bool | None:
    raw = _run_local(
        (
            "git",
            "status",
            "--porcelain",
            "--untracked-files=normal",
            "--",
            ".",
            ":(exclude)results/stage1a",
        )
    )
    return None if raw is None else bool(raw.strip())


def collect_report(
    *,
    model_snapshot_present: bool | None = None,
    transcoder_snapshot_present: bool | None = None,
) -> dict[str, Any]:
    """Collect a strict artifact envelope without network or credential access."""

    warnings: list[str] = []
    versions = {
        distribution: _distribution_version(distribution)
        for distribution in REPORTED_DISTRIBUTIONS
    }
    for distribution, expected in EXPECTED_VERSIONS.items():
        if versions.get(distribution) != expected:
            warnings.append(f"unexpected_{distribution}_version")
    if sys.version_info[:2] != (3, 11):
        warnings.append("unexpected_python_version")
    current_system = platform.system()
    current_machine = platform.machine()
    observed_macos = current_system == "Darwin" and current_machine == "arm64"
    planned_colab_platform = current_system == "Linux" and current_machine in {
        "x86_64",
        "amd64",
    }
    if not observed_macos and not planned_colab_platform:
        warnings.append("unexpected_platform_for_observed_lock")

    circuit_tracer = _circuit_tracer_metadata()
    direct_url = circuit_tracer["direct_url"]
    if (
        direct_url["commit_id"] != EXPECTED_UPSTREAM_COMMIT
        or direct_url["requested_revision"] != EXPECTED_UPSTREAM_COMMIT
        or direct_url["vcs"] != "git"
    ):
        warnings.append("circuit_tracer_direct_url_not_verified")

    accelerators, selected_device, accelerator_warnings = _accelerators()
    warnings.extend(accelerator_warnings)

    fallback_enabled = (
        os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "").strip().lower()
        in TRUTHY_ENV_VALUES
    )
    if fallback_enabled:
        warnings.append("pytorch_mps_fallback_enabled")

    model_present = (
        _snapshot_present(MODEL_SNAPSHOT, config_filename="config.json")
        if model_snapshot_present is None
        else model_snapshot_present
    )
    transcoder_present = (
        _snapshot_present(TRANSCODER_SNAPSHOT, config_filename="config.yaml")
        if transcoder_snapshot_present is None
        else transcoder_snapshot_present
    )
    if not model_present:
        warnings.append("model_snapshot_missing")
    if not transcoder_present:
        warnings.append("transcoder_snapshot_missing")

    dtype_support = accelerators["dtype_support"]
    dtype_probe_key = f"{selected_device}_bfloat16"
    selected_dtype_ready = bool(
        isinstance(dtype_support, dict)
        and isinstance(dtype_support.get(dtype_probe_key), dict)
        and dtype_support[dtype_probe_key].get("success") is True
    )
    accelerator_ready = selected_device in {"mps", "cuda"} and selected_dtype_ready
    if selected_device in {"mps", "cuda"} and not selected_dtype_ready:
        warnings.append(f"{selected_device}_bfloat16_probe_failed")
    full_model_execution_allowed = (
        accelerator_ready
        and model_present
        and transcoder_present
        and not fallback_enabled
    )
    if not full_model_execution_allowed:
        warnings.append("full_model_execution_not_ready")

    disk = shutil.disk_usage(REPOSITORY_ROOT)
    macos_version = platform.mac_ver()[0] if platform.system() == "Darwin" else None
    deviations = ["CUDA/Colab was planned but not observed in this environment."]
    if selected_device == "cpu":
        deviations.append(
            "Local accelerator execution was unavailable; CPU is limited to "
            "metadata, tests, and small semantics."
        )
    colab_observed = planned_colab_platform and selected_device == "cuda"
    if observed_macos:
        environment_lock = "requirements-lock-macos-arm64-py311.txt"
        lock_format = "pip-freeze"
        lock_provenance = "observed"
        observation_scope = "macos-arm64-py311"
    elif planned_colab_platform:
        environment_lock = "requirements-colab-py311-cu124-planned.txt"
        lock_format = "direct-pins-planned"
        lock_provenance = "planned-input-with-observed-runtime-inventory"
        observation_scope = "linux-x86_64-py311"
        warnings.append("colab_input_is_planned_not_an_observed_transitive_lock")
    else:
        environment_lock = None
        lock_format = "none"
        lock_provenance = "unrecognized-platform"
        observation_scope = "unrecognized-py311"

    payload = {
        "accelerators": accelerators,
        "assets": {
            "model": {
                "access_status": (
                    "local_snapshot_present"
                    if model_present
                    else "manual_gated_access_required"
                ),
                "local_snapshot_present": model_present,
                "metadata_reported_bytes": None,
                "repository_id": MODEL_REPOSITORY,
                "revision": MODEL_REVISION,
            },
            "transcoder": {
                "access_status": (
                    "local_snapshot_present"
                    if transcoder_present
                    else "public_metadata_only"
                ),
                "local_snapshot_present": transcoder_present,
                "metadata_reported_bytes": TRANSCODER_METADATA_BYTES,
                "repository_id": TRANSCODER_REPOSITORY,
                "revision": TRANSCODER_REVISION,
            },
        },
        "execution_policy": {
            "current_runtime": {
                "cpu_scope": "metadata_tests_and_small_semantics_only",
                "fallback_enabled": fallback_enabled,
                "fallback_used": False,
                "full_model_execution_allowed": full_model_execution_allowed,
                "offload": "cpu" if selected_device != "cpu" else "none",
                "requested_dtype": "bfloat16",
                "selected_device": selected_device,
            },
            "planned_colab": {
                "device": "cuda",
                "dtype": "bfloat16",
                "observed": colab_observed,
                "offload": "disk",
            },
        },
        "offline_only": True,
        "packages": {
            "circuit_tracer": circuit_tracer,
            "installed_inventory": _installed_distribution_inventory(),
            "versions": versions,
        },
        "platform": {
            "chip": _apple_chip(),
            "cpu_count": os.cpu_count(),
            "disk": {
                "free_bytes": disk.free,
                "total_bytes": disk.total,
            },
            "machine": current_machine,
            "macos_version": macos_version or None,
            "physical_memory_bytes": _physical_memory_bytes(),
            "system": current_system,
        },
        "privacy": {
            "credential_files_inspected": False,
            "credential_values_read": False,
            "private_paths_emitted": False,
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
    }
    head = _git_head()
    dirty = _git_dirty()
    return {
        "artifact_type": "environment_manifest",
        "deviations": sorted(deviations),
        "payload": payload,
        "provenance": {
            "base_commit": head if dirty else None,
            "code_commit": head if dirty is False else None,
            "code_revision_status": (
                "clean_commit"
                if dirty is False
                else "uncommitted_worktree"
                if dirty is True
                else "unknown"
            ),
            "environment_lock": environment_lock,
            "lock_format": lock_format,
            "lock_provenance": lock_provenance,
            "observation_scope": observation_scope,
            "stage": "stage1a",
            "upstream_commit": EXPECTED_UPSTREAM_COMMIT,
        },
        "run_id": "stage1a-local-preflight",
        "schema_version": 1,
        "status": "observed",
        "warnings": sorted(set(warnings)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Atomically write the tracked environment manifest. The only allowed "
            "destination is results/stage1a/environment_manifest.json."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = collect_report()
    if args.output is None:
        json.dump(
            report,
            sys.stdout,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")
        return 0

    expected_output = (
        REPOSITORY_ROOT / "results/stage1a/environment_manifest.json"
    ).resolve()
    requested_output = args.output.resolve()
    if requested_output != expected_output:
        raise SystemExit("--output must be results/stage1a/environment_manifest.json")

    source_root = REPOSITORY_ROOT / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from cfsus.reproduction.artifacts import write_json_atomic

    write_json_atomic(requested_output, report)
    sys.stdout.write("environment_manifest=written\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
