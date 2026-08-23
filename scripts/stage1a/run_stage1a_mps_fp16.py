#!/usr/bin/env python3
"""Produce and strictly validate the Stage 1A Apple MPS/FP16 artifact bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mps_runtime import (  # noqa: E402
    EXPLICIT_CPU_SPARSE_DEVIATION,
    MODEL_ID,
    MODEL_REVISION,
    MPS_BATCH_SEQUENCE,
    MPS_CLAIM_BOUNDARY,
    MPS_COMPLETED_STATUS,
    MPS_REPRODUCTION_CLASS,
    TRANSCODER_ID,
    TRANSCODER_REVISION,
    UPSTREAM_REVISION,
    MPSRuntimeError,
    evaluate_memory_feasibility,
    sample_mps_memory,
    should_retry_mps_attempt,
    validate_mps_configuration,
)

from cfsus.reproduction.artifacts import (  # noqa: E402
    assert_publication_safe,
    make_artifact_envelope,
    validate_json_value,
    write_json_atomic,
)
from cfsus.reproduction.config import OFFICIAL_UPSTREAM_REPOSITORY  # noqa: E402

WORKER = SCRIPT_DIRECTORY / "run_stage1a_mps_fp16_worker.py"
RESULT_DIRECTORY = REPOSITORY_ROOT / "results/stage1a_mps_fp16"
GENERATED_DIRECTORY = REPOSITORY_ROOT / "results/generated/stage1a_mps_fp16"
PREFLIGHT_OUTPUT = RESULT_DIRECTORY / "preflight/preflight_summary.json"
RUN_MANIFEST_NAME = "stage1a_mps_run_manifest.json"
CHECKSUM_NAME = "checksums.sha256"
PROJECT_BASE_COMMIT = "d965e43c34a2ba408b8ae35b13b5651bf269beed"
EXPECTED_BRANCH = "stage-1a-mps-fp16"
PROTECTED_REFS = {
    "refs/heads/main": "7aacf30d888f96a29a1cfc82d035fca489ed0c17",
    "refs/remotes/origin/main": "7aacf30d888f96a29a1cfc82d035fca489ed0c17",
    "refs/heads/stage-1a-t4-fp16": PROJECT_BASE_COMMIT,
    "refs/remotes/origin/stage-1a-t4-fp16": PROJECT_BASE_COMMIT,
}
PRESERVED_T4_DIRECTORY = REPOSITORY_ROOT / "results/stage1a_t4_fp16"
PRESERVED_T4_INDEX_PATHSPEC = ":(top,icase,literal)results/stage1a_t4_fp16"
PRESERVED_T4_SHA256 = {
    "asset_manifest.json": (
        "173ddb97159ae1e0f4308334fc0fa1244f6ac6417b4bf7203b1c30bb7f1ca5c5"
    ),
    "attribution_summary.json": (
        "f4325a3df2e92e135cf72370735776c6b179c405664b9425ca507e22dba6d36a"
    ),
    "checksums.sha256": (
        "cead4f92a80a37f1d3e669aa4341a534019d2bca888cb8459ac09ca1991ed5d8"
    ),
    "environment_manifest.json": (
        "d00f97035440a4480461714081e3f5e188f4a4386545e4f89be2f18ce201a348"
    ),
    "intervention_summary.json": (
        "43212e2dc0446ef174e44e0eb054b9bb36dc36fbdc918e19a6555d37dc33b38f"
    ),
    "semantics_summary.json": (
        "82aad89d32a3a78dc30a219dc5b6af0086f3d4af99bf8416dc216a0f490db45a"
    ),
    "stage1a_t4_fp16_run_manifest.json": (
        "9cc5acde347f9b6aa9cf5d37802cd76d7b6e5b8fcbd8902b5e50a0b65ff20c8c"
    ),
}
MODEL_METADATA_BYTES = 10_479_239_529
TRANSCODER_METADATA_BYTES = 7_855_395_802
LOADING_PLAN_ID = (
    "hf-model-mps+fp16-transcoder-encoders-mps+transformerlens-conversion-v1"
)
OBSERVED_RUNTIME_LOADING_STOP = {
    "loading_plan_id": LOADING_PLAN_ID,
    "execution_commit": "9de01b5446775f01b211acbc461f8385f9f3732a",
    "attempt_report_sha256": (
        "bae2ee7e0061de08db8f71a6c9c0734212b4666c49d27a65bb7131cbaedfc49e"
    ),
    "mps_current_allocated_peak_bytes": 35_977_178_880,
    "mps_driver_allocated_peak_bytes": 40_032_174_080,
    "swap_used_peak_bytes": 34_567_031_357,
}
RAW_PAYLOAD_NAMES = (
    "model_smoke.json",
    "environment.json",
    "asset.json",
    "semantics.json",
    "intervention.json",
    "attribution.json",
    "memory.json",
)
CANONICAL_JSON_NAMES = (
    "feasibility_report.json",
    "environment_manifest.json",
    "asset_manifest.json",
    "attribution_summary.json",
    "intervention_summary.json",
    "semantics_summary.json",
    "memory_summary.json",
    RUN_MANIFEST_NAME,
)


def _git_output(*arguments: str) -> str:
    git_environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    git_environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    result = subprocess.run(
        ("git", *arguments),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=git_environment,
    )
    if result.stderr.strip():
        raise MPSRuntimeError("Git command produced unexpected diagnostics")
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_preserved_t4(
    directory: Path = PRESERVED_T4_DIRECTORY,
    expected: Mapping[str, str] = PRESERVED_T4_SHA256,
) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise MPSRuntimeError("preserved T4 artifact directory is missing or unsafe")
    entries = list(directory.iterdir())
    observed = {entry.name for entry in entries}
    if observed != set(expected):
        raise MPSRuntimeError("preserved T4 artifact file set changed")
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise MPSRuntimeError("preserved T4 artifact contains an unsafe entry")
        if _sha256_file(entry) != expected[entry.name]:
            raise MPSRuntimeError("preserved T4 artifact content changed")


def _validate_default_index_flags(encoded_entries: str, *, source: str) -> None:
    for entry in encoded_entries.split("\0"):
        if entry and not entry.startswith("H "):
            raise MPSRuntimeError(f"Git index contains non-default {source} flags")


def _validate_no_legacy_grafts(grafts: Path) -> None:
    if not grafts.is_absolute():
        raise MPSRuntimeError("legacy Git graft path is not absolute")
    if not grafts.exists():
        return
    if grafts.is_symlink() or not grafts.is_file() or grafts.stat().st_size > 0:
        raise MPSRuntimeError("legacy Git grafts are not allowed")


def _validate_preserved_t4_index_aliases(
    encoded_paths: str,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    directory: Path = PRESERVED_T4_DIRECTORY,
) -> None:
    """Reject any tracked path that physically aliases preserved T4 evidence."""

    try:
        resolved_root = repository_root.resolve(strict=True)
        resolved_directory = directory.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise MPSRuntimeError("T4 index alias proof could not resolve paths") from error
    if not resolved_directory.is_relative_to(resolved_root):
        raise MPSRuntimeError("preserved T4 directory escapes the repository")
    preserved_files = tuple(entry for entry in directory.iterdir() if entry.is_file())
    for encoded_path in encoded_paths.split("\0"):
        if not encoded_path:
            continue
        relative = PurePosixPath(encoded_path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise MPSRuntimeError("Git index contains an unsafe path")
        candidate = repository_root.joinpath(*relative.parts)
        if not candidate.exists() and not candidate.is_symlink():
            continue
        try:
            resolved_candidate = candidate.resolve(strict=True)
            current = candidate
            physically_within_preserved_directory = False
            while True:
                if current.exists() and os.path.samefile(current, directory):
                    physically_within_preserved_directory = True
                    break
                if current == repository_root:
                    break
                current = current.parent
            physically_contains_preserved_directory = False
            if candidate.is_dir():
                current = directory
                while current != repository_root:
                    if os.path.samefile(candidate, current):
                        physically_contains_preserved_directory = True
                        break
                    current = current.parent
            physically_preserved = any(
                candidate.is_file() and os.path.samefile(candidate, preserved)
                for preserved in preserved_files
            )
        except (OSError, RuntimeError) as error:
            raise MPSRuntimeError(
                "T4 index alias proof could not inspect a path"
            ) from error
        if (
            resolved_candidate == resolved_directory
            or resolved_candidate.is_relative_to(resolved_directory)
            or physically_within_preserved_directory
            or physically_contains_preserved_directory
            or physically_preserved
        ):
            raise MPSRuntimeError("preserved T4 artifacts are unexpectedly tracked")


def _validate_external_cache_subdirectory(cache: Path, name: str) -> Path:
    candidate = cache / name
    if candidate.is_symlink() or (candidate.exists() and not candidate.is_dir()):
        raise MPSRuntimeError("Hugging Face cache subdirectory is unsafe")
    try:
        resolved_cache = cache.resolve(strict=False)
        resolved_candidate = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise MPSRuntimeError(
            "Hugging Face cache path could not be resolved"
        ) from error
    if (
        not resolved_candidate.is_relative_to(resolved_cache)
        or resolved_candidate == REPOSITORY_ROOT
        or resolved_candidate.is_relative_to(REPOSITORY_ROOT)
    ):
        raise MPSRuntimeError(
            "Hugging Face cache subdirectory must be project-external"
        )
    _validate_physical_project_separation(resolved_candidate)
    return resolved_candidate


def _validate_physical_project_separation(path: Path) -> None:
    try:
        resolved_project = REPOSITORY_ROOT.resolve(strict=True)
        current = path
        while True:
            if current.exists() and os.path.samefile(current, resolved_project):
                raise MPSRuntimeError(
                    "Hugging Face cache physically overlaps the project"
                )
            parent = current.parent
            if parent == current:
                break
            current = parent
        if path.exists():
            current = resolved_project
            while True:
                if os.path.samefile(path, current):
                    raise MPSRuntimeError(
                        "Hugging Face cache physically overlaps the project"
                    )
                parent = current.parent
                if parent == current:
                    break
                current = parent
    except MPSRuntimeError:
        raise
    except (OSError, RuntimeError) as error:
        raise MPSRuntimeError(
            "Hugging Face cache physical boundary could not be verified"
        ) from error


def _validate_external_cache_tree(cache: Path) -> None:
    if cache.is_symlink() or (cache.exists() and not cache.is_dir()):
        raise MPSRuntimeError("Hugging Face cache root is unsafe")
    if not cache.exists():
        return
    try:
        resolved_cache = cache.resolve(strict=True)
        cache_device = resolved_cache.stat().st_dev
    except (OSError, RuntimeError) as error:
        raise MPSRuntimeError(
            "Hugging Face cache tree could not be resolved"
        ) from error
    if resolved_cache == REPOSITORY_ROOT or resolved_cache.is_relative_to(
        REPOSITORY_ROOT
    ):
        raise MPSRuntimeError("Hugging Face cache tree must be project-external")
    _validate_physical_project_separation(resolved_cache)

    def fail_on_walk_error(error: OSError) -> None:
        raise MPSRuntimeError(
            "Hugging Face cache tree could not be traversed"
        ) from error

    for current_root, directory_names, filenames in os.walk(
        resolved_cache,
        followlinks=False,
        onerror=fail_on_walk_error,
    ):
        current = Path(current_root)
        for name in (*directory_names, *filenames):
            entry = current / name
            try:
                metadata = entry.lstat()
                if stat.S_ISLNK(metadata.st_mode):
                    target = entry.resolve(strict=True)
                    if (
                        not target.is_relative_to(resolved_cache)
                        or not target.is_file()
                        or target.stat().st_dev != cache_device
                    ):
                        raise MPSRuntimeError(
                            "Hugging Face cache symlink escapes the external cache"
                        )
                elif stat.S_ISDIR(metadata.st_mode):
                    if metadata.st_dev != cache_device:
                        raise MPSRuntimeError(
                            "Hugging Face cache contains a mounted directory"
                        )
                elif stat.S_ISREG(metadata.st_mode):
                    if metadata.st_dev != cache_device or metadata.st_nlink != 1:
                        raise MPSRuntimeError(
                            "Hugging Face cache contains a linked external file"
                        )
                else:
                    raise MPSRuntimeError(
                        "Hugging Face cache contains a special filesystem entry"
                    )
            except MPSRuntimeError:
                raise
            except (OSError, RuntimeError) as error:
                raise MPSRuntimeError(
                    "Hugging Face cache tree could not be inspected"
                ) from error


def _validate_protected_git_state() -> None:
    try:
        if (
            _git_output("symbolic-ref", "--quiet", "HEAD")
            != f"refs/heads/{EXPECTED_BRANCH}"
        ):
            raise MPSRuntimeError("MPS execution branch identity changed")
        for reference, expected in PROTECTED_REFS.items():
            if _git_output("show-ref", "--verify", "--hash", reference) != expected:
                raise MPSRuntimeError(f"protected Git ref changed: {reference}")
        if _git_output("for-each-ref", "--format=%(refname)", "refs/replace"):
            raise MPSRuntimeError("Git replacement refs are not allowed")
        graft_path = Path(
            _git_output(
                "rev-parse",
                "--path-format=absolute",
                "--git-path",
                "info/grafts",
            )
        )
        _validate_no_legacy_grafts(graft_path)
        _validate_default_index_flags(
            _git_output("ls-files", "-v", "-z"), source="assume-unchanged"
        )
        _validate_default_index_flags(
            _git_output("ls-files", "-f", "-z"), source="fsmonitor"
        )
        if (
            _git_output("merge-base", PROJECT_BASE_COMMIT, "HEAD")
            != PROJECT_BASE_COMMIT
        ):
            raise MPSRuntimeError("MPS branch is not descended from the exact base")
        if _git_output("ls-files", "--", PRESERVED_T4_INDEX_PATHSPEC):
            raise MPSRuntimeError("preserved T4 artifacts are unexpectedly tracked")
        _validate_preserved_t4_index_aliases(_git_output("ls-files", "-z"))
    except subprocess.CalledProcessError as error:
        raise MPSRuntimeError("protected Git state could not be verified") from error


def _validate_source_checkout() -> tuple[str, bool]:
    """Require immutable clean source while excluding declared result roots."""

    _validate_protected_git_state()
    _validate_preserved_t4()
    commit = _git_output("rev-parse", "HEAD")
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise MPSRuntimeError("project HEAD is not an immutable commit")
    status = _git_output("status", "--porcelain", "--untracked-files=all")
    allowed_untracked = (
        "?? results/stage1a_t4_fp16/",
        "?? results/stage1a_mps_fp16/",
    )
    disallowed = [
        line
        for line in status.splitlines()
        if line and not line.startswith(allowed_untracked)
    ]
    if disallowed:
        raise MPSRuntimeError("project source checkout is not clean")
    return commit, False


def _revalidate_publication_state(expected_commit: str) -> None:
    """Recheck immutable provenance immediately around result publication."""

    observed_commit, _dirty = _validate_source_checkout()
    if observed_commit != expected_commit:
        raise MPSRuntimeError("project HEAD changed during MPS execution")


def _load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as error:
        raise MPSRuntimeError("PyYAML is required for the MPS runner") from error
    if path.is_symlink() or not path.is_file():
        raise MPSRuntimeError("MPS configuration file is missing or unsafe")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MPSRuntimeError("MPS configuration must be a mapping")
    validate_mps_configuration(value)
    return value


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 5 * 1024 * 1024:
        raise MPSRuntimeError(f"unsafe or missing JSON evidence: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MPSRuntimeError(f"invalid JSON evidence: {path.name}") from error
    if not isinstance(value, dict):
        raise MPSRuntimeError(f"JSON evidence is not an object: {path.name}")
    validate_json_value(value)
    assert_publication_safe(value)
    return value


def _ensure_directory(path: Path) -> None:
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        raise MPSRuntimeError(f"artifact directory is unsafe: {path.name}")
    path.mkdir(parents=True, exist_ok=True)


def _preflight(
    output: Path | None = None,
    *,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    """Run the canonical probe after deleting its exact prior output."""

    destination = PREFLIGHT_OUTPUT if output is None else output
    _ensure_directory(destination.parent)
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise MPSRuntimeError("canonical MPS probe output is unsafe")
    destination.unlink(missing_ok=True)
    command = (
        sys.executable,
        str(SCRIPT_DIRECTORY / "probe_stage1a_mps.py"),
        "--output",
        str(destination),
    )
    probe_environment = os.environ.copy()
    probe_environment.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    probe_environment.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
    completed = runner(
        command,
        cwd=REPOSITORY_ROOT,
        env=probe_environment,
        check=False,
    )
    if not destination.is_file() or destination.is_symlink():
        raise MPSRuntimeError("canonical MPS probe did not create fresh evidence")
    report = _load_json(destination)
    expected_code = 0 if report.get("status") == "passed" else 2
    if int(getattr(completed, "returncode", -1)) not in {expected_code, 1}:
        raise MPSRuntimeError("canonical MPS probe status and exit code disagree")
    if report.get("status") == "passed" and getattr(completed, "returncode", -1) != 0:
        raise MPSRuntimeError("canonical MPS probe reported an ambiguous pass")
    return report


def _memory_gate(torch_module: Any | None = None) -> dict[str, Any]:
    try:
        physical = int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        physical = None
    telemetry = sample_mps_memory(
        torch_module if torch_module is not None else SimpleNamespace(mps=None),
        now=time.time,
    )
    return evaluate_memory_feasibility(
        {
            "model_bytes": MODEL_METADATA_BYTES,
            "transcoder_bytes": TRANSCODER_METADATA_BYTES,
        },
        physical_memory_bytes=physical,
        pressure=telemetry.get("system_memory_pressure"),
        swap_used_bytes=telemetry.get("swap_used_bytes"),
        observed_loading=OBSERVED_RUNTIME_LOADING_STOP,
    )


def _new_attempt_directory(batch_size: int) -> Path:
    _ensure_directory(GENERATED_DIRECTORY)
    return Path(
        tempfile.mkdtemp(
            prefix=f"attempt-{batch_size}-", dir=str(GENERATED_DIRECTORY.resolve())
        )
    )


def _run_attempt(
    args: argparse.Namespace,
    batch_size: int,
    environment: dict[str, str],
) -> tuple[dict[str, Any], Path]:
    attempt_directory = _new_attempt_directory(batch_size)
    report_path = attempt_directory / "attempt_report.json"
    command = [
        sys.executable,
        str(WORKER),
        "--config",
        str(args.config.resolve()),
        "--batch-size",
        str(batch_size),
        "--attempt-report",
        str(report_path),
        "--attempt-directory",
        str(attempt_directory),
    ]
    if args.allow_download:
        command.append("--allow-download")
    if args.model_snapshot is not None and args.transcoder_snapshot is not None:
        command.extend(
            (
                "--model-snapshot",
                str(args.model_snapshot.resolve()),
                "--transcoder-snapshot",
                str(args.transcoder_snapshot.resolve()),
            )
        )
    completed = subprocess.run(
        command, cwd=REPOSITORY_ROOT, env=environment, check=False
    )
    if not report_path.is_file():
        raise MPSRuntimeError("isolated worker exited without a report")
    report = _load_json(report_path)
    if (
        report.get("attempt_id") != attempt_directory.name
        or report.get("batch_size") != batch_size
    ):
        raise MPSRuntimeError("isolated worker report identity is invalid")
    report["process_exit_code"] = int(completed.returncode)
    completed_outcome = report.get("outcome") == "completed"
    if completed_outcome != (completed.returncode == 0):
        raise MPSRuntimeError("isolated worker outcome and exit code disagree")
    if completed_outcome:
        for filename in RAW_PAYLOAD_NAMES:
            _load_json(attempt_directory / filename)
    return report, attempt_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--transcoder-snapshot", type=Path)
    parser.add_argument(
        "--hf-cache", type=Path, help="exact project-external Hugging Face cache"
    )
    return parser


def _artifact_provenance(project_commit: str) -> dict[str, Any]:
    return {
        "base_commit": PROJECT_BASE_COMMIT,
        "project_commit": project_commit,
        "backend": "transformerlens",
        "accelerator_backend": "mps",
        "device": "mps",
        "dtype": "float16",
        "execution_dtype": "float16",
        "reference_dtype": "bfloat16",
        "reproduction_class": MPS_REPRODUCTION_CLASS,
        "execution_class": MPS_COMPLETED_STATUS,
        "upstream_repository": OFFICIAL_UPSTREAM_REPOSITORY,
        "upstream_revision": UPSTREAM_REVISION,
        "model_identifier": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transcoder_identifier": TRANSCODER_ID,
        "transcoder_revision": TRANSCODER_REVISION,
        "architecture": "arm64",
        "hardware_family": "Apple M2 Max",
        "offload": "disk",
        "fallback_enabled": False,
        "fallback_used": False,
        "official_bf16_reproduction": False,
        "t4_fp16_reproduction": False,
    }


def _wrap_artifact(
    *,
    artifact_type: str,
    payload: Mapping[str, Any],
    provenance: Mapping[str, Any],
    run_id: str,
    status: str,
    deviations: Sequence[str] = (),
) -> dict[str, Any]:
    return make_artifact_envelope(
        artifact_type=artifact_type,
        run_id=run_id,
        status=status,
        provenance=provenance,
        payload=payload,
        deviations=deviations,
    )


def _preflight_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    environment = raw.get("environment")
    checks = raw.get("checks")
    if not isinstance(environment, dict) or not isinstance(checks, dict):
        raise MPSRuntimeError("canonical preflight structure is invalid")
    if raw.get("status") != "passed" or raw.get("probe_status") != "passed":
        raise MPSRuntimeError("canonical preflight did not pass")
    expected_environment = {
        "python": "3.11.13",
        "system": "Darwin",
        "architecture": "arm64",
        "torch_version": "2.6.0",
        "mps_built": True,
        "mps_available": True,
        "fallback_enabled": False,
        "fallback_env_value_present": False,
        "high_watermark_override": None,
    }
    selected_environment = {key: environment.get(key) for key in expected_environment}
    if selected_environment != expected_environment:
        raise MPSRuntimeError("canonical preflight environment is invalid")
    operations = raw.get("operations")
    tolerances = raw.get("tolerances")
    if not isinstance(operations, dict) or not isinstance(tolerances, dict):
        raise MPSRuntimeError("canonical preflight evidence is incomplete")
    selected_environment["fallback_used"] = False
    return {
        "probe_status": "passed",
        "environment": selected_environment,
        "checks": dict(checks),
        "operations": dict(operations),
        "tolerances": dict(tolerances),
        "large_assets_downloaded": raw.get("large_assets_downloaded"),
        "scientific_model_result": raw.get("scientific_model_result"),
    }


def _raw_payload(attempt_directory: Path, filename: str) -> dict[str, Any]:
    return _load_json(attempt_directory / filename)


def _feasibility_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    if (
        raw.get("status") != "feasible_with_explicit_execution_deviation"
        or raw.get("downloads_authorized") is not True
    ):
        raise MPSRuntimeError("memory feasibility gate did not pass")
    pressure = raw.get("pressure")
    swap = raw.get("swap_used_bytes")
    if (
        not isinstance(pressure, str)
        or isinstance(swap, bool)
        or not isinstance(swap, int)
    ):
        raise MPSRuntimeError("memory feasibility telemetry is incomplete")
    return {
        "status": "feasible_with_explicit_execution_deviation",
        "passed": True,
        "downloads_authorized": True,
        "physical_memory_bytes": raw.get("physical_memory_bytes"),
        "system_memory_pressure": pressure,
        "swap_used_bytes": swap,
        "snapshot_sizes": raw.get("snapshot_sizes"),
        "estimate_components": raw.get("estimate_components"),
        "conservative_budget_bytes": raw.get("conservative_budget_bytes"),
        "estimated_peak_bytes": raw.get("estimated_peak_bytes"),
        "safety_reserve_bytes": 6 * 1024**3,
        "scientific_parameters_changed": False,
    }


def _require_science_payload(payload: Mapping[str, Any], label: str) -> None:
    if payload.get("status") not in {None, "completed"}:
        raise MPSRuntimeError(f"accepted {label} payload is not completed")
    if payload.get("nonfinite_count") != 0:
        raise MPSRuntimeError(f"accepted {label} payload is not finite")
    timing = payload.get("timing")
    if not isinstance(timing, dict):
        raise MPSRuntimeError(f"accepted {label} payload lacks timing")
    for name in (
        "mps_current_allocated_peak_bytes",
        "mps_driver_allocated_peak_bytes",
        "mps_recommended_max_bytes",
        "process_rss_peak_bytes",
        "swap_used_peak_bytes",
    ):
        value = timing.get(name)
        minimum = (
            1
            if name
            in {
                "mps_driver_allocated_peak_bytes",
                "mps_recommended_max_bytes",
                "process_rss_peak_bytes",
            }
            else 0
        )
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise MPSRuntimeError(f"accepted {label} timing is incomplete")


def _public_history(history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    public: list[dict[str, Any]] = []
    for item in history:
        stage_peaks = item.get("stage_peaks")
        attempt_peaks = item.get("attempt_peaks")
        if not isinstance(stage_peaks, dict) or not isinstance(attempt_peaks, dict):
            raise MPSRuntimeError("attempt telemetry report is incomplete")
        public.append(
            {
                "batch_size": item.get("batch_size"),
                "outcome": item.get("outcome"),
                "category": item.get("category"),
                "failure_stage": item.get("failure_stage"),
                "fresh_process": item.get("fresh_process"),
                "cleanup_succeeded": item.get("cleanup_succeeded"),
                "process_exit_code": item.get("process_exit_code"),
                "sample_count": item.get("sample_count"),
                "oom_classifier_match": item.get("oom_classifier_match"),
                "exception_type": item.get("exception_type"),
                "diagnostic_redacted": item.get("diagnostic_redacted"),
                "stage_peaks": stage_peaks,
                "attempt_peaks": attempt_peaks,
            }
        )
    validate_json_value(public)
    assert_publication_safe(public)
    return public


def _write_candidate_bundle(
    *,
    staging_directory: Path,
    project_commit: str,
    preflight: Mapping[str, Any],
    feasibility: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    accepted_directory: Path,
    selected_batch: int,
) -> None:
    """Write a completed candidate only inside ignored staging."""

    _ensure_directory(staging_directory)
    _ensure_directory(staging_directory / "preflight")
    provenance = _artifact_provenance(project_commit)
    run_id = f"stage1a-mps-fp16-{project_commit[:12]}-b{selected_batch}"
    environment = _raw_payload(accepted_directory, "environment.json")
    asset = _raw_payload(accepted_directory, "asset.json")
    semantics = _raw_payload(accepted_directory, "semantics.json")
    intervention = _raw_payload(accepted_directory, "intervention.json")
    attribution = _raw_payload(accepted_directory, "attribution.json")
    raw_memory = _raw_payload(accepted_directory, "memory.json")
    model_smoke = _raw_payload(accepted_directory, "model_smoke.json")
    public_history = _public_history(history)
    memory = {
        "nonfinite_count": 0,
        "timing": raw_memory.get("timing"),
        "telemetry_method": raw_memory.get("telemetry_method"),
        "sampling_interval_seconds": raw_memory.get("sampling_interval_seconds"),
        "accepted_attempt_index": len(public_history) - 1,
        "accepted_batch_size": selected_batch,
        "attempts": public_history,
    }
    for payload, label in (
        (semantics, "semantics"),
        (intervention, "intervention"),
        (attribution, "attribution"),
        (memory, "memory"),
    ):
        _require_science_payload(payload, label)
    if model_smoke.get("post_load_passed") is not True:
        raise MPSRuntimeError("accepted model-only smoke did not pass")
    if feasibility.get("status") != "feasible_with_explicit_execution_deviation":
        raise MPSRuntimeError(
            "accepted feasibility class is not the MPS deviation class"
        )
    if feasibility.get("downloads_authorized") is not True:
        raise MPSRuntimeError("accepted feasibility report did not authorize execution")

    deviation = (EXPLICIT_CPU_SPARSE_DEVIATION,)
    artifacts: dict[str, dict[str, Any]] = {
        "preflight/preflight_summary.json": _wrap_artifact(
            artifact_type="preflight_summary",
            payload=_preflight_payload(preflight),
            provenance=provenance,
            run_id=run_id,
            status="observed",
            deviations=deviation,
        ),
        "feasibility_report.json": _wrap_artifact(
            artifact_type="feasibility_report",
            payload=_feasibility_payload(feasibility),
            provenance=provenance,
            run_id=run_id,
            status="resolved",
            deviations=deviation,
        ),
        "environment_manifest.json": _wrap_artifact(
            artifact_type="environment_manifest",
            payload=environment,
            provenance=provenance,
            run_id=run_id,
            status="observed",
            deviations=deviation,
        ),
        "asset_manifest.json": _wrap_artifact(
            artifact_type="asset_manifest",
            payload=asset,
            provenance=provenance,
            run_id=run_id,
            status="resolved",
            deviations=deviation,
        ),
        "semantics_summary.json": _wrap_artifact(
            artifact_type="semantics_summary",
            payload=semantics,
            provenance=provenance,
            run_id=run_id,
            status="completed",
            deviations=deviation,
        ),
        "intervention_summary.json": _wrap_artifact(
            artifact_type="intervention_summary",
            payload=intervention,
            provenance=provenance,
            run_id=run_id,
            status="completed",
            deviations=deviation,
        ),
        "attribution_summary.json": _wrap_artifact(
            artifact_type="attribution_summary",
            payload=attribution,
            provenance=provenance,
            run_id=run_id,
            status="completed",
            deviations=deviation,
        ),
        "memory_summary.json": _wrap_artifact(
            artifact_type="memory_summary",
            payload=memory,
            provenance=provenance,
            run_id=run_id,
            status="completed",
            deviations=deviation,
        ),
    }
    for relative, record in artifacts.items():
        write_json_atomic(staging_directory / relative, record)

    timings = {
        "semantics": semantics["timing"],
        "intervention": intervention["timing"],
        "attribution": attribution["timing"],
        "memory": memory["timing"],
    }
    run_manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": MPS_COMPLETED_STATUS,
        "reproduction_class": MPS_REPRODUCTION_CLASS,
        "claim_boundary": MPS_CLAIM_BOUNDARY,
        "project": {
            "base_commit": PROJECT_BASE_COMMIT,
            "execution_commit": project_commit,
            "source_clean_excluding_preserved_t4": True,
            "preserved_t4_untracked": True,
        },
        "upstream": {
            "identifier": OFFICIAL_UPSTREAM_REPOSITORY,
            "revision": UPSTREAM_REVISION,
        },
        "model": {"identifier": MODEL_ID, "revision": MODEL_REVISION},
        "transcoder": {
            "identifier": TRANSCODER_ID,
            "revision": TRANSCODER_REVISION,
        },
        "runtime": {
            "backend": "transformerlens",
            "accelerator_backend": "mps",
            "device": "mps",
            "architecture": "arm64",
            "hardware_family": "Apple M2 Max",
            "execution_dtype": "float16",
            "reference_dtype": "bfloat16",
            "execution_class": MPS_COMPLETED_STATUS,
            "offload": "disk",
            "fallback_enabled": False,
            "fallback_used": False,
            "official_bf16_reproduction": False,
            "t4_fp16_reproduction": False,
            "execution_deviations": [EXPLICIT_CPU_SPARSE_DEVIATION],
            "accepted_batch_size": selected_batch,
            "retry_occurred": len(public_history) > 1,
        },
        "timings": timings,
        "retry_history": public_history,
        "checks": {
            "preflight_passed": True,
            "feasibility_passed": True,
            "assets_verified": True,
            "model_only_forward_passed": True,
            "loaded_runtime_semantics_passed": True,
            "attribution_passed": True,
            "intervention_passed": True,
            "telemetry_passed": True,
            "no_hidden_fallback": True,
            "nonfinite_count": 0,
        },
        "artifacts": {
            "preflight": "preflight/preflight_summary.json",
            "feasibility": "feasibility_report.json",
            "environment": "environment_manifest.json",
            "assets": "asset_manifest.json",
            "attribution": "attribution_summary.json",
            "intervention": "intervention_summary.json",
            "semantics": "semantics_summary.json",
            "memory": "memory_summary.json",
            "checksums": CHECKSUM_NAME,
        },
        "readiness": {
            "stage1b_engineering_readiness": True,
            "stage1b_empirical_claim_readiness": False,
        },
    }
    validate_json_value(run_manifest)
    assert_publication_safe(run_manifest)
    write_json_atomic(staging_directory / RUN_MANIFEST_NAME, run_manifest)


def _copy_atomic(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise MPSRuntimeError("validated staging member is unsafe")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise MPSRuntimeError("canonical artifact destination is unsafe")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(source.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _promote_validated_bundle(staging_directory: Path) -> None:
    _ensure_directory(RESULT_DIRECTORY)
    allowed_root = set(CANONICAL_JSON_NAMES) | {CHECKSUM_NAME, "preflight"}
    unexpected = [
        path.name
        for path in RESULT_DIRECTORY.iterdir()
        if path.name not in allowed_root
    ]
    if unexpected:
        raise MPSRuntimeError("canonical result directory contains unexpected entries")
    preflight_directory = RESULT_DIRECTORY / "preflight"
    _ensure_directory(preflight_directory)
    unexpected_preflight = [
        path.name
        for path in preflight_directory.iterdir()
        if path.name != "preflight_summary.json"
    ]
    if unexpected_preflight:
        raise MPSRuntimeError(
            "canonical preflight directory contains unexpected entries"
        )
    for filename in CANONICAL_JSON_NAMES:
        if filename == RUN_MANIFEST_NAME:
            continue
        _copy_atomic(staging_directory / filename, RESULT_DIRECTORY / filename)
    _copy_atomic(
        staging_directory / "preflight/preflight_summary.json",
        RESULT_DIRECTORY / "preflight/preflight_summary.json",
    )
    _copy_atomic(staging_directory / CHECKSUM_NAME, RESULT_DIRECTORY / CHECKSUM_NAME)
    # Publish the already-strictly-validated completed manifest last.
    _copy_atomic(
        staging_directory / RUN_MANIFEST_NAME,
        RESULT_DIRECTORY / RUN_MANIFEST_NAME,
    )


def _mark_canonical_failed() -> None:
    manifest_path = RESULT_DIRECTORY / RUN_MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return
    manifest = _load_json(manifest_path)
    manifest["status"] = "failed"
    readiness = manifest.get("readiness")
    if isinstance(readiness, dict):
        readiness["stage1b_engineering_readiness"] = False
    checks = manifest.get("checks")
    if isinstance(checks, dict):
        checks["telemetry_passed"] = False
    write_json_atomic(manifest_path, manifest)
    from validate_mps_fp16_artifacts import write_mps_checksums

    write_mps_checksums(RESULT_DIRECTORY)


def _validate_stage_and_promote(
    staging_directory: Path, *, project_commit: str
) -> None:
    from validate_mps_fp16_artifacts import (
        validate_mps_artifact_directory,
        write_mps_checksums,
    )

    write_mps_checksums(staging_directory)
    # The completed label first exists only in ignored staging.  It crosses the
    # publication boundary after the strict validator accepts the exact bytes.
    validate_mps_artifact_directory(staging_directory, require_complete=True)
    _revalidate_publication_state(project_commit)
    _promote_validated_bundle(staging_directory)
    try:
        _revalidate_publication_state(project_commit)
        validate_mps_artifact_directory(RESULT_DIRECTORY, require_complete=True)
    except Exception:
        _mark_canonical_failed()
        raise


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.model_snapshot is None) != (args.transcoder_snapshot is None):
        raise MPSRuntimeError("model and transcoder snapshot overrides must be paired")
    for snapshot in (args.model_snapshot, args.transcoder_snapshot):
        if snapshot is None:
            continue
        candidate = snapshot.expanduser()
        if candidate.is_symlink() or not candidate.is_dir():
            raise MPSRuntimeError("snapshot override is missing or unsafe")
        if candidate.resolve().is_relative_to(REPOSITORY_ROOT):
            raise MPSRuntimeError("snapshot override must be project-external")
    args.config = args.config.resolve()
    _load_config(args.config)
    project_commit, _dirty = _validate_source_checkout()
    _ensure_directory(GENERATED_DIRECTORY)
    preflight_directory = Path(
        tempfile.mkdtemp(prefix="preflight-", dir=str(GENERATED_DIRECTORY))
    )
    preflight = _preflight(preflight_directory / "preflight_summary.json")
    feasibility = _memory_gate()
    write_json_atomic(
        preflight_directory / "feasibility_report.json",
        feasibility,
    )
    if preflight.get("status") != "passed":
        return 2
    if feasibility.get("status") != "feasible_with_explicit_execution_deviation":
        return 2

    cache = (
        args.hf_cache.expanduser().resolve()
        if args.hf_cache is not None
        else (REPOSITORY_ROOT.parent / "hf-cache-stage1a-mps").resolve()
    )
    if cache.is_relative_to(REPOSITORY_ROOT):
        raise MPSRuntimeError("Hugging Face cache must be project-external")
    hub_cache = _validate_external_cache_subdirectory(cache, "hub")
    xet_cache = _validate_external_cache_subdirectory(cache, "xet")
    _validate_external_cache_tree(cache)
    child_environment = os.environ.copy()
    child_environment.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    child_environment.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
    for name in tuple(child_environment):
        if name.startswith("HF_XET_"):
            child_environment.pop(name)
    child_environment["HF_HUB_CACHE"] = str(hub_cache)
    child_environment["HF_XET_CACHE"] = str(xet_cache)
    if args.allow_download:
        child_environment.pop("HF_HUB_OFFLINE", None)
        child_environment.pop("TRANSFORMERS_OFFLINE", None)
    else:
        child_environment["HF_HUB_OFFLINE"] = "1"
        child_environment["TRANSFORMERS_OFFLINE"] = "1"

    history: list[dict[str, Any]] = []
    accepted_directory: Path | None = None
    selected_batch: int | None = None
    for batch_size in MPS_BATCH_SEQUENCE:
        _validate_external_cache_tree(cache)
        report, attempt_directory = _run_attempt(args, batch_size, child_environment)
        history.append(report)
        if report.get("outcome") == "completed":
            selected_batch = batch_size
            accepted_directory = attempt_directory
            break
        retry_allowed = should_retry_mps_attempt(
            batch_size=batch_size,
            category=str(report.get("category", "")),
            failure_stage=str(report.get("failure_stage", "")),
        )
        if (
            not retry_allowed
            or report.get("oom_confirmed") is not True
            or report.get("retry_eligible") is not True
            or report.get("cleanup_succeeded") is not True
            or report.get("diagnostic_redacted") is not True
        ):
            break
    if selected_batch is None or accepted_directory is None:
        return 2

    _validate_external_cache_tree(cache)
    _revalidate_publication_state(project_commit)
    staging_directory = Path(
        tempfile.mkdtemp(prefix="candidate-bundle-", dir=str(GENERATED_DIRECTORY))
    )
    _write_candidate_bundle(
        staging_directory=staging_directory,
        project_commit=project_commit,
        preflight=preflight,
        feasibility=feasibility,
        history=history,
        accepted_directory=accepted_directory,
        selected_batch=selected_batch,
    )
    _validate_stage_and_promote(
        staging_directory,
        project_commit=project_commit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
