#!/usr/bin/env python3
"""Validate and safely package Stage 1A T4/FP16 small artifacts."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import stat
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cfsus.reproduction.artifacts import (  # noqa: E402
    ArtifactValidationError,
    assert_publication_safe,
    build_checksum_manifest,
    sha256_file,
    validate_artifact_envelope,
    verify_checksum_manifest,
    write_checksum_manifest_atomic,
)
from cfsus.reproduction.config import (  # noqa: E402
    OFFICIAL_MODEL_ID,
    OFFICIAL_MODEL_REVISION,
    OFFICIAL_TRANSCODER_ID,
    OFFICIAL_TRANSCODER_REVISION,
    OFFICIAL_UPSTREAM_REVISION,
)
from cfsus.reproduction.t4_fp16 import (  # noqa: E402
    EXECUTION_DTYPE,
    MAX_BUNDLE_MEMBER_BYTES,
    MAX_BUNDLE_TOTAL_BYTES,
    PROJECT_BASE_COMMIT,
    REFERENCE_DTYPE,
    REPRODUCTION_CLASS,
    T4_RESULT_DIRECTORY,
    T4_SMALL_FILES,
    T4_SUMMARY_FILES,
    T4RunStatus,
    validate_t4_run_manifest,
)

RUN_MANIFEST_NAME = "stage1a_t4_fp16_run_manifest.json"
CHECKSUM_NAME = "checksums.sha256"
BUNDLE_PREFIX = f"{T4_RESULT_DIRECTORY}/"
SCIENCE_TYPES = {
    "attribution_summary.json": "attribution_summary",
    "intervention_summary.json": "intervention_summary",
    "semantics_summary.json": "semantics_summary",
}
T4_ENVIRONMENT_RUN_ID = "stage1a-t4-fp16-environment"
T4_DTYPE_DEVIATION = (
    "Native-BF16 reference dtype was adapted to float16 for T4 execution."
)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ArtifactValidationError(f"artifact must be a regular file: {path.name}")
    if path.stat().st_size > MAX_BUNDLE_MEMBER_BYTES:
        raise ArtifactValidationError(f"artifact exceeds size limit: {path.name}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"invalid JSON artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"JSON artifact must be an object: {path.name}")
    assert_publication_safe(value)
    return value


def _runtime_provenance(record: dict[str, Any], label: str) -> None:
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        raise ArtifactValidationError(f"{label} provenance must be an object")
    expected = {
        "upstream_revision": OFFICIAL_UPSTREAM_REVISION,
        "model_identifier": OFFICIAL_MODEL_ID,
        "model_revision": OFFICIAL_MODEL_REVISION,
        "transcoder_identifier": OFFICIAL_TRANSCODER_ID,
        "transcoder_revision": OFFICIAL_TRANSCODER_REVISION,
        "backend": "transformerlens",
        "device": "cuda",
        "dtype": EXECUTION_DTYPE,
        "reproduction_class": REPRODUCTION_CLASS,
        "reference_dtype": REFERENCE_DTYPE,
        "execution_dtype": EXECUTION_DTYPE,
        "reference_status": "pending",
        "project_dirty": False,
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ArtifactValidationError(f"{label} T4 runtime provenance is invalid")
    project_commit = provenance.get("project_commit")
    if (
        not isinstance(project_commit, str)
        or len(project_commit) != 40
        or any(character not in "0123456789abcdef" for character in project_commit)
    ):
        raise ArtifactValidationError(f"{label} project commit is invalid")
    gpu = provenance.get("gpu")
    if (
        not isinstance(gpu, dict)
        or "T4" not in str(gpu.get("name"))
        or gpu.get("compute_capability") != [7, 5]
        or not isinstance(gpu.get("bf16_supported"), bool)
    ):
        raise ArtifactValidationError(f"{label} T4 GPU provenance is invalid")
    asset_integrity = provenance.get("asset_integrity")
    if (
        not isinstance(asset_integrity, dict)
        or asset_integrity.get("verification") != "exact_file_content_hashes_matched"
    ):
        raise ArtifactValidationError(f"{label} asset integrity is invalid")
    parameter_check = provenance.get("parameter_finiteness_sample")
    threshold_check = provenance.get("threshold_finiteness")
    if (
        not isinstance(parameter_check, dict)
        or parameter_check.get("passed") is not True
    ):
        raise ArtifactValidationError(f"{label} model sample finiteness did not pass")
    if (
        not isinstance(threshold_check, dict)
        or threshold_check.get("passed") is not True
    ):
        raise ArtifactValidationError(f"{label} threshold finiteness did not pass")


def _validate_t4_environment(path: Path) -> dict[str, Any]:
    record = _load_json(path)
    validate_artifact_envelope(record, expected_type="environment_manifest")
    if (
        record.get("run_id") != T4_ENVIRONMENT_RUN_ID
        or record.get("status") != "observed"
        or record.get("deviations") != [T4_DTYPE_DEVIATION]
        or record.get("warnings")
        != ["colab_input_is_planned_not_an_observed_transitive_lock"]
    ):
        raise ArtifactValidationError("T4 environment identity is invalid")

    provenance = record.get("provenance")
    expected_provenance = {
        "device": "cuda",
        "base_commit": PROJECT_BASE_COMMIT,
        "environment_lock": "requirements-colab-py311-cu124-planned.txt",
        "execution_dtype": EXECUTION_DTYPE,
        "reference_dtype": REFERENCE_DTYPE,
        "reference_status": "pending",
        "reproduction_class": REPRODUCTION_CLASS,
        "upstream_commit": OFFICIAL_UPSTREAM_REVISION,
        "code_revision_status": "clean_commit",
    }
    if not isinstance(provenance, dict) or any(
        provenance.get(key) != value for key, value in expected_provenance.items()
    ):
        raise ArtifactValidationError("T4 environment provenance is invalid")
    code_commit = provenance.get("code_commit")
    if (
        not isinstance(code_commit, str)
        or len(code_commit) != 40
        or any(character not in "0123456789abcdef" for character in code_commit)
    ):
        raise ArtifactValidationError("T4 environment code commit is invalid")

    payload = record.get("payload")
    if not isinstance(payload, dict):
        raise ArtifactValidationError("T4 environment payload is invalid")
    execution_policy = payload.get("execution_policy")
    accelerators = payload.get("accelerators")
    if not isinstance(execution_policy, dict) or not isinstance(accelerators, dict):
        raise ArtifactValidationError("T4 environment observations are missing")
    platform = payload.get("platform")
    python = payload.get("python")
    packages = payload.get("packages")
    versions = packages.get("versions") if isinstance(packages, dict) else None
    if (
        payload.get("offline_only") is not True
        or not isinstance(platform, dict)
        or platform.get("system") != "Linux"
        or platform.get("machine") != "x86_64"
        or not isinstance(python, dict)
        or not str(python.get("version", "")).startswith("3.11.")
        or not isinstance(versions, dict)
        or versions.get("torch") != "2.6.0+cu124"
    ):
        raise ArtifactValidationError("T4 environment platform profile is invalid")
    current = execution_policy.get("current_runtime")
    planned = execution_policy.get("planned_colab")
    reference = execution_policy.get("native_reference")
    expected_current = {
        "execution_scope": "full_t4_fp16_reproduction",
        "fallback_enabled": False,
        "fallback_used": False,
        "full_model_execution_allowed": True,
        "offload": "disk",
        "requested_dtype": EXECUTION_DTYPE,
        "selected_device": "cuda",
    }
    expected_planned = {
        "device": "cuda",
        "dtype": EXECUTION_DTYPE,
        "observed": True,
        "offload": "disk",
    }
    expected_reference = {
        "device": "cuda",
        "dtype": REFERENCE_DTYPE,
        "status": "pending",
    }
    if (
        current != expected_current
        or planned != expected_planned
        or reference != expected_reference
    ):
        raise ArtifactValidationError("T4 environment execution policy is invalid")

    cuda = accelerators.get("cuda")
    dtype_support = accelerators.get("dtype_support")
    observed_device = cuda.get("observed_device") if isinstance(cuda, dict) else None
    float16_probe = (
        dtype_support.get("cuda_float16") if isinstance(dtype_support, dict) else None
    )
    if (
        not isinstance(cuda, dict)
        or cuda.get("available") is not True
        or cuda.get("compiled_version") != "12.4"
        or not isinstance(cuda.get("device_count"), int)
        or cuda["device_count"] < 1
        or not isinstance(observed_device, dict)
        or "T4" not in str(observed_device.get("name"))
        or observed_device.get("compute_capability") != [7, 5]
        or not isinstance(observed_device.get("total_memory_bytes"), int)
        or observed_device["total_memory_bytes"] <= 0
        or not isinstance(float16_probe, dict)
        or float16_probe
        != {
            "attempted": True,
            "error": None,
            "error_type": None,
            "success": True,
        }
    ):
        raise ArtifactValidationError("T4 environment CUDA/FP16 probe is invalid")
    return record


def _validate_science_summary(path: Path, expected_type: str) -> dict[str, Any]:
    record = _load_json(path)
    validate_artifact_envelope(record, expected_type=expected_type)
    if record.get("status") != "completed":
        raise ArtifactValidationError(f"{path.name} did not complete")
    _runtime_provenance(record, path.name)
    payload = record.get("payload")
    if not isinstance(payload, dict) or payload.get("nonfinite_count") != 0:
        raise ArtifactValidationError(f"{path.name} has a nonzero nonfinite count")
    boundary = payload.get("claim_boundary")
    if not isinstance(boundary, str) or "native-BF16 reference" not in boundary:
        raise ArtifactValidationError(f"{path.name} omits the BF16 claim boundary")
    if expected_type == "attribution_summary":
        graph = payload.get("graph")
        raw_validation = payload.get("raw_validation")
        if (
            not isinstance(graph, dict)
            or graph.get("finite") is not True
            or not isinstance(graph.get("selected_feature_count"), int)
            or graph["selected_feature_count"] <= 0
            or not isinstance(raw_validation, dict)
            or raw_validation.get("passed") is not True
        ):
            raise ArtifactValidationError("T4 attribution graph validation failed")
    elif expected_type == "intervention_summary":
        comparison = payload.get("baseline_noop_comparison")
        determinism = payload.get("determinism")
        desired = payload.get("desired_values")
        if (
            not isinstance(comparison, dict)
            or comparison.get("within_tolerance") is not True
            or not isinstance(determinism, dict)
            or determinism.get("within_tolerance") is not True
            or not isinstance(desired, list)
            or [item.get("alpha") for item in desired if isinstance(item, dict)]
            != [0.0, 0.5, 1.0]
        ):
            raise ArtifactValidationError("T4 intervention checks failed")
    elif expected_type == "semantics_summary":
        gate = payload.get("gate_check")
        value_check = payload.get("intervention_value_check")
        if (
            not isinstance(gate, dict)
            or gate.get("strict_greater_than") is not True
            or gate.get("equality_inactive") is not True
            or not isinstance(value_check, dict)
            or len(value_check.get("desired_values", [])) != 3
        ):
            raise ArtifactValidationError("T4 loaded-runtime semantic checks failed")
    return record


def checksum_targets(directory: Path) -> tuple[Path, ...]:
    """Return the five allowlisted JSON summaries covered by checksums."""

    return tuple(directory / name for name in sorted(T4_SUMMARY_FILES))


def write_t4_checksums(directory: Path) -> str:
    """Write deterministic filename-relative checksums for the five summaries."""

    for path in checksum_targets(directory):
        if path.is_symlink() or not path.is_file():
            raise ArtifactValidationError(f"missing checksum target: {path.name}")
    return write_checksum_manifest_atomic(
        directory / CHECKSUM_NAME,
        checksum_targets(directory),
        root=directory,
    )


def _validate_checksums(directory: Path) -> None:
    manifest = directory / CHECKSUM_NAME
    verified = verify_checksum_manifest(manifest, root=directory)
    expected_text = build_checksum_manifest(checksum_targets(directory), root=directory)
    if manifest.read_text(encoding="utf-8") != expected_text:
        raise ArtifactValidationError("T4 checksums are incomplete or unsorted")
    if set(verified) != T4_SUMMARY_FILES:
        raise ArtifactValidationError("T4 checksums do not cover every summary")


def validate_t4_artifact_directory(
    directory: Path,
    *,
    require_run_manifest: bool = True,
    require_complete: bool = True,
) -> tuple[str, ...]:
    """Validate the isolated T4 result directory and all cross-file digests."""

    if directory.is_symlink() or not directory.is_dir():
        raise ArtifactValidationError("T4 artifact directory is missing or unsafe")
    entries = tuple(sorted(directory.iterdir(), key=lambda path: path.name))
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ArtifactValidationError(
            "T4 artifact directory contains a non-regular entry"
        )
    names = {path.name for path in entries}
    if not names.issubset(T4_SMALL_FILES):
        raise ArtifactValidationError("T4 artifact directory contains an unlisted file")
    if require_complete and names != T4_SMALL_FILES:
        raise ArtifactValidationError(
            "T4 completed artifact set is not exactly seven files"
        )

    environment_record: dict[str, Any] | None = None
    if "environment_manifest.json" in names:
        environment_record = _validate_t4_environment(
            directory / "environment_manifest.json"
        )
    if "asset_manifest.json" in names:
        asset = _load_json(directory / "asset_manifest.json")
        validate_artifact_envelope(asset, expected_type="asset_manifest")
        if asset.get("status") not in {"resolved", "completed"}:
            raise ArtifactValidationError("T4 immutable asset resolution did not pass")
    science_records: dict[str, dict[str, Any]] = {}
    for filename, artifact_type in SCIENCE_TYPES.items():
        if filename in names:
            science_records[filename] = _validate_science_summary(
                directory / filename, artifact_type
            )
        elif require_complete:
            raise ArtifactValidationError(f"missing T4 science summary: {filename}")
    if CHECKSUM_NAME in names:
        _validate_checksums(directory)
    elif require_complete:
        raise ArtifactValidationError("missing T4 checksum manifest")

    if RUN_MANIFEST_NAME in names:
        manifest = _load_json(directory / RUN_MANIFEST_NAME)
        validate_t4_run_manifest(manifest)
        status = T4RunStatus(manifest["status"])
        if require_complete and status is not T4RunStatus.COMPLETED:
            raise ArtifactValidationError(
                "complete artifact set has non-completed status"
            )
        records = manifest.get("artifacts")
        if not isinstance(records, dict):
            raise ArtifactValidationError("run manifest artifact records are invalid")
        for name, metadata in records.items():
            path = directory / name
            if not isinstance(metadata, dict) or not path.is_file():
                raise ArtifactValidationError("recorded T4 artifact is missing")
            if metadata.get("size_bytes") != path.stat().st_size:
                raise ArtifactValidationError("recorded T4 artifact size mismatch")
            if metadata.get("sha256") != sha256_file(path):
                raise ArtifactValidationError("recorded T4 artifact digest mismatch")
        project = manifest.get("project")
        attribution = manifest.get("attribution")
        if not isinstance(project, dict) or not isinstance(attribution, dict):
            raise ArtifactValidationError("run manifest cross-file metadata is invalid")
        if environment_record is not None:
            environment_provenance = environment_record.get("provenance")
            if not isinstance(
                environment_provenance, dict
            ) or environment_provenance.get("code_commit") != project.get(
                "execution_commit"
            ):
                raise ArtifactValidationError("environment project commit mismatch")
        for record in science_records.values():
            provenance = record.get("provenance")
            if not isinstance(provenance, dict) or provenance.get(
                "project_commit"
            ) != project.get("execution_commit"):
                raise ArtifactValidationError("summary project commit mismatch")
        attribution_record = science_records.get("attribution_summary.json")
        if attribution_record is not None:
            payload = attribution_record.get("payload")
            parameters = (
                payload.get("parameters") if isinstance(payload, dict) else None
            )
            if not isinstance(parameters, dict) or parameters.get(
                "batch_size"
            ) != attribution.get("selected_batch_size"):
                raise ArtifactValidationError("selected attribution batch mismatch")
    elif require_run_manifest:
        raise ArtifactValidationError("missing T4 run manifest")
    return tuple(path.name for path in entries)


def _bundle_member_name(filename: str) -> str:
    return f"{BUNDLE_PREFIX}{filename}"


def build_t4_small_bundle(directory: Path, destination: Path) -> str:
    """Build a deterministic seven-file ZIP after strict artifact validation."""

    validate_t4_artifact_directory(directory)
    if destination.is_symlink() or destination.exists():
        raise ArtifactValidationError("refusing to overwrite the bundle destination")
    total = sum((directory / name).stat().st_size for name in T4_SMALL_FILES)
    if total > MAX_BUNDLE_TOTAL_BYTES:
        raise ArtifactValidationError("T4 small artifacts exceed total size limit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(
            destination, "x", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for name in sorted(T4_SMALL_FILES):
                path = directory / name
                info = zipfile.ZipInfo(
                    _bundle_member_name(name), date_time=(1980, 1, 1, 0, 0, 0)
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
        validate_t4_small_bundle(destination)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(destination)
        raise
    return sha256_file(destination)


def validate_t4_small_bundle(path: Path) -> tuple[str, ...]:
    """Reject traversal, symlinks, duplicates, bombs, and non-allowlisted members."""

    if path.is_symlink() or not path.is_file():
        raise ArtifactValidationError("T4 bundle must be a regular file")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ArtifactValidationError("T4 bundle has duplicate members")
            expected = {_bundle_member_name(name) for name in T4_SMALL_FILES}
            if set(names) != expected:
                raise ArtifactValidationError("T4 bundle member allowlist is not exact")
            total = 0
            for info in infos:
                pure = PurePosixPath(info.filename)
                mode = info.external_attr >> 16
                if (
                    pure.is_absolute()
                    or ".." in pure.parts
                    or "\\" in info.filename
                    or info.is_dir()
                    or stat.S_ISLNK(mode)
                ):
                    raise ArtifactValidationError("T4 bundle contains an unsafe member")
                if info.file_size > MAX_BUNDLE_MEMBER_BYTES:
                    raise ArtifactValidationError("T4 bundle member exceeds size limit")
                total += info.file_size
                if total > MAX_BUNDLE_TOTAL_BYTES:
                    raise ArtifactValidationError("T4 bundle exceeds total size limit")
                data = archive.read(info)
                if len(data) != info.file_size:
                    raise ArtifactValidationError(
                        "T4 bundle member size is inconsistent"
                    )
                if info.filename.endswith(".json"):
                    try:
                        value = json.loads(data, object_pairs_hook=_unique_object)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ArtifactValidationError(
                            "T4 bundle contains invalid JSON"
                        ) from exc
                    assert_publication_safe(value)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ArtifactValidationError("T4 bundle is not a valid ZIP") from exc
    return tuple(sorted(names))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=REPOSITORY_ROOT / T4_RESULT_DIRECTORY,
    )
    parser.add_argument("--write-checksums", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Accept an absent result directory before the first empirical run.",
    )
    parser.add_argument("--bundle", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report: dict[str, Any] = {"schema_version": 1, "valid": False, "errors": []}
    try:
        directory = args.artifact_dir.resolve()
        if args.allow_empty and not directory.exists():
            report["files"] = []
            report["prepared_not_executed"] = True
            report["valid"] = True
            sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
            return 0
        if args.write_checksums:
            write_t4_checksums(directory)
        files = validate_t4_artifact_directory(
            directory,
            require_complete=not args.allow_incomplete,
        )
        report["files"] = list(files)
        if args.bundle is not None:
            digest = build_t4_small_bundle(directory, args.bundle.resolve())
            report["bundle_sha256"] = digest
        report["valid"] = True
    except (OSError, ArtifactValidationError) as exc:
        report["errors"] = [str(exc).replace(str(REPOSITORY_ROOT), ".")]
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
