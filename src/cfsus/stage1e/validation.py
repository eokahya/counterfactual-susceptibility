"""Publication and standalone validation for Stage 1E offline artifacts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from cfsus.stage1c_v3.serialization import (
    detach_json,
    read_json_strict,
    write_json_new,
)
from cfsus.stage1d.validation import validate_bundle as validate_stage1d_bundle
from cfsus.stage1e.offline import (
    PROJECT_DECISION,
    TERMINAL_STATUS,
    build_run_manifest,
    compute_offline_analysis,
)

JSON_ARTIFACTS = ("offline_analysis.json", "run_manifest.json")


def _write_checksums(output: Path) -> None:
    lines = [
        f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}"
        for name in JSON_ARTIFACTS
    ]
    path = output / "checksums.sha256"
    if path.exists():
        raise ValueError("checksum sidecar already exists")
    encoded = ("\n".join(lines) + "\n").encode("ascii")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        if os.write(descriptor, encoded) != len(encoded):
            raise ValueError("short checksum write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_offline_bundle(output: Path, analysis: dict[str, Any]) -> None:
    """Publish the offline analysis, terminal manifest, and checksums once."""

    if (
        analysis.get("terminal_status") != TERMINAL_STATUS
        or analysis.get("project_decision") != PROJECT_DECISION
        or analysis.get("selected_estimator") is not None
    ):
        raise ValueError("terminal offline bundle requires the frozen negative outcome")
    output.mkdir(parents=True, exist_ok=False)
    if output.is_symlink():
        raise ValueError("offline output directory is a symlink")
    analysis_path = output / "offline_analysis.json"
    analysis_sha = write_json_new(analysis_path, analysis)
    write_json_new(output / "run_manifest.json", build_run_manifest(analysis_sha))
    _write_checksums(output)


def _load_output(output: Path) -> dict[str, dict[str, Any]]:
    expected = set(JSON_ARTIFACTS) | {"checksums.sha256"}
    if not output.is_dir() or output.is_symlink():
        raise ValueError("offline artifact directory is missing or unsafe")
    children = list(output.iterdir())
    observed = {path.name for path in children if path.is_file()}
    if observed != expected or any(path.is_symlink() for path in children):
        raise ValueError("offline artifact allowlist differs")
    records: dict[str, dict[str, Any]] = {}
    for name in JSON_ARTIFACTS:
        path = output / name
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("offline artifact exceeds the per-file cap")
        value = read_json_strict(path)
        if not isinstance(value, dict):
            raise ValueError("offline artifact must be an object")
        detach_json(value)
        records[name] = value
    lines = (output / "checksums.sha256").read_text(encoding="ascii").splitlines()
    expected_lines = [
        f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}"
        for name in JSON_ARTIFACTS
    ]
    if lines != expected_lines:
        raise ValueError("offline checksums differ")
    if sum(path.stat().st_size for path in children) > 5 * 1024 * 1024:
        raise ValueError("offline artifact bundle exceeds the total cap")
    return records


def validate_offline_bundle(
    repository: Path, stage1d_directory: Path, output: Path
) -> dict[str, Any]:
    """Recompute Phase A from committed Stage 1D inputs in a standalone process."""

    records = _load_output(output)
    stage1d_validation = validate_stage1d_bundle(repository, stage1d_directory)
    expected_analysis = compute_offline_analysis(
        stage1d_directory, stage1d_validation=stage1d_validation
    )
    observed_analysis = records["offline_analysis.json"]
    if observed_analysis != expected_analysis:
        raise ValueError("offline analysis differs from standalone recomputation")
    analysis_sha = hashlib.sha256(
        (output / "offline_analysis.json").read_bytes()
    ).hexdigest()
    expected_run = build_run_manifest(analysis_sha)
    if records["run_manifest.json"] != expected_run:
        raise ValueError("offline terminal manifest differs")
    metrics = expected_analysis["estimator_metrics"]
    gates = expected_analysis["offline_gates"]
    if (
        expected_analysis.get("terminal_status") != TERMINAL_STATUS
        or expected_analysis.get("project_decision") != PROJECT_DECISION
        or expected_analysis.get("selected_estimator") is not None
        or gates["E1"]["passed"] is not False
        or gates["E2"]["passed"] is not False
    ):
        raise ValueError("offline terminal decision differs")
    return {
        "status": "passed",
        "terminal_status": TERMINAL_STATUS,
        "project_decision": PROJECT_DECISION,
        "stage1d_development_pair_count": len(expected_analysis["trajectories"]),
        "critical_reference_pair_count": metrics["E0"]["reference_pair_count"],
        "E1_eligible_pair_count": metrics["E1"]["eligible_pair_count"],
        "E2_eligible_pair_count": metrics["E2"]["eligible_pair_count"],
        "phase_a_model_calls": 0,
        "phase_a_intervention_calls": 0,
        "phase_b_model_calls": 0,
        "phase_b_intervention_calls": 0,
        "checksums_verified": len(JSON_ARTIFACTS),
    }


__all__ = [
    "JSON_ARTIFACTS",
    "publish_offline_bundle",
    "validate_offline_bundle",
]
