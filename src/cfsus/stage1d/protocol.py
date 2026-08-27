"""Frozen-file identity helpers for the Stage 1D protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cfsus.stage1c_v3.serialization import write_json_new

PROTOCOL_FILES = (
    "configs/stage1d_multiprompt_gate_benchmark.yaml",
    "configs/stage1d_multiprompt_gate_benchmark_artifact_schema.json",
    "src/cfsus/stage1c_v3/execution_journal.py",
    "src/cfsus/stage1c_v3/intervention.py",
    "src/cfsus/stage1c_v3/intervention_runtime.py",
    "src/cfsus/stage1c_v3/prediction.py",
    "src/cfsus/stage1c_v3/runtime.py",
    "src/cfsus/stage1c_v3/serialization.py",
    "src/cfsus/stage1c_v3/worker_result.py",
    "src/cfsus/stage1d/__init__.py",
    "src/cfsus/stage1d/artifacts.py",
    "src/cfsus/stage1d/benchmark.py",
    "src/cfsus/stage1d/config.py",
    "src/cfsus/stage1d/execution.py",
    "src/cfsus/stage1d/metrics.py",
    "src/cfsus/stage1d/prediction_runtime.py",
    "src/cfsus/stage1d/preflight.py",
    "src/cfsus/stage1d/protocol.py",
    "src/cfsus/stage1d/rehearsal.py",
    "src/cfsus/stage1d/validation.py",
    "scripts/stage1d.py",
)


def sha256_file(path: Path) -> str:
    """Hash one safe single-link regular file."""

    info = path.lstat()
    if path.is_symlink() or not path.is_file() or info.st_nlink != 1:
        raise RuntimeError(f"protocol file is unsafe: {path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol_hashes(repository: Path) -> dict[str, str]:
    """Return canonical hashes for every frozen implementation file."""

    return {relative: sha256_file(repository / relative) for relative in PROTOCOL_FILES}


def protocol_map_digest(value: dict[str, str]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_protocol_manifest(
    repository: Path, *, protocol_commit: str
) -> dict[str, Any]:
    """Build the tracked immutable protocol record."""

    hashes = protocol_hashes(repository)
    return {
        "schema_version": 1,
        "artifact_type": "stage1d_protocol_manifest",
        "status": "protocol_frozen_before_baseline",
        "experiment_class": "stage1d_multiprompt_gate_benchmark",
        "base_commit": "d4fdcc2c2f0040654af17e21f396f1d26072aa0e",
        "branch": "stage-1d-multiprompt-gate-benchmark",
        "protocol_commit": protocol_commit,
        "protocol_file_sha256": hashes,
        "protocol_map_sha256": protocol_map_digest(hashes),
        "evaluation_baseline_model_calls_before_freeze": 0,
        "evaluation_source_suppression_calls_before_freeze": 0,
        "scientific_attempt_started": False,
    }


def publish_protocol_manifest(
    repository: Path, output: Path, *, protocol_commit: str
) -> dict[str, Any]:
    value = build_protocol_manifest(repository, protocol_commit=protocol_commit)
    write_json_new(output, value)
    return value


__all__ = [
    "PROTOCOL_FILES",
    "build_protocol_manifest",
    "protocol_hashes",
    "protocol_map_digest",
    "publish_protocol_manifest",
    "sha256_file",
]
