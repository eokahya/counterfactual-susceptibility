#!/usr/bin/env python3
"""Publish the detached Stage 1C-v2 prediction manifest.

The prediction assembler has no model imports and never selects or scores a
pair.  It accepts only a strict worker record, re-runs the independent
stdlib validator, and publishes one new canonical JSON file.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_stage1c_v2_artifacts as validator  # noqa: E402

from cfsus.stage1c_v2.serialization import (  # noqa: E402
    SerializationError,
    read_json_strict,
    write_json_new,
)

BASE_COMMIT = validator.BASE_COMMIT
BRANCH = validator.BRANCH
FORBIDDEN_RECURSIVE_KEYS = frozenset(
    {
        "observed",
        "observed_outcome",
        "observed_crossing",
        "realized_suppression",
        "actual_bf16_value_passed",
        "target_activation_after_intervention",
        "intervention_sweeps",
        "scientific_outcome",
    }
)


def strict_load(path: Path) -> dict[str, Any]:
    """Read one bounded, canonical, single-link JSON object."""

    value = read_json_strict(path)
    if not isinstance(value, dict):  # pragma: no cover - reader is strict
        raise ValueError("worker record must be an object")
    return cast(dict[str, Any], value)


def _walk_prediction(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.casefold() in FORBIDDEN_RECURSIVE_KEYS:
                raise ValueError(f"intervention field in prediction: {path}.{key}")
            _walk_prediction(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _walk_prediction(item, f"{path}[{index}]")


def _verify_protocol_hashes(manifest: dict[str, Any]) -> None:
    hashes = manifest.get("protocol_file_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("prediction protocol hash map is missing")
    for relative, expected in hashes.items():
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or Path(relative).is_absolute()
            or "\\" in relative
            or ".." in Path(relative).parts
            or validator.SHA256.fullmatch(expected) is None
        ):
            raise ValueError("unsafe prediction protocol hash entry")
        path = ROOT / relative
        try:
            info = path.lstat()
        except OSError as error:
            raise ValueError(
                f"prediction protocol file is missing: {relative}"
            ) from error
        if not path.is_file() or path.is_symlink() or info.st_nlink != 1:
            raise ValueError(f"prediction protocol file is unsafe: {relative}")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"prediction protocol file changed: {relative}")


def assemble_prediction(
    worker: dict[str, Any],
    *,
    output: Path | None = None,
    verify_protocol_hashes: bool = False,
) -> dict[str, Any]:
    """Validate and optionally publish a worker's detached prediction object."""

    if worker.get("schema_version") != 2:
        raise ValueError("prediction worker is not v2")
    if worker.get("artifact_type") != "stage1c_v2_prediction_worker":
        raise ValueError("prediction worker artifact type differs")
    if worker.get("status") != "passed":
        raise ValueError("prediction worker did not pass")
    prediction = worker.get("prediction_manifest")
    if not isinstance(prediction, dict):
        raise ValueError("worker has no prediction manifest")
    _walk_prediction(prediction)
    if (
        prediction.get("base_commit") != BASE_COMMIT
        or prediction.get("branch") != BRANCH
    ):
        raise ValueError("prediction manifest Git identity differs")
    try:
        validator.scan_value(prediction)
        validator.scan_prediction(prediction)
    except validator.ValidationError as error:
        raise ValueError(
            f"prediction manifest failed standalone validation: {error}"
        ) from error
    if verify_protocol_hashes:
        _verify_protocol_hashes(prediction)
    if output is not None:
        write_json_new(output, prediction)
    return prediction


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        _git("rev-parse", "HEAD") != BASE_COMMIT
        or _git("branch", "--show-current") != BRANCH
    ):
        raise RuntimeError("prediction assembly requires the exact v2 base branch")
    expected = (
        ROOT
        / "results/stage1c_v2_heldout_prospective_prediction/prediction_manifest.json"
    )
    if args.output.absolute() != expected:
        raise RuntimeError("prediction output path differs from the frozen v2 path")
    worker = strict_load(args.worker)
    assemble_prediction(worker, output=args.output, verify_protocol_hashes=True)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f'{{"prediction_manifest_sha256": "{digest}", "status": "passed"}}')
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SerializationError, ValueError, OSError) as error:
        raise SystemExit(f"prediction assembly failed: {error}") from error
