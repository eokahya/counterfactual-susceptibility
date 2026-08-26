#!/usr/bin/env python3
"""Materialize the baseline-only prediction manifest from a worker record.

The assembler is deliberately conservative: it copies only the worker's
prediction manifest, checks that it contains no intervention evidence, and
requires execution from the exact Stage 1C base before the freeze commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

BASE = "efbf70a7e462e640a0e1819a93f3b92727bbd193"
BRANCH = "stage-1c-first-prospective-prediction"
FORBIDDEN = frozenset(
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
    file_stat = path.lstat()
    if (
        not stat.S_ISREG(file_stat.st_mode)
        or file_stat.st_nlink != 1
        or file_stat.st_size > 2 * 1024 * 1024
    ):
        raise ValueError("worker record is not a bounded single-link regular file")
    raw = path.read_bytes()

    def constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    value = json.loads(
        raw.decode("utf-8"), parse_constant=constant, object_pairs_hook=unique
    )
    if not isinstance(value, dict):
        raise ValueError("worker record must be an object")
    return value


def walk_prediction(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.casefold() in FORBIDDEN:
                raise ValueError(
                    f"intervention field in prediction manifest: {path}.{key}"
                )
            walk_prediction(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            walk_prediction(item, f"{path}[{index}]")


def git(*args: str) -> str:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "-C", str(root), *args],
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if git("rev-parse", "HEAD") != BASE or git("branch", "--show-current") != BRANCH:
        raise RuntimeError("prediction freeze requires the exact Stage 1C base branch")
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError("prediction output must be a new file")
    expected_output = (
        root / "results/stage1c_first_prospective_prediction/prediction_manifest.json"
    )
    if args.output.absolute() != expected_output:
        raise RuntimeError("prediction output path differs from the frozen result path")
    worker = strict_load(args.worker)
    if worker.get("status") != "passed":
        raise RuntimeError("prediction worker did not pass")
    prediction = worker.get("prediction_manifest")
    if not isinstance(prediction, dict):
        raise RuntimeError("worker has no prediction manifest")
    walk_prediction(prediction)
    if prediction.get("status") != "prediction_frozen_ready_for_commit":
        raise RuntimeError("prediction manifest is not freeze-ready")
    if prediction.get("base_commit") != BASE or prediction.get("branch") != BRANCH:
        raise RuntimeError("prediction manifest identity differs from frozen protocol")
    protocol_hashes = prediction.get("protocol_file_sha256")
    if not isinstance(protocol_hashes, dict) or not protocol_hashes:
        raise RuntimeError("prediction manifest lacks frozen protocol hashes")
    for relative, expected_digest in protocol_hashes.items():
        if (
            not isinstance(relative, str)
            or not isinstance(expected_digest, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise RuntimeError("prediction protocol hash entry is unsafe")
        protocol_file = root / relative
        if protocol_file.is_symlink() or not protocol_file.is_file():
            raise RuntimeError(f"prediction protocol file is missing: {relative}")
        observed_digest = hashlib.sha256(protocol_file.read_bytes()).hexdigest()
        if observed_digest != expected_digest:
            raise RuntimeError(f"prediction protocol file changed: {relative}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(prediction, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "status": "passed",
                "prediction_manifest_sha256": manifest_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
