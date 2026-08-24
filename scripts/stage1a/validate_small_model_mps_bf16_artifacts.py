#!/usr/bin/env python3
"""Independently validate the accepted Stage 1A-S-BF16 artifact bundle."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cfsus.reproduction.small_model_mps_bf16 import (  # noqa: E402
    assert_fallback_disabled,
)
from cfsus.reproduction.small_model_mps_bf16_artifacts import (  # noqa: E402
    BRANCH,
    validate_bundle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    return parser


def _git(command: list[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *command],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env={
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        },
    )
    return result.stdout.strip()


def main() -> int:
    arguments = _parser().parse_args()
    assert_fallback_disabled()
    artifact_dir = arguments.artifact_dir
    if not artifact_dir.is_absolute():
        artifact_dir = REPOSITORY_ROOT / artifact_dir
    branch = _git(["branch", "--show-current"])
    if branch != BRANCH:
        raise RuntimeError("validator must run on the isolated BF16 branch")
    ancestry = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY_ROOT),
            "merge-base",
            "--is-ancestor",
            arguments.execution_commit,
            "HEAD",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if ancestry.returncode != 0:
        raise RuntimeError("execution commit is not an ancestor of validator HEAD")
    result = validate_bundle(
        artifact_dir,
        repository_root=REPOSITORY_ROOT,
        execution_commit=arguments.execution_commit,
    )
    result["branch"] = branch
    result["execution_commit_is_ancestor_of_head"] = True
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
