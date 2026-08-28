#!/usr/bin/env python3
"""Standalone hostile-input validator for the Stage 1G canonical bundle."""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfsus.stage1g import BASE_COMMIT, BRANCH, validate_bundle  # noqa: E402

FORBIDDEN_TRACKED_SUFFIXES = {
    ".bin",
    ".pt",
    ".pth",
    ".safetensors",
    ".gguf",
    ".npy",
    ".npz",
    ".pickle",
    ".pkl",
    ".journal",
    ".jsonl",
}
FORBIDDEN_TRACKED_PARTS = {
    "cache",
    "weights",
    "tokenizer_payload",
    "raw_graph",
    "adjacency",
    "gradient_tensor",
    "derivative_tensor",
    "activation_tensor",
}


def _changed_files() -> list[Path]:
    output = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--name-only", f"{BASE_COMMIT}..HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.splitlines()
    return [Path(item) for item in output if item]


def _validate_changed_tree() -> dict[str, int]:
    branch = subprocess.run(
        ["git", "-C", str(ROOT), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    if branch != BRANCH:
        raise RuntimeError("standalone Stage 1G validator is on the wrong branch")
    files = _changed_files()
    total = 0
    for relative in files:
        path = ROOT / relative
        lowered = relative.as_posix().lower()
        if (
            lowered.startswith("paper/")
            or relative.suffix.lower() in FORBIDDEN_TRACKED_SUFFIXES
        ):
            raise RuntimeError(f"forbidden Stage 1G tracked path: {relative}")
        if any(part in lowered for part in FORBIDDEN_TRACKED_PARTS):
            raise RuntimeError(f"forbidden Stage 1G tracked payload class: {relative}")
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(
                f"special or symlinked Stage 1G tracked file: {relative}"
            )
        total += info.st_size
        if info.st_size > 2 * 1024 * 1024:
            raise RuntimeError(f"oversized Stage 1G tracked file: {relative}")
    return {"changed_file_count": len(files), "changed_tree_bytes": total}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve(strict=True)
    if not bundle.is_dir() or bundle.is_symlink():
        raise RuntimeError("bundle must be a real directory")
    for path in bundle.iterdir():
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"bundle contains a non-regular file: {path.name}")
    result = validate_bundle(ROOT, bundle)
    result.update(_validate_changed_tree())
    result["hostile_input_checks"] = True
    result["fresh_process"] = True
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
