#!/usr/bin/env python3
"""Standalone strict validator for the Stage 1B compact artifact bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cfsus.stage1b import CONFIG_PATH, load_stage1b_config  # noqa: E402
from cfsus.stage1b_artifacts import validate_stage1b_artifacts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / CONFIG_PATH)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    args = parser.parse_args()
    result = validate_stage1b_artifacts(
        args.bundle,
        config=load_stage1b_config(args.config),
        execution_commit=args.execution_commit,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
