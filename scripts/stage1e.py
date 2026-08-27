#!/usr/bin/env python3
"""Offline-first Stage 1E finite-probe calibration entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfsus.stage1d.validation import (  # noqa: E402
    validate_bundle as validate_stage1d_bundle,
)
from cfsus.stage1e.offline import (  # noqa: E402
    OUTPUT_DIRECTORY,
    STAGE1D_DIRECTORY,
    compute_offline_analysis,
)
from cfsus.stage1e.validation import (  # noqa: E402
    publish_offline_bundle,
    validate_offline_bundle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("analyze-offline", "validate-offline"))
    parser.add_argument("--stage1d-dir", type=Path, default=ROOT / STAGE1D_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=ROOT / OUTPUT_DIRECTORY)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.action == "analyze-offline":
        stage1d_validation = validate_stage1d_bundle(ROOT, args.stage1d_dir)
        analysis = compute_offline_analysis(
            args.stage1d_dir, stage1d_validation=stage1d_validation
        )
        publish_offline_bundle(args.output_dir, analysis)
        result = {
            "status": "published",
            "terminal_status": analysis["terminal_status"],
            "project_decision": analysis["project_decision"],
            "selected_estimator": analysis["selected_estimator"],
            "phase_b_status": analysis["phase_b"]["status"],
        }
    else:
        result = validate_offline_bundle(ROOT, args.stage1d_dir, args.output_dir)
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
