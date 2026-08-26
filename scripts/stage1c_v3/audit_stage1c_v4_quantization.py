#!/usr/bin/env python3
"""Audit all frozen requested alphas without invoking an intervention API."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cfsus.stage1c_v3.quantization_audit import (  # noqa: E402
    audit_frozen_quantization,
)
from cfsus.stage1c_v3.serialization import (  # noqa: E402
    read_json_strict,
    write_json_new,
)

EXPECTED_SHA256 = "b2c489317852a2f54d50db783abc17dfdc08590353b0473dbab01ec3d04574cc"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = args.prediction_manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("frozen prediction manifest digest differs")
    manifest = read_json_strict(args.prediction_manifest)
    if not isinstance(manifest, dict):
        raise RuntimeError("frozen prediction manifest must be an object")
    result = audit_frozen_quantization(manifest, manifest_bytes=raw)
    if result.get("pair_count") != 28:
        raise RuntimeError("quantization audit does not cover all frozen pairs")
    write_json_new(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "pair_count": result["pair_count"],
                "requested_point_count": result["requested_point_count"],
                "distinct_applied_point_count": result["distinct_applied_point_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
