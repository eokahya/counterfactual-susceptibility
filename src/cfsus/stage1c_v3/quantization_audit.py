"""Pure metadata audit of the frozen Stage 1C-v3 BF16 schedules."""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any

from cfsus.stage1c_v3.serialization import detach_json


def bf16_round(value: float) -> float:
    """Round finite Python float through float32 to nearest-even BF16."""

    if not math.isfinite(value):
        raise ValueError("BF16 audit input must be finite")
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    lower = bits & 0xFFFF
    upper = bits >> 16
    if lower > 0x8000 or (lower == 0x8000 and upper & 1):
        upper += 1
    return float(struct.unpack(">f", struct.pack(">I", upper << 16))[0])


def audit_frozen_quantization(
    manifest: dict[str, Any], *, manifest_bytes: bytes
) -> dict[str, Any]:
    """Recompute requested, desired, applied, and realized values for every pair."""

    groups = manifest.get("selected_groups")
    if not isinstance(groups, dict) or set(groups) != {
        "primary",
        "near_boundary",
        "directional",
    }:
        raise ValueError("prediction manifest groups are malformed")
    pairs: list[dict[str, Any]] = []
    total_requested = 0
    total_distinct = 0
    pairs_with_collapses = 0
    for group in ("primary", "near_boundary", "directional"):
        rows = groups[group]
        if not isinstance(rows, list):
            raise ValueError("prediction group is not a list")
        for row in rows:
            if not isinstance(row, dict) or row.get("group") != group:
                raise ValueError("prediction pair/group identity is malformed")
            pair_id = row.get("pair_id")
            baseline = row.get("source_activation")
            requested = row.get("requested_alphas")
            if (
                not isinstance(pair_id, str)
                or isinstance(baseline, bool)
                or not isinstance(baseline, (int, float))
                or not math.isfinite(float(baseline))
                or float(baseline) <= 0.0
                or not isinstance(requested, list)
            ):
                raise ValueError("prediction pair quantization inputs are malformed")
            requested_values = tuple(float(item) for item in requested)
            if requested_values != tuple(sorted(set(requested_values))) or any(
                not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0
                for alpha in requested_values
            ):
                raise ValueError("prediction pair schedule is not canonical")
            baseline_value = float(baseline)
            points: list[dict[str, Any]] = []
            applied_groups: dict[float, list[float]] = {}
            for alpha in requested_values:
                desired = (1.0 - alpha) * baseline_value
                applied = bf16_round(desired)
                realized = 1.0 - applied / baseline_value
                if not math.isfinite(realized) or not 0.0 <= realized <= 1.0:
                    raise ValueError("realized suppression is outside [0,1]")
                applied_groups.setdefault(applied, []).append(alpha)
                points.append(
                    {
                        "requested_alpha": alpha,
                        "desired_high_precision": desired,
                        "actual_bf16_value": applied,
                        "realized_suppression": realized,
                    }
                )
            collapse_groups = [
                {
                    "actual_bf16_value": applied,
                    "requested_alphas": alphas,
                }
                for applied, alphas in sorted(applied_groups.items(), reverse=True)
                if len(alphas) > 1
            ]
            distinct = len(applied_groups)
            if collapse_groups:
                pairs_with_collapses += 1
            total_requested += len(points)
            total_distinct += distinct
            pairs.append(
                {
                    "pair_id": pair_id,
                    "group": group,
                    "baseline_source_activation": baseline_value,
                    "requested_point_count": len(points),
                    "distinct_applied_point_count": distinct,
                    "collapsed_requested_point_count": len(points) - distinct,
                    "has_distinct_nonzero_suppression": any(
                        point["realized_suppression"] > 0.0 for point in points
                    ),
                    "collapse_groups": collapse_groups,
                    "points": points,
                }
            )
    result = {
        "schema_version": 4,
        "artifact_type": "stage1c_v4_frozen_schedule_quantization_audit",
        "status": "passed",
        "prediction_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "pair_count": len(pairs),
        "requested_point_count": total_requested,
        "distinct_applied_point_count": total_distinct,
        "collapsed_requested_point_count": total_requested - total_distinct,
        "pairs_with_collapsed_requests": pairs_with_collapses,
        "scientific_schedule_changed": False,
        "scientific_intervention_calls": 0,
        "pairs": pairs,
    }
    detached = detach_json(result)
    if not isinstance(detached, dict):  # pragma: no cover - defensive
        raise ValueError("quantization audit must be a JSON object")
    return detached


__all__ = ["audit_frozen_quantization", "bf16_round"]
