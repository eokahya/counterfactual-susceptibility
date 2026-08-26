"""Frozen v2 pair analyses and scientific outcome classification."""

from __future__ import annotations

import math
from enum import StrEnum
from itertools import pairwise
from statistics import median
from typing import Any

from cfsus.exceptions import NonFiniteInputError, ScientificInputError


class ScientificOutcome(StrEnum):
    SUPPORTED = "supported"
    MIXED = "mixed"
    NOT_SUPPORTED = "not_supported"
    NO_ELIGIBLE_PAIRS = "no_eligible_pairs"
    INCONCLUSIVE_RUNTIME = "inconclusive_runtime"


def symmetric_normalized_error(reference: float, candidate: float) -> float:
    if not math.isfinite(reference) or not math.isfinite(candidate):
        raise NonFiniteInputError("movement error inputs must be finite")
    denominator = abs(reference) + abs(candidate)
    return 0.0 if denominator == 0.0 else 2.0 * abs(reference - candidate) / denominator


def _nearest_rank(values: list[float], probability: float) -> float:
    if not values:
        raise ScientificInputError("nearest-rank metric requires values")
    return sorted(values)[max(1, math.ceil(probability * len(values))) - 1]


def _first_bracket(points: list[dict[str, Any]]) -> dict[str, float] | None:
    ordered = sorted(points, key=lambda item: float(item["realized_suppression"]))
    for left, right in pairwise(ordered):
        if not bool(left["target_active"]) and bool(right["target_active"]):
            return {
                "lower_realized_suppression": float(left["realized_suppression"]),
                "upper_realized_suppression": float(right["realized_suppression"]),
            }
    return None


def _distance_to_bracket(
    alpha: float | None, bracket: dict[str, float] | None
) -> float | None:
    if alpha is None or bracket is None:
        return None
    lower = bracket["lower_realized_suppression"]
    upper = bracket["upper_realized_suppression"]
    if lower <= alpha <= upper:
        return 0.0
    return min(abs(alpha - lower), abs(alpha - upper))


def analyze_pair(
    pair: dict[str, Any], points: list[dict[str, Any]], analysis: dict[str, Any]
) -> dict[str, Any]:
    """Recompute strict crossing and local-linearity diagnostics for one pair."""

    if not points:
        raise ScientificInputError("pair analysis requires applied points")
    ordered = sorted(points, key=lambda item: float(item["realized_suppression"]))
    realized = [float(item["realized_suppression"]) for item in ordered]
    if realized != sorted(set(realized)):
        raise ScientificInputError(
            "applied points are not unique realized suppressions"
        )
    baseline = ordered[0]
    if realized[0] != 0.0 or bool(baseline["target_active"]):
        raise ScientificInputError("pair baseline is not exact inactive alpha zero")
    baseline_z = float(pair["target_preactivation"])
    if float(baseline["target_preactivation"]) != baseline_z:
        raise ScientificInputError("intervention baseline differs from prediction")
    q = float(pair["q"])
    nonzero = [item for item in ordered if float(item["realized_suppression"]) > 0.0]
    errors: list[float] = []
    signs: list[bool] = []
    for item in nonzero:
        alpha = float(item["realized_suppression"])
        predicted_delta = alpha * q
        observed_delta = float(item["target_preactivation"]) - baseline_z
        errors.append(symmetric_normalized_error(predicted_delta, observed_delta))
        signs.append(
            observed_delta > 0.0
            if predicted_delta > 0.0
            else observed_delta < 0.0
            if predicted_delta < 0.0
            else observed_delta == 0.0
        )
    sign_agreement = sum(signs) / len(signs) if signs else None
    median_error = median(errors) if errors else None
    p95_error = _nearest_rank(errors, 0.95) if errors else None
    bracket = _first_bracket(ordered)
    alpha_hat = pair.get("predicted_alpha_star")
    bracket_distance = _distance_to_bracket(
        None if alpha_hat is None else float(alpha_hat), bracket
    )
    full = next(
        (item for item in ordered if float(item["realized_suppression"]) == 1.0), None
    )
    if full is None:
        raise ScientificInputError("pair sweep lacks exact full ablation")
    full_delta = float(full["target_preactivation"]) - baseline_z
    local_pass = (
        len(nonzero) >= int(analysis["minimum_nonzero_points"])
        and sign_agreement is not None
        and sign_agreement >= float(analysis["movement_sign_agreement_minimum"])
        and median_error is not None
        and median_error <= float(analysis["median_movement_sne_maximum"])
        and p95_error is not None
        and p95_error <= float(analysis["p95_movement_sne_maximum"])
        and bracket_distance is not None
        and bracket_distance <= float(analysis["critical_bracket_distance_maximum"])
    )
    group = str(pair["group"])
    full_crossing = bool(full["target_active"])
    active_seen = False
    nonmonotonic_gate = False
    for item in ordered:
        if bool(item["target_active"]):
            active_seen = True
        elif active_seen:
            nonmonotonic_gate = True
    return {
        "pair_id": pair["pair_id"],
        "group": group,
        "point_count": len(ordered),
        "nonzero_point_count": len(nonzero),
        "predicted_full_ablation_crossing": pair["predicted_status"]
        == "definitely_crossing",
        "observed_full_ablation_crossing": full_crossing,
        "predicted_alpha_star": alpha_hat,
        "observed_critical_bracket": bracket,
        "critical_bracket_distance": bracket_distance,
        "movement_sign_agreement": sign_agreement,
        "median_movement_symmetric_normalized_error": median_error,
        "p95_movement_symmetric_normalized_error": p95_error,
        "full_ablation_observed_movement": full_delta,
        "local_calibration_passed": local_pass,
        "supporting_primary": (
            group == "primary" and full_crossing and full_delta > 0.0 and local_pass
        ),
        "directional_control_violation": group == "directional" and full_delta > 0.0,
        "near_boundary_control_crossing": group == "near_boundary"
        and any(bool(item["target_active"]) for item in ordered),
        "nonmonotonic_gate": nonmonotonic_gate,
    }


def classify_outcome(pair_analyses: list[dict[str, Any]]) -> ScientificOutcome:
    """Apply the frozen v2 outcome precedence after a valid canonical runtime."""

    primary = [item for item in pair_analyses if item["group"] == "primary"]
    if not primary:
        return ScientificOutcome.NO_ELIGIBLE_PAIRS
    supporting = sum(bool(item["supporting_primary"]) for item in primary)
    if supporting == 0:
        return ScientificOutcome.NOT_SUPPORTED
    discrepancy = (
        supporting != len(primary)
        or any(bool(item["directional_control_violation"]) for item in pair_analyses)
        or any(bool(item["near_boundary_control_crossing"]) for item in pair_analyses)
        or any(bool(item["nonmonotonic_gate"]) for item in pair_analyses)
    )
    return ScientificOutcome.MIXED if discrepancy else ScientificOutcome.SUPPORTED


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in range(start, end):
            result[order[index]] = rank
        start = end
    return result


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    x = _average_ranks(left)
    y = _average_ranks(right)
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True))
    denominator = math.sqrt(
        sum((a - x_mean) ** 2 for a in x) * sum((b - y_mean) ** 2 for b in y)
    )
    return None if denominator == 0.0 else numerator / denominator


def aggregate_analyses(pair_analyses: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce denominator-explicit primary/control summaries."""

    primary = [item for item in pair_analyses if item["group"] == "primary"]
    near = [item for item in pair_analyses if item["group"] == "near_boundary"]
    directional = [item for item in pair_analyses if item["group"] == "directional"]
    primary_crossings = sum(
        bool(item["observed_full_ablation_crossing"]) for item in primary
    )
    near_crossings = sum(bool(item["near_boundary_control_crossing"]) for item in near)
    directional_violations = sum(
        bool(item["directional_control_violation"]) for item in directional
    )
    predicted: list[float] = []
    observed: list[float] = []
    bracket_distances: list[float] = []
    pair_median_errors: list[float] = []
    pair_p95_errors: list[float] = []
    for item in primary:
        bracket = item["observed_critical_bracket"]
        alpha = item["predicted_alpha_star"]
        if bracket is not None and alpha is not None:
            predicted.append(float(alpha))
            observed.append(
                (
                    float(bracket["lower_realized_suppression"])
                    + float(bracket["upper_realized_suppression"])
                )
                / 2.0
            )
        distance = item["critical_bracket_distance"]
        if distance is not None:
            bracket_distances.append(float(distance))
        pair_median = item["median_movement_symmetric_normalized_error"]
        if pair_median is not None:
            pair_median_errors.append(float(pair_median))
        pair_p95 = item["p95_movement_symmetric_normalized_error"]
        if pair_p95 is not None:
            pair_p95_errors.append(float(pair_p95))
    correlation = _spearman(predicted, observed)
    return {
        "primary_pair_count": len(primary),
        "primary_full_ablation_crossing_count": primary_crossings,
        "primary_full_ablation_crossing_precision": (
            primary_crossings / len(primary) if primary else None
        ),
        "primary_precision_undefined_reason": None if primary else "no_primary_pairs",
        "supporting_primary_count": sum(
            bool(item["supporting_primary"]) for item in primary
        ),
        "near_boundary_pair_count": len(near),
        "near_boundary_crossing_count": near_crossings,
        "near_boundary_crossing_fraction": near_crossings / len(near) if near else None,
        "near_boundary_fraction_undefined_reason": (
            None if near else "no_near_boundary_controls"
        ),
        "directional_pair_count": len(directional),
        "directional_violation_count": directional_violations,
        "directional_violation_fraction": (
            directional_violations / len(directional) if directional else None
        ),
        "directional_fraction_undefined_reason": (
            None if directional else "no_directional_controls"
        ),
        "critical_suppression_spearman": correlation,
        "critical_suppression_spearman_pair_count": len(predicted),
        "critical_suppression_spearman_undefined_reason": (
            None
            if correlation is not None
            else "fewer_than_two_nonconstant_observed_crossings"
        ),
        "primary_bracket_distance_count": len(bracket_distances),
        "primary_bracket_distance_median": (
            median(bracket_distances) if bracket_distances else None
        ),
        "primary_bracket_distance_p95": (
            _nearest_rank(bracket_distances, 0.95) if bracket_distances else None
        ),
        "primary_bracket_distance_undefined_reason": (
            None if bracket_distances else "no_observed_primary_crossing_brackets"
        ),
        "primary_pair_median_movement_sne_median": (
            median(pair_median_errors) if pair_median_errors else None
        ),
        "primary_pair_median_movement_sne_p95": (
            _nearest_rank(pair_median_errors, 0.95) if pair_median_errors else None
        ),
        "primary_pair_movement_error_undefined_reason": (
            None if pair_median_errors else "no_primary_movement_errors"
        ),
        "primary_pair_p95_movement_sne_median": (
            median(pair_p95_errors) if pair_p95_errors else None
        ),
    }


__all__ = [
    "ScientificOutcome",
    "aggregate_analyses",
    "analyze_pair",
    "classify_outcome",
    "symmetric_normalized_error",
]
