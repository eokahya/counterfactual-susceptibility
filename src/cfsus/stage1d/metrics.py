"""Standalone deterministic Stage 1D benchmark statistics."""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from itertools import pairwise
from statistics import median
from typing import Any

from cfsus.exceptions import ScientificInputError
from cfsus.stage1c_v3.analysis import symmetric_normalized_error
from cfsus.stage1d.benchmark import METHODS


def _nearest_rank(values: list[float], probability: float) -> float:
    if not values:
        raise ScientificInputError("nearest-rank metric requires values")
    return sorted(values)[max(1, math.ceil(probability * len(values))) - 1]


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


def spearman(left: list[float], right: list[float]) -> float | None:
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


def _bootstrap_mean(values: list[float], *, count: int, seed: str) -> dict[str, Any]:
    if not values:
        return {"estimate": None, "lower": None, "upper": None, "resamples": 0}
    generator = random.Random(int(hashlib.sha256(seed.encode()).hexdigest(), 16))
    estimates = [
        sum(values[generator.randrange(len(values))] for _ in values) / len(values)
        for _ in range(count)
    ]
    return {
        "estimate": sum(values) / len(values),
        "lower": _nearest_rank(estimates, 0.025),
        "upper": _nearest_rank(estimates, 0.975),
        "resamples": count,
    }


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    return sum(
        math.comb(n, index) * probability**index * (1.0 - probability) ** (n - index)
        for index in range(k + 1)
    )


def exact_binomial_interval(successes: int, trials: int) -> tuple[float, float]:
    """Return the two-sided 95% Clopper-Pearson interval."""

    if trials <= 0 or successes < 0 or successes > trials:
        raise ScientificInputError("binomial count is invalid")
    alpha = 0.05
    if successes == 0:
        lower = 0.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            tail = 1.0 - _binomial_cdf(successes - 1, trials, mid)
            if tail > alpha / 2.0:
                hi = mid
            else:
                lo = mid
        lower = (lo + hi) / 2.0
    if successes == trials:
        upper = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            cdf = _binomial_cdf(successes, trials, mid)
            if cdf > alpha / 2.0:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2.0
    return lower, upper


def _full_point(points: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [
        point
        for point in points
        if any(
            mapping.get("requested_alpha") == 1.0
            for mapping in point.get("requested_mappings", [])
        )
    ]
    if len(matches) != 1 or float(matches[0]["realized_suppression"]) != 1.0:
        raise ScientificInputError("pair lacks one exact full-ablation point")
    return matches[0]


def _first_bracket(points: list[dict[str, Any]]) -> dict[str, float] | None:
    ordered = sorted(points, key=lambda item: float(item["realized_suppression"]))
    for left, right in pairwise(ordered):
        if not bool(left["target_active"]) and bool(right["target_active"]):
            return {
                "lower": float(left["realized_suppression"]),
                "upper": float(right["realized_suppression"]),
            }
    return None


def _pair_calibration(
    pair: dict[str, Any], points: list[dict[str, Any]]
) -> dict[str, Any]:
    ordered = sorted(points, key=lambda item: float(item["realized_suppression"]))
    if not ordered or float(ordered[0]["realized_suppression"]) != 0.0:
        raise ScientificInputError("detailed sweep lacks no-op baseline")
    active_seen = False
    nonmonotonic = False
    for point in ordered:
        if bool(point["target_active"]):
            active_seen = True
        elif active_seen:
            nonmonotonic = True
    bracket = _first_bracket(ordered)
    midpoint = None if bracket is None else (bracket["lower"] + bracket["upper"]) / 2.0
    alpha = pair.get("predicted_alpha_star")
    distance = None
    absolute_error = None
    if bracket is not None and alpha is not None:
        alpha_value = float(alpha)
        distance = (
            0.0
            if bracket["lower"] <= alpha_value <= bracket["upper"]
            else min(
                abs(alpha_value - bracket["lower"]), abs(alpha_value - bracket["upper"])
            )
        )
        midpoint_value = (bracket["lower"] + bracket["upper"]) / 2.0
        absolute_error = abs(alpha_value - midpoint_value)
    baseline_z = float(pair["target_preactivation"])
    q = float(pair["q"])
    errors: list[float] = []
    signs: list[bool] = []
    for point in ordered:
        realized = float(point["realized_suppression"])
        if realized <= 0.0:
            continue
        predicted = realized * q
        observed = float(point["target_preactivation"]) - baseline_z
        errors.append(symmetric_normalized_error(predicted, observed))
        signs.append(
            observed > 0.0
            if predicted > 0.0
            else observed < 0.0
            if predicted < 0.0
            else observed == 0.0
        )
    return {
        "prompt_id": pair["prompt_id"],
        "pair_id": pair["pair_id"],
        "detailed_role": pair["detailed_role"],
        "predicted_alpha_star": alpha,
        "observed_bracket": bracket,
        "observed_bracket_midpoint": midpoint,
        "critical_bracket_distance": distance,
        "critical_midpoint_absolute_error": absolute_error,
        "monotonic_crossing": bracket is not None and not nonmonotonic,
        "nonmonotonic_gate": nonmonotonic,
        "movement_sign_agreement": sum(signs) / len(signs) if signs else None,
        "movement_errors": errors,
        "pair_median_movement_sne": median(errors) if errors else None,
        "pair_p95_movement_sne": _nearest_rank(errors, 0.95) if errors else None,
    }


def _bootstrap_spearman(
    rows: list[dict[str, Any]], *, prompt_ids: list[str], count: int, seed: str
) -> dict[str, Any]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[str(row["prompt_id"])].append(row)
    generator = random.Random(int(hashlib.sha256(seed.encode()).hexdigest(), 16))
    values: list[float] = []
    for _ in range(count):
        sampled = [prompt_ids[generator.randrange(len(prompt_ids))] for _ in prompt_ids]
        selected = [row for prompt_id in sampled for row in by_prompt[prompt_id]]
        value = spearman(
            [float(row["predicted_alpha_star"]) for row in selected],
            [float(row["observed_bracket_midpoint"]) for row in selected],
        )
        if value is not None:
            values.append(value)
    return {
        "lower": _nearest_rank(values, 0.025) if values else None,
        "upper": _nearest_rank(values, 0.975) if values else None,
        "defined_resamples": len(values),
        "requested_resamples": count,
    }


def classify_project_decision(
    *,
    s_minus_margin: float,
    s_minus_influence: float,
    critical_spearman: float | None,
    critical_pair_count: int,
    directional_violation_fraction: float | None,
    nonmonotonic_fraction: float | None,
    rules: dict[str, Any],
) -> str:
    """Apply the frozen three-way Stage 1D project decision."""

    beats_margin = s_minus_margin >= float(rules["s_minimum_absolute_advantage"])
    beats_influence = s_minus_influence >= float(rules["s_minimum_absolute_advantage"])
    directional_pass = (
        directional_violation_fraction is not None
        and directional_violation_fraction
        <= float(rules["directional_violation_fraction_maximum"])
    )
    calibration_pass = (
        critical_spearman is not None
        and critical_spearman >= float(rules["critical_spearman_minimum"])
        and critical_pair_count >= int(rules["critical_pair_count_minimum"])
    )
    monotonic_pass = (
        nonmonotonic_fraction is not None
        and nonmonotonic_fraction
        <= float(rules["detailed_nonmonotonic_fraction_maximum"])
    )
    if (
        beats_margin
        and beats_influence
        and calibration_pass
        and directional_pass
        and monotonic_pass
    ):
        return "continue_first_order_to_behavioral_stage"
    if (not beats_margin and not beats_influence) or not directional_pass:
        return "redesign_method_before_scaling"
    return "retain_crossing_ranker_but_redesign_calibration"


def compute_benchmark_summary(
    prediction: dict[str, Any],
    points_by_pair: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Recompute every decision metric from frozen panels and serialized points."""

    prompts = prediction["prompts"]
    prompt_ids = [str(item["id"]) for item in prompts]
    pair_rows = {
        row["pair_id"]: row for prompt in prompts for row in prompt["execution_pairs"]
    }
    per_prompt: list[dict[str, Any]] = []
    for prompt in prompts:
        method_metrics: dict[str, Any] = {}
        for method in METHODS:
            ids = prompt["method_pair_ids"][method]
            crossings = sum(
                bool(_full_point(points_by_pair[pair_id])["target_active"])
                for pair_id in ids
            )
            method_metrics[method] = {
                "selected_count": len(ids),
                "crossing_count": crossings,
                "precision_at_4": crossings / 4.0,
                "missing_slot_count": 4 - len(ids),
            }
        directional_ids = prompt["directional_pair_ids"]
        violations = sum(
            float(_full_point(points_by_pair[pair_id])["target_preactivation"])
            - float(pair_rows[pair_id]["target_preactivation"])
            > 0.0
            for pair_id in directional_ids
        )
        overlap = {
            f"{left}__{right}": len(
                set(prompt["method_pair_ids"][left])
                & set(prompt["method_pair_ids"][right])
            )
            for index, left in enumerate(METHODS)
            for right in METHODS[index + 1 :]
        }
        per_prompt.append(
            {
                "prompt_id": prompt["id"],
                "methods": method_metrics,
                "directional_selected_count": len(directional_ids),
                "directional_violation_count": violations,
                "method_overlap_counts": overlap,
            }
        )
    bootstrap_count = int(config["metrics"]["bootstrap_resamples"])
    seed = str(config["metrics"]["bootstrap_seed"])
    full_metrics: dict[str, Any] = {}
    for method in METHODS:
        values = [float(row["methods"][method]["precision_at_4"]) for row in per_prompt]
        full_metrics[method] = {
            "mean_prompt_precision_at_4": sum(values) / len(values),
            "pooled_crossing_count": sum(
                int(row["methods"][method]["crossing_count"]) for row in per_prompt
            ),
            "pooled_denominator": len(per_prompt) * 4,
            "pooled_precision_at_4": sum(
                int(row["methods"][method]["crossing_count"]) for row in per_prompt
            )
            / (len(per_prompt) * 4),
            "prompt_bootstrap_95": _bootstrap_mean(
                values, count=bootstrap_count, seed=f"{seed}|{method}"
            ),
        }
    differences: dict[str, Any] = {}
    for baseline in ("margin_only", "influence_only", "random_positive"):
        values = [
            float(row["methods"]["S"]["precision_at_4"])
            - float(row["methods"][baseline]["precision_at_4"])
            for row in per_prompt
        ]
        differences[f"S_minus_{baseline}"] = _bootstrap_mean(
            values, count=bootstrap_count, seed=f"{seed}|S-minus-{baseline}"
        )
    directional_successes = sum(
        int(row["directional_violation_count"]) for row in per_prompt
    )
    directional_trials = sum(
        int(row["directional_selected_count"]) for row in per_prompt
    )
    directional_interval = (
        exact_binomial_interval(directional_successes, directional_trials)
        if directional_trials
        else None
    )
    detailed_rows = [
        _pair_calibration(row, points_by_pair[pair_id])
        for pair_id, row in pair_rows.items()
        if row.get("detailed_role") is not None
    ]
    calibration_rows = [
        row
        for row in detailed_rows
        if row.get("detailed_role") in {"B1", "B2", "B3"}
        and bool(
            pair_rows[str(row["pair_id"])]
            .get("quantization_evidence", {})
            .get("calibration_resolvable")
        )
    ]
    monotonic = [row for row in calibration_rows if row["monotonic_crossing"]]
    predicted = [float(row["predicted_alpha_star"]) for row in monotonic]
    observed = [float(row["observed_bracket_midpoint"]) for row in monotonic]
    correlation = spearman(predicted, observed)
    distances = [float(row["critical_bracket_distance"]) for row in monotonic]
    absolute_errors = [
        float(row["critical_midpoint_absolute_error"]) for row in monotonic
    ]
    by_bin: dict[str, Any] = {}
    for name in ("B1", "B2", "B3"):
        selected = [row for row in monotonic if row["detailed_role"] == name]
        by_bin[name] = {
            "pair_count": len(selected),
            "spearman": spearman(
                [float(row["predicted_alpha_star"]) for row in selected],
                [float(row["observed_bracket_midpoint"]) for row in selected],
            ),
            "median_absolute_error": median(
                [float(row["critical_midpoint_absolute_error"]) for row in selected]
            )
            if selected
            else None,
        }
    movement_errors = [
        error for row in detailed_rows for error in row["movement_errors"]
    ]
    sign_values = [
        float(row["movement_sign_agreement"])
        for row in detailed_rows
        if row["movement_sign_agreement"] is not None
    ]
    nonmonotonic_count = sum(bool(row["nonmonotonic_gate"]) for row in detailed_rows)
    selected_quant = [
        row["quantization_evidence"]
        for row in pair_rows.values()
        if isinstance(row.get("quantization_evidence"), dict)
    ]
    critical = {
        "monotonic_resolvable_crossing_pair_count": len(monotonic),
        "spearman": correlation,
        "median_bracket_distance": median(distances) if distances else None,
        "p95_bracket_distance": _nearest_rank(distances, 0.95) if distances else None,
        "median_midpoint_absolute_error": median(absolute_errors)
        if absolute_errors
        else None,
        "prompt_bootstrap_spearman_95": _bootstrap_spearman(
            monotonic,
            prompt_ids=prompt_ids,
            count=bootstrap_count,
            seed=f"{seed}|critical-spearman",
        ),
        "by_predicted_alpha_bin": by_bin,
        "pair_results": calibration_rows,
    }
    nonmonotonic_fraction_value = (
        nonmonotonic_count / len(detailed_rows) if detailed_rows else None
    )
    movement: dict[str, Any] = {
        "pair_count": len(detailed_rows),
        "movement_sign_agreement_mean": sum(sign_values) / len(sign_values)
        if sign_values
        else None,
        "point_median_symmetric_normalized_error": median(movement_errors)
        if movement_errors
        else None,
        "point_p95_symmetric_normalized_error": _nearest_rank(movement_errors, 0.95)
        if movement_errors
        else None,
        "per_pair_median_error_distribution": [
            row["pair_median_movement_sne"] for row in detailed_rows
        ],
        "nonmonotonic_gate_count": nonmonotonic_count,
        "nonmonotonic_gate_fraction": nonmonotonic_fraction_value,
        "selected_quantization_collapsed_request_count": sum(
            int(row["collapsed_requested_point_count"]) for row in selected_quant
        ),
        "selected_quantization_predicted_noop_count": sum(
            "predicted_alpha_bf16_noop" in row["limitation_reasons"]
            for row in selected_quant
        ),
    }
    advantage_margin = float(differences["S_minus_margin_only"]["estimate"])
    advantage_influence = float(differences["S_minus_influence_only"]["estimate"])
    directional_fraction = (
        directional_successes / directional_trials if directional_trials else None
    )
    nonmonotonic_fraction = nonmonotonic_fraction_value
    rules = config["decision_rule"]
    directional_pass = (
        directional_fraction is not None
        and directional_fraction
        <= float(rules["directional_violation_fraction_maximum"])
    )
    calibration_pass = (
        correlation is not None
        and correlation >= float(rules["critical_spearman_minimum"])
        and len(monotonic) >= int(rules["critical_pair_count_minimum"])
    )
    monotonic_pass = (
        nonmonotonic_fraction is not None
        and nonmonotonic_fraction
        <= float(rules["detailed_nonmonotonic_fraction_maximum"])
    )
    beats_margin = advantage_margin >= float(rules["s_minimum_absolute_advantage"])
    beats_influence = advantage_influence >= float(
        rules["s_minimum_absolute_advantage"]
    )
    decision = classify_project_decision(
        s_minus_margin=advantage_margin,
        s_minus_influence=advantage_influence,
        critical_spearman=correlation,
        critical_pair_count=len(monotonic),
        directional_violation_fraction=directional_fraction,
        nonmonotonic_fraction=nonmonotonic_fraction,
        rules=rules,
    )
    return {
        "schema_version": 1,
        "artifact_type": "stage1d_benchmark_summary",
        "status": "passed",
        "per_prompt": per_prompt,
        "full_ablation_discovery": full_metrics,
        "paired_prompt_differences": differences,
        "directional_controls": {
            "violation_count": directional_successes,
            "trial_count": directional_trials,
            "violation_fraction": directional_fraction,
            "exact_binomial_95": (
                {
                    "lower": directional_interval[0],
                    "upper": directional_interval[1],
                }
                if directional_interval is not None
                else None
            ),
        },
        "critical_suppression_calibration": critical,
        "local_movement_fidelity": movement,
        "decision_inputs": {
            "beats_margin_by_0.10": beats_margin,
            "beats_influence_by_0.10": beats_influence,
            "critical_calibration_pass": calibration_pass,
            "directional_control_pass": directional_pass,
            "nonmonotonic_fraction_pass": monotonic_pass,
        },
        "project_decision": decision,
    }


__all__ = [
    "classify_project_decision",
    "compute_benchmark_summary",
    "exact_binomial_interval",
    "spearman",
]
