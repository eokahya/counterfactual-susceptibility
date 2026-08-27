"""Pure offline Stage 1E finite-probe development-set reanalysis."""

from __future__ import annotations

import hashlib
import math
import random
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any

from cfsus.stage1c_v3.quantization_audit import bf16_round
from cfsus.stage1c_v3.serialization import read_json_strict
from cfsus.stage1d.metrics import spearman

EXPERIMENT_CLASS = "stage1e_finite_probe_calibration"
BRANCH = "stage-1e-finite-probe-calibration"
BASE_COMMIT = "b71df55fdeb2fb66601af56207b6fbe5238e57d8"
AUDITED_STAGE1D_ARTIFACT_COMMIT = "2a5c3e63a838e7547fe1b30fe888610ec21ee46e"
TERMINAL_STATUS = "completed_stage1e_offline_negative"
PROJECT_DECISION = "simple_finite_probe_calibration_not_supported"
STAGE1D_DIRECTORY = Path("results/stage1d_multiprompt_gate_benchmark")
OUTPUT_DIRECTORY = Path("results/stage1e_finite_probe_calibration")
E1_GRID = (0.125, 0.1875, 0.25)
E2_GRID = (0.5, 0.625)
REFERENCE_ROLES = frozenset({"B1", "B2", "B3"})
ROLE_ORDER = {"B1": 0, "B2": 1, "B3": 2, "near_boundary": 3}
BOOTSTRAP_COUNT = 10_000
BOOTSTRAP_SEED = "stage1e-offline-prompt-bootstrap-v1"
STAGE1D_FILES = (
    "protocol_manifest.json",
    "prediction_manifest.json",
    "panel_membership.json",
    "quantization_audit.json",
    "full_ablation_points.json",
    "calibration_sweeps.json",
    "benchmark_summary.json",
    "run_manifest.json",
    "environment_manifest.json",
    "checksums.sha256",
)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _nearest_rank(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    index = max(1, math.ceil(probability * len(values))) - 1
    return sorted(values)[index]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _first_crossing_bracket(
    points: list[dict[str, Any]],
) -> tuple[dict[str, float] | None, bool]:
    ordered = sorted(
        points,
        key=lambda row: _number(row.get("realized_suppression"), "realized alpha"),
    )
    active_seen = False
    nonmonotonic = False
    for point in ordered:
        z = _number(point.get("target_preactivation"), "target preactivation")
        threshold = _number(point.get("target_threshold"), "target threshold")
        active = z > threshold
        if active:
            active_seen = True
        elif active_seen:
            nonmonotonic = True
    bracket: dict[str, float] | None = None
    for left, right in pairwise(ordered):
        left_active = _number(left["target_preactivation"], "left z") > _number(
            left["target_threshold"], "left threshold"
        )
        right_active = _number(right["target_preactivation"], "right z") > _number(
            right["target_threshold"], "right threshold"
        )
        if not left_active and right_active:
            bracket = {
                "lower_realized_alpha": _number(
                    left["realized_suppression"], "lower realized alpha"
                ),
                "upper_realized_alpha": _number(
                    right["realized_suppression"], "upper realized alpha"
                ),
            }
            break
    return bracket, nonmonotonic


def select_serialized_probe(
    pair: dict[str, Any],
    points: list[dict[str, Any]],
    nominal_grid: tuple[float, ...],
) -> dict[str, float] | None:
    """Select the first nonzero BF16 probe that exists in the serialized sweep."""

    baseline = _number(pair.get("source_activation"), "source activation")
    if baseline <= 0.0:
        raise ValueError("source activation must be positive")
    by_applied: dict[float, dict[str, Any]] = {}
    for point in points:
        applied = _number(point.get("actual_bf16_value_passed"), "applied source")
        if applied in by_applied:
            raise ValueError("serialized sweep repeats an applied BF16 value")
        by_applied[applied] = point
    for nominal in nominal_grid:
        desired = (1.0 - nominal) * baseline
        applied = bf16_round(desired)
        realized = 1.0 - applied / baseline
        if realized <= 0.0:
            continue
        serialized_point = by_applied.get(applied)
        if serialized_point is None:
            continue
        observed_realized = _number(
            serialized_point.get("realized_suppression"), "observed realized alpha"
        )
        if observed_realized != realized:
            raise ValueError("serialized realized alpha differs from BF16 mapping")
        return {
            "nominal_requested_alpha": nominal,
            "desired_source_activation": desired,
            "applied_bf16_source_activation": applied,
            "realized_alpha": realized,
            "target_preactivation": _number(
                serialized_point.get("target_preactivation"),
                "probe target preactivation",
            ),
        }
    return None


def estimate_e0(*, margin: float, q: float) -> dict[str, Any]:
    """Compute the zero-probe local estimate."""

    if not math.isfinite(margin) or not math.isfinite(q) or q <= 0.0:
        return {
            "status": "abstained",
            "abstention_reason": "nonfinite_or_nonpositive_q",
            "predicted_alpha": None,
            "probe_count": 0,
            "probes": [],
        }
    alpha = margin / q
    if not math.isfinite(alpha):
        return {
            "status": "abstained",
            "abstention_reason": "nonfinite_estimate",
            "predicted_alpha": None,
            "probe_count": 0,
            "probes": [],
        }
    return {
        "status": "accepted",
        "abstention_reason": None,
        "predicted_alpha": alpha,
        "probe_count": 0,
        "probes": [],
    }


def estimate_e1(
    *, margin: float, baseline_z: float, probe: dict[str, float] | None
) -> dict[str, Any]:
    """Compute the one-probe secant estimate using realized alpha."""

    if probe is None:
        return {
            "status": "abstained",
            "abstention_reason": "no_serialized_quantization_resolvable_probe",
            "predicted_alpha": None,
            "estimated_drive": None,
            "probe_count": 0,
            "probes": [],
        }
    delta = probe["realized_alpha"]
    estimated_drive = (probe["target_preactivation"] - baseline_z) / delta
    if not math.isfinite(estimated_drive) or estimated_drive <= 0.0:
        return {
            "status": "abstained",
            "abstention_reason": "nonpositive_or_nonfinite_secant_drive",
            "predicted_alpha": None,
            "estimated_drive": estimated_drive,
            "probe_count": 1,
            "probes": [probe],
        }
    alpha = margin / estimated_drive
    if not math.isfinite(alpha):
        return {
            "status": "abstained",
            "abstention_reason": "nonfinite_estimate",
            "predicted_alpha": None,
            "estimated_drive": estimated_drive,
            "probe_count": 1,
            "probes": [probe],
        }
    return {
        "status": "accepted",
        "abstention_reason": None,
        "predicted_alpha": alpha,
        "estimated_drive": estimated_drive,
        "probe_count": 1,
        "probes": [probe],
    }


def estimate_e2(
    *,
    margin: float,
    baseline_z: float,
    first_probe: dict[str, float] | None,
    second_probe: dict[str, float] | None,
) -> dict[str, Any]:
    """Fit the frozen two-probe quadratic and return its admissible root."""

    probes = [probe for probe in (first_probe, second_probe) if probe is not None]
    if first_probe is None or second_probe is None:
        return {
            "status": "abstained",
            "abstention_reason": "missing_serialized_quantization_resolvable_probe",
            "predicted_alpha": None,
            "linear_coefficient": None,
            "quadratic_coefficient": None,
            "probe_count": len(probes),
            "probes": probes,
        }
    delta_1 = first_probe["realized_alpha"]
    delta_2 = second_probe["realized_alpha"]
    linear_coefficient: float | None
    quadratic_coefficient: float | None
    if delta_1 <= 0.0 or delta_2 <= delta_1:
        reason = "singular_probe_geometry"
        linear_coefficient = None
        quadratic_coefficient = None
    else:
        movement_1 = first_probe["target_preactivation"] - baseline_z
        movement_2 = second_probe["target_preactivation"] - baseline_z
        quadratic_coefficient = (movement_2 / delta_2 - movement_1 / delta_1) / (
            delta_2 - delta_1
        )
        linear_coefficient = movement_1 / delta_1 - quadratic_coefficient * delta_1
        reason = ""
    if (
        linear_coefficient is None
        or quadratic_coefficient is None
        or not math.isfinite(linear_coefficient)
        or not math.isfinite(quadratic_coefficient)
    ):
        return {
            "status": "abstained",
            "abstention_reason": reason or "nonfinite_fit",
            "predicted_alpha": None,
            "linear_coefficient": linear_coefficient,
            "quadratic_coefficient": quadratic_coefficient,
            "probe_count": 2,
            "probes": probes,
        }
    roots: list[float] = []
    if abs(quadratic_coefficient) <= 1.0e-15:
        if linear_coefficient != 0.0:
            roots.append(margin / linear_coefficient)
    else:
        discriminant = (
            linear_coefficient * linear_coefficient
            + 4.0 * quadratic_coefficient * margin
        )
        if discriminant >= 0.0 and math.isfinite(discriminant):
            square_root = math.sqrt(discriminant)
            roots.extend(
                (
                    (-linear_coefficient - square_root) / (2.0 * quadratic_coefficient),
                    (-linear_coefficient + square_root) / (2.0 * quadratic_coefficient),
                )
            )
    admissible = sorted(
        root for root in roots if math.isfinite(root) and 0.0 < root <= 1.0
    )
    if not admissible:
        return {
            "status": "abstained",
            "abstention_reason": "no_admissible_root",
            "predicted_alpha": None,
            "linear_coefficient": linear_coefficient,
            "quadratic_coefficient": quadratic_coefficient,
            "probe_count": 2,
            "probes": probes,
        }
    root = admissible[0]
    if quadratic_coefficient != 0.0:
        derivative_zero = -linear_coefficient / (2.0 * quadratic_coefficient)
        if 0.0 < derivative_zero < root:
            return {
                "status": "abstained",
                "abstention_reason": "fitted_derivative_changes_sign_before_root",
                "predicted_alpha": None,
                "linear_coefficient": linear_coefficient,
                "quadratic_coefficient": quadratic_coefficient,
                "probe_count": 2,
                "probes": probes,
            }
    return {
        "status": "accepted",
        "abstention_reason": None,
        "predicted_alpha": root,
        "linear_coefficient": linear_coefficient,
        "quadratic_coefficient": quadratic_coefficient,
        "probe_count": 2,
        "probes": probes,
    }


def _serialized_trajectory(
    pair: dict[str, Any], points: list[dict[str, Any]]
) -> dict[str, Any]:
    baseline_z = _number(pair.get("target_preactivation"), "baseline z")
    threshold = _number(pair.get("target_threshold"), "threshold")
    margin = threshold - baseline_z
    if margin != _number(pair.get("margin"), "margin"):
        raise ValueError("serialized margin differs from threshold minus baseline z")
    q = _number(pair.get("q"), "q")
    if q <= 0.0:
        raise ValueError("Stage 1E detailed pair must have positive q")
    ordered = sorted(
        points,
        key=lambda row: _number(row.get("realized_suppression"), "realized alpha"),
    )
    if (
        not ordered
        or _number(ordered[0]["realized_suppression"], "baseline alpha") != 0.0
    ):
        raise ValueError("detailed trajectory lacks the serialized no-op")
    full_points = [
        point
        for point in ordered
        if _number(point.get("realized_suppression"), "full alpha") == 1.0
    ]
    if len(full_points) != 1:
        raise ValueError("detailed trajectory lacks one full-ablation point")
    bracket, nonmonotonic = _first_crossing_bracket(ordered)
    midpoint = (
        None
        if bracket is None
        else (bracket["lower_realized_alpha"] + bracket["upper_realized_alpha"]) / 2.0
    )
    first_probe = select_serialized_probe(pair, ordered, E1_GRID)
    second_probe = select_serialized_probe(pair, ordered, E2_GRID)
    estimator_results = {
        "E0": estimate_e0(margin=margin, q=q),
        "E1": estimate_e1(margin=margin, baseline_z=baseline_z, probe=first_probe),
        "E2": estimate_e2(
            margin=margin,
            baseline_z=baseline_z,
            first_probe=first_probe,
            second_probe=second_probe,
        ),
    }
    role = pair.get("detailed_role")
    quantization = _object(pair.get("quantization_evidence"), "quantization evidence")
    reference_eligible = (
        role in REFERENCE_ROLES
        and quantization.get("calibration_resolvable") is True
        and bracket is not None
        and not nonmonotonic
    )
    trajectory_points: list[dict[str, Any]] = []
    for point in ordered:
        z = _number(point.get("target_preactivation"), "point target z")
        point_threshold = _number(point.get("target_threshold"), "point threshold")
        if point_threshold != threshold:
            raise ValueError("trajectory threshold drifted")
        mappings = _list(point.get("requested_mappings"), "requested mappings")
        trajectory_points.append(
            {
                "requested_mappings": [
                    {
                        "requested_alpha": _number(
                            _object(mapping, "mapping").get("requested_alpha"),
                            "requested alpha",
                        ),
                        "desired_source_activation": _number(
                            _object(mapping, "mapping").get("desired_high_precision"),
                            "desired source activation",
                        ),
                        "applied_bf16_source_activation": _number(
                            _object(mapping, "mapping").get("actual_bf16_value_passed"),
                            "applied source activation",
                        ),
                        "realized_alpha": _number(
                            _object(mapping, "mapping").get("realized_suppression"),
                            "mapping realized alpha",
                        ),
                    }
                    for mapping in mappings
                ],
                "applied_bf16_source_activation": _number(
                    point.get("actual_bf16_value_passed"), "point applied source"
                ),
                "realized_alpha": _number(
                    point.get("realized_suppression"), "point realized alpha"
                ),
                "target_preactivation": z,
                "target_threshold": point_threshold,
                "strict_crossing": z > point_threshold,
            }
        )
    full_z = _number(full_points[0].get("target_preactivation"), "full z")
    return {
        "prompt_id": pair.get("prompt_id"),
        "pair_id": pair.get("pair_id"),
        "detailed_role": role,
        "source": pair.get("source"),
        "target": pair.get("target"),
        "baseline_source_activation": _number(
            pair.get("source_activation"), "source activation"
        ),
        "baseline_target_preactivation": baseline_z,
        "target_threshold": threshold,
        "margin": margin,
        "q": q,
        "zero_probe_predicted_alpha": margin / q,
        "quantization_resolvable": quantization.get("calibration_resolvable"),
        "observed_crossing_bracket": bracket,
        "observed_critical_midpoint": midpoint,
        "observed_full_ablation_crossing": full_z > threshold,
        "nonmonotonic_trajectory": nonmonotonic,
        "reference_eligible": reference_eligible,
        "trajectory_points": trajectory_points,
        "estimators": estimator_results,
    }


def _bootstrap_interval(
    rows: list[dict[str, Any]],
    prompt_ids: list[str],
    estimator: str,
    metric: str,
) -> dict[str, Any]:
    by_prompt: dict[str, list[dict[str, Any]]] = {prompt: [] for prompt in prompt_ids}
    for row in rows:
        by_prompt[str(row["prompt_id"])].append(row)
    generator = random.Random(
        int(
            hashlib.sha256(
                f"{BOOTSTRAP_SEED}|{estimator}|{metric}".encode()
            ).hexdigest(),
            16,
        )
    )
    values: list[float] = []
    for _ in range(BOOTSTRAP_COUNT):
        sampled = [prompt_ids[generator.randrange(len(prompt_ids))] for _ in prompt_ids]
        selected = [row for prompt in sampled for row in by_prompt[prompt]]
        if metric == "spearman":
            value = spearman(
                [float(row["predicted_alpha"]) for row in selected],
                [float(row["observed_midpoint"]) for row in selected],
            )
        elif metric == "median_absolute_error":
            value = (
                median(float(row["absolute_error"]) for row in selected)
                if selected
                else None
            )
        else:  # pragma: no cover - internal contract
            raise ValueError("unknown bootstrap metric")
        if value is not None and math.isfinite(value):
            values.append(value)
    return {
        "lower": _nearest_rank(values, 0.025),
        "upper": _nearest_rank(values, 0.975),
        "defined_resamples": len(values),
        "requested_resamples": BOOTSTRAP_COUNT,
    }


def _estimator_metrics(
    trajectories: list[dict[str, Any]],
    prompt_ids: list[str],
    estimator: str,
) -> dict[str, Any]:
    reference = [row for row in trajectories if row["reference_eligible"]]
    accepted_reference: list[dict[str, Any]] = []
    for row in reference:
        result = _object(row["estimators"][estimator], "estimator result")
        alpha = result.get("predicted_alpha")
        if alpha is None:
            continue
        estimate = _number(alpha, "predicted alpha")
        midpoint = _number(row["observed_critical_midpoint"], "critical midpoint")
        bracket = _object(row["observed_crossing_bracket"], "crossing bracket")
        lower = _number(bracket["lower_realized_alpha"], "bracket lower")
        upper = _number(bracket["upper_realized_alpha"], "bracket upper")
        accepted_reference.append(
            {
                "prompt_id": row["prompt_id"],
                "pair_id": row["pair_id"],
                "predicted_alpha": estimate,
                "observed_midpoint": midpoint,
                "absolute_error": abs(estimate - midpoint),
                "bracket_distance": (
                    0.0
                    if lower <= estimate <= upper
                    else min(abs(estimate - lower), abs(estimate - upper))
                ),
            }
        )
    predicted = [float(row["predicted_alpha"]) for row in accepted_reference]
    observed = [float(row["observed_midpoint"]) for row in accepted_reference]
    absolute_errors = [float(row["absolute_error"]) for row in accepted_reference]
    bracket_distances = [float(row["bracket_distance"]) for row in accepted_reference]
    confusion = {
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": 0,
        "false_negative": 0,
    }
    for row in trajectories:
        alpha = _object(row["estimators"][estimator], "estimator").get(
            "predicted_alpha"
        )
        predicted_crossing = (
            alpha is not None and 0.0 < _number(alpha, "predicted alpha") <= 1.0
        )
        observed_crossing = bool(row["observed_full_ablation_crossing"])
        key = (
            "true_positive"
            if predicted_crossing and observed_crossing
            else "false_positive"
            if predicted_crossing
            else "false_negative"
            if observed_crossing
            else "true_negative"
        )
        confusion[key] += 1
    accepted_all = sum(
        _object(row["estimators"][estimator], "estimator").get("predicted_alpha")
        is not None
        for row in trajectories
    )
    nonmonotonic = [row for row in trajectories if row["nonmonotonic_trajectory"]]
    rejected_nonmonotonic = sum(
        _object(row["estimators"][estimator], "estimator").get("predicted_alpha")
        is None
        for row in nonmonotonic
    )
    total = len(trajectories)
    reference_count = len(reference)
    eligible = len(accepted_reference)
    return {
        "reference_pair_count": reference_count,
        "eligible_pair_count": eligible,
        "coverage": eligible / reference_count if reference_count else 0.0,
        "spearman": spearman(predicted, observed),
        "median_absolute_error": median(absolute_errors) if absolute_errors else None,
        "p95_absolute_error": _nearest_rank(absolute_errors, 0.95),
        "median_bracket_distance": median(bracket_distances)
        if bracket_distances
        else None,
        "p95_bracket_distance": _nearest_rank(bracket_distances, 0.95),
        "full_ablation_crossing_classification": {
            **confusion,
            "accuracy": (confusion["true_positive"] + confusion["true_negative"])
            / total,
            "evaluated_pair_count": total,
            "abstention_treated_as_no_crossing": True,
        },
        "accepted_prediction_count_all_detailed_pairs": accepted_all,
        "abstention_count_all_detailed_pairs": total - accepted_all,
        "abstention_rate_all_detailed_pairs": (total - accepted_all) / total,
        "nonmonotonic_pair_count": len(nonmonotonic),
        "nonmonotonic_rejection_count": rejected_nonmonotonic,
        "nonmonotonic_rejection_rate": rejected_nonmonotonic / len(nonmonotonic)
        if nonmonotonic
        else None,
        "finite_intervention_calls_per_accepted_prediction": {
            "E0": 0,
            "E1": 1,
            "E2": 2,
        }[estimator],
        "prompt_bootstrap_95": {
            "spearman": _bootstrap_interval(
                accepted_reference, prompt_ids, estimator, "spearman"
            ),
            "median_absolute_error": _bootstrap_interval(
                accepted_reference,
                prompt_ids,
                estimator,
                "median_absolute_error",
            ),
        },
        "reference_pair_results": accepted_reference,
    }


def _gate(metrics: dict[str, dict[str, Any]], estimator: str) -> dict[str, Any]:
    candidate = metrics[estimator]
    baseline = metrics["E0"]
    e0_median = _number(baseline["median_absolute_error"], "E0 median error")
    e0_spearman = _number(baseline["spearman"], "E0 Spearman")
    candidate_median = candidate.get("median_absolute_error")
    candidate_spearman = candidate.get("spearman")
    checks = {
        "eligible_pairs_at_least_10": int(candidate["eligible_pair_count"]) >= 10,
        "coverage_at_least_0.60": _number(candidate["coverage"], "coverage") >= 0.60,
        "median_error_at_most_0.80_times_e0": (
            candidate_median is not None
            and _number(candidate_median, "candidate median error") <= 0.80 * e0_median
        ),
        "spearman_at_least_max_0.70_or_e0": (
            candidate_spearman is not None
            and _number(candidate_spearman, "candidate Spearman")
            >= max(0.70, e0_spearman)
        ),
        "full_ablation_classification_no_worse_than_e0": _number(
            _object(
                candidate["full_ablation_crossing_classification"],
                "candidate classification",
            )["accuracy"],
            "candidate accuracy",
        )
        >= _number(
            _object(
                baseline["full_ablation_crossing_classification"],
                "E0 classification",
            )["accuracy"],
            "E0 accuracy",
        ),
    }
    return {
        "estimator": estimator,
        "thresholds": {
            "eligible_pair_minimum": 10,
            "coverage_minimum": 0.60,
            "median_absolute_error_ratio_to_e0_maximum": 0.80,
            "spearman_minimum": max(0.70, e0_spearman),
            "full_ablation_classification_must_not_be_worse": True,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def compute_offline_analysis(
    stage1d_directory: Path,
    *,
    stage1d_validation: dict[str, Any],
) -> dict[str, Any]:
    """Recompute E0/E1/E2 from only the accepted serialized Stage 1D bundle."""

    if stage1d_validation.get("status") != "passed":
        raise ValueError("Stage 1D standalone validation did not pass")
    prediction = _object(
        read_json_strict(stage1d_directory / "prediction_manifest.json"),
        "prediction manifest",
    )
    calibration = _object(
        read_json_strict(stage1d_directory / "calibration_sweeps.json"),
        "calibration sweeps",
    )
    stage1d_summary = _object(
        read_json_strict(stage1d_directory / "benchmark_summary.json"),
        "Stage 1D summary",
    )
    if (
        prediction.get("experiment_class") != "stage1d_multiprompt_gate_benchmark"
        or calibration.get("status") != "passed"
        or stage1d_summary.get("status") != "passed"
    ):
        raise ValueError("Stage 1D evidence identity or status differs")
    prompt_rows = _list(prediction.get("prompts"), "prediction prompts")
    prompt_ids = [str(_object(prompt, "prompt").get("id")) for prompt in prompt_rows]
    pairs = {
        str(pair["pair_id"]): pair
        for prompt in prompt_rows
        for pair in _list(_object(prompt, "prompt").get("execution_pairs"), "pairs")
        if _object(pair, "pair").get("detailed_role") is not None
    }
    sweep_rows = _list(calibration.get("sweeps"), "sweeps")
    sweeps = {
        str(_object(sweep, "sweep").get("pair_id")): _list(
            _object(sweep, "sweep").get("points"), "sweep points"
        )
        for sweep in sweep_rows
    }
    if len(pairs) != 32 or set(pairs) != set(sweeps):
        raise ValueError("Stage 1D detailed positive pair set differs")
    trajectories = [
        _serialized_trajectory(
            pair, [_object(point, "point") for point in sweeps[pair_id]]
        )
        for pair_id, pair in pairs.items()
    ]
    trajectories.sort(
        key=lambda row: (
            str(row["prompt_id"]),
            ROLE_ORDER[str(row["detailed_role"])],
            str(row["pair_id"]),
        )
    )
    metrics = {
        estimator: _estimator_metrics(trajectories, prompt_ids, estimator)
        for estimator in ("E0", "E1", "E2")
    }
    stage1d_critical = _object(
        stage1d_summary.get("critical_suppression_calibration"), "Stage 1D critical"
    )
    if (
        metrics["E0"]["reference_pair_count"]
        != stage1d_critical.get("monotonic_resolvable_crossing_pair_count")
        or metrics["E0"]["spearman"] != stage1d_critical.get("spearman")
        or metrics["E0"]["median_absolute_error"]
        != stage1d_critical.get("median_midpoint_absolute_error")
    ):
        raise ValueError("E0 reconstruction differs from accepted Stage 1D metrics")
    gates = {estimator: _gate(metrics, estimator) for estimator in ("E1", "E2")}
    selected = next(
        (estimator for estimator in ("E1", "E2") if gates[estimator]["passed"]),
        None,
    )
    terminal_status = (
        TERMINAL_STATUS if selected is None else "stage1e_offline_gate_passed"
    )
    project_decision = (
        PROJECT_DECISION
        if selected is None
        else "finite_probe_selected_for_confirmatory_freeze"
    )
    phase_b_status = "not_run" if selected is None else "protocol_freeze_required"
    phase_b_reason = (
        "neither_finite_probe_estimator_passed_frozen_offline_gate"
        if selected is None
        else "selected_estimator_requires_pushed_protocol_before_model_calls"
    )
    finite_probe_status = (
        "not_supported" if selected is None else "development_selected_not_confirmed"
    )
    return {
        "schema_version": 1,
        "artifact_type": "stage1e_offline_finite_probe_analysis",
        "status": "passed",
        "terminal_status": terminal_status,
        "project_decision": project_decision,
        "experiment_class": EXPERIMENT_CLASS,
        "branch": BRANCH,
        "base_commit": BASE_COMMIT,
        "audited_stage1d_artifact_commit": AUDITED_STAGE1D_ARTIFACT_COMMIT,
        "phase": "offline_development_reanalysis_only",
        "stage1d_development_set_only": True,
        "stage1d_input_file_sha256": {
            name: _sha256(stage1d_directory / name) for name in STAGE1D_FILES
        },
        "stage1d_standalone_validation": stage1d_validation,
        "guards": {
            "model_calls": 0,
            "nnsight_runtime_loads": 0,
            "intervention_api_calls": 0,
            "ignored_temporary_outputs_read": False,
            "requested_alpha_used_for_fitting": False,
            "realized_alpha_used_for_fitting": True,
            "phase_b_started": False,
        },
        "frozen_estimators": {
            "E0": "alpha=m/q for q>0",
            "E1": "secant drive from first available serialized BF16 probe",
            "E2": "quadratic through origin and two realized-alpha probes",
            "E1_nominal_grid": list(E1_GRID),
            "E2_nominal_grid": list(E2_GRID),
            "offline_probe_matching": (
                "first nominal nonzero BF16-applied value present in the "
                "serialized trajectory"
            ),
            "E2_root_rule": (
                "smallest real root in (0,1] with no prior derivative sign change"
            ),
            "preference_order": ["E1", "E2"],
        },
        "reference_definition": {
            "roles": ["B1", "B2", "B3"],
            "requires_stage1d_quantization_resolvable": True,
            "requires_observed_crossing_bracket": True,
            "requires_monotonic_trajectory": True,
            "reference_pair_count": metrics["E0"]["reference_pair_count"],
            "all_detailed_positive_pair_count": len(trajectories),
        },
        "trajectories": trajectories,
        "estimator_metrics": metrics,
        "offline_gates": gates,
        "selected_estimator": selected,
        "phase_b": {
            "status": phase_b_status,
            "reason": phase_b_reason,
            "fresh_prompt_model_calls": 0,
            "confirmatory_intervention_calls": 0,
        },
        "claim_boundary": {
            "q_ranker_status": "retained_for_candidate_discovery",
            "finite_probe_calibration_status": finite_probe_status,
            "stage1f_behavioral_readiness": False,
            "counterfactual_behavioral_importance_result": "none",
            "mediation_result": "none",
            "official_bf16_reproduction": "pending",
            "reference_clt_reproduction": "pending",
            "paper_results_readiness": False,
        },
    }


def build_run_manifest(analysis_sha256: str) -> dict[str, Any]:
    """Build the compact terminal manifest for an offline-negative result."""

    return {
        "schema_version": 1,
        "artifact_type": "stage1e_run_manifest",
        "status": TERMINAL_STATUS,
        "project_decision": PROJECT_DECISION,
        "experiment_class": EXPERIMENT_CLASS,
        "branch": BRANCH,
        "base_commit": BASE_COMMIT,
        "audited_stage1d_artifact_commit": AUDITED_STAGE1D_ARTIFACT_COMMIT,
        "offline_analysis_sha256": analysis_sha256,
        "selected_estimator": None,
        "phase_a": {
            "status": "completed",
            "model_calls": 0,
            "intervention_api_calls": 0,
        },
        "phase_b": {
            "status": "not_run_offline_gate_failed",
            "model_calls": 0,
            "intervention_api_calls": 0,
            "canonical_attempt_count": 0,
            "scientific_retry_count": 0,
        },
        "claim_boundary": {
            "q_ranker_status": "retained_for_candidate_discovery",
            "finite_probe_calibration_status": "not_supported",
            "stage1f_behavioral_readiness": False,
            "counterfactual_behavioral_importance_result": "none",
            "mediation_result": "none",
            "official_bf16_reproduction": "pending",
            "reference_clt_reproduction": "pending",
            "paper_results_readiness": False,
        },
    }


__all__ = [
    "AUDITED_STAGE1D_ARTIFACT_COMMIT",
    "BASE_COMMIT",
    "BRANCH",
    "E1_GRID",
    "E2_GRID",
    "EXPERIMENT_CLASS",
    "OUTPUT_DIRECTORY",
    "PROJECT_DECISION",
    "STAGE1D_DIRECTORY",
    "STAGE1D_FILES",
    "TERMINAL_STATUS",
    "build_run_manifest",
    "compute_offline_analysis",
    "estimate_e0",
    "estimate_e1",
    "estimate_e2",
    "select_serialized_probe",
]
