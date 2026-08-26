#!/usr/bin/env python3
"""Run the one fresh, frozen Stage 1C source-suppression protocol.

This worker is deliberately separate from the prediction worker.  It accepts
only a prediction manifest produced and committed before intervention, checks
the commit and protocol identity, then executes each selected pair exactly
once at each deduplicated BF16 value.  It never changes a target state or
reselects a pair after observing an intervention.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cfsus.mps_telemetry import MPSTelemetrySampler  # noqa: E402
from cfsus.reproduction.artifacts import write_json_atomic  # noqa: E402
from cfsus.reproduction.small_model_mps_bf16 import (  # noqa: E402
    assert_fallback_disabled,
)
from cfsus.stage1b_runtime import (  # noqa: E402
    build_mps_bf16_replacement,
    resolve_offline_snapshots,
)
from cfsus.stage1c.analysis import (  # noqa: E402
    aggregate_analyses,
    analyze_pair,
    classify_outcome,
)
from cfsus.stage1c.config import (  # noqa: E402
    BASE_COMMIT,
    BRANCH,
    CONFIG_PATH,
    load_stage1c_config,
)
from cfsus.stage1c.intervention import (  # noqa: E402
    bisection_requested_alpha,
    plan_applied_values,
)
from cfsus.stage1c.intervention_runtime import (  # noqa: E402
    Stage1CInterventionBackend,
)
from cfsus.types import FeatureActivity, FeatureRef  # noqa: E402

FORBIDDEN_PREDICTION_KEYS = frozenset(
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
PREDICTION_MANIFEST_RELATIVE = (
    "results/stage1c_first_prospective_prediction/prediction_manifest.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / CONFIG_PATH)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--prediction-manifest-sha256", required=True)
    parser.add_argument("--pre-intervention-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emergency-output", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required file is missing or unsafe: {path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("prediction manifest must be a regular file")

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_pairs,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise RuntimeError("prediction manifest is not strict finite JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError("prediction manifest must be a JSON object")
    return value


def _verify_tracked_prediction_manifest(path: Path) -> None:
    """Require the exact committed prediction manifest, byte-for-byte."""

    expected = REPOSITORY_ROOT / PREDICTION_MANIFEST_RELATIVE
    if path.resolve() != expected.resolve() or path.is_symlink() or not path.is_file():
        raise RuntimeError(
            "canonical worker must read the tracked Stage 1C prediction manifest"
        )
    try:
        tracked = subprocess.run(
            [
                "git",
                "-C",
                str(REPOSITORY_ROOT),
                "show",
                f"HEAD:{PREDICTION_MANIFEST_RELATIVE}",
            ],
            check=True,
            capture_output=True,
            timeout=15,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise RuntimeError("tracked prediction manifest is absent from HEAD") from error
    if path.read_bytes() != tracked:
        raise RuntimeError(
            "working prediction manifest differs from the tracked HEAD bytes"
        )


def _scan_prediction_keys(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RuntimeError(f"prediction manifest key is not text at {path}")
            if key in FORBIDDEN_PREDICTION_KEYS:
                raise RuntimeError(
                    f"prediction manifest contains intervention field {path}.{key}"
                )
            _scan_prediction_keys(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_prediction_keys(item, path=f"{path}[{index}]")


def _git_run(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
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


def _git_identity(expected_head: str) -> dict[str, Any]:
    if len(expected_head) != 40 or any(
        char not in "0123456789abcdef" for char in expected_head
    ):
        raise RuntimeError("pre-intervention commit is not a lowercase SHA-1")
    branch = _git_run("branch", "--show-current")
    head = _git_run("rev-parse", "HEAD")
    if branch != BRANCH or head != expected_head:
        raise RuntimeError(
            "canonical worker is not on the exact pre-intervention branch/head"
        )
    if _git_run("status", "--porcelain"):
        raise RuntimeError("canonical intervention requires a clean worktree")
    origin_ref = f"refs/remotes/origin/{BRANCH}"
    try:
        origin_head = _git_run("rev-parse", "--verify", origin_ref)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "pre-intervention commit is not available from origin"
        ) from error
    if origin_head != expected_head:
        raise RuntimeError(
            "origin branch does not point at the pre-intervention commit"
        )
    upstream = _git_run("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    if upstream != f"origin/{BRANCH}":
        raise RuntimeError("Stage 1C branch does not track its origin branch")
    return {
        "branch": branch,
        "head": head,
        "origin_head": origin_head,
        "upstream": upstream,
        "working_tree_clean": True,
    }


def _protocol_hashes() -> dict[str, str]:
    relative_paths = (
        "configs/stage1c_first_prospective_prediction.yaml",
        "configs/stage1c_first_prospective_prediction_artifact_schema.json",
        "src/cfsus/stage1c/config.py",
        "src/cfsus/stage1c/prediction.py",
        "src/cfsus/stage1c/vjp.py",
        "src/cfsus/stage1c/runtime.py",
        "src/cfsus/stage1c/intervention.py",
        "src/cfsus/stage1c/intervention_runtime.py",
        "src/cfsus/stage1c/analysis.py",
        "scripts/stage1c/run_stage1c_prediction_worker.py",
        "scripts/stage1c/run_stage1c_intervention_worker.py",
        "scripts/stage1c/preflight_stage1c.py",
        "scripts/stage1c/run_stage1c.py",
        "scripts/stage1c/assemble_stage1c_prediction.py",
        "scripts/stage1c/assemble_stage1c_artifacts.py",
        "scripts/stage1c/validate_stage1c_artifacts.py",
    )
    return {
        relative: _sha256(REPOSITORY_ROOT / relative) for relative in relative_paths
    }


def _feature(value: Any, label: str) -> FeatureRef:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a feature object")
    try:
        result = FeatureRef(
            int(value["layer"]), int(value["position"]), int(value["feature_id"])
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"{label} is malformed") from error
    if (
        result.layer >= 18
        or result.position not in {1, 2, 3, 4, 5}
        or result.feature_id >= 16_384
    ):
        raise RuntimeError(f"{label} is outside the frozen PLT domain")
    return result


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} must be finite")
    return result


def _validate_prediction_manifest(
    manifest: dict[str, Any], config: dict[str, Any], args: argparse.Namespace
) -> dict[str, list[dict[str, Any]]]:
    _scan_prediction_keys(manifest)
    if manifest.get("artifact_type") != "stage1c_prediction_manifest":
        raise RuntimeError("prediction manifest artifact type is invalid")
    if manifest.get("status") != "prediction_frozen_ready_for_commit":
        raise RuntimeError("prediction manifest is not frozen for intervention")
    if manifest.get("base_commit") != BASE_COMMIT or manifest.get("branch") != BRANCH:
        raise RuntimeError("prediction manifest Git identity is invalid")
    if manifest.get("config_sha256") != _sha256(args.config):
        raise RuntimeError("prediction manifest config digest differs")
    schema = (
        REPOSITORY_ROOT
        / "configs/stage1c_first_prospective_prediction_artifact_schema.json"
    )
    if manifest.get("artifact_schema_sha256") != _sha256(schema):
        raise RuntimeError("prediction manifest schema digest differs")
    protocol = manifest.get("protocol_file_sha256")
    if not isinstance(protocol, dict) or protocol != _protocol_hashes():
        raise RuntimeError("frozen protocol file hashes differ from the manifest")
    runtime = manifest.get("runtime_identity")
    if (
        not isinstance(runtime, dict)
        or runtime.get("backend") != "nnsight"
        or runtime.get("device") != "mps:0"
        or runtime.get("dtype") != "torch.bfloat16"
    ):
        raise RuntimeError("prediction runtime identity is invalid")
    prompt = manifest.get("prompt")
    expected_prompt = config["prompt"]
    if (
        not isinstance(prompt, dict)
        or prompt.get("id") != expected_prompt["id"]
        or prompt.get("text") != expected_prompt["text"]
        or prompt.get("token_ids") != expected_prompt["expected_token_ids"]
    ):
        raise RuntimeError("prediction prompt identity differs from the frozen config")
    protocol_config = manifest.get("protocol")
    if not isinstance(protocol_config, dict):
        raise RuntimeError("prediction protocol is missing")
    if protocol_config.get("intervention_regime") != config["intervention"]:
        raise RuntimeError("prediction intervention regime differs from config")
    if protocol_config.get("schedule") != config["schedule"]:
        raise RuntimeError("prediction schedule differs from config")
    if config["intervention"]["canonical_attempts"] != 1:
        raise RuntimeError("canonical intervention retry count is not one")
    groups = manifest.get("selected_groups")
    if not isinstance(groups, dict) or set(groups) != {
        "primary",
        "near_boundary",
        "directional",
    }:
        raise RuntimeError("selected prediction groups are malformed")
    all_ids: set[str] = set()
    all_targets: dict[str, set[FeatureRef]] = {name: set() for name in groups}
    source_counts: Counter[FeatureRef] = Counter()
    normalized: dict[str, list[dict[str, Any]]] = {}
    for group, rows in groups.items():
        if not isinstance(rows, list):
            raise RuntimeError(f"prediction group {group} is not a list")
        normalized[group] = []
        for raw in rows:
            if not isinstance(raw, dict):
                raise RuntimeError("selected pair is not an object")
            pair_id = raw.get("pair_id")
            if (
                not isinstance(pair_id, str)
                or len(pair_id) != 64
                or any(char not in "0123456789abcdef" for char in pair_id)
            ):
                raise RuntimeError("selected pair ID is malformed")
            if pair_id in all_ids:
                raise RuntimeError("selected pair groups are not disjoint")
            all_ids.add(pair_id)
            source = _feature(raw.get("source"), f"{group}.source")
            target = _feature(raw.get("target"), f"{group}.target")
            if source.layer >= target.layer or source.position > target.position:
                raise RuntimeError("selected pair violates causal ordering")
            if target in all_targets[group]:
                raise RuntimeError("selected control group repeats a target")
            all_targets[group].add(target)
            source_counts[source] += int(group == "primary")
            for name in (
                "source_activation",
                "target_preactivation",
                "target_threshold",
                "margin",
                "targeted_response",
                "q",
                "susceptibility",
            ):
                _finite(raw.get(name), f"{group}.{name}")
            predicted_alpha = raw.get("predicted_alpha_star")
            if predicted_alpha is not None:
                _finite(predicted_alpha, f"{group}.predicted_alpha_star")
            requested = raw.get("requested_alphas")
            if not isinstance(requested, list) or not requested:
                raise RuntimeError("selected pair requested schedule is missing")
            requested_floats = tuple(
                _finite(item, "requested alpha") for item in requested
            )
            if (
                requested_floats != tuple(sorted(set(requested_floats)))
                or requested_floats[0] < 0.0
                or requested_floats[-1] > 1.0
            ):
                raise RuntimeError("selected pair requested schedule is not canonical")
            expected = {float(item) for item in config["schedule"]["coarse_alphas"]}
            alpha = None if predicted_alpha is None else float(predicted_alpha)
            if alpha is not None and 0.0 <= alpha <= 1.0:
                offset = float(config["schedule"]["alpha_hat_offset"])
                expected.update(
                    {
                        max(0.0, min(1.0, alpha - offset)),
                        alpha,
                        max(0.0, min(1.0, alpha + offset)),
                    }
                )
            if requested_floats != tuple(sorted(expected)):
                raise RuntimeError(
                    "selected pair schedule is not reproducible from the frozen "
                    "protocol"
                )
            normalized[group].append(raw)
    if any(count > 2 for count in source_counts.values()):
        raise RuntimeError("primary source diversity cap is violated")
    return normalized


def _verify_baselines(
    backend: Any, groups: dict[str, list[dict[str, Any]]]
) -> dict[tuple[int, int, int], Any]:
    features: list[FeatureRef] = []
    for rows in groups.values():
        for row in rows:
            features.extend(
                (
                    _feature(row["source"], "source"),
                    _feature(row["target"], "target"),
                )
            )
    unique = tuple(sorted(set(features)))
    states = backend.measure_states(unique)
    for rows in groups.values():
        for row in rows:
            source = _feature(row["source"], "source")
            target = _feature(row["target"], "target")
            source_state = states[source]
            target_state = states[target]
            if (
                source_state.activity is not FeatureActivity.ACTIVE
                or source_state.activation <= 0.0
            ):
                raise RuntimeError("selected source is no longer baseline active")
            if (
                target_state.activity is not FeatureActivity.INACTIVE
                or target_state.activation != 0.0
                or target_state.preactivation > target_state.threshold
            ):
                raise RuntimeError(
                    "selected target is no longer exact baseline inactive"
                )
            for field, observed in (
                ("source_activation", source_state.activation),
                ("target_preactivation", target_state.preactivation),
                ("target_threshold", target_state.threshold),
            ):
                expected = _finite(row[field], field)
                if float(observed) != expected:
                    raise RuntimeError(f"remeasured baseline differs for {field}")
            if target_state.threshold - target_state.preactivation != _finite(
                row["margin"], "margin"
            ):
                raise RuntimeError("remeasured target margin differs")
    return states


def _first_bracket(
    points: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    ordered = sorted(points, key=lambda item: float(item["realized_suppression"]))
    for left, right in pairwise(ordered):
        if not bool(left["target_active"]) and bool(right["target_active"]):
            return left, right
    return None


def _symmetric_normalized_error(reference: float, observed: float) -> float:
    denominator = abs(reference) + abs(observed)
    return 0.0 if denominator == 0.0 else 2.0 * abs(reference - observed) / denominator


def _run_pair(
    backend: Stage1CInterventionBackend,
    row: dict[str, Any],
    source_baseline: float,
    config: dict[str, Any],
    torch: Any,
    sampler: Any,
) -> dict[str, Any]:
    source_tensor = torch.tensor(
        source_baseline, device="mps", dtype=torch.bfloat16
    ).reshape(())
    requested = tuple(float(item) for item in row["requested_alphas"])
    plans = plan_applied_values(source_tensor, requested, torch)
    points: list[dict[str, Any]] = []
    for index, plan in enumerate(plans):
        with sampler.stage(f"intervention_{row['pair_id'][:12]}_grid_{index}"):
            point = backend.measure_point(
                row,
                plan,
                freeze_attention=True,
                constrained_layers=None,
                stage="coarse_and_alpha_hat_schedule",
            )
        point["representative_requested_alpha"] = plan.representative_requested_alpha
        representative = min(plan.requests, key=lambda item: item.requested_alpha)
        point["requested_alpha"] = representative.requested_alpha
        point["desired_high_precision"] = representative.desired_high_precision
        point["bisection_step"] = None
        _add_local_prediction_fields(point, row)
        points.append(point)
    bisection_steps = 0
    maximum_steps = int(config["schedule"]["maximum_bisection_steps"])
    while bisection_steps < maximum_steps:
        bracket = _first_bracket(points)
        if bracket is None:
            break
        lower, upper = bracket
        requested_midpoint = bisection_requested_alpha(
            float(lower["realized_suppression"]), float(upper["realized_suppression"])
        )
        midpoint_plan = plan_applied_values(source_tensor, (requested_midpoint,), torch)
        if len(midpoint_plan) != 1:
            raise RuntimeError("bisection produced an invalid BF16 application plan")
        plan = midpoint_plan[0]
        realized = float(plan.realized_suppression)
        existing = {float(item["realized_suppression"]) for item in points}
        if realized in existing or not (
            float(lower["realized_suppression"])
            < realized
            < float(upper["realized_suppression"])
        ):
            break
        with sampler.stage(
            f"intervention_{row['pair_id'][:12]}_bisect_{bisection_steps}"
        ):
            point = backend.measure_point(
                row,
                plan,
                freeze_attention=True,
                constrained_layers=None,
                stage="deterministic_realized_suppression_bisection",
            )
        point["representative_requested_alpha"] = plan.representative_requested_alpha
        representative = min(plan.requests, key=lambda item: item.requested_alpha)
        point["requested_alpha"] = representative.requested_alpha
        point["desired_high_precision"] = representative.desired_high_precision
        point["bisection_step"] = bisection_steps
        _add_local_prediction_fields(point, row)
        points.append(point)
        bisection_steps += 1
    ordered = sorted(points, key=lambda item: float(item["realized_suppression"]))
    realized = [float(item["realized_suppression"]) for item in ordered]
    if realized != sorted(set(realized)) or realized[0] != 0.0 or realized[-1] != 1.0:
        raise RuntimeError(
            "canonical sweep lacks unique zero/full realized suppression"
        )
    if bool(ordered[0]["target_active"]):
        raise RuntimeError("selected target was active at baseline intervention point")
    return {
        "pair_id": row["pair_id"],
        "group": row["group"],
        "source": row["source"],
        "target": row["target"],
        "target_preactivation": row["target_preactivation"],
        "target_threshold": row["target_threshold"],
        "q": row["q"],
        "predicted_alpha_star": row.get("predicted_alpha_star"),
        "predicted_status": row.get("predicted_status"),
        "baseline_source_activation": source_baseline,
        "point_count": len(ordered),
        "bisection_step_count": bisection_steps,
        "points": ordered,
    }


def _add_local_prediction_fields(point: dict[str, Any], row: dict[str, Any]) -> None:
    """Record the frozen local prediction beside every observed scalar point."""

    baseline_z = _finite(row["target_preactivation"], "target_preactivation")
    threshold = _finite(row["target_threshold"], "target_threshold")
    q = _finite(row["q"], "q")
    realized = _finite(point["realized_suppression"], "realized_suppression")
    observed = _finite(point["target_preactivation"], "observed target preactivation")
    predicted_z = baseline_z + realized * q
    if not math.isfinite(predicted_z):
        raise RuntimeError("local predicted target preactivation is non-finite")
    predicted_active = predicted_z > threshold
    predicted_activation = predicted_z if predicted_active else 0.0
    point.update(
        {
            "predicted_target_preactivation": predicted_z,
            "predicted_target_activation": predicted_activation,
            "predicted_target_active": predicted_active,
            "target_preactivation_absolute_error": abs(observed - predicted_z),
            "target_preactivation_symmetric_normalized_error": (
                _symmetric_normalized_error(predicted_z, observed)
            ),
        }
    )


def _pair_analysis(
    row: dict[str, Any], sweep: dict[str, Any], analysis: dict[str, Any]
) -> dict[str, Any]:
    """Materialize the frozen pair-level crossing/locality summary."""

    points = sorted(
        sweep["points"], key=lambda item: float(item["realized_suppression"])
    )
    baseline_z = _finite(row["target_preactivation"], "baseline target z")
    q = _finite(row["q"], "q")
    errors: list[float] = []
    signs: list[bool] = []
    for point in points[1:]:
        realized = _finite(point["realized_suppression"], "realized suppression")
        predicted_delta = realized * q
        observed_delta = _finite(point["target_preactivation"], "target z") - baseline_z
        errors.append(_symmetric_normalized_error(predicted_delta, observed_delta))
        signs.append(
            observed_delta > 0.0
            if predicted_delta > 0.0
            else observed_delta < 0.0
            if predicted_delta < 0.0
            else observed_delta == 0.0
        )
    bracket = None
    for left, right in pairwise(points):
        if not bool(left["target_active"]) and bool(right["target_active"]):
            bracket = {
                "lower_realized_suppression": float(left["realized_suppression"]),
                "upper_realized_suppression": float(right["realized_suppression"]),
            }
            break
    alpha = row.get("predicted_alpha_star")
    distance = None
    if alpha is not None and bracket is not None:
        alpha_value = _finite(alpha, "predicted alpha")
        lower = bracket["lower_realized_suppression"]
        upper = bracket["upper_realized_suppression"]
        distance = (
            0.0
            if lower <= alpha_value <= upper
            else min(abs(alpha_value - lower), abs(alpha_value - upper))
        )
    full = next(
        point for point in points if float(point["realized_suppression"]) == 1.0
    )
    sign_agreement = sum(signs) / len(signs) if signs else None
    median_error = median(errors) if errors else None
    p95_error = (
        sorted(errors)[max(1, math.ceil(0.95 * len(errors))) - 1] if errors else None
    )
    local_pass = (
        len(errors) >= int(analysis["minimum_nonzero_points"])
        and sign_agreement is not None
        and sign_agreement >= float(analysis["movement_sign_agreement_minimum"])
        and median_error is not None
        and median_error <= float(analysis["median_movement_sne_maximum"])
        and p95_error is not None
        and p95_error <= float(analysis["p95_movement_sne_maximum"])
        and distance is not None
        and distance <= float(analysis["critical_bracket_distance_maximum"])
    )
    group = str(row["group"])
    full_crossing = bool(full["target_active"])
    active_seen = False
    nonmonotonic = False
    for point in points:
        if bool(point["target_active"]):
            active_seen = True
        elif active_seen:
            nonmonotonic = True
    return {
        "pair_id": row["pair_id"],
        "group": group,
        "observed_full_ablation_crossing": full_crossing,
        "observed_critical_bracket": bracket,
        "critical_bracket_distance": distance,
        "movement_sign_agreement": sign_agreement,
        "median_movement_symmetric_normalized_error": median_error,
        "p95_movement_symmetric_normalized_error": p95_error,
        "local_calibration_passed": local_pass,
        "supporting_primary": (
            group == "primary"
            and full_crossing
            and float(full["target_preactivation"]) - baseline_z > 0.0
            and local_pass
        ),
        "directional_control_violation": (
            group == "directional"
            and float(full["target_preactivation"]) - baseline_z > 0.0
        ),
        "near_boundary_control_crossing": (
            group == "near_boundary"
            and any(bool(point["target_active"]) for point in points)
        ),
        "nonmonotonic_gate": nonmonotonic,
    }


def _artifact_bundle(
    sweeps: list[dict[str, Any]], config: dict[str, Any], telemetry: Any
) -> tuple[dict[str, Any], str]:
    analyses = [
        analyze_pair(
            {
                "pair_id": sweep["pair_id"],
                "group": sweep["group"],
                "target_preactivation": sweep["target_preactivation"],
                "target_threshold": sweep["target_threshold"],
                "q": sweep["q"],
                "predicted_alpha_star": sweep.get("predicted_alpha_star"),
                "predicted_status": sweep["predicted_status"],
            },
            sweep["points"],
            config["analysis"],
        )
        for sweep in sweeps
    ]
    aggregates = aggregate_analyses(analyses)
    outcome = classify_outcome(analyses).value
    local_summary = {
        "artifact_type": "stage1c_local_linearity_summary",
        "pairs": analyses,
        "aggregate_metrics": aggregates,
        "scientific_outcome": outcome,
    }
    artifact_map = {
        "intervention_sweeps": {
            "artifact_type": "stage1c_intervention_sweeps",
            "pairs": sweeps,
        },
        "crossing_summary": {
            "artifact_type": "stage1c_crossing_summary",
            "pairs": analyses,
            "aggregate_metrics": aggregates,
            "scientific_outcome": outcome,
        },
        "local_linearity_summary": local_summary,
        "memory_timing_summary": {
            "artifact_type": "stage1c_memory_timing_summary",
            "telemetry": telemetry,
        },
        "attempts": {
            "artifact_type": "stage1c_attempts",
            "attempt_count": 1 if sweeps else 0,
            "scientific_retry_count": 0,
            "intervention_required": bool(sweeps),
        },
    }
    return artifact_map, outcome


def _verify_assets(cache: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            str(
                REPOSITORY_ROOT
                / "scripts/stage1a/verify_small_model_mps_bf16_assets.py"
            ),
            "--hf-cache",
            str(cache),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
        env={
            **{
                key: value
                for key, value in os.environ.items()
                if key
                not in {
                    "HF_TOKEN",
                    "HUGGING_FACE_HUB_TOKEN",
                    "GITHUB_TOKEN",
                    "GH_TOKEN",
                    "PYTORCH_ENABLE_MPS_FALLBACK",
                }
            },
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )
    record = json.loads(result.stdout)
    if record.get("status") != "verified":
        raise RuntimeError("immutable asset verification failed")
    return dict(record)


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    config = load_stage1c_config(args.config)
    if config["phase"] != "prediction_only_open":
        raise RuntimeError(
            "canonical worker config phase changed after prediction freeze"
        )
    assert_fallback_disabled()
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
    ):
        raise RuntimeError("canonical worker requires offline mode")
    expected_head = args.pre_intervention_commit
    git_start = _git_identity(expected_head)
    _verify_tracked_prediction_manifest(args.prediction_manifest)
    manifest = _strict_json(args.prediction_manifest)
    if _sha256(args.prediction_manifest) != args.prediction_manifest_sha256:
        raise RuntimeError(
            "prediction manifest SHA-256 does not match the supplied digest"
        )
    groups = _validate_prediction_manifest(manifest, config, args)

    # The protocol explicitly terminates at the baseline-only phase when no
    # primary prediction exists.  Controls must not be used to manufacture an
    # intervention run in that case.  Still emit a machine-readable, valid
    # worker result so a supervisor can record the terminal scientific class.
    if not groups["primary"]:
        artifact_map, _ = _artifact_bundle([], config, None)
        return {
            "schema_version": 1,
            "artifact_type": "stage1c_intervention_worker",
            "status": "passed",
            "scientific_outcome": "no_eligible_pairs",
            "attempt_count": 0,
            "git": _git_identity(args.pre_intervention_commit),
            "pre_intervention_commit": args.pre_intervention_commit,
            "prediction_manifest_sha256": args.prediction_manifest_sha256,
            "prediction_manifest_verification": {
                "status": "passed",
                "tracked_manifest": True,
                "protocol_hashes_match": True,
                "prediction_outcome_fields_absent": True,
            },
            "canonical_source_suppression_api_calls": 0,
            "scientific_retry_count": 0,
            "intervention_skipped": True,
            "sweeps": [],
            "intervention_artifacts": artifact_map,
            "telemetry": None,
        }

    import nnsight  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    import transformers  # type: ignore[import-not-found]

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("native MPS is unavailable")
    if torch.is_autocast_enabled():
        raise RuntimeError("outer autocast must be disabled")
    asset_manifest = _verify_assets(args.hf_cache)
    model_snapshot, transcoder_snapshot = resolve_offline_snapshots(
        args.hf_cache, REPOSITORY_ROOT
    )
    sampler = MPSTelemetrySampler(torch, config["safety_limits"], args.emergency_output)
    sampler_finished = False
    model: Any = None
    sweeps: list[dict[str, Any]] = []
    try:
        with sampler.stage("replacement_runtime_loading"):
            model, module_guard = build_mps_bf16_replacement(
                model_snapshot, transcoder_snapshot, torch
            )
            prompt = str(config["prompt"]["text"])
            tokens = model.ensure_tokenized(prompt)
            token_ids = [int(item) for item in tokens.detach().cpu().tolist()]
            if token_ids != config["prompt"]["expected_token_ids"]:
                raise RuntimeError("prompt token identity changed")
            from cfsus.stage1c.runtime import Stage1CPredictionBackend

            prediction_backend = Stage1CPredictionBackend(
                model,
                prompt=prompt,
                prompt_id=str(config["prompt"]["id"]),
                torch=torch,
            )
        with sampler.stage("remeasure_selected_baselines"):
            states = _verify_baselines(prediction_backend, groups)
        intervention_backend = Stage1CInterventionBackend(
            model, prompt=prompt, torch=torch
        )
        for group in ("primary", "near_boundary", "directional"):
            for row in groups[group]:
                source = _feature(row["source"], "source")
                sweeps.append(
                    _run_pair(
                        intervention_backend,
                        row,
                        float(states[source].activation),
                        config,
                        torch,
                        sampler,
                    )
                )
        telemetry = sampler.finish()
        sampler_finished = True
        if telemetry["violations"] or telemetry["telemetry_failures"]:
            raise RuntimeError(
                "canonical intervention telemetry contains a safety failure"
            )
        git_end = _git_identity(expected_head)
        if git_end != git_start:
            raise RuntimeError("canonical worktree identity changed during execution")
        intervention_artifacts, scientific_outcome = _artifact_bundle(
            sweeps, config, telemetry
        )
        return {
            "schema_version": 1,
            "artifact_type": "stage1c_intervention_worker",
            "status": "passed",
            "attempt_count": 1,
            "git": git_end,
            "pre_intervention_commit": expected_head,
            "prediction_manifest_sha256": args.prediction_manifest_sha256,
            "prediction_manifest_verification": {
                "status": "passed",
                "tracked_manifest": True,
                "protocol_hashes_match": True,
                "prediction_outcome_fields_absent": True,
            },
            "asset_manifest": asset_manifest,
            "environment": {
                "machine": platform.machine(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "nnsight": nnsight.__version__,
                "transformers": transformers.__version__,
                "mps_built": torch.backends.mps.is_built(),
                "mps_available": torch.backends.mps.is_available(),
                "fallback_variable_present": (
                    "PYTORCH_ENABLE_MPS_FALLBACK" in os.environ
                ),
                "outer_autocast_enabled": torch.is_autocast_enabled(),
            },
            "module_guard": module_guard,
            "intervention_regime": dict(config["intervention"]),
            "remeasured_selected_baseline_count": len(states),
            "canonical_source_suppression_api_calls": sum(
                int(item["point_count"]) for item in sweeps
            ),
            "scientific_retry_count": 0,
            "sweeps": sweeps,
            "intervention_artifacts": intervention_artifacts,
            "scientific_outcome": scientific_outcome,
            "telemetry": telemetry,
        }
    finally:
        if not sampler_finished:
            with contextlib.suppress(Exception):
                sampler.finish()
        sweeps.clear()
        if model is not None:
            del model
        gc.collect()
        if "torch" in locals():
            torch.mps.empty_cache()


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists() or args.emergency_output.exists():
        raise RuntimeError("intervention worker output paths must be new")
    if not args.output.parent.is_dir() or not args.emergency_output.parent.is_dir():
        raise RuntimeError("intervention worker output parents must exist")
    result = _execute(args)
    write_json_atomic(args.output, result)
    print(
        json.dumps(
            {"status": result["status"], "phase": "intervention"},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
