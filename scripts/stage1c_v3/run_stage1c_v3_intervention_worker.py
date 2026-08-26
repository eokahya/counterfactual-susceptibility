#!/usr/bin/env python3
"""Execute exactly one frozen Stage 1C-v3 intervention attempt.

The worker accepts only the committed v3 prediction manifest.  It performs
all identity checks before importing model/intervention runtime packages and
constructs a recursively detached result before the cleanup ``finally``
block can clear live sweep state.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cfsus.mps_telemetry import MPSTelemetrySampler  # noqa: E402
from cfsus.reproduction.small_model_mps_bf16 import (  # noqa: E402
    assert_fallback_disabled,
)
from cfsus.stage1b_runtime import (  # noqa: E402
    build_mps_bf16_replacement,
    resolve_offline_snapshots,
)
from cfsus.stage1c_v3.analysis import (  # noqa: E402
    aggregate_analyses,
    analyze_pair,
    classify_outcome,
)
from cfsus.stage1c_v3.config import (  # noqa: E402
    BASE_COMMIT,
    BRANCH,
    CONFIG_PATH,
    EXPERIMENT_CLASS,
    SCHEMA_PATH,
    load_stage1c_v3_config,
    selected_positions_for_token_ids,
    validate_prompt_token_ids,
)
from cfsus.stage1c_v3.execution_journal import (  # noqa: E402
    CanonicalExecutionJournal,
)
from cfsus.stage1c_v3.historical import (  # noqa: E402
    DENYLIST_PATH,
    HISTORICAL_MANIFEST_FREEZE_COMMIT,
    HISTORICAL_MANIFEST_PATH,
    assert_expected_prompt_derivation,
    endpoint_overlap_category,
    exact_pair_key,
    exact_pair_record,
    load_authenticated_historical_metadata,
)
from cfsus.stage1c_v3.intervention import (  # noqa: E402
    bisection_requested_alpha,
    plan_applied_values,
)
from cfsus.stage1c_v3.intervention_runtime import (  # noqa: E402
    Stage1CInterventionBackend,
    Stage1CInterventionBackendProtocol,
)
from cfsus.stage1c_v3.prediction import canonical_v3_pair_id  # noqa: E402
from cfsus.stage1c_v3.serialization import (  # noqa: E402
    read_json_strict,
    write_json_new,
)
from cfsus.stage1c_v3.worker_result import (  # noqa: E402
    build_detached_worker_result,
)
from cfsus.types import FeatureActivity, FeatureRef  # noqa: E402

PREDICTION_MANIFEST_RELATIVE = (
    "results/stage1c_v3_preregistered_prospective_prediction/prediction_manifest.json"
)
RUNTIME_FINGERPRINT = (
    "gemma3-270m@9b0cfec892e2/plt@fada11860ac1/"
    "circuit-tracer@8f1e2438df61/nnsight/mps/bf16/stage1c-v3"
)
PREDICTION_FREEZE_COMMIT = "10f7234a036562e9337514fc085415a017e99102"
PREDICTION_MANIFEST_SHA256 = (
    "b2c489317852a2f54d50db783abc17dfdc08590353b0473dbab01ec3d04574cc"
)
PREDICTION_PROTOCOL_MAP_SHA256 = (
    "9ac9c59aeef736ed495d688bdeb6866250b089a191430c9d175661c702ec3db8"
)
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / CONFIG_PATH)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--prediction-manifest-sha256", required=True)
    parser.add_argument("--pre-intervention-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emergency-output", type=Path, required=True)
    parser.add_argument("--attempt-lock", type=Path, required=True)
    parser.add_argument("--point-journal", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    try:
        info = path.lstat()
    except OSError as error:
        raise RuntimeError(f"required protocol file is unreadable: {path}") from error
    if path.is_symlink() or not path.is_file() or info.st_nlink != 1:
        raise RuntimeError(f"required protocol file is not a single-link file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prediction_protocol_map_digest(value: Any) -> str:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in value.items()
    ):
        raise RuntimeError("prediction protocol hash map is malformed")
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} must be finite")
    return result


def _feature(value: Any, label: str, *, token_count: int) -> FeatureRef:
    if not isinstance(value, dict) or set(value) != {"layer", "position", "feature_id"}:
        raise RuntimeError(f"{label} feature record is malformed")
    values = (value["layer"], value["position"], value["feature_id"])
    if any(isinstance(item, bool) or not isinstance(item, int) for item in values):
        raise RuntimeError(f"{label} feature record is malformed")
    try:
        result = FeatureRef(*values)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} feature record is malformed") from error
    if not (0 <= result.layer < 18 and 1 <= result.position < token_count):
        raise RuntimeError(f"{label} is outside the selected v3 prompt positions")
    if not 0 <= result.feature_id < 16_384:
        raise RuntimeError(f"{label} is outside the frozen feature domain")
    return result


def _scan_prediction_keys(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RuntimeError(f"prediction manifest key is not text at {path}")
            normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
            if normalized in FORBIDDEN_PREDICTION_KEYS:
                raise RuntimeError(
                    f"prediction manifest contains intervention field {path}.{key}"
                )
            _scan_prediction_keys(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_prediction_keys(item, path=f"{path}[{index}]")


def _verify_tracked_prediction_manifest(path: Path) -> None:
    expected = REPOSITORY_ROOT / PREDICTION_MANIFEST_RELATIVE
    if path.resolve() != expected.resolve() or path.is_symlink() or not path.is_file():
        raise RuntimeError("worker must read the tracked v3 prediction manifest")
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
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "tracked v3 prediction manifest is absent from HEAD"
        ) from error
    if path.read_bytes() != tracked:
        raise RuntimeError(
            "working prediction manifest differs from tracked HEAD bytes"
        )
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PREDICTION_MANIFEST_SHA256:
        raise RuntimeError("tracked prediction manifest digest differs")
    try:
        frozen = subprocess.run(
            [
                "git",
                "-C",
                str(REPOSITORY_ROOT),
                "show",
                f"{PREDICTION_FREEZE_COMMIT}:{PREDICTION_MANIFEST_RELATIVE}",
            ],
            check=True,
            capture_output=True,
            timeout=15,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("prediction-freeze manifest is unavailable") from error
    if raw != frozen:
        raise RuntimeError(
            "tracked manifest is not byte-identical to prediction freeze"
        )


def _validate_prediction_manifest(
    manifest: dict[str, Any], config: dict[str, Any], args: argparse.Namespace
) -> dict[str, list[dict[str, Any]]]:
    _scan_prediction_keys(manifest)
    prompt_derivation = assert_expected_prompt_derivation()
    historical = load_authenticated_historical_metadata(REPOSITORY_ROOT)
    denylist = frozenset(historical.exact_pairs)
    if (
        manifest.get("schema_version") != 3
        or manifest.get("artifact_type") != "stage1c_v3_prediction_manifest"
        or manifest.get("experiment_class") != EXPERIMENT_CLASS
        or manifest.get("status") != "prediction_frozen_ready_for_commit"
    ):
        raise RuntimeError("prediction manifest v3 identity or status is invalid")
    if manifest.get("base_commit") != BASE_COMMIT or manifest.get("branch") != BRANCH:
        raise RuntimeError("prediction manifest Git identity is invalid")
    if manifest.get("config_sha256") != _sha256(args.config):
        raise RuntimeError("prediction manifest config digest differs")
    if manifest.get("artifact_schema_sha256") != _sha256(REPOSITORY_ROOT / SCHEMA_PATH):
        raise RuntimeError("prediction manifest schema digest differs")
    if (
        _prediction_protocol_map_digest(manifest.get("protocol_file_sha256"))
        != PREDICTION_PROTOCOL_MAP_SHA256
    ):
        raise RuntimeError(
            "prediction protocol hashes differ from the authenticated freeze commit"
        )
    if manifest.get("prompt_derivation") != {
        "algorithm": config["prompt_derivation"]["algorithm"],
        "base_commit": config["prompt_derivation"]["base_commit"],
        "salt": config["prompt_derivation"]["salt"],
        "message": prompt_derivation.message,
        "sha256_hex": prompt_derivation.sha256_hex,
        "index": prompt_derivation.index,
        "prompt": prompt_derivation.prompt,
        "prompt_id": prompt_derivation.prompt_id,
        "pool": list(config["prompt_derivation"]["pool"]),
    }:
        raise RuntimeError("prediction prompt derivation differs")
    if manifest.get("historical_independence") != {
        "source_manifest_path": HISTORICAL_MANIFEST_PATH.as_posix(),
        "source_manifest_sha256": historical.source_manifest_sha256,
        "source_manifest_git_blob_sha1": (historical.source_manifest_git_blob_sha1),
        "source_manifest_freeze_commit": HISTORICAL_MANIFEST_FREEZE_COMMIT,
        "denylist_path": DENYLIST_PATH.as_posix(),
        "denylist_sha256": historical.denylist_sha256,
        "exact_pair_count": len(historical.exact_pairs),
        "historical_endpoint_count": len(historical.historical_endpoints),
        "mask_applied_before_ranking": True,
        "endpoint_overlap_policy": "audit_only",
        "historical_intervention_outcome_read": False,
        "v2_temporary_baseline_artifact_read": False,
    }:
        raise RuntimeError("prediction historical-independence record differs")
    runtime = manifest.get("runtime_identity")
    if not isinstance(runtime, dict) or {
        runtime.get("backend"),
        runtime.get("device"),
        runtime.get("dtype"),
    } != {"nnsight", "mps:0", "torch.bfloat16"}:
        raise RuntimeError("prediction runtime identity is invalid")
    prompt = manifest.get("prompt")
    expected_prompt = config["prompt"]
    if (
        not isinstance(prompt, dict)
        or prompt.get("id") != expected_prompt["id"]
        or prompt.get("text") != expected_prompt["text"]
    ):
        raise RuntimeError("prediction prompt identity differs from config")
    token_ids = prompt.get("token_ids")
    if not isinstance(token_ids, list):
        raise RuntimeError("prediction prompt token IDs are missing")
    try:
        validate_prompt_token_ids(config, token_ids)
    except Exception as error:
        raise RuntimeError("prediction prompt token identity differs") from error
    token_count = len(token_ids)
    scanner = manifest.get("protocol", {}).get("scanner")
    if not isinstance(scanner, dict) or scanner.get(
        "selected_positions"
    ) != selected_positions_for_token_ids(token_ids):
        raise RuntimeError(
            "prediction scanner positions are not dynamic non-BOS positions"
        )
    protocol = manifest.get("protocol")
    if (
        not isinstance(protocol, dict)
        or protocol.get("schedule") != config["schedule"]
        or protocol.get("intervention_regime") != config["intervention"]
    ):
        raise RuntimeError("prediction protocol differs from frozen v3 config")
    if int(config["intervention"]["canonical_attempts"]) != 1:
        raise RuntimeError("v3 canonical intervention attempts must equal one")
    if int(config["intervention"]["scientific_retries"]) != 0:
        raise RuntimeError("v3 scientific retries must equal zero")
    groups = manifest.get("selected_groups")
    expected_groups = {"primary", "near_boundary", "directional"}
    if not isinstance(groups, dict) or set(groups) != expected_groups:
        raise RuntimeError("selected v3 groups are malformed")
    normalized: dict[str, list[dict[str, Any]]] = {}
    all_ids: set[str] = set()
    group_targets: dict[str, set[FeatureRef]] = {
        name: set() for name in expected_groups
    }
    source_counts: Counter[FeatureRef] = Counter()
    for group in ("primary", "near_boundary", "directional"):
        rows = groups[group]
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
                or any(item not in "0123456789abcdef" for item in pair_id)
            ):
                raise RuntimeError("selected pair ID is malformed")
            if pair_id in all_ids:
                raise RuntimeError("selected groups are not disjoint")
            all_ids.add(pair_id)
            source = _feature(
                raw.get("source"), f"{group}.source", token_count=token_count
            )
            target = _feature(
                raw.get("target"), f"{group}.target", token_count=token_count
            )
            pair_key = exact_pair_key(source, target)
            if pair_key in denylist:
                raise RuntimeError("historical exact pair entered the v3 manifest")
            if raw.get("exact_pair_key") != exact_pair_record(pair_key):
                raise RuntimeError("serialized exact pair key differs")
            expected_overlap = endpoint_overlap_category(
                source,
                target,
                denylist=denylist,
                endpoints=historical.historical_endpoints,
            ).value
            if raw.get("endpoint_overlap_category") != expected_overlap:
                raise RuntimeError("endpoint-overlap audit category differs")
            if pair_id != canonical_v3_pair_id(
                source=source,
                target=target,
                runtime_fingerprint=RUNTIME_FINGERPRINT,
                prompt_id=str(expected_prompt["id"]),
                seed=str(config["scoring"]["pair_seed"]),
            ):
                raise RuntimeError(
                    "selected pair ID differs from the frozen v3 identity"
                )
            if source.layer >= target.layer or source.position > target.position:
                raise RuntimeError("selected pair violates causal ordering")
            if target in group_targets[group]:
                raise RuntimeError("selected group repeats a target")
            group_targets[group].add(target)
            if group == "primary":
                source_counts[source] += 1
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
            alpha = raw.get("predicted_alpha_star")
            if alpha is not None:
                _finite(alpha, f"{group}.predicted_alpha_star")
            requested = raw.get("requested_alphas")
            if not isinstance(requested, list) or tuple(
                _finite(item, "requested alpha") for item in requested
            ) != tuple(
                sorted({_finite(item, "requested alpha") for item in requested})
            ):
                raise RuntimeError("selected pair requested schedule is not canonical")
            expected_schedule = {
                float(item) for item in config["schedule"]["coarse_alphas"]
            }
            if alpha is not None and 0.0 <= float(alpha) <= 1.0:
                offset = float(config["schedule"]["alpha_hat_offset"])
                expected_schedule.update(
                    {
                        max(0.0, min(1.0, float(alpha) - offset)),
                        float(alpha),
                        max(0.0, min(1.0, float(alpha) + offset)),
                    }
                )
            if tuple(_finite(item, "requested alpha") for item in requested) != tuple(
                sorted(expected_schedule)
            ):
                raise RuntimeError(
                    "selected pair schedule differs from frozen protocol"
                )
            normalized[group].append(raw)
    if len(normalized["primary"]) > int(config["selection"]["primary_maximum"]):
        raise RuntimeError("primary selection exceeds frozen maximum")
    if not normalized["primary"] and (
        normalized["near_boundary"] or normalized["directional"]
    ):
        raise RuntimeError("no-primary manifest must not retain control pairs")
    if any(
        count > int(config["selection"]["maximum_primary_per_source"])
        for count in source_counts.values()
    ):
        raise RuntimeError("primary source diversity cap is violated")
    return normalized


def _verify_baselines(
    backend: Stage1CInterventionBackendProtocol,
    groups: dict[str, list[dict[str, Any]]],
    *,
    token_count: int,
) -> dict[FeatureRef, Any]:
    features: list[FeatureRef] = []
    for rows in groups.values():
        for row in rows:
            features.extend(
                (
                    _feature(row["source"], "source", token_count=token_count),
                    _feature(row["target"], "target", token_count=token_count),
                )
            )
    states = backend.measure_states(tuple(sorted(set(features))))
    for rows in groups.values():
        for row in rows:
            source = FeatureRef(**row["source"])
            target = FeatureRef(**row["target"])
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
            for name, observed in (
                ("source_activation", source_state.activation),
                ("target_preactivation", target_state.preactivation),
                ("target_threshold", target_state.threshold),
            ):
                if float(observed) != _finite(row[name], name):
                    raise RuntimeError(f"remeasured baseline differs for {name}")
            if target_state.threshold - target_state.preactivation != _finite(
                row["margin"], "margin"
            ):
                raise RuntimeError("remeasured target margin differs")
    return cast(dict[FeatureRef, Any], states)


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


def _add_local_prediction_fields(point: dict[str, Any], row: dict[str, Any]) -> None:
    baseline = _finite(row["target_preactivation"], "baseline target preactivation")
    threshold = _finite(row["target_threshold"], "target threshold")
    q = _finite(row["q"], "q")
    realized = _finite(point["realized_suppression"], "realized suppression")
    observed = _finite(point["target_preactivation"], "observed target preactivation")
    predicted = baseline + realized * q
    if not math.isfinite(predicted):
        raise RuntimeError("local predicted target preactivation is non-finite")
    point.update(
        {
            "predicted_target_preactivation": predicted,
            "predicted_target_activation": predicted if predicted > threshold else 0.0,
            "predicted_target_active": predicted > threshold,
            "target_preactivation_absolute_error": abs(observed - predicted),
            "target_preactivation_symmetric_normalized_error": (
                _symmetric_normalized_error(predicted, observed)
            ),
        }
    )


def _bind_point_evidence(
    point: dict[str, Any], row: dict[str, Any], *, telemetry_stage: str
) -> None:
    """Make each point independently attributable and recomputable."""

    point.update(
        {
            "pair_id": row["pair_id"],
            "group": row["group"],
            "source": row["source"],
            "target": row["target"],
            "exact_pair_key": row["exact_pair_key"],
            "endpoint_overlap_category": row["endpoint_overlap_category"],
            "baseline_source_activation": row["source_activation"],
            "baseline_target_preactivation": row["target_preactivation"],
            "targeted_response": row["targeted_response"],
            "q": row["q"],
            "predicted_alpha_star": row.get("predicted_alpha_star"),
            "predicted_status": row["predicted_status"],
            "memory_stage_identity": telemetry_stage,
            "finite_value_checks_passed": True,
        }
    )


def _run_pair(
    backend: Stage1CInterventionBackend,
    row: dict[str, Any],
    source_baseline: float,
    config: dict[str, Any],
    torch: Any,
    sampler: Any,
    journal: CanonicalExecutionJournal,
) -> dict[str, Any]:
    source_tensor = torch.tensor(
        source_baseline, device="mps", dtype=torch.bfloat16
    ).reshape(())
    requested = tuple(float(item) for item in row["requested_alphas"])
    plans = plan_applied_values(source_tensor, requested, torch)
    points: list[dict[str, Any]] = []
    for index, plan in enumerate(plans):
        telemetry_stage = f"intervention_{row['pair_id'][:12]}_grid_{index}"
        with sampler.stage(telemetry_stage):
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
        _bind_point_evidence(point, row, telemetry_stage=telemetry_stage)
        _add_local_prediction_fields(point, row)
        journal.append_completed_point(point)
        points.append(point)
    bisection_steps = 0
    while bisection_steps < int(config["schedule"]["maximum_bisection_steps"]):
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
        if realized in existing or not float(
            lower["realized_suppression"]
        ) < realized < float(upper["realized_suppression"]):
            break
        telemetry_stage = f"intervention_{row['pair_id'][:12]}_bisect_{bisection_steps}"
        with sampler.stage(telemetry_stage):
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
        _bind_point_evidence(point, row, telemetry_stage=telemetry_stage)
        _add_local_prediction_fields(point, row)
        journal.append_completed_point(point)
        points.append(point)
        bisection_steps += 1
    ordered = sorted(points, key=lambda item: float(item["realized_suppression"]))
    realized_values = [float(item["realized_suppression"]) for item in ordered]
    if (
        not ordered
        or realized_values != sorted(set(realized_values))
        or realized_values[0] != 0.0
        or realized_values[-1] != 1.0
        or bool(ordered[0]["target_active"])
    ):
        raise RuntimeError("canonical sweep lacks exact zero/full inactive baseline")
    if (
        float(ordered[0]["target_preactivation"])
        != _finite(row["target_preactivation"], "frozen target preactivation")
        or float(ordered[0]["target_threshold"])
        != _finite(row["target_threshold"], "frozen target threshold")
        or float(ordered[0]["actual_bf16_value_passed"]) != source_baseline
    ):
        raise RuntimeError("zero-suppression point differs from frozen baseline")
    return {
        "pair_id": row["pair_id"],
        "group": row["group"],
        "source": row["source"],
        "target": row["target"],
        "exact_pair_key": row["exact_pair_key"],
        "endpoint_overlap_category": row["endpoint_overlap_category"],
        "source_activation": row["source_activation"],
        "targeted_response": row["targeted_response"],
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


def _artifact_bundle(
    sweeps: list[dict[str, Any]], config: dict[str, Any], telemetry: Any
) -> tuple[dict[str, Any], str]:
    analyses = [
        analyze_pair(
            {
                "pair_id": item["pair_id"],
                "group": item["group"],
                "exact_pair_key": item["exact_pair_key"],
                "endpoint_overlap_category": item["endpoint_overlap_category"],
                "target_preactivation": item["target_preactivation"],
                "target_threshold": item["target_threshold"],
                "q": item["q"],
                "predicted_alpha_star": item.get("predicted_alpha_star"),
                "predicted_status": item["predicted_status"],
            },
            item["points"],
            config["analysis"],
        )
        for item in sweeps
    ]
    aggregates = aggregate_analyses(analyses)
    outcome = classify_outcome(analyses).value
    return {
        "schema_version": 3,
        "experiment_class": EXPERIMENT_CLASS,
        "intervention_sweeps": {
            "artifact_type": "stage1c_v3_intervention_sweeps",
            "pairs": sweeps,
        },
        "crossing_summary": {
            "artifact_type": "stage1c_v3_crossing_summary",
            "pairs": analyses,
            "aggregate_metrics": aggregates,
            "scientific_outcome": outcome,
        },
        "local_linearity_summary": {
            "artifact_type": "stage1c_v3_local_linearity_summary",
            "pairs": analyses,
            "aggregate_metrics": aggregates,
            "scientific_outcome": outcome,
        },
        "memory_timing_summary": {
            "artifact_type": "stage1c_v3_memory_timing_summary",
            "telemetry": telemetry,
        },
        "attempts": {
            "artifact_type": "stage1c_v3_attempts",
            "attempt_count": 1 if sweeps else 0,
            "scientific_retry_count": 0,
            "intervention_required": bool(sweeps),
        },
    }, outcome


def _execute_production_sweeps(
    backend: Stage1CInterventionBackendProtocol,
    groups: dict[str, list[dict[str, Any]]],
    *,
    token_count: int,
    config: dict[str, Any],
    torch: Any,
    sampler: Any,
    journal: CanonicalExecutionJournal,
) -> tuple[list[dict[str, Any]], dict[FeatureRef, Any]]:
    """Run the exact baseline-and-point path shared by production and rehearsals."""

    if not isinstance(backend, Stage1CInterventionBackendProtocol):
        raise RuntimeError(
            "production backend does not satisfy the complete worker protocol"
        )
    states = _verify_baselines(backend, groups, token_count=token_count)
    sweeps: list[dict[str, Any]] = []
    for group in ("primary", "near_boundary", "directional"):
        for row in groups[group]:
            source = FeatureRef(**row["source"])
            sweeps.append(
                _run_pair(
                    backend,
                    row,
                    float(states[source].activation),
                    config,
                    torch,
                    sampler,
                    journal,
                )
            )
    return sweeps, states


def _verify_assets(cache: Path) -> dict[str, Any]:
    model, transcoder = resolve_offline_snapshots(cache, REPOSITORY_ROOT)
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
        raise RuntimeError("immutable v3 asset verification failed")
    model_record = record.get("model")
    transcoder_record = record.get("transcoder")
    if not isinstance(model_record, dict) or not isinstance(transcoder_record, dict):
        raise RuntimeError("immutable v3 asset byte evidence is missing")
    return {
        "status": "verified",
        "download_performed": False,
        "network_accessed": False,
        "authentication_used": False,
        "authentication_value_recorded": False,
        "actual_total_bytes": record.get("actual_total_bytes"),
        "model_total_bytes": model_record.get("total_bytes"),
        "transcoder_total_bytes": transcoder_record.get("total_bytes"),
        "model_revision": model.name,
        "transcoder_revision": transcoder.name,
        "exact_allowlist_hashes_verified": True,
    }


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    config = load_stage1c_v3_config(args.config, require_token_ids=True)
    assert_fallback_disabled()
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
    ):
        raise RuntimeError("v3 intervention worker requires offline mode")
    from preflight_stage1c_v3 import verify_git

    git_start = verify_git("intervention", args.pre_intervention_commit)
    _verify_tracked_prediction_manifest(args.prediction_manifest)
    manifest_value = read_json_strict(args.prediction_manifest)
    if not isinstance(manifest_value, dict):
        raise RuntimeError("prediction manifest must be an object")
    manifest = cast(dict[str, Any], manifest_value)
    observed_digest = hashlib.sha256(args.prediction_manifest.read_bytes()).hexdigest()
    if observed_digest != args.prediction_manifest_sha256:
        raise RuntimeError("prediction manifest SHA-256 differs")
    groups = _validate_prediction_manifest(manifest, config, args)
    if not groups["primary"]:
        artifacts, outcome = _artifact_bundle([], config, None)
        return build_detached_worker_result(
            [],
            intervention_artifacts=artifacts,
            canonical_source_suppression_api_calls=0,
            instrumented_source_suppression_api_calls=0,
            schema_version=3,
            artifact_type="stage1c_v3_intervention_worker",
            status="passed",
            scientific_outcome=outcome,
            attempt_count=0,
            git=git_start,
            pre_intervention_commit=args.pre_intervention_commit,
            prediction_manifest_sha256=args.prediction_manifest_sha256,
            prediction_manifest_verification={
                "status": "passed",
                "tracked_manifest": True,
                "protocol_hashes_match": True,
            },
            scientific_retry_count=0,
            intervention_skipped=True,
            telemetry=None,
        )

    import nnsight  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    import transformers  # type: ignore[import-not-found]

    if (
        not torch.backends.mps.is_built()
        or not torch.backends.mps.is_available()
        or torch.is_autocast_enabled()
    ):
        raise RuntimeError("native MPS/BF16 runtime identity is unavailable")
    asset_manifest = _verify_assets(args.hf_cache)
    model_snapshot, transcoder_snapshot = resolve_offline_snapshots(
        args.hf_cache, REPOSITORY_ROOT
    )
    sampler = MPSTelemetrySampler(torch, config["safety_limits"], args.emergency_output)
    sampler_finished = False
    model: Any = None
    sweeps: list[dict[str, Any]] = []
    pair_ids = tuple(
        row["pair_id"]
        for group in ("primary", "near_boundary", "directional")
        for row in groups[group]
    )
    journal = CanonicalExecutionJournal(
        args.point_journal,
        args.attempt_lock,
        frozen_pair_ids=pair_ids,
        pre_intervention_commit=args.pre_intervention_commit,
        prediction_manifest_sha256=args.prediction_manifest_sha256,
    )
    try:
        with sampler.stage("replacement_runtime_loading"):
            model, module_guard = build_mps_bf16_replacement(
                model_snapshot, transcoder_snapshot, torch
            )
            prompt = str(config["prompt"]["text"])
            tokens = model.ensure_tokenized(prompt)
            token_ids = [int(item) for item in tokens.detach().cpu().tolist()]
            validate_prompt_token_ids(config, token_ids)
            if token_ids != manifest["prompt"]["token_ids"]:
                raise RuntimeError(
                    "intervention prompt token identity differs from manifest"
                )
            backend = Stage1CInterventionBackend(
                model,
                prompt=prompt,
                torch=torch,
                token_count=len(token_ids),
                attempt_recorder=journal.before_source_suppression,
            )
        sweeps, states = _execute_production_sweeps(
            backend,
            groups,
            token_count=len(manifest["prompt"]["token_ids"]),
            config=config,
            torch=torch,
            sampler=sampler,
            journal=journal,
        )
        telemetry = sampler.finish()
        sampler_finished = True
        if telemetry["violations"] or telemetry["telemetry_failures"]:
            raise RuntimeError("v3 intervention telemetry contains a safety failure")
        serialized_point_count = sum(int(item["point_count"]) for item in sweeps)
        observed_api_calls = int(backend.source_suppression_api_calls)
        if observed_api_calls != serialized_point_count:
            raise RuntimeError(
                "instrumented suppression API calls differ from serialized points"
            )
        journal.verify_complete(expected_point_count=serialized_point_count)
        git_end = verify_git("intervention", args.pre_intervention_commit)
        if git_end != git_start:
            raise RuntimeError("v3 worktree identity changed during intervention")
        artifacts, outcome = _artifact_bundle(sweeps, config, telemetry)
        return build_detached_worker_result(
            sweeps,
            intervention_artifacts=artifacts,
            canonical_source_suppression_api_calls=observed_api_calls,
            instrumented_source_suppression_api_calls=observed_api_calls,
            schema_version=3,
            artifact_type="stage1c_v3_intervention_worker",
            status="passed",
            attempt_count=1,
            git=git_end,
            pre_intervention_commit=args.pre_intervention_commit,
            prediction_manifest_sha256=args.prediction_manifest_sha256,
            prediction_manifest_verification={
                "status": "passed",
                "tracked_manifest": True,
                "protocol_hashes_match": True,
            },
            asset_manifest=asset_manifest,
            environment={
                "system": platform.system(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "circuit-tracer": importlib.metadata.version("circuit-tracer"),
                "torch": str(torch.__version__),
                "nnsight": nnsight.__version__,
                "transformers": transformers.__version__,
                "mps_built": torch.backends.mps.is_built(),
                "mps_available": torch.backends.mps.is_available(),
                "fallback_variable_present": "PYTORCH_ENABLE_MPS_FALLBACK"
                in os.environ,
                "outer_autocast_enabled": torch.is_autocast_enabled(),
            },
            module_guard=module_guard,
            intervention_regime=dict(config["intervention"]),
            remeasured_selected_baseline_count=len(states),
            scientific_retry_count=0,
            scientific_outcome=outcome,
            telemetry=telemetry,
        )
    finally:
        journal.close()
        if not sampler_finished:
            with contextlib.suppress(Exception):
                sampler.finish()
        sweeps.clear()
        if model is not None:
            del model
        gc.collect()
        if "torch" in locals():
            with contextlib.suppress(Exception):
                torch.mps.empty_cache()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    expected_attempt_parent = (
        REPOSITORY_ROOT
        / "results/generated/stage1c_v3_preregistered_prospective_prediction"
    ).resolve()
    if (
        args.output.exists()
        or args.emergency_output.exists()
        or args.attempt_lock.exists()
        or args.point_journal.exists()
        or not args.output.parent.is_dir()
        or not args.emergency_output.parent.is_dir()
        or not args.attempt_lock.parent.is_dir()
        or args.attempt_lock.resolve(strict=False).parent != expected_attempt_parent
        or args.attempt_lock.name != "canonical_attempt_v1.lock"
        or args.point_journal.parent.resolve() != args.output.parent.resolve()
        or args.point_journal.name != "point_journal.jsonl"
    ):
        raise RuntimeError(
            "v4 intervention output, journal, or attempt-lock path is unsafe"
        )
    result = _execute(args)
    write_json_new(args.output, result)
    print(
        json.dumps(
            {"status": result["status"], "phase": "intervention"}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
