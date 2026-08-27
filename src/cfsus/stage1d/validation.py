"""Standalone hostile-input validator for the Stage 1D compact bundle."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, cast

from cfsus.stage1c_v3.analysis import symmetric_normalized_error
from cfsus.stage1c_v3.prediction import canonical_v3_pair_id
from cfsus.stage1c_v3.quantization_audit import bf16_round
from cfsus.stage1c_v3.serialization import detach_json, read_json_strict
from cfsus.stage1d.artifacts import JSON_ARTIFACTS
from cfsus.stage1d.benchmark import (
    METHODS,
    detailed_requested_alphas,
    quantization_evidence,
)
from cfsus.stage1d.config import (
    BRANCH,
    COMPLETED_STATUS,
    EXPERIMENT_CLASS,
    PROMPTS,
    load_stage1d_config,
)
from cfsus.stage1d.metrics import compute_benchmark_summary
from cfsus.stage1d.prediction_runtime import RUNTIME_FINGERPRINT
from cfsus.stage1d.protocol import PROTOCOL_FILES, protocol_map_digest, sha256_file
from cfsus.types import FeatureRef

SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")


def _object(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _feature(value: Any, label: str, token_count: int) -> FeatureRef:
    row = _object(value, label)
    if set(row) != {"layer", "position", "feature_id"} or any(
        type(row[key]) is not int for key in row
    ):
        raise ValueError(f"{label} is malformed")
    feature = FeatureRef(**row)
    if not (
        0 <= feature.layer < 18
        and 1 <= feature.position < token_count
        and 0 <= feature.feature_id < 16_384
    ):
        raise ValueError(f"{label} is outside the frozen domain")
    return feature


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def validate_protocol(
    repository: Path, protocol: dict[str, Any], config: dict[str, Any]
) -> None:
    if (
        protocol.get("schema_version") != 1
        or protocol.get("artifact_type") != "stage1d_protocol_manifest"
        or protocol.get("status") != "protocol_frozen_before_baseline"
        or protocol.get("experiment_class") != EXPERIMENT_CLASS
        or protocol.get("branch") != BRANCH
        or protocol.get("base_commit") != config["base_commit"]
        or protocol.get("evaluation_baseline_model_calls_before_freeze") != 0
        or protocol.get("evaluation_source_suppression_calls_before_freeze") != 0
        or protocol.get("scientific_attempt_started") is not False
    ):
        raise ValueError("protocol manifest identity or freeze boundary differs")
    commit = protocol.get("protocol_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("protocol commit is malformed")
    hashes = _object(protocol.get("protocol_file_sha256"), "protocol hashes")
    if set(hashes) != set(PROTOCOL_FILES) or len(hashes) != len(PROTOCOL_FILES):
        raise ValueError("protocol file allowlist/order differs")
    for relative, expected in hashes.items():
        if (
            SHA256.fullmatch(str(expected)) is None
            or sha256_file(repository / relative) != expected
        ):
            raise ValueError(f"protocol file digest differs: {relative}")
    if protocol.get("protocol_map_sha256") != protocol_map_digest(hashes):
        raise ValueError("protocol hash-map digest differs")


def _ranking_key(
    method: str, row: dict[str, Any], prompt_id: str, random_domain: str
) -> tuple[Any, ...]:
    source = row["source"]
    target = row["target"]
    stable = (
        prompt_id,
        source["layer"],
        source["position"],
        source["feature_id"],
        target["layer"],
        target["position"],
        target["feature_id"],
        row["pair_id"],
    )
    if method == "S":
        return (-float(row["susceptibility"]), *stable)
    if method == "margin_only":
        return (-float(row["margin_only_score"]), *stable)
    if method == "influence_only":
        return (-float(row["q"]), *stable)
    digest = hashlib.sha256(
        f"{random_domain}|{prompt_id}|{row['pair_id']}".encode()
    ).hexdigest()
    return (digest, *stable)


def validate_prediction(
    prediction: dict[str, Any], config: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    if (
        prediction.get("schema_version") != 1
        or prediction.get("artifact_type") != "stage1d_prediction_manifest"
        or prediction.get("status") != "prediction_frozen_ready_for_commit"
        or prediction.get("experiment_class") != EXPERIMENT_CLASS
        or prediction.get("branch") != BRANCH
        or prediction.get("base_commit") != config["base_commit"]
        or prediction.get("protocol_commit") != protocol["protocol_commit"]
        or prediction.get("protocol_map_sha256") != protocol["protocol_map_sha256"]
        or prediction.get("prompt_order") != [item[0] for item in PROMPTS]
        or prediction.get("prediction_only_guards")
        != {
            "evaluation_source_suppression_api_calls": 0,
            "historical_intervention_outcomes_read": False,
            "norway_development_outcome_used": False,
            "graph_edge_input_used_for_inactive_predictions": False,
            "network_accessed": False,
        }
        or prediction.get("claim_boundary") != config["claim_boundary"]
        or prediction.get("runtime_identity") != config["runtime"]
        or prediction.get("protocol")
        != {
            key: config[key]
            for key in (
                "scanner",
                "source_pool",
                "responses",
                "scoring",
                "quantization_resolvability",
                "full_ablation_panel",
                "detailed_panel",
                "schedules",
                "metrics",
                "decision_rule",
                "intervention",
            )
        }
    ):
        raise ValueError("prediction manifest identity or no-outcome guard differs")
    prompts = prediction.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != 8:
        raise ValueError("prediction prompt records differ")
    all_pairs: dict[str, dict[str, Any]] = {}
    for observed, frozen in zip(prompts, config["prompts"], strict=True):
        prompt = _object(observed, "prediction prompt")
        for key in ("id", "text", "token_ids", "selected_positions"):
            if prompt.get(key) != frozen[key]:
                raise ValueError(f"prediction prompt {frozen['id']} {key} differs")
        rows = prompt.get("execution_pairs")
        if not isinstance(rows, list):
            raise ValueError("prediction execution pair rows are malformed")
        by_id: dict[str, dict[str, Any]] = {}
        for raw in rows:
            row = _object(raw, "prediction pair")
            pair_id = row.get("pair_id")
            if not isinstance(pair_id, str) or SHA256.fullmatch(pair_id) is None:
                raise ValueError("prediction pair ID is malformed")
            if pair_id in all_pairs or pair_id in by_id:
                raise ValueError("prediction pair ID is not globally unique")
            source = _feature(row.get("source"), "source", len(frozen["token_ids"]))
            target = _feature(row.get("target"), "target", len(frozen["token_ids"]))
            if source.layer >= target.layer or source.position > target.position:
                raise ValueError("prediction pair violates causal order")
            expected_id = canonical_v3_pair_id(
                source=source,
                target=target,
                runtime_fingerprint=RUNTIME_FINGERPRINT,
                prompt_id=frozen["id"],
                seed=config["scoring"]["pair_seed"],
                experiment_class=EXPERIMENT_CLASS,
            )
            if pair_id != expected_id or row.get("prompt_id") != frozen["id"]:
                raise ValueError("prediction pair identity differs")
            activation = _number(row.get("source_activation"), "source activation")
            z = _number(row.get("target_preactivation"), "target preactivation")
            threshold = _number(row.get("target_threshold"), "target threshold")
            response = _number(row.get("targeted_response"), "targeted response")
            margin = _number(row.get("margin"), "margin")
            q = _number(row.get("q"), "q")
            susceptibility = _number(row.get("susceptibility"), "susceptibility")
            if (
                activation <= 0.0
                or z > threshold
                or margin != threshold - z
                or q != -activation * response
                or susceptibility != q / (margin + float(config["scoring"]["epsilon"]))
                or _number(row.get("margin_only_score"), "margin score")
                != 1.0 / (margin + 1.0e-12)
                or _number(row.get("influence_only_score"), "influence score")
                != max(q, 0.0)
            ):
                raise ValueError("prediction pair scalar definition differs")
            alpha = row.get("predicted_alpha_star")
            expected_alpha = margin / q if q > 0.0 else None
            if alpha != expected_alpha:
                raise ValueError("predicted alpha differs from margin/q")
            requested = list(
                detailed_requested_alphas(_row_pair(row, source=source, target=target))
                if row.get("detailed_role") is not None
                else (1.0,)
            )
            if row.get("requested_alphas") != requested:
                raise ValueError("frozen requested schedule differs")
            expected_quantization = (
                quantization_evidence(_row_pair(row, source=source, target=target))
                if q > 0.0 and alpha is not None
                else None
            )
            if row.get("quantization_evidence") != expected_quantization:
                raise ValueError("selected-pair quantization evidence differs")
            by_id[pair_id] = row
            all_pairs[pair_id] = row
        memberships: dict[str, set[str]] = {pair_id: set() for pair_id in by_id}
        for method in METHODS:
            ids = prompt.get("method_pair_ids", {}).get(method)
            if not isinstance(ids, list) or len(ids) > 4 or len(ids) != len(set(ids)):
                raise ValueError(f"{method} panel quota or uniqueness differs")
            if any(pair_id not in by_id for pair_id in ids):
                raise ValueError(f"{method} panel references an absent pair")
            selected = [by_id[pair_id] for pair_id in ids]
            if method == "S" and any(
                float(row["q"]) <= 0.0
                or row["predicted_alpha_star"] is None
                or not 0.0 < float(row["predicted_alpha_star"]) < 1.0
                for row in selected
            ):
                raise ValueError("S panel contains an ineligible pair")
            if method != "S" and any(float(row["q"]) <= 0.0 for row in selected):
                raise ValueError(f"{method} panel contains a non-positive-q pair")
            if selected != sorted(
                selected,
                key=lambda row: _ranking_key(
                    method,
                    row,
                    frozen["id"],
                    config["full_ablation_panel"]["random_hash_domain"],
                ),
            ):
                raise ValueError(f"{method} panel order differs")
            for pair_id in ids:
                memberships[pair_id].add(method)
        directional = prompt.get("directional_pair_ids")
        if (
            not isinstance(directional, list)
            or len(directional) > 2
            or any(pair_id not in by_id for pair_id in directional)
            or any(float(by_id[pair_id]["q"]) > 0.0 for pair_id in directional)
        ):
            raise ValueError("directional panel differs")
        for pair_id in directional:
            memberships[pair_id].add("directional")
        detailed = _object(prompt.get("detailed_pair_ids"), "detailed pair IDs")
        if set(detailed) != {"B1", "B2", "B3", "near_boundary"}:
            raise ValueError("detailed panel roles differ")
        for role, pair_id in detailed.items():
            if pair_id is None:
                continue
            if pair_id not in by_id or by_id[pair_id].get("detailed_role") != role:
                raise ValueError("detailed panel identity differs")
            alpha = float(by_id[pair_id]["predicted_alpha_star"])
            if role == "B1" and not 0.02 <= alpha < 0.10:
                raise ValueError("B1 alpha range differs")
            if role == "B2" and not 0.10 <= alpha < 0.40:
                raise ValueError("B2 alpha range differs")
            if role == "B3" and not 0.40 <= alpha <= 0.95:
                raise ValueError("B3 alpha range differs")
            if role == "near_boundary" and not 1.05 <= alpha <= 2.0:
                raise ValueError("near-boundary alpha range differs")
        for pair_id, row in by_id.items():
            expected_memberships = sorted(
                memberships[pair_id],
                key=lambda item: (*METHODS, "directional").index(item),
            )
            if row.get("method_memberships") != expected_memberships:
                raise ValueError("pair method membership differs")
            if row.get("full_ablation_selected") is not bool(expected_memberships):
                raise ValueError("pair full-ablation selection flag differs")
    return all_pairs


def _row_pair(row: dict[str, Any], *, source: FeatureRef, target: FeatureRef) -> Any:
    """Construct the frozen dataclass for schedule recomputation."""

    from cfsus.stage1c_v3.prediction import ProspectivePair
    from cfsus.types import CrossingStatus

    return ProspectivePair(
        pair_id=row["pair_id"],
        source=source,
        target=target,
        source_activation=float(row["source_activation"]),
        target_preactivation=float(row["target_preactivation"]),
        target_threshold=float(row["target_threshold"]),
        margin=float(row["margin"]),
        targeted_response=float(row["targeted_response"]),
        q=float(row["q"]),
        susceptibility=float(row["susceptibility"]),
        predicted_alpha_star=(
            None
            if row["predicted_alpha_star"] is None
            else float(row["predicted_alpha_star"])
        ),
        status=CrossingStatus(row["predicted_status"]),
    )


def _load_bundle(output: Path) -> dict[str, Any]:
    expected = set(JSON_ARTIFACTS) | {"checksums.sha256"}
    observed = {path.name for path in output.iterdir() if path.is_file()}
    if observed != expected or any(path.is_symlink() for path in output.iterdir()):
        raise ValueError("artifact bundle allowlist differs")
    records: dict[str, Any] = {}
    for name in JSON_ARTIFACTS:
        path = output / name
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError(f"artifact exceeds the per-file cap: {name}")
        value = read_json_strict(path)
        detach_json(value)
        records[name] = value
    if sum((output / name).stat().st_size for name in expected) > 5 * 1024 * 1024:
        raise ValueError("artifact bundle exceeds the frozen cap")
    lines = (output / "checksums.sha256").read_text(encoding="ascii").splitlines()
    expected_lines = [
        f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}"
        for name in JSON_ARTIFACTS
    ]
    if lines != expected_lines:
        raise ValueError("artifact checksums differ")
    return records


def validate_bundle(repository: Path, output: Path) -> dict[str, Any]:
    """Recompute every result from only the compact serialized artifacts."""

    config = load_stage1d_config(
        repository / "configs/stage1d_multiprompt_gate_benchmark.yaml"
    )
    records = _load_bundle(output)
    protocol = _object(records["protocol_manifest.json"], "protocol")
    prediction = _object(records["prediction_manifest.json"], "prediction")
    validate_protocol(repository, protocol, config)
    pairs = validate_prediction(prediction, config, protocol)
    panel = _object(records["panel_membership.json"], "panel membership")
    expected_panel_prompts = [
        {
            "id": prompt["id"],
            "method_pair_ids": prompt["method_pair_ids"],
            "directional_pair_ids": prompt["directional_pair_ids"],
            "detailed_pair_ids": prompt["detailed_pair_ids"],
            "missing_strata": prompt["missing_strata"],
        }
        for prompt in prediction["prompts"]
    ]
    if panel.get("prompts") != expected_panel_prompts:
        raise ValueError("panel-membership artifact differs from prediction freeze")
    quant = _object(records["quantization_audit.json"], "quantization audit")
    expected_quant = [
        {"id": prompt["id"], **prompt["quantization_audit"]}
        for prompt in prediction["prompts"]
    ]
    if quant.get("prompts") != expected_quant:
        raise ValueError("quantization audit differs from prediction freeze")
    full = _object(records["full_ablation_points.json"], "full points")
    calibration = _object(records["calibration_sweeps.json"], "calibration")
    points_by_pair: dict[str, list[dict[str, Any]]] = {}
    point_rows: list[dict[str, Any]] = []
    full_points = full.get("points")
    sweeps = calibration.get("sweeps")
    if not isinstance(full_points, list) or not isinstance(sweeps, list):
        raise ValueError("serialized point collections are malformed")
    for point in full_points:
        row = _object(point, "full point")
        pair_id = str(row.get("pair_id"))
        points_by_pair[pair_id] = [row]
        point_rows.append(row)
    for raw in sweeps:
        sweep = _object(raw, "calibration sweep")
        rows = sweep.get("points")
        if (
            not isinstance(rows, list)
            or sweep.get("point_count") != len(rows)
            or sweep.get("pair_id") in points_by_pair
        ):
            raise ValueError("calibration sweep point count or uniqueness differs")
        points_by_pair[sweep["pair_id"]] = [
            _object(point, "calibration point") for point in rows
        ]
        point_rows.extend(points_by_pair[sweep["pair_id"]])
    if set(points_by_pair) != set(pairs):
        raise ValueError("serialized pair set differs from prediction freeze")
    expected_references = sorted(
        pair_id
        for pair_id, pair in pairs.items()
        if pair.get("detailed_role") is not None
        and pair.get("full_ablation_selected") is True
    )
    if sorted(full.get("detailed_full_point_references", [])) != expected_references:
        raise ValueError("detailed full-ablation references differ")
    point_rows.sort(key=lambda item: int(item["source_suppression_api_call_index"]))
    if [item["source_suppression_api_call_index"] for item in point_rows] != list(
        range(1, len(point_rows) + 1)
    ):
        raise ValueError("serialized source-suppression call indices differ")
    for point in point_rows:
        pair = pairs[str(point["pair_id"])]
        z = _number(point.get("target_preactivation"), "target preactivation")
        threshold = _number(point.get("target_threshold"), "target threshold")
        requested_mappings = point.get("requested_mappings")
        if not isinstance(requested_mappings, list) or not requested_mappings:
            raise ValueError("point requested mapping provenance is missing")
        baseline = float(pair["source_activation"])
        applied = _number(point.get("actual_bf16_value_passed"), "applied value")
        realized = 1.0 - applied / baseline
        baseline_z = float(pair["target_preactivation"])
        observed_movement = z - baseline_z
        predicted_movement = realized * float(pair["q"])
        expected_sign = (
            observed_movement > 0.0
            if predicted_movement > 0.0
            else observed_movement < 0.0
            if predicted_movement < 0.0
            else observed_movement == 0.0
        )
        if (
            point.get("prompt_id") != pair["prompt_id"]
            or threshold != float(pair["target_threshold"])
            or _number(
                point.get("baseline_source_activation"), "baseline source activation"
            )
            != baseline
            or _number(
                point.get("baseline_target_preactivation"),
                "baseline target preactivation",
            )
            != baseline_z
            or _number(
                point.get("baseline_target_threshold"), "baseline target threshold"
            )
            != float(pair["target_threshold"])
            or point.get("baseline_target_active") is not False
            or point.get("target_active") is not (z > threshold)
            or point.get("strict_crossing") is not (z > threshold)
            or _number(point.get("target_activation"), "target activation")
            != (z if z > threshold else 0.0)
            or _number(point.get("realized_suppression"), "realized suppression")
            != realized
            or _number(point.get("target_preactivation_movement"), "observed movement")
            != observed_movement
            or _number(
                point.get("first_order_predicted_movement"), "predicted movement"
            )
            != predicted_movement
            or point.get("movement_sign_agreement") is not expected_sign
            or _number(
                point.get("target_preactivation_symmetric_normalized_error"),
                "movement symmetric normalized error",
            )
            != symmetric_normalized_error(predicted_movement, observed_movement)
            or point.get("source_value_device") != "mps:0"
            or point.get("source_value_dtype") != "torch.bfloat16"
        ):
            raise ValueError("serialized point gate or mapping differs")
        for mapping in requested_mappings:
            requested = _number(mapping.get("requested_alpha"), "requested alpha")
            desired = (1.0 - requested) * baseline
            mapping_applied = _number(
                mapping.get("actual_bf16_value_passed"), "mapping applied value"
            )
            if (
                _number(mapping.get("desired_high_precision"), "desired value")
                != desired
                or mapping_applied != applied
                or mapping_applied != bf16_round(desired)
                or _number(
                    mapping.get("realized_suppression"),
                    "mapping realized suppression",
                )
                != realized
            ):
                raise ValueError("desired source activation mapping differs")
    for pair_id, pair in pairs.items():
        rows = points_by_pair[pair_id]
        frozen_requested = {float(item) for item in pair["requested_alphas"]}
        observed_requested = {
            float(mapping["requested_alpha"])
            for point in rows
            for mapping in point["requested_mappings"]
        }
        if not frozen_requested.issubset(observed_requested):
            raise ValueError("frozen requested schedule is incomplete")
        initial_distinct = {
            bf16_round((1.0 - alpha) * float(pair["source_activation"]))
            for alpha in frozen_requested
        }
        if pair.get("detailed_role") is None:
            if len(rows) != 1 or observed_requested != {1.0}:
                raise ValueError("non-detailed pair schedule differs")
        elif len(rows) > len(initial_distinct) + int(
            config["schedules"]["maximum_bisection_steps"]
        ):
            raise ValueError("detailed pair exceeds the bisection cap")
    recomputed = compute_benchmark_summary(prediction, points_by_pair, config)
    recomputed["experiment_class"] = EXPERIMENT_CLASS
    if records["benchmark_summary.json"] != recomputed:
        raise ValueError("benchmark summary differs from standalone recomputation")
    run = _object(records["run_manifest.json"], "run manifest")
    if (
        run.get("status") != COMPLETED_STATUS
        or run.get("canonical_attempt_count") != 1
        or run.get("scientific_retry_count") != 0
        or run.get("instrumented_evaluation_api_calls") != len(point_rows)
        or run.get("completed_journal_points") != len(point_rows)
        or run.get("serialized_unique_point_rows") != len(point_rows)
        or run.get("project_decision") != recomputed["project_decision"]
        or run.get("claim_boundary") != config["claim_boundary"]
    ):
        raise ValueError("run manifest counts, outcome, or claim boundary differs")
    environment = _object(records["environment_manifest.json"], "environment")
    telemetry = _object(environment.get("telemetry"), "telemetry")
    if telemetry.get("violations") != [] or telemetry.get("telemetry_failures") != 0:
        raise ValueError("canonical telemetry contains a safety failure")
    return {
        "status": "passed",
        "terminal_status": COMPLETED_STATUS,
        "project_decision": recomputed["project_decision"],
        "prompt_count": 8,
        "evaluation_call_count": len(point_rows),
        "journal_completed_point_count": len(point_rows),
        "serialized_unique_point_count": len(point_rows),
        "checksums_verified": len(JSON_ARTIFACTS),
    }


__all__ = [
    "SHA256",
    "validate_bundle",
    "validate_prediction",
    "validate_protocol",
]
