#!/usr/bin/env python3
"""Run and independently validate one active-only Stage 1C-v4 rehearsal."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cfsus.mps_telemetry import MPSTelemetrySampler  # noqa: E402
from cfsus.reproduction.small_model_mps_bf16 import (  # noqa: E402
    assert_fallback_disabled,
)
from cfsus.stage1b_runtime import (  # noqa: E402
    build_mps_bf16_replacement,
    resolve_offline_snapshots,
)
from cfsus.stage1c_v3.config import (  # noqa: E402
    load_stage1c_v3_config,
    validate_prompt_token_ids,
)
from cfsus.stage1c_v3.intervention import plan_applied_values  # noqa: E402
from cfsus.stage1c_v3.intervention_runtime import (  # noqa: E402
    Stage1CInterventionBackend,
    Stage1CInterventionBackendProtocol,
)
from cfsus.stage1c_v3.quantization_audit import bf16_round  # noqa: E402
from cfsus.stage1c_v3.serialization import (  # noqa: E402
    detach_json,
    read_json_strict,
    write_json_new,
)
from cfsus.stage1c_v3.worker_result import (  # noqa: E402
    build_detached_worker_result,
)
from cfsus.types import FeatureActivity, FeatureRef  # noqa: E402

EXPECTED_MANIFEST_SHA256 = (
    "b2c489317852a2f54d50db783abc17dfdc08590353b0473dbab01ec3d04574cc"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", type=Path)
    parser.add_argument("--hf-cache", type=Path)
    parser.add_argument("--prediction-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--emergency-output", type=Path)
    return parser


def _frozen_pairs(
    manifest: dict[str, Any],
) -> tuple[
    tuple[FeatureRef, ...],
    frozenset[tuple[FeatureRef, FeatureRef]],
]:
    groups = manifest.get("selected_groups")
    if not isinstance(groups, dict):
        raise RuntimeError("frozen manifest groups are missing")
    sources: set[FeatureRef] = set()
    pairs: set[tuple[FeatureRef, FeatureRef]] = set()
    for group in ("primary", "near_boundary", "directional"):
        rows = groups.get(group)
        if not isinstance(rows, list):
            raise RuntimeError("frozen manifest group is malformed")
        for row in rows:
            source = FeatureRef(**row["source"])
            target = FeatureRef(**row["target"])
            sources.add(source)
            pairs.add((source, target))
    if len(pairs) != 28:
        raise RuntimeError("frozen scientific pair count differs")
    return tuple(sorted(sources)), frozenset(pairs)


def _active_only_pair(
    sources: tuple[FeatureRef, ...],
    frozen_pairs: frozenset[tuple[FeatureRef, FeatureRef]],
) -> tuple[FeatureRef, FeatureRef]:
    candidates = sorted(
        (source, target)
        for source in sources
        for target in sources
        if source.layer < target.layer
        and source.position <= target.position
        and (source, target) not in frozen_pairs
    )
    if not candidates:
        raise RuntimeError("no deterministic active-only rehearsal pair exists")
    return candidates[0]


def _state_record(state: Any) -> dict[str, Any]:
    return {
        "feature": {
            "layer": state.feature.layer,
            "position": state.feature.position,
            "feature_id": state.feature.feature_id,
        },
        "preactivation": state.preactivation,
        "activation": state.activation,
        "threshold": state.threshold,
        "activity": state.activity.value,
        "device": state.device,
        "dtype": state.dtype,
    }


def _execute(
    cache: Path,
    manifest_path: Path,
    emergency_output: Path,
) -> dict[str, Any]:
    assert_fallback_disabled()
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
    ):
        raise RuntimeError("active-only rehearsal requires enforced offline mode")
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("frozen prediction manifest digest differs")
    value = read_json_strict(manifest_path)
    if not isinstance(value, dict):
        raise RuntimeError("frozen prediction manifest must be an object")
    manifest = cast(dict[str, Any], value)
    sources, frozen_pairs = _frozen_pairs(manifest)
    source, target = _active_only_pair(sources, frozen_pairs)

    import torch  # type: ignore[import-not-found]

    if (
        not torch.backends.mps.is_built()
        or not torch.backends.mps.is_available()
        or torch.is_autocast_enabled()
    ):
        raise RuntimeError("native MPS/BF16 runtime is unavailable")
    config = load_stage1c_v3_config(require_token_ids=True)
    model_snapshot, transcoder_snapshot = resolve_offline_snapshots(cache, ROOT)
    sampler = MPSTelemetrySampler(torch, config["safety_limits"], emergency_output)
    sampler_finished = False
    model: Any = None
    try:
        with sampler.stage("rehearsal_runtime_loading"):
            model, module_guard = build_mps_bf16_replacement(
                model_snapshot, transcoder_snapshot, torch
            )
            prompt = str(config["prompt"]["text"])
            token_ids = [
                int(item)
                for item in model.ensure_tokenized(prompt).detach().cpu().tolist()
            ]
            validate_prompt_token_ids(config, token_ids)
            backend = Stage1CInterventionBackend(
                model,
                prompt=prompt,
                torch=torch,
                token_count=len(token_ids),
            )
            if not isinstance(backend, Stage1CInterventionBackendProtocol):
                raise RuntimeError("production backend protocol is incomplete")
        with sampler.stage("rehearsal_selected_baseline"):
            states = backend.measure_states((source, target))
        source_state = states[source]
        target_state = states[target]
        if (
            source_state.activity is not FeatureActivity.ACTIVE
            or target_state.activity is not FeatureActivity.ACTIVE
            or source_state.activation <= 0.0
            or target_state.activation <= 0.0
        ):
            raise RuntimeError("deterministic rehearsal endpoints are not active")
        pair_id = hashlib.sha256(
            json.dumps(
                {
                    "domain": "stage1c-v4-active-only-rehearsal",
                    "source": [source.layer, source.position, source.feature_id],
                    "target": [target.layer, target.position, target.feature_id],
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        pair = {
            "pair_id": pair_id,
            "source": _state_record(source_state)["feature"],
            "target": _state_record(target_state)["feature"],
        }
        source_tensor = torch.tensor(
            source_state.activation, device="mps", dtype=torch.bfloat16
        ).reshape(())
        plans = plan_applied_values(source_tensor, (0.0, 0.25), torch)
        points: list[dict[str, Any]] = []
        for index, plan in enumerate(plans):
            stage = f"rehearsal_active_point_{index}"
            with sampler.stage(stage):
                point = backend.measure_point(
                    pair,
                    plan,
                    freeze_attention=True,
                    constrained_layers=None,
                    stage="active_only_engineering_rehearsal",
                )
            representative = min(plan.requests, key=lambda item: item.requested_alpha)
            point.update(
                {
                    "pair_id": pair_id,
                    "memory_stage_identity": stage,
                    "requested_alpha": representative.requested_alpha,
                    "desired_high_precision": (representative.desired_high_precision),
                }
            )
            detached = detach_json(point)
            if not isinstance(detached, dict):
                raise RuntimeError("rehearsal point did not detach")
            points.append(cast(dict[str, Any], detached))
        telemetry = sampler.finish()
        sampler_finished = True
        if telemetry["violations"] or telemetry["telemetry_failures"]:
            raise RuntimeError("rehearsal telemetry contains a safety failure")
        sweep = {
            "pair_id": pair_id,
            "point_count": len(points),
            "points": points,
        }
        worker = build_detached_worker_result(
            [sweep],
            intervention_artifacts={"intervention_sweeps": {"pairs": [sweep]}},
            canonical_source_suppression_api_calls=(
                backend.source_suppression_api_calls
            ),
            schema_version=4,
            artifact_type="stage1c_v4_active_only_rehearsal_worker",
            status="passed",
        )
        return {
            "schema_version": 4,
            "artifact_type": "stage1c_v4_active_only_production_path_rehearsal",
            "status": "passed",
            "prediction_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "scientific_pair_overlap": False,
            "scientific_attempt_started": False,
            "scientific_intervention_calls": 0,
            "engineering_intervention_calls": backend.source_suppression_api_calls,
            "source_baseline": _state_record(source_state),
            "target_baseline": _state_record(target_state),
            "worker": worker,
            "telemetry": telemetry,
            "module_guard": module_guard,
        }
    finally:
        if not sampler_finished:
            with contextlib.suppress(Exception):
                sampler.finish()
        if model is not None:
            del model
        gc.collect()
        with contextlib.suppress(Exception):
            torch.mps.empty_cache()


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} must be finite")
    return result


def validate_rehearsal(value: dict[str, Any]) -> dict[str, Any]:
    """Recompute the active-only engineering record without model/runtime imports."""

    if (
        value.get("schema_version") != 4
        or value.get("artifact_type")
        != "stage1c_v4_active_only_production_path_rehearsal"
        or value.get("status") != "passed"
        or value.get("prediction_manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or value.get("scientific_pair_overlap") is not False
        or value.get("scientific_attempt_started") is not False
        or value.get("scientific_intervention_calls") != 0
    ):
        raise RuntimeError("rehearsal identity or scientific boundary differs")
    source = value.get("source_baseline")
    target = value.get("target_baseline")
    worker = value.get("worker")
    if not isinstance(source, dict) or not isinstance(target, dict):
        raise RuntimeError("rehearsal baselines are missing")
    if source.get("activity") != "active" or target.get("activity") != "active":
        raise RuntimeError("rehearsal endpoints are not baseline active")
    manifest_path = (
        ROOT
        / "results/stage1c_v3_preregistered_prospective_prediction"
        / "prediction_manifest.json"
    )
    manifest_raw = manifest_path.read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("standalone rehearsal manifest digest differs")
    manifest_value = read_json_strict(manifest_path)
    if not isinstance(manifest_value, dict):
        raise RuntimeError("standalone rehearsal manifest is malformed")
    active_sources, frozen_pairs = _frozen_pairs(manifest_value)
    source_feature = FeatureRef(**source["feature"])
    target_feature = FeatureRef(**target["feature"])
    if (
        source_feature not in active_sources
        or target_feature not in active_sources
        or (source_feature, target_feature) in frozen_pairs
        or source_feature.layer >= target_feature.layer
        or source_feature.position > target_feature.position
    ):
        raise RuntimeError("rehearsal pair is not an active-only non-scientific pair")
    if not isinstance(worker, dict):
        raise RuntimeError("rehearsal worker record is missing")
    top = worker.get("sweeps")
    nested = (
        worker.get("intervention_artifacts", {})
        .get("intervention_sweeps", {})
        .get("pairs")
    )
    if not isinstance(top, list) or top != nested or len(top) != 1:
        raise RuntimeError("rehearsal detached sweep representations differ")
    points = top[0].get("points")
    if not isinstance(points, list) or len(points) != 2:
        raise RuntimeError("rehearsal must contain exact no-op/nonzero points")
    baseline = _number(source.get("activation"), "source activation")
    target_threshold = _number(target.get("threshold"), "target threshold")
    for index, (point, requested) in enumerate(zip(points, (0.0, 0.25), strict=True)):
        if not isinstance(point, dict):
            raise RuntimeError("rehearsal point is malformed")
        desired = (1.0 - requested) * baseline
        applied = bf16_round(desired)
        realized = 1.0 - applied / baseline
        z = _number(point.get("target_preactivation"), "target preactivation")
        active = z > target_threshold
        if (
            point.get("source_suppression_api_call_index") != index + 1
            or _number(point.get("requested_alpha"), "requested alpha") != requested
            or _number(point.get("desired_high_precision"), "desired activation")
            != desired
            or _number(point.get("actual_bf16_value_passed"), "applied activation")
            != applied
            or _number(point.get("realized_suppression"), "realized suppression")
            != realized
            or point.get("target_active") is not active
            or _number(point.get("target_activation"), "target activation")
            != (z if active else 0.0)
            or point.get("loaded_gate") != "a=z*1[z>tau]"
            or point.get("threshold_equality_activity") != "inactive"
            or point.get("source_value_device") != "mps:0"
            or point.get("source_value_dtype") != "torch.bfloat16"
            or point.get("target_value_device") != "mps:0"
            or point.get("target_value_dtype") != "torch.bfloat16"
            or point.get("logits_finite") is not True
        ):
            raise RuntimeError("rehearsal point failed independent recomputation")
        if index == 0 and z != _number(
            target.get("preactivation"), "baseline target preactivation"
        ):
            raise RuntimeError("rehearsal no-op differs from remeasured baseline")
    call_count = worker.get("canonical_source_suppression_api_calls")
    if call_count != len(points) or value.get("engineering_intervention_calls") != len(
        points
    ):
        raise RuntimeError("rehearsal call and serialized point counts differ")
    return {
        "status": "passed",
        "pair_count": 1,
        "point_count": len(points),
        "engineering_intervention_calls": call_count,
        "scientific_intervention_calls": 0,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.validate_only is not None:
        if any(
            item is not None
            for item in (
                args.hf_cache,
                args.prediction_manifest,
                args.output,
                args.emergency_output,
            )
        ):
            raise RuntimeError("validate-only mode accepts only its artifact path")
        value = read_json_strict(args.validate_only)
        if not isinstance(value, dict):
            raise RuntimeError("rehearsal artifact must be an object")
        print(json.dumps(validate_rehearsal(value), sort_keys=True))
        return 0
    if any(
        item is None
        for item in (
            args.hf_cache,
            args.prediction_manifest,
            args.output,
            args.emergency_output,
        )
    ):
        raise RuntimeError("execution mode requires cache, manifest, and output paths")
    result = _execute(
        cast(Path, args.hf_cache),
        cast(Path, args.prediction_manifest),
        cast(Path, args.emergency_output),
    )
    validate_rehearsal(result)
    write_json_new(cast(Path, args.output), result)
    print(json.dumps({"status": "passed", "phase": "active_rehearsal"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
