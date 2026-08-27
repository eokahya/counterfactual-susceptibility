"""Active-target engineering rehearsal over the exact production adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from cfsus.stage1c_v3.execution_journal import CanonicalExecutionJournal
from cfsus.stage1c_v3.intervention import plan_applied_values
from cfsus.stage1c_v3.intervention_runtime import Stage1CInterventionBackend
from cfsus.stage1c_v3.serialization import detach_json, read_json_strict, write_json_new
from cfsus.stage1c_v3.worker_result import build_detached_worker_result
from cfsus.types import FeatureActivity, FeatureRef

NORWAY_TOKEN_IDS = [2, 818, 5279, 529, 32649, 563]


def _candidate_sources(repository: Path) -> tuple[FeatureRef, ...]:
    manifest = read_json_strict(
        repository
        / "results/stage1c_v4_protocol_preserving_execution/prediction_manifest.json"
    )
    if not isinstance(manifest, dict):
        raise RuntimeError("v4 development prediction manifest is malformed")
    manifest_object = cast(dict[str, Any], manifest)
    groups = manifest_object.get("selected_groups")
    if not isinstance(groups, dict):
        raise RuntimeError("v4 development selected groups are missing")
    groups_object = cast(dict[str, Any], groups)
    sources: set[FeatureRef] = set()
    for name in ("primary", "near_boundary", "directional"):
        rows = groups_object.get(name)
        if not isinstance(rows, list):
            raise RuntimeError("v4 development group is malformed")
        for raw in rows:
            if not isinstance(raw, dict) or not isinstance(raw.get("source"), dict):
                raise RuntimeError("v4 development source is malformed")
            sources.add(FeatureRef(**cast(dict[str, int], raw["source"])))
    return tuple(sorted(sources))


def _active_pair(
    states: dict[FeatureRef, Any],
) -> tuple[FeatureRef, FeatureRef]:
    candidates = sorted(
        (source, target)
        for source, source_state in states.items()
        for target, target_state in states.items()
        if source.layer < target.layer
        and source.position <= target.position
        and source_state.activity is FeatureActivity.ACTIVE
        and target_state.activity is FeatureActivity.ACTIVE
        and source_state.activation > 0.0
        and target_state.activation > 0.0
    )
    if not candidates:
        raise RuntimeError("no deterministic Norway active-only rehearsal pair exists")
    return candidates[0]


def _evaluation_pair_ids(repository: Path) -> set[str]:
    path = (
        repository
        / "results/stage1d_multiprompt_gate_benchmark/prediction_manifest.json"
    )
    value = read_json_strict(path)
    if not isinstance(value, dict) or not isinstance(value.get("prompts"), list):
        raise RuntimeError("Stage 1D prediction manifest is unavailable for rehearsal")
    value_object = cast(dict[str, Any], value)
    prompts = cast(list[Any], value_object["prompts"])
    result: set[str] = set()
    for prompt in prompts:
        if not isinstance(prompt, dict) or not isinstance(
            prompt.get("execution_pairs"), list
        ):
            raise RuntimeError("Stage 1D evaluation pair list is malformed")
        prompt_object = cast(dict[str, Any], prompt)
        for pair in cast(list[Any], prompt_object["execution_pairs"]):
            if not isinstance(pair, dict) or not isinstance(pair.get("pair_id"), str):
                raise RuntimeError("Stage 1D evaluation pair identity is malformed")
            result.add(cast(str, pair["pair_id"]))
    return result


def run_rehearsal(
    *,
    repository: Path,
    model: Any,
    torch: Any,
    sampler: Any,
    journal_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Exercise baseline, no-op, nonzero call, journal, and detachment."""

    prompt = "The capital of Norway is"
    token_ids = [
        int(item) for item in model.ensure_tokenized(prompt).detach().cpu().tolist()
    ]
    if token_ids != NORWAY_TOKEN_IDS:
        raise RuntimeError("Norway rehearsal token IDs differ")
    preliminary = Stage1CInterventionBackend(
        model,
        prompt=prompt,
        prompt_id="development_norway_v4",
        torch=torch,
        token_count=len(token_ids),
    )
    sources = _candidate_sources(repository)
    with sampler.stage("rehearsal_baseline_candidate_measurement"):
        states = preliminary.measure_states(sources)
    source, target = _active_pair(states)
    pair_id = hashlib.sha256(
        json.dumps(
            {
                "domain": "stage1d-active-target-rehearsal",
                "prompt_id": "development_norway_v4",
                "source": [source.layer, source.position, source.feature_id],
                "target": [target.layer, target.position, target.feature_id],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if pair_id in _evaluation_pair_ids(repository):
        raise RuntimeError("engineering rehearsal overlaps an evaluation pair identity")
    pair = {
        "pair_id": pair_id,
        "source": {
            "layer": source.layer,
            "position": source.position,
            "feature_id": source.feature_id,
        },
        "target": {
            "layer": target.layer,
            "position": target.position,
            "feature_id": target.feature_id,
        },
    }
    with CanonicalExecutionJournal(
        journal_path,
        None,
        frozen_pair_ids=(pair_id,),
        pre_intervention_commit="engineering_rehearsal",
        prediction_manifest_sha256="0" * 64,
        experiment_class="stage1d_engineering_rehearsal",
        attempt_boundary="engineering_only_no_scientific_attempt",
        attempt_lock_artifact_type="stage1d_engineering_rehearsal_lock",
    ) as journal:
        backend = Stage1CInterventionBackend(
            model,
            prompt=prompt,
            prompt_id="development_norway_v4",
            torch=torch,
            token_count=len(token_ids),
            attempt_recorder=journal.before_source_suppression,
        )
        with sampler.stage("rehearsal_exact_baseline_remeasurement"):
            exact = backend.measure_states((source, target))
        source_state = exact[source]
        target_state = exact[target]
        if (
            source_state.activity is not FeatureActivity.ACTIVE
            or target_state.activity is not FeatureActivity.ACTIVE
        ):
            raise RuntimeError("rehearsal endpoints changed activity")
        source_tensor = torch.tensor(
            source_state.activation, device="mps", dtype=torch.bfloat16
        ).reshape(())
        plans = plan_applied_values(source_tensor, (0.0, 0.25), torch)
        points: list[dict[str, Any]] = []
        for index, plan in enumerate(plans):
            stage = f"rehearsal_point_{index}"
            with sampler.stage(stage):
                point = backend.measure_point(
                    pair,
                    plan,
                    freeze_attention=True,
                    constrained_layers=None,
                    stage=stage,
                )
            point.update(
                {
                    "pair_id": pair_id,
                    "prompt_id": "development_norway_v4",
                    "baseline_source_activation": source_state.activation,
                    "baseline_target_preactivation": target_state.preactivation,
                    "baseline_target_threshold": target_state.threshold,
                }
            )
            detached = detach_json(point)
            if not isinstance(detached, dict):
                raise RuntimeError("rehearsal point did not detach")
            journal.append_completed_point(detached)
            points.append(detached)
        journal.verify_complete(expected_point_count=2)
    sweep = {"pair_id": pair_id, "point_count": 2, "points": points}
    worker = build_detached_worker_result(
        [sweep],
        intervention_artifacts={"intervention_sweeps": {"pairs": [sweep]}},
        canonical_source_suppression_api_calls=2,
        instrumented_source_suppression_api_calls=2,
        schema_version=1,
        artifact_type="stage1d_active_target_rehearsal_worker",
        status="passed",
    )
    record = {
        "schema_version": 1,
        "artifact_type": "stage1d_active_target_production_rehearsal",
        "status": "passed",
        "prompt_id": "development_norway_v4",
        "evaluation_pair_calls": 0,
        "evaluation_pair_id_overlap": False,
        "scientific_attempt_started": False,
        "scientific_attempt_lock_created": False,
        "engineering_call_count": 2,
        "journal_call_start_count": 2,
        "journal_completed_point_count": 2,
        "worker": worker,
    }
    write_json_new(output, record)
    return record


def validate_rehearsal(path: Path, journal_path: Path) -> dict[str, Any]:
    """Validate rehearsal in a model-free standalone process."""

    value = read_json_strict(path)
    if not isinstance(value, dict):
        raise ValueError("rehearsal artifact is malformed")
    if (
        value.get("status") != "passed"
        or value.get("evaluation_pair_calls") != 0
        or value.get("evaluation_pair_id_overlap") is not False
        or value.get("scientific_attempt_started") is not False
        or value.get("scientific_attempt_lock_created") is not False
        or value.get("engineering_call_count") != 2
    ):
        raise ValueError("rehearsal scientific boundary differs")
    worker = value.get("worker")
    if not isinstance(worker, dict):
        raise ValueError("rehearsal worker is missing")
    worker_object = cast(dict[str, Any], worker)
    top = worker_object.get("sweeps")
    artifacts = worker_object.get("intervention_artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("rehearsal artifact sweep map is missing")
    intervention = artifacts.get("intervention_sweeps")
    if not isinstance(intervention, dict):
        raise ValueError("rehearsal intervention sweep map is missing")
    nested = intervention.get("pairs")
    if top != nested or not isinstance(top, list) or len(top) != 1:
        raise ValueError("rehearsal detached sweep copies differ")
    sweep = top[0]
    if not isinstance(sweep, dict):
        raise ValueError("rehearsal sweep is malformed")
    points = sweep.get("points")
    if not isinstance(points, list) or len(points) != 2:
        raise ValueError("rehearsal point rows differ")
    requested: list[float] = []
    for point in points:
        if not isinstance(point, dict) or not isinstance(
            point.get("requested_mappings"), list
        ):
            raise ValueError("rehearsal requested mappings are malformed")
        mappings = cast(list[Any], point["requested_mappings"])
        if any(not isinstance(item, dict) for item in mappings):
            raise ValueError("rehearsal requested mapping is malformed")
        requested.append(
            min(
                float(cast(dict[str, Any], item)["requested_alpha"])
                for item in mappings
            )
        )
    if requested != [0.0, 0.25]:
        raise ValueError("rehearsal no-op/nonzero schedule differs")
    lines = journal_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 4 or [json.loads(line)["record_type"] for line in lines] != [
        "source_suppression_call_started",
        "point_completed",
        "source_suppression_call_started",
        "point_completed",
    ]:
        raise ValueError("rehearsal journal order differs")
    return {
        "status": "passed",
        "engineering_call_count": 2,
        "journal_completed_point_count": 2,
        "detached_serialization": True,
        "standalone_validation": True,
    }


__all__ = ["run_rehearsal", "validate_rehearsal"]
