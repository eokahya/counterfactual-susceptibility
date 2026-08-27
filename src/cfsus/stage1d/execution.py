"""Shared Stage 1D production pair executor over the accepted v4 adapter."""

from __future__ import annotations

import gc
import itertools
import math
from collections.abc import Callable, Iterable
from typing import Any

from cfsus.stage1c_v3.analysis import symmetric_normalized_error
from cfsus.stage1c_v3.execution_journal import CanonicalExecutionJournal
from cfsus.stage1c_v3.intervention import (
    AppliedValuePlan,
    bisection_requested_alpha,
    plan_applied_values,
)
from cfsus.stage1c_v3.serialization import detach_json
from cfsus.types import FeatureActivity, FeatureRef


def _feature(value: dict[str, Any]) -> FeatureRef:
    return FeatureRef(
        layer=int(value["layer"]),
        position=int(value["position"]),
        feature_id=int(value["feature_id"]),
    )


def _matching_baseline(pair: dict[str, Any], states: dict[FeatureRef, Any]) -> Any:
    source = _feature(pair["source"])
    target = _feature(pair["target"])
    source_state = states[source]
    target_state = states[target]
    if (
        source_state.activity is not FeatureActivity.ACTIVE
        or source_state.activation <= 0.0
        or target_state.activity is not FeatureActivity.INACTIVE
        or source_state.activation != float(pair["source_activation"])
        or target_state.preactivation != float(pair["target_preactivation"])
        or target_state.threshold != float(pair["target_threshold"])
        or target_state.preactivation > target_state.threshold
    ):
        raise RuntimeError("canonical baseline remeasurement differs from prediction")
    return source_state, target_state


def _enrich_point(
    pair: dict[str, Any],
    point: dict[str, Any],
    *,
    source_state: Any,
    target_state: Any,
) -> dict[str, Any]:
    realized = float(point["realized_suppression"])
    observed = float(point["target_preactivation"]) - target_state.preactivation
    predicted = realized * float(pair["q"])
    point.update(
        {
            "pair_id": pair["pair_id"],
            "prompt_id": pair["prompt_id"],
            "method_memberships": list(pair["method_memberships"]),
            "detailed_role": pair["detailed_role"],
            "baseline_source_activation": source_state.activation,
            "baseline_target_preactivation": target_state.preactivation,
            "baseline_target_threshold": target_state.threshold,
            "baseline_target_activation": target_state.activation,
            "baseline_target_active": False,
            "target_preactivation_movement": observed,
            "first_order_predicted_movement": predicted,
            "movement_sign_agreement": (
                observed > 0.0
                if predicted > 0.0
                else observed < 0.0
                if predicted < 0.0
                else observed == 0.0
            ),
            "target_preactivation_symmetric_normalized_error": (
                symmetric_normalized_error(predicted, observed)
            ),
            "strict_crossing": bool(point["target_active"]),
        }
    )
    if not all(
        math.isfinite(float(point[key]))
        for key in (
            "baseline_source_activation",
            "baseline_target_preactivation",
            "baseline_target_threshold",
            "target_preactivation_movement",
            "first_order_predicted_movement",
            "target_preactivation_symmetric_normalized_error",
        )
    ):
        raise RuntimeError("canonical point contains a non-finite scalar")
    detached = detach_json(point)
    if not isinstance(detached, dict):  # pragma: no cover
        raise RuntimeError("canonical point did not detach")
    return detached


def _run_plan(
    backend: Any,
    journal: CanonicalExecutionJournal,
    sampler: Any,
    pair: dict[str, Any],
    plan: AppliedValuePlan,
    *,
    source_state: Any,
    target_state: Any,
    stage: str,
) -> dict[str, Any]:
    with sampler.stage(stage):
        point = backend.measure_point(
            pair,
            plan,
            freeze_attention=True,
            constrained_layers=None,
            stage=stage,
        )
    detached = _enrich_point(
        pair,
        point,
        source_state=source_state,
        target_state=target_state,
    )
    journal.append_completed_point(detached)
    return detached


def _crossing_bracket(
    points: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    ordered = sorted(points, key=lambda item: float(item["realized_suppression"]))
    active_seen = False
    for point in ordered:
        if bool(point["target_active"]):
            active_seen = True
        elif active_seen:
            return None
    for left, right in itertools.pairwise(ordered):
        if not bool(left["target_active"]) and bool(right["target_active"]):
            return left, right
    return None


def execute_frozen_pairs(
    *,
    model: Any,
    torch: Any,
    sampler: Any,
    prompts: list[dict[str, Any]],
    journal: CanonicalExecutionJournal,
    backend_factory: Callable[..., Any],
    maximum_bisection_steps: int,
) -> tuple[list[dict[str, Any]], int]:
    """Execute each frozen pair once through one global durable journal."""

    sweeps: list[dict[str, Any]] = []
    global_calls = 0
    for prompt in prompts:
        token_count = len(prompt["token_ids"])
        backend = backend_factory(
            model,
            prompt=str(prompt["text"]),
            prompt_id=str(prompt["id"]),
            torch=torch,
            token_count=token_count,
            attempt_recorder=journal.before_source_suppression,
            call_index_offset=global_calls,
        )
        for pair in prompt["execution_pairs"]:
            source = _feature(pair["source"])
            target = _feature(pair["target"])
            with sampler.stage(f"{prompt['id']}_{pair['pair_id'][:12]}_baseline"):
                states = backend.measure_states((source, target))
            source_state, target_state = _matching_baseline(pair, states)
            source_tensor = torch.tensor(
                source_state.activation, device="mps", dtype=torch.bfloat16
            ).reshape(())
            requested = tuple(float(item) for item in pair["requested_alphas"])
            plans = plan_applied_values(source_tensor, requested, torch)
            points: list[dict[str, Any]] = []
            applied_values: set[float] = set()
            for index, plan in enumerate(plans):
                applied_values.add(plan.applied_bf16)
                points.append(
                    _run_plan(
                        backend,
                        journal,
                        sampler,
                        pair,
                        plan,
                        source_state=source_state,
                        target_state=target_state,
                        stage=f"{prompt['id']}_{pair['pair_id'][:12]}_grid_{index}",
                    )
                )
            if pair["detailed_role"] is not None:
                for step in range(maximum_bisection_steps):
                    bracket = _crossing_bracket(points)
                    if bracket is None:
                        break
                    requested_alpha = bisection_requested_alpha(
                        float(bracket[0]["realized_suppression"]),
                        float(bracket[1]["realized_suppression"]),
                    )
                    plan = plan_applied_values(
                        source_tensor, (requested_alpha,), torch
                    )[0]
                    if plan.applied_bf16 in applied_values or not float(
                        bracket[0]["realized_suppression"]
                    ) < plan.realized_suppression < float(
                        bracket[1]["realized_suppression"]
                    ):
                        break
                    applied_values.add(plan.applied_bf16)
                    points.append(
                        _run_plan(
                            backend,
                            journal,
                            sampler,
                            pair,
                            plan,
                            source_state=source_state,
                            target_state=target_state,
                            stage=(
                                f"{prompt['id']}_{pair['pair_id'][:12]}_bisect_{step}"
                            ),
                        )
                    )
            points.sort(key=lambda item: float(item["realized_suppression"]))
            sweeps.append(
                {
                    "prompt_id": prompt["id"],
                    "pair_id": pair["pair_id"],
                    "point_count": len(points),
                    "points": points,
                }
            )
            global_calls = backend.source_suppression_api_calls
            del states, source_tensor, plans, points
            gc.collect()
            torch.mps.empty_cache()
    journal.verify_complete(expected_point_count=global_calls)
    return sweeps, global_calls


__all__ = ["execute_frozen_pairs"]
