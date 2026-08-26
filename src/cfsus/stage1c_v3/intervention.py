"""Frozen v3 source-suppression schedule and BF16 application mapping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from cfsus.exceptions import NonFiniteInputError, ScientificInputError


@dataclass(frozen=True, slots=True)
class RequestedApplication:
    requested_alpha: float
    desired_high_precision: float
    applied_bf16: float
    realized_suppression: float


@dataclass(frozen=True, slots=True)
class AppliedValuePlan:
    applied_bf16: float
    realized_suppression: float
    requests: tuple[RequestedApplication, ...]
    tensor: Any

    @property
    def representative_requested_alpha(self) -> float:
        return min(item.requested_alpha for item in self.requests)


def plan_applied_values(
    baseline: Any, requested_alphas: tuple[float, ...], torch: Any
) -> tuple[AppliedValuePlan, ...]:
    """Compute high-precision desired values and deduplicate applied BF16 values."""

    if (
        not isinstance(baseline, torch.Tensor)
        or baseline.device.type != "mps"
        or baseline.dtype != torch.bfloat16
        or baseline.numel() != 1
    ):
        raise ScientificInputError("baseline source activation must be scalar MPS/BF16")
    baseline_value = float(baseline.item())
    if not math.isfinite(baseline_value) or baseline_value <= 0.0:
        raise ScientificInputError("baseline source activation must be finite positive")
    if not requested_alphas or tuple(requested_alphas) != tuple(
        sorted(set(requested_alphas))
    ):
        raise ScientificInputError("requested alphas must be unique canonical order")
    grouped: dict[float, list[RequestedApplication]] = {}
    tensors: dict[float, Any] = {}
    for alpha in requested_alphas:
        if (
            isinstance(alpha, bool)
            or not math.isfinite(alpha)
            or not 0.0 <= alpha <= 1.0
        ):
            raise ScientificInputError("requested alpha is outside [0,1]")
        desired = (1.0 - float(alpha)) * baseline_value
        if not math.isfinite(desired):
            raise NonFiniteInputError("desired source activation is non-finite")
        tensor = torch.tensor(desired, device="mps", dtype=torch.bfloat16).reshape(())
        if tensor.device.type != "mps" or tensor.dtype != torch.bfloat16:
            raise ScientificInputError("applied source value moved off MPS/BF16")
        applied = float(tensor.item())
        realized = 1.0 - applied / baseline_value
        if not math.isfinite(realized) or not -1.0e-12 <= realized <= 1.0 + 1.0e-12:
            raise ScientificInputError("realized suppression is outside [0,1]")
        record = RequestedApplication(float(alpha), desired, applied, realized)
        grouped.setdefault(applied, []).append(record)
        tensors[applied] = tensor
    plans = tuple(
        AppliedValuePlan(
            applied_bf16=applied,
            realized_suppression=items[0].realized_suppression,
            requests=tuple(sorted(items, key=lambda item: item.requested_alpha)),
            tensor=tensors[applied],
        )
        for applied, items in grouped.items()
    )
    ordered = tuple(sorted(plans, key=lambda item: item.realized_suppression))
    if any(
        later.realized_suppression <= earlier.realized_suppression
        for earlier, later in pairwise(ordered)
    ):
        raise ScientificInputError(
            "deduplicated realized suppressions are not increasing"
        )
    return ordered


def bisection_requested_alpha(lower_realized: float, upper_realized: float) -> float:
    """Return the deterministic midpoint of a strict realized bracket."""

    for name, value in (("lower", lower_realized), ("upper", upper_realized)):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ScientificInputError(f"{name} realized suppression is invalid")
    if lower_realized >= upper_realized:
        raise ScientificInputError("bisection requires a strict bracket")
    return (lower_realized + upper_realized) / 2.0


def applied_plan_record(plan: AppliedValuePlan) -> dict[str, Any]:
    """Serialize requested-versus-applied mappings without a tensor payload."""

    return {
        "representative_requested_alpha": plan.representative_requested_alpha,
        "requested_mappings": [
            {
                "requested_alpha": item.requested_alpha,
                "desired_high_precision": item.desired_high_precision,
                "actual_bf16_value_passed": item.applied_bf16,
                "realized_suppression": item.realized_suppression,
            }
            for item in plan.requests
        ],
        "actual_bf16_value_passed": plan.applied_bf16,
        "realized_suppression": plan.realized_suppression,
        "collapsed_request_count": len(plan.requests),
        "source_value_device": "mps:0",
        "source_value_dtype": "torch.bfloat16",
    }


__all__ = [
    "AppliedValuePlan",
    "RequestedApplication",
    "applied_plan_record",
    "bisection_requested_alpha",
    "plan_applied_values",
]
