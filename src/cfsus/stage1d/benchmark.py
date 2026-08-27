"""Deterministic Stage 1D panel selection and BF16 audit."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cfsus.exceptions import ScientificInputError
from cfsus.stage1c_v3.prediction import ProspectivePair, prospective_pair_record
from cfsus.stage1c_v3.quantization_audit import bf16_round
from cfsus.types import FeatureRef

METHODS = ("S", "margin_only", "influence_only", "random_positive")
DETAILED_BINS = (
    ("B1", 0.02, 0.10, False),
    ("B2", 0.10, 0.40, False),
    ("B3", 0.40, 0.95, True),
)


def _feature_key(feature: FeatureRef) -> tuple[int, int, int]:
    return feature.layer, feature.position, feature.feature_id


def _stable_key(pair: ProspectivePair, prompt_id: str) -> tuple[Any, ...]:
    return (
        prompt_id,
        _feature_key(pair.source),
        _feature_key(pair.target),
        pair.pair_id,
    )


def detailed_requested_alphas(pair: ProspectivePair) -> tuple[float, ...]:
    """Return the frozen coarse plus clipped alpha-relative schedule."""

    values = {0.0, 0.25, 0.5, 0.75, 1.0}
    alpha = pair.predicted_alpha_star
    if alpha is not None and math.isfinite(alpha) and alpha > 0.0:
        values.update(min(1.0, multiplier * alpha) for multiplier in (0.5, 1.0, 1.5))
    return tuple(sorted(values))


def quantization_evidence(pair: ProspectivePair) -> dict[str, Any]:
    """Audit the exact frozen detailed schedule without an intervention call."""

    alpha = pair.predicted_alpha_star
    if pair.q <= 0.0 or alpha is None or not math.isfinite(alpha):
        raise ScientificInputError(
            "quantization audit requires finite positive-q alpha"
        )
    baseline = pair.source_activation
    requested = detailed_requested_alphas(pair)
    mappings: list[dict[str, float]] = []
    applied: dict[float, list[float]] = {}
    for requested_alpha in requested:
        desired = (1.0 - requested_alpha) * baseline
        actual = bf16_round(desired)
        realized = 1.0 - actual / baseline
        if not all(math.isfinite(item) for item in (desired, actual, realized)):
            raise ScientificInputError("quantization mapping is non-finite")
        applied.setdefault(actual, []).append(requested_alpha)
        mappings.append(
            {
                "requested_alpha": requested_alpha,
                "desired_source_activation": desired,
                "applied_bf16_source_activation": actual,
                "realized_suppression": realized,
            }
        )
    predicted_desired = (1.0 - alpha) * baseline
    predicted_applied = bf16_round(predicted_desired)
    predicted_realized = 1.0 - predicted_applied / baseline
    bound = min(1.0, 2.0 * alpha)
    distinct_nonzero_below = len(
        {
            row["applied_bf16_source_activation"]
            for row in mappings
            if 0.0 < row["realized_suppression"] <= bound
        }
    )
    derivative = -pair.q / baseline
    reasons: list[str] = []
    if not 0.02 <= alpha <= 0.95:
        reasons.append("predicted_alpha_outside_0.02_0.95")
    if predicted_applied == bf16_round(baseline):
        reasons.append("predicted_alpha_bf16_noop")
    if distinct_nonzero_below < 3:
        reasons.append("fewer_than_three_distinct_nonzero_below_twice_alpha")
    if not all(
        math.isfinite(item)
        for item in (
            baseline,
            pair.margin,
            derivative,
            predicted_desired,
            predicted_realized,
        )
    ):
        reasons.append("non_finite_required_field")
    return {
        "pair_id": pair.pair_id,
        "predicted_alpha_star": alpha,
        "predicted_alpha_desired_source_activation": predicted_desired,
        "predicted_alpha_applied_bf16_source_activation": predicted_applied,
        "predicted_alpha_realized_suppression": predicted_realized,
        "distinct_applied_value_count": len(applied),
        "distinct_nonzero_at_or_below_twice_alpha": distinct_nonzero_below,
        "collapsed_requested_point_count": len(mappings) - len(applied),
        "calibration_resolvable": not reasons,
        "limitation_reasons": reasons,
        "requested_mappings": mappings,
    }


def _random_key(pair: ProspectivePair, prompt_id: str, domain: str) -> tuple[Any, ...]:
    digest = hashlib.sha256(f"{domain}|{prompt_id}|{pair.pair_id}".encode()).hexdigest()
    return digest, *_stable_key(pair, prompt_id)


def _top(
    candidates: Sequence[ProspectivePair],
    *,
    count: int,
    key: Any,
) -> tuple[ProspectivePair, ...]:
    return tuple(sorted(candidates, key=key)[:count])


@dataclass(frozen=True, slots=True)
class PromptPanelSelection:
    prompt_id: str
    method_pair_ids: dict[str, list[str]]
    directional_pair_ids: list[str]
    detailed_pair_ids: dict[str, str | None]
    execution_pairs: list[dict[str, Any]]
    quantization_audit: dict[str, Any]
    missing_strata: dict[str, str | None]


def select_prompt_panels(
    pairs: Sequence[ProspectivePair],
    *,
    prompt_id: str,
    config: Mapping[str, Any],
) -> PromptPanelSelection:
    """Select all method and detailed panels before any intervention outcome."""

    if not prompt_id.strip():
        raise ScientificInputError("prompt panel selection requires a prompt ID")
    if len({item.pair_id for item in pairs}) != len(pairs):
        raise ScientificInputError("candidate pair IDs are not unique")
    k = int(config["full_ablation_panel"]["per_method_k"])
    positive = tuple(item for item in pairs if item.q > 0.0)
    predicted_crossing = tuple(
        item
        for item in positive
        if item.predicted_alpha_star is not None
        and 0.0 < item.predicted_alpha_star < 1.0
    )

    def stable(item: ProspectivePair) -> tuple[Any, ...]:
        return _stable_key(item, prompt_id)

    selected: dict[str, tuple[ProspectivePair, ...]] = {
        "S": _top(
            predicted_crossing,
            count=k,
            key=lambda item: (-item.susceptibility, *stable(item)),
        ),
        "margin_only": _top(
            positive,
            count=k,
            key=lambda item: (-(1.0 / (item.margin + 1.0e-12)), *stable(item)),
        ),
        "influence_only": _top(
            positive,
            count=k,
            key=lambda item: (-item.q, *stable(item)),
        ),
        "random_positive": _top(
            positive,
            count=k,
            key=lambda item: _random_key(
                item,
                prompt_id,
                str(config["full_ablation_panel"]["random_hash_domain"]),
            ),
        ),
    }
    directional = _top(
        tuple(item for item in pairs if item.q <= 0.0),
        count=int(config["full_ablation_panel"]["directional_k"]),
        key=lambda item: (item.q, *stable(item)),
    )
    audit_hasher = hashlib.sha256()
    reason_counts: Counter[str] = Counter()
    resolvable: dict[str, tuple[ProspectivePair, dict[str, Any]]] = {}
    audit_count = 0
    limited_count = 0
    for pair in sorted(positive, key=stable):
        if pair.predicted_alpha_star is None:
            continue
        evidence = quantization_evidence(pair)
        encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
        audit_hasher.update(encoded)
        audit_hasher.update(b"\n")
        audit_count += 1
        if evidence["calibration_resolvable"]:
            resolvable[pair.pair_id] = (pair, evidence)
        else:
            limited_count += 1
            reason_counts.update(evidence["limitation_reasons"])
    detailed: dict[str, ProspectivePair | None] = {}
    for name, low, high, inclusive_high in DETAILED_BINS:
        candidates = [
            pair
            for pair, _ in resolvable.values()
            if pair.predicted_alpha_star is not None
            and low <= pair.predicted_alpha_star
            and (
                pair.predicted_alpha_star <= high
                if inclusive_high
                else pair.predicted_alpha_star < high
            )
        ]
        detailed[name] = min(
            candidates,
            key=lambda item: (-item.susceptibility, *stable(item)),
            default=None,
        )
    near_candidates = [
        item
        for item in positive
        if item.predicted_alpha_star is not None
        and 1.05 <= item.predicted_alpha_star <= 2.0
    ]
    detailed["near_boundary"] = min(
        near_candidates,
        key=lambda item: (
            item.predicted_alpha_star,
            -item.susceptibility,
            *stable(item),
        ),
        default=None,
    )
    by_id = {item.pair_id: item for item in pairs}
    memberships: dict[str, list[str]] = {}
    for method in METHODS:
        for item in selected[method]:
            memberships.setdefault(item.pair_id, []).append(method)
    for item in directional:
        memberships.setdefault(item.pair_id, []).append("directional")
    detailed_roles: dict[str, str] = {}
    for role, detailed_pair in detailed.items():
        if detailed_pair is not None:
            detailed_roles[detailed_pair.pair_id] = role
    execution_ids = sorted(
        set(memberships) | set(detailed_roles),
        key=lambda pair_id: stable(by_id[pair_id]),
    )
    execution_rows: list[dict[str, Any]] = []
    selected_quantization: list[dict[str, Any]] = []
    for pair_id in execution_ids:
        pair = by_id[pair_id]
        row = prospective_pair_record(pair)
        row.update(
            {
                "prompt_id": prompt_id,
                "margin_only_score": 1.0 / (pair.margin + 1.0e-12),
                "influence_only_score": max(pair.q, 0.0),
                "method_memberships": memberships.get(pair_id, []),
                "full_ablation_selected": pair_id in memberships,
                "detailed_role": detailed_roles.get(pair_id),
                "requested_alphas": list(
                    detailed_requested_alphas(pair)
                    if pair_id in detailed_roles
                    else (1.0,)
                ),
            }
        )
        if pair.q > 0.0 and pair.predicted_alpha_star is not None:
            evidence = quantization_evidence(pair)
            row["quantization_evidence"] = evidence
            selected_quantization.append(evidence)
        else:
            row["quantization_evidence"] = None
        execution_rows.append(row)
    missing: dict[str, str | None] = {
        method: None if len(selected[method]) == k else "fewer_than_four_eligible_pairs"
        for method in METHODS
    }
    missing["directional"] = (
        None
        if len(directional) == int(config["full_ablation_panel"]["directional_k"])
        else "fewer_than_two_q_nonpositive_pairs"
    )
    for role, detailed_pair in detailed.items():
        missing[f"detailed_{role}"] = (
            None if detailed_pair is not None else "no_eligible_pair"
        )
    return PromptPanelSelection(
        prompt_id=prompt_id,
        method_pair_ids={
            name: [item.pair_id for item in selected[name]] for name in METHODS
        },
        directional_pair_ids=[item.pair_id for item in directional],
        detailed_pair_ids={
            name: None if item is None else item.pair_id
            for name, item in detailed.items()
        },
        execution_pairs=execution_rows,
        quantization_audit={
            "audited_positive_defined_alpha_count": audit_count,
            "calibration_resolvable_count": len(resolvable),
            "quantization_limited_count": limited_count,
            "limitation_reason_counts": dict(sorted(reason_counts.items())),
            "full_audit_sha256": audit_hasher.hexdigest(),
            "selected_pair_evidence": selected_quantization,
            "full_pair_rows_persisted": False,
        },
        missing_strata=missing,
    )


__all__ = [
    "DETAILED_BINS",
    "METHODS",
    "PromptPanelSelection",
    "detailed_requested_alphas",
    "quantization_evidence",
    "select_prompt_panels",
]
