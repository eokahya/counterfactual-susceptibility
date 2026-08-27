"""Baseline-only v3 scoring and deterministic preregistered pair selection."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cfsus.exceptions import NonFiniteInputError, ScientificInputError
from cfsus.stage1c_v3.config import (
    EXPERIMENT_CLASS,
    PAIR_SEED,
    PROMPT_ID,
)
from cfsus.stage1c_v3.historical import (
    EndpointKey,
    ExactPairKey,
    assert_no_historical_exact_pairs,
    endpoint_overlap_category,
    exact_pair_key,
    exact_pair_record,
)
from cfsus.susceptibility.pairwise import (
    activation_margin,
    classify_predicted_crossing,
    critical_suppression_fraction,
    pairwise_susceptibility,
    suppression_response,
)
from cfsus.types import (
    CrossingStatus,
    FeatureActivity,
    FeatureRef,
    MeasuredFeatureState,
    NearThresholdCandidate,
)


@dataclass(frozen=True, slots=True)
class ProspectivePair:
    """One fresh preregistered Norway-prompt baseline source/target response."""

    pair_id: str
    source: FeatureRef
    target: FeatureRef
    source_activation: float
    target_preactivation: float
    target_threshold: float
    margin: float
    targeted_response: float
    q: float
    susceptibility: float
    predicted_alpha_star: float | None
    status: CrossingStatus

    def __post_init__(self) -> None:
        if len(self.pair_id) != 64 or any(
            character not in "0123456789abcdef" for character in self.pair_id
        ):
            raise ScientificInputError("v3 prospective pair ID is not a SHA-256")
        if self.source.layer >= self.target.layer:
            raise ScientificInputError("v3 source is not layer-upstream")
        if self.source.position > self.target.position:
            raise ScientificInputError("v3 source violates causal order")
        for name in (
            "source_activation",
            "target_preactivation",
            "target_threshold",
            "margin",
            "targeted_response",
            "q",
            "susceptibility",
        ):
            if not math.isfinite(getattr(self, name)):
                raise NonFiniteInputError(f"{name} is non-finite")
        if self.source_activation <= 0.0 or self.margin < 0.0:
            raise ScientificInputError("v3 baseline state is invalid")
        if self.margin != self.target_threshold - self.target_preactivation:
            raise ScientificInputError("v3 margin is inconsistent")
        if self.predicted_alpha_star is not None and not math.isfinite(
            self.predicted_alpha_star
        ):
            raise NonFiniteInputError("predicted alpha is non-finite")


@dataclass(frozen=True, slots=True)
class SelectedPairGroups:
    """Three disjoint prospectively selected v3 groups."""

    primary: tuple[ProspectivePair, ...]
    near_boundary: tuple[ProspectivePair, ...]
    directional: tuple[ProspectivePair, ...]
    near_overlap_fallback_count: int
    directional_overlap_fallback_count: int

    def __post_init__(self) -> None:
        groups = (self.primary, self.near_boundary, self.directional)
        ids = [item.pair_id for group in groups for item in group]
        if len(ids) != len(set(ids)):
            raise ScientificInputError("v3 selected pair groups overlap")
        if len({item.target for item in self.primary}) != len(self.primary):
            raise ScientificInputError("v3 primary selection repeats a target")
        if any(
            count > 2
            for count in Counter(item.source for item in self.primary).values()
        ):
            raise ScientificInputError("v3 primary selection exceeds source cap")


def _feature_record(feature: FeatureRef) -> dict[str, int]:
    return {
        "layer": feature.layer,
        "position": feature.position,
        "feature_id": feature.feature_id,
    }


def source_pool_digest(sources: Sequence[MeasuredFeatureState]) -> str:
    """Digest the exact fresh ordered active-source pool without persisting it."""

    records: list[dict[str, Any]] = []
    expected = tuple(sorted(sources, key=lambda item: item.feature))
    if tuple(sources) != expected or len({item.feature for item in sources}) != len(
        sources
    ):
        raise ScientificInputError("v3 source pool is not unique canonical order")
    for item in sources:
        if item.activity is not FeatureActivity.ACTIVE or item.activation <= 0.0:
            raise ScientificInputError("v3 source pool contains an inactive feature")
        records.append(
            {
                "feature": _feature_record(item.feature),
                "preactivation": item.preactivation,
                "activation": item.activation,
                "threshold": item.threshold,
            }
        )
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def target_pool_digest(targets: Sequence[NearThresholdCandidate]) -> str:
    """Digest the exact fresh ordered inactive-target pool."""

    records = [
        {
            "feature": _feature_record(item.feature),
            "preactivation": item.preactivation,
            "activation": item.activation,
            "threshold": item.threshold,
            "margin": item.margin,
        }
        for item in targets
    ]
    if tuple(targets) != tuple(sorted(targets, key=lambda item: item.sort_key)):
        raise ScientificInputError("v3 target pool order differs from scanner order")
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def causally_eligible(source: FeatureRef, target: FeatureRef) -> bool:
    """Apply strict layer-upstream and non-decreasing position ordering."""

    return source.layer < target.layer and source.position <= target.position


def filter_source_pool(
    sources: Sequence[MeasuredFeatureState],
    targets: Sequence[NearThresholdCandidate],
    *,
    maximum_sources: int,
) -> tuple[MeasuredFeatureState, ...]:
    """Keep every fresh active feature causal to at least one target."""

    if isinstance(maximum_sources, bool) or maximum_sources < 1:
        raise ScientificInputError("maximum source count must be positive")
    result = tuple(
        sorted(
            (
                source
                for source in sources
                if source.activity is FeatureActivity.ACTIVE
                and source.activation > 0.0
                and any(
                    causally_eligible(source.feature, target.feature)
                    for target in targets
                )
            ),
            key=lambda item: item.feature,
        )
    )
    if len(result) > maximum_sources:
        raise ScientificInputError("v3 active source pool exceeds the frozen cap")
    if len({item.feature for item in result}) != len(result):
        raise ScientificInputError("v3 active source pool contains duplicates")
    return result


def filter_target_pool(
    targets: Sequence[NearThresholdCandidate], sources: Sequence[MeasuredFeatureState]
) -> tuple[NearThresholdCandidate, ...]:
    """Exclude only fresh targets with no causally upstream active source."""

    return tuple(
        target
        for target in targets
        if any(causally_eligible(source.feature, target.feature) for source in sources)
    )


def canonical_v3_pair_id(
    *,
    source: FeatureRef,
    target: FeatureRef,
    runtime_fingerprint: str,
    prompt_id: str = PROMPT_ID,
    seed: str = PAIR_SEED,
    experiment_class: str = EXPERIMENT_CLASS,
) -> str:
    """Hash the explicit v3 class/prompt/runtime endpoint identity only."""

    if (
        not seed.strip()
        or not prompt_id.strip()
        or not runtime_fingerprint.strip()
        or not experiment_class.strip()
    ):
        raise ScientificInputError("v3 pair identity strings must be non-empty")
    if not isinstance(source, FeatureRef) or not isinstance(target, FeatureRef):
        raise ScientificInputError("v3 pair endpoints must be FeatureRef values")
    payload = {
        "experiment_class": experiment_class,
        "prompt_id": prompt_id,
        "runtime_fingerprint": runtime_fingerprint,
        "seed": seed,
        "source": [source.layer, source.position, source.feature_id],
        "target": [target.layer, target.position, target.feature_id],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_prospective_pair(
    *,
    source: MeasuredFeatureState,
    target: NearThresholdCandidate,
    targeted_response: float,
    seed: str = PAIR_SEED,
    prompt_id: str = PROMPT_ID,
    runtime_fingerprint: str,
    epsilon: float,
    tolerance: float,
    experiment_class: str = EXPERIMENT_CLASS,
) -> ProspectivePair:
    """Build one signed score from fresh baseline scalar measurements."""

    if source.activity is not FeatureActivity.ACTIVE or source.activation <= 0.0:
        raise ScientificInputError("v3 source is not baseline active")
    if not causally_eligible(source.feature, target.feature):
        raise ScientificInputError("v3 endpoints are not causally eligible")
    margin = activation_margin(
        target.preactivation, target.threshold, tolerance=tolerance
    )
    if margin < 0.0:
        raise ScientificInputError("v3 exact inactive target has a negative margin")
    q = suppression_response(source.activation, targeted_response)
    susceptibility = pairwise_susceptibility(
        q, margin, epsilon=epsilon, tolerance=tolerance
    )
    alpha = critical_suppression_fraction(margin, q, tolerance=tolerance)
    status = classify_predicted_crossing(margin, q, tolerance=tolerance)
    return ProspectivePair(
        pair_id=canonical_v3_pair_id(
            seed=seed,
            prompt_id=prompt_id,
            source=source.feature,
            target=target.feature,
            runtime_fingerprint=runtime_fingerprint,
            experiment_class=experiment_class,
        ),
        source=source.feature,
        target=target.feature,
        source_activation=source.activation,
        target_preactivation=target.preactivation,
        target_threshold=target.threshold,
        margin=margin,
        targeted_response=float(targeted_response),
        q=q,
        susceptibility=susceptibility,
        predicted_alpha_star=alpha,
        status=status,
    )


def prospective_pair_record(pair: ProspectivePair) -> dict[str, Any]:
    """Serialize baseline quantities only; intervention outcome is absent."""

    return {
        "pair_id": pair.pair_id,
        "source": _feature_record(pair.source),
        "target": _feature_record(pair.target),
        "source_activation": pair.source_activation,
        "target_preactivation": pair.target_preactivation,
        "target_threshold": pair.target_threshold,
        "margin": pair.margin,
        "targeted_response": pair.targeted_response,
        "q": pair.q,
        "susceptibility": pair.susceptibility,
        "predicted_alpha_star": pair.predicted_alpha_star,
        "predicted_status": pair.status.value,
    }


def pair_score_digest(pairs: Sequence[ProspectivePair]) -> str:
    """Digest fresh pair scores in endpoint order without a derivative matrix."""

    ordered = tuple(sorted(pairs, key=lambda item: (item.target, item.source)))
    if len({item.pair_id for item in ordered}) != len(ordered):
        raise ScientificInputError("v3 eligible pair set contains duplicate IDs")
    hasher = hashlib.sha256()
    for item in ordered:
        hasher.update(
            json.dumps(
                prospective_pair_record(item),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        hasher.update(b"\n")
    return hasher.hexdigest()


def _select_primary(
    candidates: Sequence[ProspectivePair], *, maximum: int
) -> tuple[ProspectivePair, ...]:
    ordered = sorted(
        (
            item
            for item in candidates
            if item.status is CrossingStatus.DEFINITELY_CROSSING
            and item.predicted_alpha_star is not None
            and 0.0 < item.predicted_alpha_star < 1.0
        ),
        key=lambda item: (
            -item.susceptibility,
            item.predicted_alpha_star,
            item.target,
            item.source,
        ),
    )
    selected: list[ProspectivePair] = []
    targets: set[FeatureRef] = set()
    source_counts: Counter[FeatureRef] = Counter()
    for item in ordered:
        if item.target in targets or source_counts[item.source] >= 2:
            continue
        selected.append(item)
        targets.add(item.target)
        source_counts[item.source] += 1
        if len(selected) == maximum:
            break
    return tuple(selected)


def _select_controls(
    candidates: Sequence[ProspectivePair],
    *,
    maximum: int,
    previously_used_targets: set[FeatureRef],
    key: Any,
) -> tuple[tuple[ProspectivePair, ...], int]:
    ordered = sorted(candidates, key=key)
    selected: list[ProspectivePair] = []
    group_targets: set[FeatureRef] = set()
    for prefer_unused in (True, False):
        for item in ordered:
            if item in selected or item.target in group_targets:
                continue
            if (item.target not in previously_used_targets) is not prefer_unused:
                continue
            selected.append(item)
            group_targets.add(item.target)
            if len(selected) == maximum:
                break
        if len(selected) == maximum:
            break
    fallback_count = sum(item.target in previously_used_targets for item in selected)
    return tuple(selected), fallback_count


def select_pair_groups(
    pairs: Sequence[ProspectivePair], *, selection: Mapping[str, Any], tolerance: float
) -> SelectedPairGroups:
    """Apply the frozen deterministic v3 group rules."""

    primary = _select_primary(pairs, maximum=int(selection["primary_maximum"]))
    if not primary:
        return SelectedPairGroups(
            primary=(),
            near_boundary=(),
            directional=(),
            near_overlap_fallback_count=0,
            directional_overlap_fallback_count=0,
        )
    primary_targets = {item.target for item in primary}
    near_candidates = tuple(
        item
        for item in pairs
        if item.q > 0.0
        and item.status is CrossingStatus.NOT_CROSSING
        and item.predicted_alpha_star is not None
        and item.predicted_alpha_star > 1.0
        and item.margin - item.q > tolerance
    )
    near, near_fallback = _select_controls(
        near_candidates,
        maximum=int(selection["near_boundary_maximum"]),
        previously_used_targets=primary_targets,
        key=lambda item: (item.predicted_alpha_star - 1.0, item.target, item.source),
    )
    used_targets = primary_targets | {item.target for item in near}
    directional_candidates = tuple(
        item
        for item in pairs
        if item.q <= 0.0
        and item.status is CrossingStatus.NOT_CROSSING
        and item.margin > tolerance
    )
    directional, directional_fallback = _select_controls(
        directional_candidates,
        maximum=int(selection["directional_maximum"]),
        previously_used_targets=used_targets,
        key=lambda item: (
            -(abs(item.q) / (item.margin + 1.0e-12)),
            item.target,
            item.source,
        ),
    )
    return SelectedPairGroups(
        primary=primary,
        near_boundary=near,
        directional=directional,
        near_overlap_fallback_count=near_fallback,
        directional_overlap_fallback_count=directional_fallback,
    )


def requested_schedule(
    pair: ProspectivePair, *, coarse_alphas: Iterable[float], alpha_hat_offset: float
) -> tuple[float, ...]:
    """Return the fixed coarse grid plus clipped alpha-star probes."""

    values = {float(item) for item in coarse_alphas}
    alpha = pair.predicted_alpha_star
    if alpha is not None and 0.0 <= alpha <= 1.0:
        values.update(
            {
                max(0.0, min(1.0, alpha - alpha_hat_offset)),
                alpha,
                max(0.0, min(1.0, alpha + alpha_hat_offset)),
            }
        )
    if any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in values):
        raise ScientificInputError("v3 requested schedule contains an invalid alpha")
    return tuple(sorted(values))


def selected_group_records(
    groups: SelectedPairGroups,
    *,
    schedule: Mapping[str, Any],
    denylist: frozenset[ExactPairKey],
    historical_endpoints: frozenset[EndpointKey],
) -> dict[str, list[dict[str, Any]]]:
    selected = tuple(
        pair
        for pairs in (groups.primary, groups.near_boundary, groups.directional)
        for pair in pairs
    )
    assert_no_historical_exact_pairs(selected, denylist)
    result: dict[str, list[dict[str, Any]]] = {}
    for name, pairs in (
        ("primary", groups.primary),
        ("near_boundary", groups.near_boundary),
        ("directional", groups.directional),
    ):
        rows: list[dict[str, Any]] = []
        for pair in pairs:
            row = prospective_pair_record(pair)
            row["group"] = name
            pair_key = exact_pair_key(pair.source, pair.target)
            row["exact_pair_key"] = exact_pair_record(pair_key)
            row["endpoint_overlap_category"] = endpoint_overlap_category(
                pair.source,
                pair.target,
                denylist=denylist,
                endpoints=historical_endpoints,
            ).value
            row["requested_alphas"] = list(
                requested_schedule(
                    pair,
                    coarse_alphas=schedule["coarse_alphas"],
                    alpha_hat_offset=float(schedule["alpha_hat_offset"]),
                )
            )
            rows.append(row)
        result[name] = rows
    return result


__all__ = [
    "ProspectivePair",
    "SelectedPairGroups",
    "build_prospective_pair",
    "canonical_v3_pair_id",
    "causally_eligible",
    "filter_source_pool",
    "filter_target_pool",
    "pair_score_digest",
    "prospective_pair_record",
    "requested_schedule",
    "select_pair_groups",
    "selected_group_records",
    "source_pool_digest",
    "target_pool_digest",
]
