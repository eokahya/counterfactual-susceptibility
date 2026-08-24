"""Graph-independent definitions for active-pair prospective validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any

from cfsus.exceptions import NonFiniteInputError, ScientificInputError
from cfsus.types import ActivePairReference, FeatureRef, LocalResponseEstimate


def symmetric_normalized_error(reference: float, candidate: float) -> float:
    """Return ``2|x-y|/(|x|+|y|)`` with an explicit zero policy."""

    if not math.isfinite(reference) or not math.isfinite(candidate):
        raise NonFiniteInputError("symmetric error inputs must be finite")
    denominator = abs(reference) + abs(candidate)
    if denominator == 0.0:
        return 0.0
    result = 2.0 * abs(reference - candidate) / denominator
    if not math.isfinite(result):
        raise NonFiniteInputError("symmetric normalized error overflowed")
    return result


def canonical_pair_id(
    *,
    seed: str,
    prompt_id: str,
    source: FeatureRef,
    target: FeatureRef,
    runtime_fingerprint: str,
) -> str:
    """Hash a canonical endpoint/runtime tuple without numeric outcomes."""

    if not seed.strip() or not prompt_id.strip() or not runtime_fingerprint.strip():
        raise ScientificInputError("pair identity strings must be non-empty")
    if not isinstance(source, FeatureRef) or not isinstance(target, FeatureRef):
        raise ScientificInputError("pair endpoints must be FeatureRef values")
    payload = {
        "prompt_id": prompt_id,
        "runtime_fingerprint": runtime_fingerprint,
        "seed": seed,
        "source": [source.layer, source.position, source.feature_id],
        "target": [target.layer, target.position, target.feature_id],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def select_disjoint_pair_ids(
    pair_ids: tuple[str, ...], *, calibration_count: int, canonical_count: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Select disjoint deterministic splits from unique pre-hashed IDs."""

    for name, value in (
        ("calibration_count", calibration_count),
        ("canonical_count", canonical_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ScientificInputError(f"{name} must be a positive integer")
    if len(pair_ids) != len(set(pair_ids)):
        raise ScientificInputError("pair IDs must be unique")
    if any(
        len(item) != 64
        or any(character not in "0123456789abcdef" for character in item)
        for item in pair_ids
    ):
        raise ScientificInputError("pair IDs must be lowercase SHA-256 values")
    ordered = tuple(sorted(pair_ids))
    needed = calibration_count + canonical_count
    if len(ordered) < needed:
        raise ScientificInputError("eligible pair set is too small for frozen splits")
    calibration = ordered[:calibration_count]
    canonical = ordered[calibration_count:needed]
    if set(calibration) & set(canonical):
        raise AssertionError("deterministic pair split is not disjoint")
    return calibration, canonical


def _plain_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ScientificInputError("graph tensor did not reduce to a list")
    return value


def extract_active_pair_references(
    graph: Any,
    *,
    seed: str,
    prompt_id: str,
    runtime_fingerprint: str,
    per_target_per_sign: int,
) -> tuple[ActivePairReference, ...]:
    """Derive a bounded balanced pair pool from raw target-row/source-column edges."""

    if isinstance(per_target_per_sign, bool) or per_target_per_sign < 1:
        raise ScientificInputError("per_target_per_sign must be positive")
    active_rows = _plain_list(graph.active_features)
    activation_values = _plain_list(graph.activation_values)
    selected_values = _plain_list(graph.selected_features)
    adjacency = _plain_list(graph.adjacency_matrix)
    if not active_rows or len(active_rows) != len(activation_values):
        raise ScientificInputError("raw graph active-feature structure is invalid")
    selected = tuple(int(value) for value in selected_values)
    if selected != tuple(sorted(set(selected))):
        raise ScientificInputError("raw graph selected-feature order is not canonical")
    if any(index < 0 or index >= len(active_rows) for index in selected):
        raise ScientificInputError("raw graph selected-feature index is invalid")
    feature_node_count = len(selected)
    if len(adjacency) < feature_node_count or any(
        not isinstance(row, list) or len(row) < feature_node_count
        for row in adjacency[:feature_node_count]
    ):
        raise ScientificInputError("raw graph adjacency shape is invalid")

    features = tuple(
        FeatureRef(int(row[0]), int(row[1]), int(row[2])) for row in active_rows
    )
    references: list[ActivePairReference] = []
    seen: set[str] = set()
    for target_offset, target_index in enumerate(selected):
        target = features[target_index]
        target_activation = float(activation_values[target_index])
        if not math.isfinite(target_activation) or target_activation == 0.0:
            raise ScientificInputError("raw graph selected target is not active")
        by_sign: dict[int, list[tuple[float, FeatureRef, int, float]]] = {
            -1: [],
            1: [],
        }
        for source_offset, source_index in enumerate(selected):
            source = features[source_index]
            source_activation = float(activation_values[source_index])
            raw_edge = float(adjacency[target_offset][source_offset])
            if (
                source == target
                or source.layer >= target.layer
                or source.position > target.position
                or not math.isfinite(source_activation)
                or source_activation <= 0.0
                or not math.isfinite(raw_edge)
                or raw_edge == 0.0
            ):
                continue
            sign = 1 if raw_edge > 0.0 else -1
            by_sign[sign].append((-abs(raw_edge), source, source_index, raw_edge))
        for sign in (-1, 1):
            by_sign[sign].sort(key=lambda item: (item[0], item[1]))
            for _, source, source_index, raw_edge in by_sign[sign][
                :per_target_per_sign
            ]:
                pair_id = canonical_pair_id(
                    seed=seed,
                    prompt_id=prompt_id,
                    source=source,
                    target=target,
                    runtime_fingerprint=runtime_fingerprint,
                )
                if pair_id in seen:
                    raise ScientificInputError("eligible graph pair is duplicated")
                seen.add(pair_id)
                references.append(
                    ActivePairReference(
                        pair_id=pair_id,
                        source=source,
                        target=target,
                        source_activation=float(activation_values[source_index]),
                        raw_edge=raw_edge,
                    )
                )
    if not references:
        raise ScientificInputError("raw graph has no eligible active pairs")
    return tuple(sorted(references, key=lambda item: item.pair_id))


def select_disjoint_pair_references(
    references: Sequence[ActivePairReference],
    *,
    calibration_count: int,
    canonical_count: int,
    minimum_target_layers: int,
    minimum_target_positions: int,
) -> tuple[tuple[ActivePairReference, ...], tuple[ActivePairReference, ...]]:
    """Freeze disjoint hash-ordered splits with deterministic diversity coverage."""

    if len({item.pair_id for item in references}) != len(references):
        raise ScientificInputError("eligible references contain duplicate IDs")
    ordered = tuple(sorted(references, key=lambda item: item.pair_id))
    if len(ordered) < calibration_count + canonical_count:
        raise ScientificInputError("eligible references are too few for both splits")
    calibration = ordered[:calibration_count]
    remaining = ordered[calibration_count:]
    selected: list[ActivePairReference] = []

    def add_first(predicate: Any) -> None:
        for item in remaining:
            if item not in selected and predicate(item):
                selected.append(item)
                return
        raise ScientificInputError("eligible references cannot satisfy diversity")

    layers = sorted({item.target.layer for item in remaining})
    positions = sorted({item.target.position for item in remaining})
    if len(layers) < minimum_target_layers or len(positions) < minimum_target_positions:
        raise ScientificInputError("eligible references lack target diversity")
    for layer in layers[:minimum_target_layers]:
        add_first(lambda item, layer=layer: item.target.layer == layer)
    for position in positions[:minimum_target_positions]:
        add_first(lambda item, position=position: item.target.position == position)
    add_first(lambda item: item.raw_edge < 0.0)
    add_first(lambda item: item.raw_edge > 0.0)
    for item in remaining:
        if item not in selected:
            selected.append(item)
        if len(selected) == canonical_count:
            break
    if len(selected) != canonical_count:
        raise ScientificInputError(
            "eligible references are too few for canonical split"
        )
    canonical = tuple(selected)
    if {item.pair_id for item in calibration} & {item.pair_id for item in canonical}:
        raise AssertionError("calibration and canonical splits overlap")
    return calibration, canonical


def validate_pair_distribution(
    references: Sequence[ActivePairReference],
    *,
    minimum_pairs: int,
    minimum_target_layers: int,
    minimum_target_positions: int,
    require_both_signs: bool,
) -> None:
    """Fail closed when a frozen pair set lacks the required coverage."""

    if len(references) < minimum_pairs:
        raise ScientificInputError("canonical pair count is below the hard floor")
    if len({item.target.layer for item in references}) < minimum_target_layers:
        raise ScientificInputError("canonical target-layer coverage is insufficient")
    if len({item.target.position for item in references}) < minimum_target_positions:
        raise ScientificInputError("canonical target-position coverage is insufficient")
    if require_both_signs and {
        1 if item.raw_edge > 0.0 else -1 for item in references
    } != {-1, 1}:
        raise ScientificInputError("canonical raw edges do not cover both signs")


def _average_ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average = (start + 1 + end) / 2.0
        for index in range(start, end):
            ranks[ordered[index][0]] = average
        start = end
    return tuple(ranks)


def _pearson(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ScientificInputError("correlation requires aligned vectors of length >=2")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True)
    )
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_ss * right_ss)
    if denominator == 0.0:
        raise ScientificInputError("correlation is undefined for constant ranks")
    result = numerator / denominator
    if not math.isfinite(result):
        raise NonFiniteInputError("correlation is non-finite")
    return result


def _quantile_nearest_rank(values: tuple[float, ...], probability: float) -> float:
    if not values:
        raise ScientificInputError("quantile requires at least one value")
    ordered = sorted(values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


@dataclass(frozen=True, slots=True)
class LocalResponseMetrics:
    """Independently recomputed prospective active-pair metrics."""

    pair_count: int
    above_edge_floor_count: int
    spearman: float
    sign_agreement: float
    median_symmetric_normalized_error: float
    p95_symmetric_normalized_error: float


def compute_local_response_metrics(
    references: tuple[ActivePairReference, ...],
    estimates: tuple[LocalResponseEstimate, ...],
    *,
    edge_floor: float,
) -> LocalResponseMetrics:
    """Reconstruct ``a_j*J_ij`` and compare it with raw unnormalized edges."""

    if not math.isfinite(edge_floor) or edge_floor < 0.0:
        raise ScientificInputError("edge_floor must be finite and non-negative")
    if len(references) != len(estimates) or len(references) < 2:
        raise ScientificInputError("validation requires aligned pairs of length >=2")
    if len({item.pair_id for item in references}) != len(references):
        raise ScientificInputError("validation references contain duplicate pair IDs")

    raw_edges: list[float] = []
    reconstructed: list[float] = []
    errors: list[float] = []
    floor_sign_matches: list[bool] = []
    for reference, estimate in zip(references, estimates, strict=True):
        if reference.source != estimate.source or reference.target != estimate.target:
            raise ScientificInputError("estimate endpoints differ from reference")
        if reference.source_activation != estimate.source_activation:
            raise ScientificInputError("source activation differs from reference")
        rebuilt = estimate.source_activation * estimate.response
        if not math.isfinite(rebuilt):
            raise NonFiniteInputError("reconstructed edge is non-finite")
        raw_edges.append(reference.raw_edge)
        reconstructed.append(rebuilt)
        errors.append(symmetric_normalized_error(reference.raw_edge, rebuilt))
        if abs(reference.raw_edge) >= edge_floor:
            floor_sign_matches.append((reference.raw_edge > 0.0) == (rebuilt > 0.0))
    if not floor_sign_matches:
        raise ScientificInputError("no validation edges meet the frozen edge floor")
    spearman = _pearson(
        _average_ranks(tuple(raw_edges)), _average_ranks(tuple(reconstructed))
    )
    return LocalResponseMetrics(
        pair_count=len(references),
        above_edge_floor_count=len(floor_sign_matches),
        spearman=spearman,
        sign_agreement=sum(floor_sign_matches) / len(floor_sign_matches),
        median_symmetric_normalized_error=median(errors),
        p95_symmetric_normalized_error=_quantile_nearest_rank(tuple(errors), 0.95),
    )
