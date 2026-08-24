"""Chunked near-threshold scanner with an ephemeral dense oracle."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from cfsus.exceptions import ScientificInputError
from cfsus.types import (
    FeatureActivity,
    MeasuredFeatureState,
    NearThresholdCandidate,
)

ChunkProjector = Callable[[int, int], Sequence[MeasuredFeatureState]]
GroupProjector = Callable[[int, int, int, int], Sequence[MeasuredFeatureState]]


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ScientificInputError(f"{name} must be a positive integer")
    return value


def _candidate(state: MeasuredFeatureState) -> NearThresholdCandidate | None:
    if not isinstance(state, MeasuredFeatureState):
        raise ScientificInputError("projector returned a non-state value")
    if state.activity is FeatureActivity.ACTIVE:
        return None
    if state.activation != 0.0 or state.preactivation > state.threshold:
        raise ScientificInputError(
            "loaded inactive state violates exact gate semantics"
        )
    return NearThresholdCandidate(
        feature=state.feature,
        preactivation=state.preactivation,
        activation=state.activation,
        threshold=state.threshold,
        margin=state.inactive_margin,
        device=state.device,
        dtype=state.dtype,
    )


@dataclass(frozen=True, slots=True)
class GroupScanResult:
    """Bounded retained candidates for one layer/token-position group."""

    layer: int
    position: int
    feature_count: int
    chunk_size: int
    top_k: int
    inactive_count: int
    active_count: int
    chunks_processed: int
    maximum_retained_candidates: int
    candidates: tuple[NearThresholdCandidate, ...]


@dataclass(frozen=True, slots=True)
class ScannerResult:
    """Compact per-group and global scanner output."""

    chunk_size: int
    top_k_per_group: int
    global_top_k: int
    groups: tuple[GroupScanResult, ...]
    global_candidates: tuple[NearThresholdCandidate, ...]


def scan_feature_group(
    *,
    layer: int,
    position: int,
    feature_count: int,
    chunk_size: int,
    top_k: int,
    project_chunk: ChunkProjector,
) -> GroupScanResult:
    """Scan one group without retaining more than ``top_k+chunk_size`` records."""

    for name, value in (
        ("feature_count", feature_count),
        ("chunk_size", chunk_size),
        ("top_k", top_k),
    ):
        _positive_integer(name, value)
    if top_k > feature_count:
        raise ScientificInputError("top_k must not exceed feature_count")
    if layer < 0 or position < 0:
        raise ScientificInputError("layer and position must be non-negative")

    retained: list[NearThresholdCandidate] = []
    seen: set[int] = set()
    active_count = 0
    inactive_count = 0
    chunks_processed = 0
    maximum_retained = 0
    for start in range(0, feature_count, chunk_size):
        end = min(feature_count, start + chunk_size)
        states = tuple(project_chunk(start, end))
        if len(states) != end - start:
            raise ScientificInputError("projector chunk length is incorrect")
        expected_ids = tuple(range(start, end))
        observed_ids = tuple(state.feature.feature_id for state in states)
        if observed_ids != expected_ids:
            raise ScientificInputError(
                "projector feature IDs are missing or out of order"
            )
        for state in states:
            if state.feature.layer != layer or state.feature.position != position:
                raise ScientificInputError(
                    "projector returned a state for another group"
                )
            if state.feature.feature_id in seen:
                raise ScientificInputError("projector returned a duplicate feature")
            seen.add(state.feature.feature_id)
            if state.activity is FeatureActivity.ACTIVE:
                active_count += 1
            else:
                inactive_count += 1
            candidate = _candidate(state)
            if candidate is not None:
                retained.append(candidate)
        maximum_retained = max(maximum_retained, len(retained))
        retained.sort(key=lambda item: item.sort_key)
        del retained[top_k:]
        chunks_processed += 1

    if seen != set(range(feature_count)):
        raise ScientificInputError("scanner did not cover every feature exactly once")
    if active_count + inactive_count != feature_count:
        raise ScientificInputError("scanner activity counts do not cover the group")
    return GroupScanResult(
        layer=layer,
        position=position,
        feature_count=feature_count,
        chunk_size=chunk_size,
        top_k=top_k,
        inactive_count=inactive_count,
        active_count=active_count,
        chunks_processed=chunks_processed,
        maximum_retained_candidates=maximum_retained,
        candidates=tuple(retained),
    )


def dense_group_oracle(
    *,
    layer: int,
    position: int,
    feature_count: int,
    top_k: int,
    project_dense: ChunkProjector,
) -> GroupScanResult:
    """Run the bounded oracle by projecting exactly one complete feature group."""

    return scan_feature_group(
        layer=layer,
        position=position,
        feature_count=feature_count,
        chunk_size=feature_count,
        top_k=top_k,
        project_chunk=project_dense,
    )


def scan_groups(
    *,
    groups: Iterable[tuple[int, int]],
    feature_count: int,
    chunk_size: int,
    top_k_per_group: int,
    global_top_k: int,
    project_group_chunk: GroupProjector,
) -> ScannerResult:
    """Scan unique groups and perform a second bounded deterministic merge."""

    _positive_integer("global_top_k", global_top_k)
    normalized_groups = tuple(groups)
    if not normalized_groups or len(normalized_groups) != len(set(normalized_groups)):
        raise ScientificInputError("groups must be non-empty and unique")
    group_results: list[GroupScanResult] = []
    global_candidates: list[NearThresholdCandidate] = []
    for layer, position in normalized_groups:

        def project_chunk(
            start: int,
            end: int,
            *,
            selected_layer: int = layer,
            selected_position: int = position,
        ) -> Sequence[MeasuredFeatureState]:
            return project_group_chunk(selected_layer, selected_position, start, end)

        result = scan_feature_group(
            layer=layer,
            position=position,
            feature_count=feature_count,
            chunk_size=chunk_size,
            top_k=top_k_per_group,
            project_chunk=project_chunk,
        )
        group_results.append(result)
        global_candidates.extend(result.candidates)
        global_candidates.sort(key=lambda item: item.sort_key)
        del global_candidates[global_top_k:]
    return ScannerResult(
        chunk_size=chunk_size,
        top_k_per_group=top_k_per_group,
        global_top_k=global_top_k,
        groups=tuple(group_results),
        global_candidates=tuple(global_candidates),
    )


def compare_scanner_results(reference: ScannerResult, candidate: ScannerResult) -> None:
    """Require exact candidate identity and order while ignoring chunk diagnostics."""

    if not isinstance(reference, ScannerResult) or not isinstance(
        candidate, ScannerResult
    ):
        raise ScientificInputError("scanner comparisons require ScannerResult values")
    if len(reference.groups) != len(candidate.groups):
        raise ScientificInputError("scanner group counts differ")
    for expected, observed in zip(reference.groups, candidate.groups, strict=True):
        if (expected.layer, expected.position) != (observed.layer, observed.position):
            raise ScientificInputError("scanner group order differs")
        if expected.feature_count != observed.feature_count:
            raise ScientificInputError("scanner feature counts differ")
        if expected.inactive_count != observed.inactive_count:
            raise ScientificInputError("scanner inactive counts differ")
        if expected.active_count != observed.active_count:
            raise ScientificInputError("scanner active counts differ")
        if expected.candidates != observed.candidates:
            raise ScientificInputError("scanner per-group candidates differ")
    if reference.global_candidates != candidate.global_candidates:
        raise ScientificInputError("scanner global candidates differ")
