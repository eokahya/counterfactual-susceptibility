from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from cfsus.exceptions import NonFiniteInputError, ScientificInputError
from cfsus.scanning.near_threshold import (
    compare_scanner_results,
    dense_group_oracle,
    scan_feature_group,
    scan_groups,
)
from cfsus.types import FeatureActivity, FeatureRef, MeasuredFeatureState


def _states(layer: int, position: int) -> tuple[MeasuredFeatureState, ...]:
    preactivations = (-2.0, 0.0, 0.5, 1.0, 1.0, 1.5, 3.0, -1.0, 0.75, 2.0)
    thresholds = (0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 2.0, -0.5, 1.0, 3.0)
    result: list[MeasuredFeatureState] = []
    for feature_id, (z_value, threshold) in enumerate(
        zip(preactivations, thresholds, strict=True)
    ):
        active = z_value > threshold
        result.append(
            MeasuredFeatureState(
                feature=FeatureRef(layer, position, feature_id),
                preactivation=z_value,
                activation=z_value if active else 0.0,
                threshold=threshold,
                activity=(
                    FeatureActivity.ACTIVE if active else FeatureActivity.INACTIVE
                ),
                device="mps:0",
                dtype="torch.bfloat16",
            )
        )
    return tuple(result)


@pytest.mark.parametrize("chunk_size", [3, 4, 7])
def test_chunked_scanner_exactly_matches_dense_oracle(chunk_size: int) -> None:
    states = _states(2, 4)
    oracle = dense_group_oracle(
        layer=2,
        position=4,
        feature_count=len(states),
        top_k=5,
        project_dense=lambda start, end: states[start:end],
    )
    observed = scan_feature_group(
        layer=2,
        position=4,
        feature_count=len(states),
        chunk_size=chunk_size,
        top_k=5,
        project_chunk=lambda start, end: states[start:end],
    )
    assert observed.candidates == oracle.candidates
    assert [item.feature.feature_id for item in observed.candidates] == [1, 3, 8, 2, 7]
    assert observed.active_count == 2
    assert observed.inactive_count == 8
    assert observed.maximum_retained_candidates <= 5 + chunk_size


def test_threshold_equality_is_inactive_and_active_features_are_excluded() -> None:
    states = _states(2, 4)
    result = scan_feature_group(
        layer=2,
        position=4,
        feature_count=len(states),
        chunk_size=3,
        top_k=8,
        project_chunk=lambda start, end: states[start:end],
    )
    retained = {item.feature.feature_id for item in result.candidates}
    assert {1, 3}.issubset(retained)
    assert {5, 6}.isdisjoint(retained)
    assert all(item.activation == 0.0 for item in result.candidates)


def test_group_and_global_tie_order_is_exact_and_serializable() -> None:
    groups = ((0, 1), (0, 2), (1, 1))
    states = {group: _states(*group) for group in groups}
    result = scan_groups(
        groups=groups,
        feature_count=10,
        chunk_size=4,
        top_k_per_group=3,
        global_top_k=5,
        project_group_chunk=lambda layer, position, start, end: states[
            (layer, position)
        ][start:end],
    )
    assert tuple(item.sort_key for item in result.global_candidates) == tuple(
        sorted(item.sort_key for item in result.global_candidates)
    )
    json.dumps(asdict(result), allow_nan=False)


def test_compare_scanner_results_rejects_changed_candidate_order() -> None:
    states = _states(2, 4)
    reference = scan_groups(
        groups=((2, 4),),
        feature_count=10,
        chunk_size=3,
        top_k_per_group=5,
        global_top_k=5,
        project_group_chunk=lambda _layer, _position, start, end: states[start:end],
    )
    changed_group = replace(
        reference.groups[0], candidates=tuple(reversed(reference.groups[0].candidates))
    )
    changed = replace(reference, groups=(changed_group,))
    with pytest.raises(ScientificInputError, match="per-group candidates"):
        compare_scanner_results(reference, changed)


def test_scanner_rejects_missing_duplicate_or_foreign_feature_refs() -> None:
    states = _states(2, 4)

    def duplicate(start: int, end: int) -> tuple[MeasuredFeatureState, ...]:
        chunk = list(states[start:end])
        if start == 0:
            chunk[1] = replace(chunk[1], feature=FeatureRef(2, 4, 0))
        return tuple(chunk)

    with pytest.raises(ScientificInputError, match="missing or out of order"):
        scan_feature_group(
            layer=2,
            position=4,
            feature_count=10,
            chunk_size=3,
            top_k=5,
            project_chunk=duplicate,
        )

    foreign = replace(states[0], feature=FeatureRef(3, 4, 0))
    with pytest.raises(ScientificInputError, match="another group"):
        scan_feature_group(
            layer=2,
            position=4,
            feature_count=10,
            chunk_size=3,
            top_k=5,
            project_chunk=lambda start, end: (
                (foreign, *states[1:end]) if start == 0 else states[start:end]
            ),
        )


def test_scanner_rejects_empty_groups_invalid_bounds_and_nonfinite_state() -> None:
    with pytest.raises(ScientificInputError, match="non-empty and unique"):
        scan_groups(
            groups=(),
            feature_count=10,
            chunk_size=3,
            top_k_per_group=5,
            global_top_k=5,
            project_group_chunk=lambda *_args: (),
        )
    with pytest.raises(ScientificInputError, match="must not exceed"):
        scan_feature_group(
            layer=0,
            position=0,
            feature_count=2,
            chunk_size=1,
            top_k=3,
            project_chunk=lambda *_args: (),
        )
    with pytest.raises(NonFiniteInputError):
        MeasuredFeatureState(
            feature=FeatureRef(0, 0, 0),
            preactivation=float("nan"),
            activation=0.0,
            threshold=1.0,
            activity=FeatureActivity.INACTIVE,
            device="mps:0",
            dtype="torch.bfloat16",
        )


def test_loaded_state_rejects_inconsistent_activity() -> None:
    with pytest.raises(ScientificInputError, match="inactive loaded state"):
        MeasuredFeatureState(
            feature=FeatureRef(0, 0, 0),
            preactivation=2.0,
            activation=0.0,
            threshold=1.0,
            activity=FeatureActivity.INACTIVE,
            device="mps:0",
            dtype="torch.bfloat16",
        )
