from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from cfsus.exceptions import NonFiniteInputError, ScientificInputError
from cfsus.responses.validation import (
    canonical_pair_id,
    compute_local_response_metrics,
    extract_active_pair_references,
    select_disjoint_pair_ids,
    select_disjoint_pair_references,
    symmetric_normalized_error,
    validate_pair_distribution,
)
from cfsus.types import ActivePairReference, FeatureRef, LocalResponseEstimate


def _pair(
    index: int, raw_edge: float
) -> tuple[ActivePairReference, LocalResponseEstimate]:
    source = FeatureRef(index % 2, index % 3, 100 + index)
    target = FeatureRef(6 + index % 7, 2 + index % 3, 500 + index)
    pair_id = canonical_pair_id(
        seed="stage1b-fixed-seed",
        prompt_id="pilot",
        source=source,
        target=target,
        runtime_fingerprint="gemma3-270m-mps-bf16-pinned",
    )
    source_activation = 2.0 + index
    reference = ActivePairReference(
        pair_id=pair_id,
        source=source,
        target=target,
        source_activation=source_activation,
        raw_edge=raw_edge,
    )
    estimate = LocalResponseEstimate(
        source=source,
        target=target,
        source_activation=source_activation,
        target_preactivation=3.0,
        response=raw_edge / source_activation,
        device="mps:0",
        dtype="torch.bfloat16",
        method="target_encoder_reverse_vjp_source_decoder_contraction",
        convention="attribution_matched_target_preactivation",
    )
    return reference, estimate


def test_symmetric_normalized_error_zero_and_sign_policy() -> None:
    assert symmetric_normalized_error(0.0, 0.0) == 0.0
    assert symmetric_normalized_error(1.0, 0.0) == 2.0
    assert symmetric_normalized_error(-1.0, 1.0) == 2.0
    with pytest.raises(NonFiniteInputError):
        symmetric_normalized_error(float("inf"), 1.0)


def test_pair_ids_are_endpoint_deterministic_and_splits_disjoint() -> None:
    ids = tuple(_pair(index, 1.0 + index)[0].pair_id for index in range(10))
    calibration, canonical = select_disjoint_pair_ids(
        tuple(reversed(ids)), calibration_count=3, canonical_count=5
    )
    assert calibration == tuple(sorted(ids))[:3]
    assert canonical == tuple(sorted(ids))[3:8]
    assert set(calibration).isdisjoint(canonical)
    with pytest.raises(ScientificInputError, match="unique"):
        select_disjoint_pair_ids((*ids, ids[0]), calibration_count=3, canonical_count=5)


def test_exact_reconstruction_passes_all_metric_floors() -> None:
    pairs = tuple(
        _pair(index, (-1.0 if index % 2 else 1.0) * (index + 1)) for index in range(8)
    )
    metrics = compute_local_response_metrics(
        tuple(item[0] for item in pairs),
        tuple(item[1] for item in pairs),
        edge_floor=0.5,
    )
    assert metrics.pair_count == 8
    assert metrics.spearman == pytest.approx(1.0)
    assert metrics.sign_agreement == 1.0
    assert metrics.median_symmetric_normalized_error == 0.0
    assert metrics.p95_symmetric_normalized_error == 0.0


def test_sign_reversal_is_visible_to_independent_metrics() -> None:
    pairs = tuple(_pair(index, float(index + 1)) for index in range(4))
    estimates = tuple(replace(item[1], response=-item[1].response) for item in pairs)
    metrics = compute_local_response_metrics(
        tuple(item[0] for item in pairs), estimates, edge_floor=0.0
    )
    assert metrics.sign_agreement == 0.0
    assert metrics.median_symmetric_normalized_error == 2.0


def test_response_validation_rejects_endpoint_activation_and_duplicate_mismatch() -> (
    None
):
    pairs = tuple(_pair(index, float(index + 1)) for index in range(3))
    references = tuple(item[0] for item in pairs)
    estimates = tuple(item[1] for item in pairs)
    with pytest.raises(ScientificInputError, match="endpoints"):
        compute_local_response_metrics(
            references,
            (replace(estimates[0], target=FeatureRef(9, 2, 999)), *estimates[1:]),
            edge_floor=0.0,
        )
    with pytest.raises(ScientificInputError, match="activation"):
        compute_local_response_metrics(
            references,
            (replace(estimates[0], source_activation=9.0), *estimates[1:]),
            edge_floor=0.0,
        )
    duplicated = (references[0], references[0], references[2])
    with pytest.raises(ScientificInputError, match="duplicate"):
        compute_local_response_metrics(duplicated, estimates, edge_floor=0.0)


def test_local_response_rejects_invalid_order_inactive_source_or_graph_use() -> None:
    source = FeatureRef(3, 3, 1)
    target = FeatureRef(2, 3, 2)
    with pytest.raises(ScientificInputError, match="upstream"):
        LocalResponseEstimate(
            source=source,
            target=target,
            source_activation=1.0,
            target_preactivation=1.0,
            response=1.0,
            device="mps:0",
            dtype="torch.bfloat16",
            method="vjp",
            convention="frozen",
        )
    valid_source = FeatureRef(1, 2, 1)
    valid_target = FeatureRef(2, 3, 2)
    with pytest.raises(ScientificInputError, match="baseline-active"):
        LocalResponseEstimate(
            source=valid_source,
            target=valid_target,
            source_activation=0.0,
            target_preactivation=1.0,
            response=1.0,
            device="mps:0",
            dtype="torch.bfloat16",
            method="vjp",
            convention="frozen",
        )
    with pytest.raises(ScientificInputError, match="must not use"):
        LocalResponseEstimate(
            source=valid_source,
            target=valid_target,
            source_activation=1.0,
            target_preactivation=1.0,
            response=1.0,
            device="mps:0",
            dtype="torch.bfloat16",
            method="vjp",
            convention="frozen",
            graph_edge_used=True,
        )


def test_raw_graph_pair_extraction_uses_target_rows_and_source_columns() -> None:
    active = [
        [0, 0, 10],
        [0, 1, 11],
        [1, 1, 12],
        [2, 2, 13],
        [3, 2, 14],
        [4, 3, 15],
    ]
    adjacency = [[0.0] * 6 for _ in range(6)]
    adjacency[2][0] = 0.25
    adjacency[2][1] = -0.5
    adjacency[3][0] = -0.75
    adjacency[3][2] = 1.0
    adjacency[4][1] = 1.5
    adjacency[4][3] = -2.0
    adjacency[5][0] = 3.0
    adjacency[5][4] = -4.0
    graph = SimpleNamespace(
        active_features=active,
        activation_values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        selected_features=list(range(6)),
        adjacency_matrix=adjacency,
    )
    references = extract_active_pair_references(
        graph,
        seed="fixed",
        prompt_id="pilot",
        runtime_fingerprint="runtime",
        per_target_per_sign=1,
    )
    endpoints = {
        (item.source.feature_id, item.target.feature_id, item.raw_edge)
        for item in references
    }
    assert (11, 12, -0.5) in endpoints
    assert (10, 12, 0.25) in endpoints
    assert (14, 15, -4.0) in endpoints
    assert (10, 15, 3.0) in endpoints


def test_reference_split_enforces_disjoint_diversity_and_both_signs() -> None:
    references = tuple(
        _pair(index, (-1.0 if index % 2 else 1.0) * (index + 1))[0]
        for index in range(20)
    )
    calibration, canonical = select_disjoint_pair_references(
        references,
        calibration_count=3,
        canonical_count=10,
        minimum_target_layers=4,
        minimum_target_positions=3,
    )
    assert {item.pair_id for item in calibration}.isdisjoint(
        item.pair_id for item in canonical
    )
    validate_pair_distribution(
        canonical,
        minimum_pairs=10,
        minimum_target_layers=4,
        minimum_target_positions=3,
        require_both_signs=True,
    )
    with pytest.raises(ScientificInputError, match="both signs"):
        validate_pair_distribution(
            tuple(replace(item, raw_edge=abs(item.raw_edge)) for item in canonical),
            minimum_pairs=10,
            minimum_target_layers=4,
            minimum_target_positions=3,
            require_both_signs=True,
        )
