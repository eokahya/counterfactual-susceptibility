from __future__ import annotations

import hashlib
import json

import pytest

from cfsus.exceptions import ScientificInputError
from cfsus.stage1c_v2.prediction import (
    ProspectivePair,
    build_prospective_pair,
    canonical_v2_pair_id,
    causally_eligible,
    filter_source_pool,
    pair_score_digest,
    prospective_pair_record,
    requested_schedule,
    select_pair_groups,
)
from cfsus.types import (
    CrossingStatus,
    FeatureActivity,
    FeatureRef,
    MeasuredFeatureState,
    NearThresholdCandidate,
)


def _source(
    layer: int,
    feature_id: int,
    *,
    position: int = 1,
    activation: float = 1.0,
) -> MeasuredFeatureState:
    return MeasuredFeatureState(
        feature=FeatureRef(layer, position, feature_id),
        preactivation=activation,
        activation=activation,
        threshold=0.0,
        activity=FeatureActivity.ACTIVE,
        device="mps:0",
        dtype="torch.bfloat16",
    )


def _target(
    layer: int,
    feature_id: int,
    *,
    position: int = 1,
    preactivation: float = 0.0,
    threshold: float = 1.0,
) -> NearThresholdCandidate:
    return NearThresholdCandidate(
        feature=FeatureRef(layer, position, feature_id),
        preactivation=preactivation,
        activation=0.0,
        threshold=threshold,
        margin=threshold - preactivation,
        device="mps:0",
        dtype="torch.bfloat16",
    )


def _pair(
    source: MeasuredFeatureState,
    target: NearThresholdCandidate,
    response: float,
) -> ProspectivePair:
    return build_prospective_pair(
        source=source,
        target=target,
        targeted_response=response,
        runtime_fingerprint="gemma3-v2-test-runtime",
        epsilon=1.0e-12,
        tolerance=1.0e-9,
    )


def _selection() -> dict[str, object]:
    return {
        "primary_maximum": 12,
        "near_boundary_maximum": 8,
        "directional_maximum": 8,
        "maximum_per_target": 1,
        "maximum_primary_per_source": 2,
        "primary_order": ["susceptibility_desc", "alpha_hat_asc", "target", "source"],
        "near_order": ["distance_above_one_asc", "target", "source"],
        "directional_order": ["movement_over_margin_desc", "target", "source"],
        "prefer_unused_control_targets": True,
        "control_overlap_fallback": "deterministic_after_unique_exhausted",
    }


def test_v2_pair_id_is_explicitly_namespaced_and_score_is_baseline_only() -> None:
    source = _source(0, 3, activation=4.0).feature
    target = _target(2, 8).feature
    pair_id = canonical_v2_pair_id(
        source=source, target=target, runtime_fingerprint="runtime"
    )
    payload = {
        "experiment_class": "stage1c_v2_heldout_prospective_prediction",
        "prompt_id": "capital_germany_heldout_v2",
        "runtime_fingerprint": "runtime",
        "seed": "stage1c-v2-heldout-prospective-prediction",
        "source": [0, 1, 3],
        "target": [2, 1, 8],
    }
    assert (
        pair_id
        == hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    pair = _pair(_source(0, 3, activation=4.0), _target(2, 8), response=-0.5)
    record = prospective_pair_record(pair)
    assert pair.q == 2.0
    assert pair.predicted_alpha_star == pytest.approx(0.5)
    assert pair.status is CrossingStatus.DEFINITELY_CROSSING
    assert not any("observed" in key or "intervention" in key for key in record)


def test_selection_preserves_frozen_order_and_caps() -> None:
    source = _source(0, 1)
    candidates = tuple(
        _pair(source, _target(2, feature), response=-response)
        for feature, response in ((1, 3.0), (2, 2.0), (3, 1.0))
    )
    selected = select_pair_groups(candidates, selection=_selection(), tolerance=1.0e-9)
    assert [item.target.feature_id for item in selected.primary] == [1, 2]
    assert len(selected.primary) == 2


def test_source_filter_and_schedule_remain_deterministic() -> None:
    source = _source(0, 1)
    target = _target(2, 4)
    assert filter_source_pool((source,), (target,), maximum_sources=10) == (source,)
    pair = _pair(source, target, response=-2.0)
    schedule = requested_schedule(
        pair,
        coarse_alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
        alpha_hat_offset=1.0 / 64.0,
    )
    assert schedule == tuple(sorted(set(schedule)))
    assert {0.0, 0.25, 0.5, 0.75, 1.0}.issubset(schedule)
    assert len(pair_score_digest((pair,))) == 64


def test_empty_eligible_pool_remains_a_deterministic_no_eligible_result() -> None:
    source = _source(2, 1)
    target = _target(1, 4)
    assert filter_source_pool((source,), (target,), maximum_sources=10) == ()
    selected = select_pair_groups((), selection=_selection(), tolerance=1.0e-9)
    assert selected.primary == ()
    assert selected.near_boundary == ()
    assert selected.directional == ()
    assert pair_score_digest(()) == hashlib.sha256(b"").hexdigest()


def test_selection_rejects_historical_endpoint_without_reranking() -> None:
    pair = _pair(_source(14, 234), _target(15, 771), response=-3.0)
    with pytest.raises(ScientificInputError, match="historical selected endpoint"):
        select_pair_groups((pair,), selection=_selection(), tolerance=1.0e-9)


def test_causal_order_is_strict_in_layer_and_non_decreasing_in_position() -> None:
    assert causally_eligible(FeatureRef(0, 2, 1), FeatureRef(1, 2, 2))
    assert not causally_eligible(FeatureRef(1, 2, 1), FeatureRef(1, 2, 2))
    assert not causally_eligible(FeatureRef(0, 3, 1), FeatureRef(1, 2, 2))
