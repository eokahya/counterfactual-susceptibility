from __future__ import annotations

import inspect

import pytest

from cfsus.exceptions import ScientificInputError
from cfsus.stage1c.prediction import (
    ProspectivePair,
    build_prospective_pair,
    causally_eligible,
    filter_source_pool,
    filter_target_pool,
    prospective_pair_record,
    requested_schedule,
    select_pair_groups,
)
from cfsus.stage1c.runtime import Stage1CPredictionBackend
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
        seed="stage1c-test",
        prompt_id="pilot",
        runtime_fingerprint="test-runtime",
        epsilon=1.0e-12,
        tolerance=1.0e-9,
    )


def _selection(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "primary_maximum": 12,
        "near_boundary_maximum": 8,
        "directional_maximum": 8,
        "maximum_per_target": 1,
        "maximum_primary_per_source": 2,
    }
    result.update(overrides)
    return result


def test_signed_score_formula_and_predicted_crossing_are_baseline_only() -> None:
    pair = _pair(_source(0, 3, activation=4.0), _target(2, 8), response=-0.5)
    assert pair.margin == 1.0
    assert pair.q == 2.0
    assert pair.susceptibility == pytest.approx(2.0)
    assert pair.predicted_alpha_star == pytest.approx(0.5)
    assert pair.status is CrossingStatus.DEFINITELY_CROSSING
    record = prospective_pair_record(pair)
    assert set(record) == {
        "pair_id",
        "source",
        "target",
        "source_activation",
        "target_preactivation",
        "target_threshold",
        "margin",
        "targeted_response",
        "q",
        "susceptibility",
        "predicted_alpha_star",
        "predicted_status",
    }
    assert not any("intervention" in key or "observed" in key for key in record)


def test_full_ablation_equality_is_not_a_definite_crossing() -> None:
    pair = _pair(_source(0, 1), _target(2, 2), response=-1.0)
    assert pair.q == pair.margin
    assert pair.status is CrossingStatus.BOUNDARY_AMBIGUOUS


def test_causal_source_filter_requires_strict_layer_and_non_decreasing_position() -> (
    None
):
    source = _source(2, 1, position=3)
    target = _target(2, 9, position=2)
    assert not causally_eligible(source.feature, target.feature)
    with pytest.raises(ScientificInputError, match="causal"):
        _pair(source, target, response=-1.0)

    sources = (_source(0, 1), _source(3, 2), _source(1, 3, position=3))
    targets = (_target(2, 4, position=1), _target(2, 5, position=2))
    filtered = filter_source_pool(sources, targets, maximum_sources=10)
    assert tuple(item.feature for item in filtered) == (FeatureRef(0, 1, 1),)
    assert filter_target_pool(targets, filtered) == targets


def test_pair_selection_is_sorted_capped_and_groups_are_disjoint() -> None:
    common = _source(0, 1, activation=1.0)
    primary = (
        _pair(common, _target(2, 1), response=-3.0),
        _pair(common, _target(2, 2), response=-2.0),
        _pair(common, _target(2, 3), response=-1.5),
    )
    near = _pair(_source(0, 2), _target(2, 4), response=-0.5)
    directional = _pair(_source(0, 3), _target(2, 5), response=0.5)
    selected = select_pair_groups(
        (*primary, near, directional),
        selection=_selection(
            primary_maximum=12,
            near_boundary_maximum=1,
            directional_maximum=1,
        ),
        tolerance=1.0e-9,
    )
    assert [item.target.feature_id for item in selected.primary] == [1, 2]
    assert len(selected.primary) == 2  # source cap, not an outcome-dependent truncation
    assert selected.near_boundary == (near,)
    assert selected.directional == (directional,)
    ids = [
        item.pair_id
        for group in (selected.primary, selected.near_boundary, selected.directional)
        for item in group
    ]
    assert len(ids) == len(set(ids))


def test_controls_use_signed_q_and_requested_schedule_is_deterministic() -> None:
    pair = _pair(_source(0, 1), _target(2, 1), response=-0.4)
    assert pair.q > 0.0 and pair.predicted_alpha_star > 1.0
    controls = select_pair_groups(
        (pair, _pair(_source(0, 2), _target(2, 2), response=0.0)),
        selection=_selection(
            primary_maximum=1,
            near_boundary_maximum=1,
            directional_maximum=1,
        ),
        tolerance=1.0e-9,
    )
    assert controls.primary == ()
    assert controls.near_boundary == (pair,)
    assert len(controls.directional) == 1
    assert controls.directional[0].q == 0.0
    schedule = requested_schedule(
        _pair(_source(0, 3), _target(2, 3), response=-2.0),
        coarse_alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
        alpha_hat_offset=1.0 / 64.0,
    )
    assert schedule == tuple(sorted(set(schedule)))
    assert {0.0, 0.25, 0.5, 0.75, 1.0}.issubset(schedule)


def test_prediction_backend_response_tile_has_no_graph_or_intervention_parameter() -> (
    None
):
    signature = inspect.signature(Stage1CPredictionBackend.response_tile)
    assert "graph" not in signature.parameters
    assert "adjacency" not in signature.parameters
    assert "feature_intervention" not in inspect.getsource(Stage1CPredictionBackend)
