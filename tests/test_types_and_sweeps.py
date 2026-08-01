from __future__ import annotations

import pytest

from cfsus.evaluation.crossings import first_observed_crossing
from cfsus.exceptions import NonFiniteInputError, ScientificInputError
from cfsus.interventions.sweep import make_suppression_sweep, regular_alpha_grid
from cfsus.types import (
    FeatureRef,
    ModelSetting,
    ObservedInterventionPoint,
    ObservedSweepResult,
    SourceSuppression,
)


@pytest.mark.parametrize("bad_index", [-1, True, 1.5])
def test_feature_ref_rejects_invalid_indices(bad_index: object) -> None:
    with pytest.raises(ScientificInputError):
        FeatureRef(layer=bad_index, position=0, feature_id=0)  # type: ignore[arg-type]


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_source_suppression_rejects_alpha_outside_closed_interval(
    alpha: float,
) -> None:
    with pytest.raises(ScientificInputError, match=r"\[0, 1\]"):
        SourceSuppression(source=FeatureRef(0, 0, 0), alpha=alpha)


def test_source_suppression_uses_declared_activation_mapping() -> None:
    intervention = SourceSuppression(source=FeatureRef(0, 0, 1), alpha=0.25)

    assert intervention.suppressed_activation(4.0) == pytest.approx(3.0)
    with pytest.raises(NonFiniteInputError):
        intervention.suppressed_activation(float("inf"))


def test_regular_grid_and_sweep_are_deterministic() -> None:
    source = FeatureRef(1, 2, 3)
    grid = regular_alpha_grid(4)
    sweep = make_suppression_sweep(source, grid)

    assert grid == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert tuple(point.alpha for point in sweep) == grid


def test_sweep_rejects_unsorted_or_duplicate_alpha() -> None:
    source = FeatureRef(0, 0, 0)
    with pytest.raises(ScientificInputError, match="strictly increasing"):
        make_suppression_sweep(source, (0.0, 0.5, 0.5, 1.0))


def test_observed_crossing_uses_backend_reported_activity() -> None:
    points = (
        ObservedInterventionPoint(0.0, 0.2, 0.0, False),
        ObservedInterventionPoint(0.5, 0.4, 0.0, False),
        ObservedInterventionPoint(0.75, 0.6, 0.6, True),
    )
    result = ObservedSweepResult(
        source=FeatureRef(0, 0, 1),
        target=FeatureRef(1, 0, 2),
        setting=ModelSetting.REPLACEMENT_MODEL,
        points=points,
        observed_critical_alpha=0.75,
    )

    assert first_observed_crossing(result.points) == 0.75


def test_observed_crossing_returns_none_when_gate_never_activates() -> None:
    points = (
        ObservedInterventionPoint(0.0, 0.2, 0.0, False),
        ObservedInterventionPoint(1.0, 0.4, 0.0, False),
    )

    assert first_observed_crossing(points) is None
