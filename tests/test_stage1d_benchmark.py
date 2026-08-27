from __future__ import annotations

from pathlib import Path

import pytest

from cfsus.exceptions import ScientificInputError
from cfsus.stage1c_v3.prediction import ProspectivePair
from cfsus.stage1d.benchmark import quantization_evidence, select_prompt_panels
from cfsus.stage1d.config import load_stage1d_config
from cfsus.stage1d.metrics import (
    classify_project_decision,
    exact_binomial_interval,
    spearman,
)
from cfsus.types import CrossingStatus, FeatureRef


def _pair(index: int, *, alpha: float | None, q: float) -> ProspectivePair:
    requested_margin = 0.1 if alpha is None else alpha * q
    preactivation = 1.0 - requested_margin
    margin = 1.0 - preactivation
    return ProspectivePair(
        pair_id=f"{index:064x}",
        source=FeatureRef(index % 8, 1, index),
        target=FeatureRef(10 + index % 8, 1, 100 + index),
        source_activation=1.0,
        target_preactivation=preactivation,
        target_threshold=1.0,
        margin=margin,
        targeted_response=-q,
        q=q,
        susceptibility=q / (margin + 1.0e-12),
        predicted_alpha_star=alpha,
        status=(
            CrossingStatus.DEFINITELY_CROSSING
            if q > margin
            else CrossingStatus.NOT_CROSSING
        ),
    )


def test_config_and_frozen_prompt_order() -> None:
    config = load_stage1d_config()
    assert [item["id"] for item in config["prompts"]] == [
        f"P{index:02d}" for index in range(1, 9)
    ]
    assert config["full_ablation_panel"]["per_method_k"] == 4
    assert config["metrics"]["bootstrap_resamples"] == 10_000


def test_config_rejects_duplicate_yaml_key(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")
    with pytest.raises(ScientificInputError, match="duplicate YAML key"):
        load_stage1d_config(path)


def test_quantization_rule_and_deterministic_panels() -> None:
    config = load_stage1d_config()
    pairs = [
        _pair(1, alpha=0.05, q=0.2),
        _pair(2, alpha=0.20, q=0.2),
        _pair(3, alpha=0.60, q=0.2),
        _pair(4, alpha=1.50, q=0.2),
        *[_pair(index, alpha=0.8, q=0.1 + index / 100) for index in range(5, 15)],
        _pair(15, alpha=None, q=-0.3),
        _pair(16, alpha=None, q=-0.2),
    ]
    evidence = quantization_evidence(pairs[0])
    assert evidence["calibration_resolvable"] is True
    assert evidence["distinct_nonzero_at_or_below_twice_alpha"] >= 3
    first = select_prompt_panels(pairs, prompt_id="P01", config=config)
    second = select_prompt_panels(
        tuple(reversed(pairs)), prompt_id="P01", config=config
    )
    assert first == second
    assert first.detailed_pair_ids == {
        "B1": pairs[0].pair_id,
        "B2": pairs[1].pair_id,
        "B3": pairs[2].pair_id,
        "near_boundary": pairs[3].pair_id,
    }
    assert len(first.method_pair_ids["S"]) == 4
    assert len(first.directional_pair_ids) == 2
    limited = quantization_evidence(_pair(17, alpha=0.01, q=0.2))
    assert limited["calibration_resolvable"] is False
    assert "predicted_alpha_outside_0.02_0.95" in limited["limitation_reasons"]
    empty = select_prompt_panels([], prompt_id="P01", config=config)
    assert empty.execution_pairs == []
    assert all(value is not None for value in empty.missing_strata.values())


def test_statistics_primitives() -> None:
    assert spearman([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert spearman([1.0, 1.0], [2.0, 3.0]) is None
    lower, upper = exact_binomial_interval(0, 16)
    assert lower == 0.0
    assert upper == pytest.approx(0.2059072142)
    lower, upper = exact_binomial_interval(16, 16)
    assert lower == pytest.approx(0.7940927858)
    assert upper == 1.0


def test_frozen_three_way_decision_rule() -> None:
    rules = load_stage1d_config()["decision_rule"]
    common = {
        "critical_spearman": 0.6,
        "critical_pair_count": 20,
        "directional_violation_fraction": 0.0,
        "nonmonotonic_fraction": 0.0,
        "rules": rules,
    }
    assert (
        classify_project_decision(s_minus_margin=0.1, s_minus_influence=0.1, **common)
        == "continue_first_order_to_behavioral_stage"
    )
    assert (
        classify_project_decision(
            s_minus_margin=0.1,
            s_minus_influence=0.0,
            **common,
        )
        == "retain_crossing_ranker_but_redesign_calibration"
    )
    assert (
        classify_project_decision(
            s_minus_margin=0.0,
            s_minus_influence=0.0,
            **common,
        )
        == "redesign_method_before_scaling"
    )
