from __future__ import annotations

import json
from pathlib import Path

import pytest

from cfsus.exceptions import ScientificInputError
from cfsus.stage1c_v3.config import (
    BASE_COMMIT,
    BRANCH,
    EXPECTED_TOKEN_IDS,
    EXPERIMENT_CLASS,
    PAIR_SEED,
    PROMPT_ID,
    PROMPT_TEXT,
    SELECTED_POSITIONS,
    load_stage1c_v3_config,
    selected_positions_for_token_ids,
    validate_prompt_token_ids,
    validate_stage1c_v3_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage1c_v3_preregistered_prospective_prediction.yaml"
SCHEMA = (
    ROOT
    / "configs/stage1c_v3_preregistered_prospective_prediction_artifact_schema.json"
)


def test_v3_config_has_new_identity_and_frozen_token_ids() -> None:
    config = load_stage1c_v3_config(CONFIG)
    assert config["schema_version"] == 3
    assert config["experiment_class"] == EXPERIMENT_CLASS
    assert config["branch"] == BRANCH
    assert config["base_commit"] == BASE_COMMIT
    assert config["prompt"] == {
        "id": PROMPT_ID,
        "text": PROMPT_TEXT,
        "expected_token_ids": list(EXPECTED_TOKEN_IDS),
    }
    assert config["scanner"]["selected_positions"] == list(SELECTED_POSITIONS)
    assert load_stage1c_v3_config(CONFIG, require_token_ids=True) == config


def test_v3_artifact_schema_has_disjoint_explicit_type_identities() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert schema["artifact_type"] == "stage1c_v3_artifact_schema"
    assert schema["prediction_manifest_type"] == "stage1c_v3_prediction_manifest"
    assert schema["worker_artifact_type"] == "stage1c_v3_intervention_worker"
    assert schema["final_bundle_type"] == "stage1c_v3_final_bundle"


def test_only_non_bos_positions_are_resolved() -> None:
    assert selected_positions_for_token_ids([2, 10, 11]) == [1, 2]
    assert selected_positions_for_token_ids([0, 10]) == [1]
    with pytest.raises(ScientificInputError, match="non-empty"):
        selected_positions_for_token_ids([])


def test_token_identity_is_exactly_frozen_before_execution() -> None:
    config = load_stage1c_v3_config(CONFIG)
    validate_prompt_token_ids(config, EXPECTED_TOKEN_IDS)
    with pytest.raises(ScientificInputError, match="token identity"):
        validate_prompt_token_ids(config, [2, 818, 5279, 529, 9406, 563])


def test_config_rejects_v1_identity_and_pair_seed() -> None:
    config = load_stage1c_v3_config(CONFIG)
    bad = {**config, "experiment_class": "stage1c_first_prospective_prediction"}
    with pytest.raises(ScientificInputError, match="differs"):
        validate_stage1c_v3_config(bad)
    bad_scoring = {
        **config,
        "scoring": {
            **config["scoring"],
            "pair_seed": "stage1c-first-prospective-v1",
        },
    }
    with pytest.raises(ScientificInputError, match="differs"):
        validate_stage1c_v3_config(bad_scoring)
    assert PAIR_SEED != "stage1c-first-prospective-v1"


def test_config_freezes_exact_pair_only_history_and_v2_output_guard() -> None:
    config = load_stage1c_v3_config(CONFIG)
    historical = config["historical_exclusion"]
    assert historical["unit"] == "context_independent_exact_pair_key"
    assert historical["mask_stage"] == "before_ranking_and_quota_selection"
    assert historical["single_endpoint_overlap"] == "audit_only"
    assert historical["v2_temporary_baseline_reads"] == "forbidden"
