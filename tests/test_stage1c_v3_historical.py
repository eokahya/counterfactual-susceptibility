from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from cfsus.exceptions import ScientificInputError
from cfsus.stage1c_v3.historical import (
    PROMPT_SELECTION_INDEX,
    PROMPT_SELECTION_SHA256,
    EndpointOverlapCategory,
    assert_expected_prompt_derivation,
    assert_no_historical_exact_pairs,
    endpoint_overlap_category,
    extract_historical_exact_pairs,
    load_authenticated_historical_metadata,
    mask_historical_exact_pairs,
)
from cfsus.types import FeatureRef

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Pair:
    source: FeatureRef
    target: FeatureRef


def _feature(value: tuple[int, int, int]) -> FeatureRef:
    return FeatureRef(*value)


def test_authenticated_denylist_is_exact_and_prompt_derivation_is_independent() -> None:
    metadata = load_authenticated_historical_metadata(ROOT)
    assert len(metadata.exact_pairs) == 28
    assert len(metadata.historical_endpoints) == 53
    derivation = assert_expected_prompt_derivation()
    assert derivation.sha256_hex == PROMPT_SELECTION_SHA256
    assert derivation.index == PROMPT_SELECTION_INDEX == 7
    assert derivation.prompt == "The capital of Norway is"
    assert derivation.prompt_id == "capital_norway_preregistered_v3"


def test_exact_pair_mask_runs_before_ranking_and_preserves_nonexact_endpoints() -> None:
    metadata = load_authenticated_historical_metadata(ROOT)
    denylist = frozenset(metadata.exact_pairs)
    endpoints = metadata.historical_endpoints
    exact_source, exact_target = metadata.exact_pairs[0]
    other_source, _ = next(
        pair
        for pair in metadata.exact_pairs[1:]
        if pair[0] != exact_source and (pair[0], exact_target) not in denylist
    )
    novel_source = FeatureRef(0, 1, 16_383)
    novel_target = FeatureRef(17, 5, 16_382)
    while (
        novel_source.layer,
        novel_source.position,
        novel_source.feature_id,
    ) in endpoints:
        novel_source = FeatureRef(0, 1, novel_source.feature_id - 1)
    while (
        novel_target.layer,
        novel_target.position,
        novel_target.feature_id,
    ) in endpoints:
        novel_target = FeatureRef(17, 5, novel_target.feature_id - 1)

    exact = Pair(_feature(exact_source), _feature(exact_target))
    source_only = Pair(_feature(exact_source), novel_target)
    target_only = Pair(novel_source, _feature(exact_target))
    both_separate = Pair(_feature(other_source), _feature(exact_target))
    neither = Pair(novel_source, novel_target)
    kept, excluded = mask_historical_exact_pairs(
        (source_only, exact, target_only, both_separate), denylist
    )
    assert kept == (source_only, target_only, both_separate)
    assert excluded == 1
    assert (
        endpoint_overlap_category(
            source_only.source,
            source_only.target,
            denylist=denylist,
            endpoints=endpoints,
        )
        is EndpointOverlapCategory.SOURCE_ONLY
    )
    assert (
        endpoint_overlap_category(
            target_only.source,
            target_only.target,
            denylist=denylist,
            endpoints=endpoints,
        )
        is EndpointOverlapCategory.TARGET_ONLY
    )
    assert (
        endpoint_overlap_category(
            both_separate.source,
            both_separate.target,
            denylist=denylist,
            endpoints=endpoints,
        )
        is EndpointOverlapCategory.BOTH_SEPARATE
    )
    assert (
        endpoint_overlap_category(
            neither.source,
            neither.target,
            denylist=denylist,
            endpoints=endpoints,
        )
        is EndpointOverlapCategory.NEITHER
    )
    assert_no_historical_exact_pairs(kept, denylist)
    with pytest.raises(ScientificInputError, match="historical exact pair"):
        assert_no_historical_exact_pairs((exact,), denylist)


def test_historical_extraction_rejects_normalized_outcome_fields_recursively() -> None:
    import copy
    import json

    source = json.loads(
        (
            ROOT
            / "results/stage1c_first_prospective_prediction/prediction_manifest.json"
        ).read_text()
    )
    hostile = copy.deepcopy(source)
    hostile["selected_groups"]["primary"][0]["Observed-Outcome"] = "forbidden"
    with pytest.raises(ScientificInputError, match="outcome field"):
        extract_historical_exact_pairs(hostile)
