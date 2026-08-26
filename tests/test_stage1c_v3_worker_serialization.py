"""Synthetic regression tests for the v3 worker lifetime boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from cfsus.stage1c_v3.serialization import (
    SerializationError,
    read_json_strict,
    write_json_new,
)
from cfsus.stage1c_v3.worker_result import (
    build_detached_worker_result,
    construct_then_cleanup,
)


def _sweeps() -> list[dict[str, Any]]:
    return [
        {
            "pair_id": "a" * 64,
            "group": "primary",
            "point_count": 2,
            "points": [
                {
                    "realized_suppression": 0.0,
                    "target_active": False,
                    "trace": [1, {"v": "baseline"}],
                },
                {
                    "realized_suppression": 1.0,
                    "target_active": True,
                    "trace": [2, {"v": "full"}],
                },
            ],
        },
        {
            "pair_id": "b" * 64,
            "group": "near_boundary",
            "point_count": 2,
            "points": [
                {
                    "realized_suppression": 0.0,
                    "target_active": False,
                    "trace": [3, {"v": "baseline"}],
                },
                {
                    "realized_suppression": 1.0,
                    "target_active": False,
                    "trace": [4, {"v": "full"}],
                },
            ],
        },
    ]


def _assert_recursive_disjoint(left: Any, right: Any) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        assert left.keys() == right.keys()
        for key in left:
            if isinstance(left[key], (dict, list)) or isinstance(
                right[key], (dict, list)
            ):
                assert left[key] is not right[key]
            _assert_recursive_disjoint(left[key], right[key])
    elif isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            if isinstance(first, (dict, list)) or isinstance(second, (dict, list)):
                assert first is not second
            _assert_recursive_disjoint(first, second)
    else:
        assert left == right


def test_nonempty_worker_result_detaches_top_and_artifact_sweeps() -> None:
    working = _sweeps()
    artifacts = {
        "intervention_sweeps": {"pairs": working},
        "nested": {"copy": working[0]},
    }
    result = build_detached_worker_result(
        working,
        intervention_artifacts=artifacts,
        canonical_source_suppression_api_calls=4,
        artifact_type="stage1c_v3_intervention_worker",
    )
    top = result["sweeps"]
    pairs = result["intervention_artifacts"]["intervention_sweeps"]["pairs"]
    assert top == pairs
    assert top is not pairs
    _assert_recursive_disjoint(top, pairs)
    working[0]["points"][0]["trace"][1]["v"] = "mutated"
    working.clear()
    assert result["sweeps"][0]["points"][0]["trace"][1]["v"] == "baseline"
    assert len(result["intervention_artifacts"]["intervention_sweeps"]["pairs"]) == 2


def test_construct_then_cleanup_returns_nonempty_result_after_finally() -> None:
    working = _sweeps()

    def cleanup() -> None:
        working[0]["points"].clear()
        working.clear()

    result = construct_then_cleanup(
        working,
        intervention_artifacts={"intervention_sweeps": {"pairs": working}},
        cleanup=cleanup,
        artifact_type="stage1c_v3_intervention_worker",
    )
    assert working == []
    assert len(result["sweeps"]) == 2
    assert result["sweeps"][0]["point_count"] == 2
    assert result["sweeps"][0]["points"][1]["realized_suppression"] == 1.0


def test_cleanup_exception_still_clears_working_state() -> None:
    working = _sweeps()

    def cleanup() -> None:
        working[0]["points"].append({"unexpected": True})
        raise RuntimeError("synthetic cleanup failure")

    with pytest.raises(RuntimeError, match="synthetic cleanup failure"):
        construct_then_cleanup(working, cleanup=cleanup)
    assert working == []


@pytest.mark.parametrize("calls", [0, 3, 5, -1, True])
def test_api_call_count_must_equal_serialized_point_count(calls: Any) -> None:
    expected = 4
    if calls == expected:
        result = build_detached_worker_result(
            _sweeps(), canonical_source_suppression_api_calls=calls
        )
        assert result["canonical_source_suppression_api_calls"] == expected
    else:
        with pytest.raises(SerializationError, match="API-call count"):
            build_detached_worker_result(
                _sweeps(), canonical_source_suppression_api_calls=calls
            )


def test_point_count_must_equal_point_array_length() -> None:
    working = _sweeps()
    working[0]["point_count"] = 99
    with pytest.raises(SerializationError, match="point_count"):
        build_detached_worker_result(working)


def test_instrumented_api_count_must_equal_serialized_points() -> None:
    with pytest.raises(SerializationError, match="instrumented API-call count"):
        build_detached_worker_result(
            _sweeps(),
            canonical_source_suppression_api_calls=4,
            instrumented_source_suppression_api_calls=3,
        )
    result = build_detached_worker_result(
        _sweeps(),
        canonical_source_suppression_api_calls=4,
        instrumented_source_suppression_api_calls=4,
    )
    assert result["instrumented_source_suppression_api_calls"] == 4


def test_strict_write_read_preserves_pair_ids_and_point_counts(tmp_path: Path) -> None:
    result = build_detached_worker_result(
        _sweeps(), artifact_type="stage1c_v3_intervention_worker"
    )
    path = tmp_path / "worker.json"
    write_json_new(path, result)
    reread = read_json_strict(path)
    assert isinstance(reread, dict)
    record = cast(dict[str, Any], reread)
    sweeps = cast(list[dict[str, Any]], record["sweeps"])
    artifacts = cast(dict[str, Any], record["intervention_artifacts"])
    intervention = cast(dict[str, Any], artifacts["intervention_sweeps"])
    assert [item["pair_id"] for item in sweeps] == ["a" * 64, "b" * 64]
    assert [item["point_count"] for item in sweeps] == [2, 2]
    assert sweeps == intervention["pairs"]


def test_no_eligible_pairs_is_zero_attempt_zero_call_terminal_result() -> None:
    result = build_detached_worker_result(
        [],
        intervention_artifacts={
            "attempts": {
                "artifact_type": "stage1c_v3_attempts",
                "attempt_count": 0,
                "scientific_retry_count": 0,
                "intervention_required": False,
            }
        },
        canonical_source_suppression_api_calls=0,
        scientific_outcome="no_eligible_pairs",
        attempt_count=0,
        intervention_skipped=True,
    )
    assert result["sweeps"] == []
    assert result["intervention_artifacts"]["intervention_sweeps"]["pairs"] == []
    assert result["canonical_source_suppression_api_calls"] == 0
    assert result["attempt_count"] == 0
    assert result["scientific_outcome"] == "no_eligible_pairs"
