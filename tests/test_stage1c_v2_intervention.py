from __future__ import annotations

import inspect

import pytest

from cfsus.exceptions import ScientificInputError
from cfsus.stage1c_v2.intervention import (
    AppliedValuePlan,
    RequestedApplication,
    applied_plan_record,
    bisection_requested_alpha,
)
from cfsus.stage1c_v2.intervention_runtime import _feature


def test_bisection_and_plan_records_are_frozen() -> None:
    assert bisection_requested_alpha(0.25, 0.75) == 0.5
    with pytest.raises(ScientificInputError, match="strict bracket"):
        bisection_requested_alpha(0.5, 0.5)
    plan = AppliedValuePlan(
        applied_bf16=0.5,
        realized_suppression=0.5,
        requests=(RequestedApplication(0.5, 0.5, 0.5, 0.5),),
        tensor=object(),
    )
    record = applied_plan_record(plan)
    assert record["collapsed_request_count"] == 1
    assert record["requested_mappings"][0]["realized_suppression"] == 0.5


def test_v2_intervention_rejects_bool_feature_coordinates_and_uses_dynamic_length() -> (
    None
):
    with pytest.raises(ScientificInputError, match="feature record"):
        _feature({"layer": True, "position": 1, "feature_id": 2}, "source")
    source = inspect.getsource(
        __import__("cfsus.stage1c_v2.intervention_runtime", fromlist=["x"])
    )
    assert "self.token_count" in source
    assert "(18, 6)" not in source
