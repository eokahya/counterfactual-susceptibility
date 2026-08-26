from __future__ import annotations

import importlib.util
import math
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from cfsus.exceptions import ScientificInputError
from cfsus.stage1c.intervention import (
    applied_plan_record,
    bisection_requested_alpha,
    plan_applied_values,
)

_SCRIPT = (
    Path(__file__).parents[1] / "scripts/stage1c/run_stage1c_intervention_worker.py"
)
_SPEC = importlib.util.spec_from_file_location("stage1c_intervention_worker", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_WORKER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_WORKER)


def _round_to_bf16(value: float) -> float:
    """A small software BF16 round-to-nearest-even test double."""

    bits = struct.unpack(">I", struct.pack(">f", float(value)))[0]
    lower = bits & 0xFFFF
    upper = bits >> 16
    if lower > 0x8000 or (lower == 0x8000 and upper & 1):
        upper += 1
    return struct.unpack(">f", struct.pack(">I", upper << 16))[0]


class _FakeTensor:
    def __init__(self, value: float, torch: _FakeTorch) -> None:
        self._value = float(value)
        self.device = SimpleNamespace(type="mps")
        self.dtype = torch.bfloat16

    def numel(self) -> int:
        return 1

    def item(self) -> float:
        return self._value

    def reshape(self, *_shape: object) -> _FakeTensor:
        return self


class _FakeTorch:
    bfloat16 = object()
    Tensor = _FakeTensor

    def tensor(self, value: float, *, device: str, dtype: object) -> _FakeTensor:
        assert device == "mps"
        assert dtype is self.bfloat16
        return _FakeTensor(_round_to_bf16(value), self)


def test_high_precision_mapping_records_actual_bf16_and_collapses_requests() -> None:
    torch = _FakeTorch()
    baseline = torch.tensor(1.0, device="mps", dtype=torch.bfloat16)
    plans = plan_applied_values(baseline, (0.0, 0.0001, 0.5, 1.0), torch)
    assert [plan.applied_bf16 for plan in plans] == [1.0, 0.5, 0.0]
    assert plans[0].requests[0].requested_alpha == 0.0
    assert plans[0].requests[1].requested_alpha == 0.0001
    assert plans[0].requests[1].desired_high_precision == pytest.approx(0.9999)
    assert plans[0].requests[1].applied_bf16 == 1.0
    assert plans[0].realized_suppression == 0.0
    assert plans[1].realized_suppression == pytest.approx(0.5)
    assert plans[2].realized_suppression == pytest.approx(1.0)
    record = applied_plan_record(plans[0])
    assert record["collapsed_request_count"] == 2
    assert record["source_value_device"] == "mps:0"
    assert record["source_value_dtype"] == "torch.bfloat16"


def test_mapping_rejects_noncanonical_or_invalid_requests() -> None:
    torch = _FakeTorch()
    baseline = torch.tensor(2.0, device="mps", dtype=torch.bfloat16)
    with pytest.raises(ScientificInputError, match="unique canonical"):
        plan_applied_values(baseline, (0.5, 0.0), torch)
    with pytest.raises(ScientificInputError, match="outside"):
        plan_applied_values(baseline, (-0.1,), torch)
    with pytest.raises(ScientificInputError, match="outside"):
        plan_applied_values(baseline, (True,), torch)


def test_bisection_is_strict_midpoint_and_has_finite_bounds() -> None:
    assert bisection_requested_alpha(0.25, 0.75) == 0.5
    with pytest.raises(ScientificInputError, match="strict bracket"):
        bisection_requested_alpha(0.5, 0.5)
    with pytest.raises(ScientificInputError, match="invalid"):
        bisection_requested_alpha(float("nan"), 0.5)


def test_first_bracket_is_inactive_to_active_and_equality_remains_inactive() -> None:
    points = [
        {"realized_suppression": 0.0, "target_active": False},
        {"realized_suppression": 0.5, "target_active": False},
        {"realized_suppression": 0.75, "target_active": True},
        {"realized_suppression": 1.0, "target_active": True},
    ]
    bracket = _WORKER._first_bracket(points)
    assert bracket is not None
    assert bracket[0]["realized_suppression"] == 0.5
    assert bracket[1]["realized_suppression"] == 0.75
    equality = [
        {"realized_suppression": 0.0, "target_active": False},
        {"realized_suppression": 1.0, "target_active": False},
    ]
    assert _WORKER._first_bracket(equality) is None


def test_applied_value_plan_serialization_has_no_tensor_payload() -> None:
    torch = _FakeTorch()
    baseline = torch.tensor(1.0, device="mps", dtype=torch.bfloat16)
    plan = plan_applied_values(baseline, (0.0,), torch)[0]
    record = applied_plan_record(plan)
    assert "tensor" not in record
    assert all(
        math.isfinite(float(value))
        for value in record["requested_mappings"][0].values()
    )
