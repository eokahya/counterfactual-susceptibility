from __future__ import annotations

import pytest

from cfsus.backends.circuit_tracer import CircuitTracerAdapter
from cfsus.exceptions import (
    BackendUnavailableError,
    ScientificInputError,
    UnsupportedBackendOperationError,
)
from cfsus.types import (
    BackendOperation,
    CapabilityStatus,
    FeatureRef,
    InterventionRegime,
    ModelSetting,
    SourceSuppression,
)


def test_adapter_reports_every_operation_without_claiming_local_support() -> None:
    report = CircuitTracerAdapter().capability_report()

    assert {item.operation for item in report.evidence} == set(BackendOperation)
    assert all(
        item.status is not CapabilityStatus.SUPPORTED for item in report.evidence
    )
    assert (
        report.status_for(BackendOperation.INACTIVE_PREACTIVATION_ACCESS)
        is CapabilityStatus.UPSTREAM_SUPPORTED
    )
    assert (
        report.status_for(BackendOperation.TARGETED_PREACTIVATION_ACCESS)
        is CapabilityStatus.UNSUPPORTED
    )


def test_adapter_is_import_safe_without_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cfsus.backends.circuit_tracer._dependency_available", lambda: False
    )
    adapter = CircuitTracerAdapter()

    assert adapter.capability_report().dependency_available is False
    with pytest.raises(BackendUnavailableError, match="not installed"):
        adapter.require_capability(BackendOperation.THRESHOLD_ACCESS)


def test_suppression_dispatches_by_declared_model_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cfsus.backends.circuit_tracer._dependency_available", lambda: True
    )
    monkeypatch.setattr(
        "cfsus.backends.circuit_tracer._dependency_version", lambda: "0.5.2"
    )
    adapter = CircuitTracerAdapter()
    intervention = SourceSuppression(FeatureRef(0, 0, 1), alpha=0.5)

    with pytest.raises(
        UnsupportedBackendOperationError,
        match=BackendOperation.UNDERLYING_SOURCE_SUPPRESSION.value,
    ):
        adapter.run_source_suppression(
            intervention,
            FeatureRef(1, 0, 2),
            ModelSetting.UNDERLYING_MODEL,
            InterventionRegime.UNDERLYING_NONLINEAR,
        )


def test_suppression_rejects_setting_regime_mismatches() -> None:
    adapter = CircuitTracerAdapter()
    intervention = SourceSuppression(FeatureRef(0, 0, 1), alpha=0.5)
    target = FeatureRef(1, 0, 2)

    with pytest.raises(ScientificInputError, match="does not accept"):
        adapter.run_source_suppression(
            intervention,
            target,
            ModelSetting.REPLACEMENT_MODEL,
            InterventionRegime.ATTRIBUTION_MATCHED,
        )
    with pytest.raises(ScientificInputError, match="requires a declared"):
        adapter.run_source_suppression(
            intervention,
            target,
            ModelSetting.UNDERLYING_MODEL,
            None,
        )
