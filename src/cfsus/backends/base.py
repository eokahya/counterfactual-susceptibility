"""Backend-neutral protocol for capability-gated feature operations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from cfsus.types import (
    BackendCapabilityReport,
    BackendOperation,
    BaselineFeatureState,
    BehaviorMetricMetadata,
    FeatureRef,
    InterventionRegime,
    ModelSetting,
    ObservedInterventionPoint,
    SourceSuppression,
)


@runtime_checkable
class FeatureBackend(Protocol):
    """Local adapter boundary; it does not describe any upstream package API.

    Implementations must preserve the project's suppression convention and must
    capability-gate every operation whose upstream semantics are not verified.
    """

    def capability_report(self) -> BackendCapabilityReport:
        """Return granular support status with source-audit evidence."""
        ...

    def require_capability(self, operation: BackendOperation) -> None:
        """Raise a project exception unless ``operation`` is verified supported."""
        ...

    def baseline_feature_state(self, feature: FeatureRef) -> BaselineFeatureState:
        """Return backend-reported baseline state for exactly one feature."""
        ...

    def run_source_suppression(
        self,
        intervention: SourceSuppression,
        target: FeatureRef,
        setting: ModelSetting,
        regime: InterventionRegime | None,
        behavior_metric: BehaviorMetricMetadata | None = None,
    ) -> ObservedInterventionPoint:
        """Run one intervention point after an adapter maps local semantics."""
        ...
