"""Import-safe, deliberately non-executing ``circuit-tracer`` adapter skeleton.

Stage 0 does not load a model or execute upstream code. The authoritative source
mapping lives in ``docs/UPSTREAM_API_AUDIT.md``. In particular, consult its
sections on threshold/preactivation semantics, attribution/Jacobian access, and
replacement-versus-underlying-model intervention behavior before implementing
any method below. This module never infers support from a symbol name alone.
"""

from __future__ import annotations

from importlib import metadata, util

from cfsus.exceptions import (
    BackendUnavailableError,
    ScientificInputError,
    UnsupportedBackendOperationError,
)
from cfsus.types import (
    BackendCapabilityReport,
    BackendOperation,
    BaselineFeatureState,
    BehaviorMetricMetadata,
    CapabilityEvidence,
    CapabilityStatus,
    FeatureRef,
    InterventionRegime,
    ModelSetting,
    ObservedInterventionPoint,
    SourceSuppression,
)

_DISTRIBUTION_NAME = "circuit-tracer"
_IMPORT_NAME = "circuit_tracer"


def _dependency_version() -> str | None:
    try:
        return metadata.version(_DISTRIBUTION_NAME)
    except metadata.PackageNotFoundError:
        return None


def _dependency_available() -> bool:
    return util.find_spec(_IMPORT_NAME) is not None


class CircuitTracerAdapter:
    """Capability-gated placeholder for an audited future upstream mapping.

    Dependency presence is only environment metadata. It never promotes an
    operation to ``SUPPORTED``: that requires both source-level verification in
    ``docs/UPSTREAM_API_AUDIT.md`` and a tested local mapping. The audited commit
    must also be checked against the installed distribution before model use.
    """

    def capability_report(self) -> BackendCapabilityReport:
        """Report audited upstream support separately from local implementation.

        ``UPSTREAM_SUPPORTED`` means source inspection verified a public route,
        but the deliberately non-executing Stage 0 adapter still cannot run it.
        Only ``SUPPORTED`` would make ``require_capability`` succeed.
        """

        available = _dependency_available()
        dependency_version = _dependency_version() if available else None
        evidence = (
            CapabilityEvidence(
                BackendOperation.THRESHOLD_ACCESS,
                CapabilityStatus.UPSTREAM_SUPPORTED,
                "Public raw JumpReLU thresholds verified; audit section 4. Local "
                "checkpoint-shape mapping remains deferred.",
            ),
            CapabilityEvidence(
                BackendOperation.ACTIVE_ACTIVATION_ACCESS,
                CapabilityStatus.UPSTREAM_SUPPORTED,
                "Public dense/sparse post-gate cache verified; audit section 6.",
            ),
            CapabilityEvidence(
                BackendOperation.INACTIVE_PREACTIVATION_ACCESS,
                CapabilityStatus.UPSTREAM_SUPPORTED,
                "Public bias-inclusive dense all-feature cache verified; it is not "
                "scalable or targeted. See audit section 6.",
            ),
            CapabilityEvidence(
                BackendOperation.TARGETED_PREACTIVATION_ACCESS,
                CapabilityStatus.UNSUPPORTED,
                "No public feature-targeted/chunked access; audit sections 6 and 9.",
            ),
            CapabilityEvidence(
                BackendOperation.LOCAL_JACOBIAN,
                CapabilityStatus.UNSUPPORTED,
                "No public raw J_ij API, especially for inactive targets; "
                "audit section 7.",
            ),
            CapabilityEvidence(
                BackendOperation.JACOBIAN_VECTOR_PRODUCT,
                CapabilityStatus.UNSUPPORTED,
                "No public JVP API; audit section 7.",
            ),
            CapabilityEvidence(
                BackendOperation.VECTOR_JACOBIAN_PRODUCT,
                CapabilityStatus.UNSUPPORTED,
                "No public VJP API; audit section 7.",
            ),
            CapabilityEvidence(
                BackendOperation.DIRECT_EFFECT,
                CapabilityStatus.UPSTREAM_SUPPORTED,
                "Graph edges expose a_j*J_ij only for selected baseline-active targets "
                "under the frozen attribution convention; audit section 7.",
            ),
            CapabilityEvidence(
                BackendOperation.REPLACEMENT_SOURCE_SUPPRESSION,
                CapabilityStatus.UNSUPPORTED,
                "No distinct replacement-only feature intervention; audit section 8.",
            ),
            CapabilityEvidence(
                BackendOperation.REPLACEMENT_TARGET_CLAMP,
                CapabilityStatus.UNSUPPORTED,
                "No distinct replacement-only target clamp; audit section 8.",
            ),
            CapabilityEvidence(
                BackendOperation.UNDERLYING_SOURCE_SUPPRESSION,
                CapabilityStatus.UPSTREAM_SUPPORTED,
                "Absolute post-gate decoder edit can encode suppression in the "
                "underlying LM; nonlinear sweeps require freeze_attention=False and "
                "constrained_layers=None. Audit section 8; local mapping deferred.",
            ),
            CapabilityEvidence(
                BackendOperation.UNDERLYING_TARGET_CLAMP,
                CapabilityStatus.UPSTREAM_SUPPORTED,
                "An absolute zero post-gate decoder-coordinate edit is available at "
                "the feature output; it is not a persistent preactivation clamp. "
                "Audit section 8; local mapping remains deferred.",
            ),
            CapabilityEvidence(
                BackendOperation.UNDERLYING_MODEL_INTERVENTION,
                CapabilityStatus.UPSTREAM_SUPPORTED,
                "Public feature_intervention edits the underlying LM residual "
                "computation; audit section 8.",
            ),
            CapabilityEvidence(
                BackendOperation.SELECTED_STATE_CAPTURE,
                CapabilityStatus.UNSUPPORTED,
                "Only a full activation/preactivation cache is public; "
                "audit section 9.",
            ),
            CapabilityEvidence(
                BackendOperation.FULL_STATE_CACHE_CAPTURE,
                CapabilityStatus.UPSTREAM_SUPPORTED,
                "Public interventions can return the full activation or preactivation "
                "cache; audit section 9. Local mapping remains deferred.",
            ),
            CapabilityEvidence(
                BackendOperation.OUTPUT_LOGIT_CAPTURE,
                CapabilityStatus.UPSTREAM_SUPPORTED,
                "Public feature_intervention returns logits; audit section 9.",
            ),
        )
        return BackendCapabilityReport(
            backend_name="decoderesearch/circuit-tracer",
            dependency_available=available,
            dependency_version=dependency_version,
            evidence=evidence,
        )

    def require_capability(self, operation: BackendOperation) -> None:
        """Reject every operation until an audited local mapping exists."""

        report = self.capability_report()
        if not report.dependency_available:
            raise BackendUnavailableError(
                "optional circuit-tracer dependency is not installed; Stage 0 does "
                "not install or download model tooling"
            )
        status = report.status_for(operation)
        if status is CapabilityStatus.SUPPORTED:
            return
        raise UnsupportedBackendOperationError(
            f"{operation.value} is {status.value} in the conservative Stage 0 adapter; "
            "consult docs/UPSTREAM_API_AUDIT.md before implementing the mapping"
        )

    def baseline_feature_state(self, feature: FeatureRef) -> BaselineFeatureState:
        """Reject baseline execution; no upstream tensor API is guessed here."""

        self.require_capability(BackendOperation.TARGETED_PREACTIVATION_ACCESS)
        raise AssertionError("require_capability must raise")

    def run_source_suppression(
        self,
        intervention: SourceSuppression,
        target: FeatureRef,
        setting: ModelSetting,
        regime: InterventionRegime | None,
        behavior_metric: BehaviorMetricMetadata | None = None,
    ) -> ObservedInterventionPoint:
        """Reject intervention execution until semantics are mapped and tested."""

        if not isinstance(setting, ModelSetting):
            raise ScientificInputError("setting must be a ModelSetting")
        if setting is ModelSetting.REPLACEMENT_MODEL:
            if regime is not None:
                raise ScientificInputError(
                    "replacement-model suppression does not accept an underlying "
                    "intervention regime"
                )
            operation = BackendOperation.REPLACEMENT_SOURCE_SUPPRESSION
        else:
            if not isinstance(regime, InterventionRegime):
                raise ScientificInputError(
                    "underlying-model suppression requires a declared "
                    "InterventionRegime"
                )
            operation = BackendOperation.UNDERLYING_SOURCE_SUPPRESSION
        self.require_capability(operation)
        raise AssertionError("require_capability must raise")
