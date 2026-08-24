"""Typed scientific records shared by mathematical and backend layers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from math import isfinite

from cfsus.exceptions import NonFiniteInputError, ScientificInputError


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise NonFiniteInputError(f"{name} must be finite, got {value!r}")


def _require_non_negative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ScientificInputError(f"{name} must be a non-negative integer")


class FeatureActivity(StrEnum):
    """Activity reported by the backend's exact loaded feature gate."""

    ACTIVE = "active"
    INACTIVE = "inactive"


class CrossingStatus(StrEnum):
    """Tolerance-aware classification of a predicted gate crossing."""

    DEFINITELY_CROSSING = "definitely_crossing"
    BOUNDARY_AMBIGUOUS = "boundary_ambiguous"
    NOT_CROSSING = "not_crossing"
    INVALID_TARGET = "invalid_target"
    NON_FINITE_INPUT = "non_finite_input"


class ModelSetting(StrEnum):
    """Where an intervention's consequences were measured."""

    REPLACEMENT_MODEL = "replacement_model"
    UNDERLYING_MODEL = "underlying_model"


class InterventionRegime(StrEnum):
    """Declared nonlinear/frozen state convention for an intervention run."""

    UNDERLYING_NONLINEAR = "underlying_nonlinear"
    ATTENTION_FROZEN = "attention_frozen"
    ATTRIBUTION_MATCHED = "attribution_matched"


class BackendOperation(StrEnum):
    """Granular operations required by the research specification."""

    THRESHOLD_ACCESS = "threshold_access"
    ACTIVE_ACTIVATION_ACCESS = "active_activation_access"
    INACTIVE_PREACTIVATION_ACCESS = "inactive_preactivation_access"
    TARGETED_PREACTIVATION_ACCESS = "targeted_preactivation_access"
    LOCAL_JACOBIAN = "local_jacobian"
    JACOBIAN_VECTOR_PRODUCT = "jacobian_vector_product"
    VECTOR_JACOBIAN_PRODUCT = "vector_jacobian_product"
    DIRECT_EFFECT = "direct_effect"
    REPLACEMENT_SOURCE_SUPPRESSION = "replacement_source_suppression"
    REPLACEMENT_TARGET_CLAMP = "replacement_target_clamp"
    UNDERLYING_SOURCE_SUPPRESSION = "underlying_source_suppression"
    UNDERLYING_TARGET_CLAMP = "underlying_target_clamp"
    UNDERLYING_MODEL_INTERVENTION = "underlying_model_intervention"
    SELECTED_STATE_CAPTURE = "selected_state_capture"
    FULL_STATE_CACHE_CAPTURE = "full_state_cache_capture"
    OUTPUT_LOGIT_CAPTURE = "output_logit_capture"


class CapabilityStatus(StrEnum):
    """Audit status for one backend operation."""

    SUPPORTED = "supported"
    UPSTREAM_SUPPORTED = "upstream_supported_adapter_deferred"
    UNSUPPORTED = "unsupported"
    UNVERIFIED = "unverified"


@dataclass(frozen=True, slots=True, order=True)
class FeatureRef:
    """A feature at one zero-based model layer and token position."""

    layer: int
    position: int
    feature_id: int

    def __post_init__(self) -> None:
        _require_non_negative_integer("layer", self.layer)
        _require_non_negative_integer("position", self.position)
        _require_non_negative_integer("feature_id", self.feature_id)


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    """Immutable identity attached to every Stage 1B measurement."""

    backend: str
    device: str
    dtype: str
    model_revision: str
    transcoder_revision: str
    upstream_revision: str
    prompt_id: str

    def __post_init__(self) -> None:
        for name in (
            "backend",
            "device",
            "dtype",
            "model_revision",
            "transcoder_revision",
            "upstream_revision",
            "prompt_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ScientificInputError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class MeasuredFeatureState:
    """One exact loaded JumpReLU state measured by the runtime.

    The record deliberately contains scalar values only. Tensor projections and
    loaded gate evaluation stay inside the backend on its declared device and
    dtype before crossing this boundary.
    """

    feature: FeatureRef
    preactivation: float
    activation: float
    threshold: float
    activity: FeatureActivity
    device: str
    dtype: str

    def __post_init__(self) -> None:
        if not isinstance(self.feature, FeatureRef):
            raise ScientificInputError("feature must be a FeatureRef")
        for name in ("preactivation", "activation", "threshold"):
            _require_finite(name, getattr(self, name))
        if not isinstance(self.activity, FeatureActivity):
            raise ScientificInputError("activity must be a FeatureActivity")
        if not self.device.strip() or not self.dtype.strip():
            raise ScientificInputError("device and dtype must be non-empty")
        if self.activity is FeatureActivity.INACTIVE:
            if self.activation != 0.0 or self.preactivation > self.threshold:
                raise ScientificInputError(
                    "inactive loaded state must have a=0 and z<=threshold"
                )
        elif (
            self.preactivation <= self.threshold
            or self.activation != self.preactivation
        ):
            raise ScientificInputError(
                "active loaded state must have z>threshold and a=z"
            )

    @property
    def inactive_margin(self) -> float:
        """Return ``threshold-preactivation`` for an inactive state."""

        if self.activity is not FeatureActivity.INACTIVE:
            raise ScientificInputError("inactive margin is undefined for active state")
        margin = self.threshold - self.preactivation
        _require_finite("inactive_margin", margin)
        if margin < 0.0:
            raise ScientificInputError("inactive margin must be non-negative")
        return margin


@dataclass(frozen=True, slots=True)
class NearThresholdCandidate:
    """Compact inactive feature retained by the Stage 1B scanner."""

    feature: FeatureRef
    preactivation: float
    activation: float
    threshold: float
    margin: float
    device: str
    dtype: str

    def __post_init__(self) -> None:
        if not isinstance(self.feature, FeatureRef):
            raise ScientificInputError("feature must be a FeatureRef")
        for name in ("preactivation", "activation", "threshold", "margin"):
            _require_finite(name, getattr(self, name))
        if self.activation != 0.0:
            raise ScientificInputError("near-threshold candidate must be inactive")
        if self.preactivation > self.threshold:
            raise ScientificInputError(
                "near-threshold candidate must have z<=threshold"
            )
        if self.margin < 0.0 or self.margin != self.threshold - self.preactivation:
            raise ScientificInputError("candidate margin must equal threshold-z")
        if not self.device.strip() or not self.dtype.strip():
            raise ScientificInputError("device and dtype must be non-empty")

    @property
    def sort_key(self) -> tuple[float, int, int, int]:
        """Frozen deterministic scanner order."""

        return (
            self.margin,
            self.feature.layer,
            self.feature.position,
            self.feature.feature_id,
        )


@dataclass(frozen=True, slots=True)
class LocalResponseEstimate:
    """Independent scalar ``partial z_i / partial a_j`` measurement."""

    source: FeatureRef
    target: FeatureRef
    source_activation: float
    target_preactivation: float
    response: float
    device: str
    dtype: str
    method: str
    convention: str
    graph_edge_used: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.source, FeatureRef) or not isinstance(
            self.target, FeatureRef
        ):
            raise ScientificInputError("source and target must be FeatureRef values")
        if self.source == self.target:
            raise ScientificInputError("source and target must differ")
        if self.source.layer >= self.target.layer:
            raise ScientificInputError("source layer must be strictly upstream")
        if self.source.position > self.target.position:
            raise ScientificInputError("source token position must be causal upstream")
        for name in ("source_activation", "target_preactivation", "response"):
            _require_finite(name, getattr(self, name))
        if self.source_activation <= 0.0:
            raise ScientificInputError("source must be baseline-active and positive")
        if not self.device.strip() or not self.dtype.strip():
            raise ScientificInputError("device and dtype must be non-empty")
        if not self.method.strip() or not self.convention.strip():
            raise ScientificInputError("method and convention must be non-empty")
        if self.graph_edge_used:
            raise ScientificInputError("targeted response must not use a graph edge")


@dataclass(frozen=True, slots=True)
class ActivePairReference:
    """Compact raw-graph reference admitted only by independent validation."""

    pair_id: str
    source: FeatureRef
    target: FeatureRef
    source_activation: float
    raw_edge: float

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or len(self.pair_id) != 64:
            raise ScientificInputError("pair_id must be a lowercase SHA-256 string")
        if any(character not in "0123456789abcdef" for character in self.pair_id):
            raise ScientificInputError("pair_id must be a lowercase SHA-256 string")
        if not isinstance(self.source, FeatureRef) or not isinstance(
            self.target, FeatureRef
        ):
            raise ScientificInputError("source and target must be FeatureRef values")
        if self.source == self.target or self.source.layer >= self.target.layer:
            raise ScientificInputError("reference pair must be strictly layer-upstream")
        if self.source.position > self.target.position:
            raise ScientificInputError("reference pair must be causal upstream")
        _require_finite("source_activation", self.source_activation)
        _require_finite("raw_edge", self.raw_edge)
        if self.source_activation <= 0.0 or self.raw_edge == 0.0:
            raise ScientificInputError(
                "reference requires positive source and nonzero edge"
            )


@dataclass(frozen=True, slots=True)
class BaselineFeatureState:
    """Baseline values under the backend's exact activation convention.

    Preactivation and threshold have identical units. Activation may have a
    backend-defined scale. ``activity`` is reported explicitly because Stage 0
    must not reimplement the loaded transcoder's gate from memory.
    """

    feature: FeatureRef
    preactivation: float
    activation: float
    threshold: float
    activity: FeatureActivity

    def __post_init__(self) -> None:
        if not isinstance(self.feature, FeatureRef):
            raise ScientificInputError("feature must be a FeatureRef")
        _require_finite("preactivation", self.preactivation)
        _require_finite("activation", self.activation)
        _require_finite("threshold", self.threshold)
        if not isinstance(self.activity, FeatureActivity):
            raise ScientificInputError("activity must be a FeatureActivity")


@dataclass(frozen=True, slots=True)
class SourceSuppression:
    """Suppress a source post-gate activation by ``alpha`` in ``[0, 1]``.

    The local scientific convention is always ``a -> (1 - alpha) * a``.
    Backends are responsible for mapping this convention to verified upstream
    intervention semantics without changing it.
    """

    source: FeatureRef
    alpha: float

    def __post_init__(self) -> None:
        if not isinstance(self.source, FeatureRef):
            raise ScientificInputError("source must be a FeatureRef")
        _require_finite("alpha", self.alpha)
        if not 0.0 <= self.alpha <= 1.0:
            raise ScientificInputError("alpha must lie in the closed interval [0, 1]")

    def suppressed_activation(self, baseline_activation: float) -> float:
        """Return ``(1 - alpha) * baseline_activation`` after validation."""

        _require_finite("baseline_activation", baseline_activation)
        return (1.0 - self.alpha) * baseline_activation


@dataclass(frozen=True, slots=True)
class PredictedCrossing:
    """Complete diagnostic record for one local pairwise prediction.

    Margin and ``q`` share preactivation units; susceptibility and critical
    alpha are dimensionless. Nullable derived values denote a quantity that is
    undefined under the declared status, never an artificial infinity.
    """

    margin: float | None
    q: float | None
    susceptibility: float | None
    predicted_critical_alpha: float | None
    status: CrossingStatus
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, CrossingStatus):
            raise ScientificInputError("status must be a CrossingStatus")
        if not self.reason.strip():
            raise ScientificInputError("reason must be non-empty")
        if self.status is not CrossingStatus.NON_FINITE_INPUT:
            for name in ("margin", "q", "susceptibility"):
                value = getattr(self, name)
                if value is not None:
                    _require_finite(name, value)
            if self.predicted_critical_alpha is not None:
                _require_finite(
                    "predicted_critical_alpha", self.predicted_critical_alpha
                )


@dataclass(frozen=True, slots=True)
class ObservedInterventionPoint:
    """One point from a source-suppression sweep.

    ``target_active`` must be computed with the exact loaded backend gate. The
    optional behavior value uses the metric declared by ``BehaviorMetricMetadata``.
    """

    alpha: float
    target_preactivation: float
    target_activation: float
    target_active: bool
    behavior_value: float | None = None

    def __post_init__(self) -> None:
        _require_finite("alpha", self.alpha)
        if not 0.0 <= self.alpha <= 1.0:
            raise ScientificInputError("alpha must lie in the closed interval [0, 1]")
        _require_finite("target_preactivation", self.target_preactivation)
        _require_finite("target_activation", self.target_activation)
        if not isinstance(self.target_active, bool):
            raise ScientificInputError("target_active must be a bool")
        if self.behavior_value is not None:
            _require_finite("behavior_value", self.behavior_value)


@dataclass(frozen=True, slots=True)
class ObservedSweepResult:
    """Typed observations from one backend and model setting."""

    source: FeatureRef
    target: FeatureRef
    setting: ModelSetting
    points: tuple[ObservedInterventionPoint, ...]
    observed_critical_alpha: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.source, FeatureRef) or not isinstance(
            self.target, FeatureRef
        ):
            raise ScientificInputError("source and target must be FeatureRef values")
        if not isinstance(self.setting, ModelSetting):
            raise ScientificInputError("setting must be a ModelSetting")
        if not self.points:
            raise ScientificInputError(
                "an observed sweep must contain at least one point"
            )
        if any(
            later.alpha <= earlier.alpha for earlier, later in pairwise(self.points)
        ):
            raise ScientificInputError("sweep alpha values must be strictly increasing")
        if self.observed_critical_alpha is not None:
            _require_finite("observed_critical_alpha", self.observed_critical_alpha)
            if not 0.0 <= self.observed_critical_alpha <= 1.0:
                raise ScientificInputError("observed_critical_alpha must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class BehaviorMetricMetadata:
    """Metadata placeholder for a future, explicitly declared scalar metric.

    Stage 0 intentionally stores no behavioral-salience or mediation formula:
    their exact backend interface and numerical stability policy remain deferred.
    """

    name: str
    description: str
    units: str
    sign_convention: str

    def __post_init__(self) -> None:
        for name in ("name", "description", "units", "sign_convention"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ScientificInputError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class CapabilityEvidence:
    """Audited status and evidence summary for one backend operation."""

    operation: BackendOperation
    status: CapabilityStatus
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.operation, BackendOperation):
            raise ScientificInputError("operation must be a BackendOperation")
        if not isinstance(self.status, CapabilityStatus):
            raise ScientificInputError("status must be a CapabilityStatus")
        if not self.detail.strip():
            raise ScientificInputError("capability detail must be non-empty")


@dataclass(frozen=True, slots=True)
class BackendCapabilityReport:
    """Granular, typed capability report for a backend adapter."""

    backend_name: str
    dependency_available: bool
    dependency_version: str | None
    evidence: tuple[CapabilityEvidence, ...]

    def __post_init__(self) -> None:
        if not self.backend_name.strip():
            raise ScientificInputError("backend_name must be non-empty")
        if not isinstance(self.dependency_available, bool):
            raise ScientificInputError("dependency_available must be a bool")
        if self.dependency_version is not None and not self.dependency_version.strip():
            raise ScientificInputError("dependency_version must be non-empty or None")
        operations = tuple(item.operation for item in self.evidence)
        if len(operations) != len(set(operations)):
            raise ScientificInputError("capability operations must be unique")

    def status_for(self, operation: BackendOperation) -> CapabilityStatus:
        """Return the audited status for ``operation`` or ``UNVERIFIED``."""

        for item in self.evidence:
            if item.operation is operation:
                return item.status
        return CapabilityStatus.UNVERIFIED
