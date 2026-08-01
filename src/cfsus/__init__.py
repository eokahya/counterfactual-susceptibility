"""Counterfactual Susceptibility research scaffold."""

from cfsus.config import SusceptibilityConfig
from cfsus.types import (
    BackendCapabilityReport,
    BaselineFeatureState,
    BehaviorMetricMetadata,
    CrossingStatus,
    FeatureActivity,
    FeatureRef,
    InterventionRegime,
    ModelSetting,
    ObservedInterventionPoint,
    ObservedSweepResult,
    PredictedCrossing,
    SourceSuppression,
)

__all__ = [
    "BackendCapabilityReport",
    "BaselineFeatureState",
    "BehaviorMetricMetadata",
    "CrossingStatus",
    "FeatureActivity",
    "FeatureRef",
    "InterventionRegime",
    "ModelSetting",
    "ObservedInterventionPoint",
    "ObservedSweepResult",
    "PredictedCrossing",
    "SourceSuppression",
    "SusceptibilityConfig",
]

__version__ = "0.0.0"
