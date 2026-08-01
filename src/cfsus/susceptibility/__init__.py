"""Backend-independent susceptibility mathematics."""

from cfsus.susceptibility.pairwise import (
    activation_margin,
    classify_predicted_crossing,
    critical_suppression_fraction,
    pairwise_susceptibility,
    predict_crossing,
    predicted_target_preactivation,
    suppression_response,
)

__all__ = [
    "activation_margin",
    "classify_predicted_crossing",
    "critical_suppression_fraction",
    "pairwise_susceptibility",
    "predict_crossing",
    "predicted_target_preactivation",
    "suppression_response",
]
