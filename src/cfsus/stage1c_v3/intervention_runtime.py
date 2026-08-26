"""Held-out v3 single-source intervention adapter with dynamic token length."""

from __future__ import annotations

import gc
import math
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from cfsus.backends.nnsight_plt import NNSightPLTMeasurementBackend
from cfsus.exceptions import ScientificInputError
from cfsus.stage1c_v3.intervention import AppliedValuePlan, applied_plan_record
from cfsus.types import FeatureRef, MeasuredFeatureState

AttemptRecorder = Callable[[dict[str, Any], int], None]


@runtime_checkable
class Stage1CInterventionBackendProtocol(Protocol):
    """Complete production surface consumed by the intervention worker."""

    source_suppression_api_calls: int

    def measure_states(
        self, features: Sequence[FeatureRef]
    ) -> dict[FeatureRef, MeasuredFeatureState]: ...

    def measure_point(
        self,
        pair: dict[str, Any],
        plan: AppliedValuePlan,
        *,
        freeze_attention: bool,
        constrained_layers: None,
        stage: str,
    ) -> dict[str, Any]: ...


def _feature(value: dict[str, Any], label: str) -> FeatureRef:
    if set(value) != {"layer", "position", "feature_id"}:
        raise ScientificInputError(f"{label} feature record is invalid")
    values = (value["layer"], value["position"], value["feature_id"])
    if any(isinstance(item, bool) or not isinstance(item, int) for item in values):
        raise ScientificInputError(f"{label} feature record is invalid")
    try:
        return FeatureRef(
            layer=value["layer"],
            position=value["position"],
            feature_id=value["feature_id"],
        )
    except (TypeError, ValueError) as error:
        raise ScientificInputError(f"{label} feature record is invalid") from error


class Stage1CVersion3InterventionBackend:
    """Execute one absolute source edit and return one downstream target scalar."""

    def __init__(
        self,
        model: Any,
        *,
        prompt: str,
        torch: Any,
        token_count: int | None = None,
        attempt_recorder: AttemptRecorder | None = None,
    ) -> None:
        self.model = model
        self.prompt = prompt
        self.torch = torch
        if getattr(model, "backend", None) != "nnsight":
            raise ScientificInputError("intervention model backend is not NNsight")
        if (
            str(getattr(model, "device", "")) != "mps:0"
            or model.dtype != torch.bfloat16
        ):
            raise ScientificInputError("intervention model is not MPS/BF16")
        if token_count is None:
            token_count = int(model.ensure_tokenized(prompt).shape[-1])
        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count < 2
        ):
            raise ScientificInputError(
                "preregistered prompt must have a BOS and content token"
            )
        self.token_count = token_count
        self.transcoders = getattr(model.transcoders, "_module", model.transcoders)
        self._measurement_backend = NNSightPLTMeasurementBackend(
            model,
            prompt=prompt,
            prompt_id="capital_norway_preregistered_v3",
            torch=torch,
        )
        self.source_suppression_api_calls = 0
        self._attempt_recorder = attempt_recorder

    def measure_states(
        self, features: Sequence[FeatureRef]
    ) -> dict[FeatureRef, MeasuredFeatureState]:
        """Remeasure endpoints through the loaded runtime's canonical primitive."""

        unique = tuple(sorted(set(features)))
        if not unique or len(unique) != len(features):
            raise ScientificInputError(
                "baseline feature set must be nonempty and duplicate-free"
            )
        if any(
            not isinstance(feature, FeatureRef)
            or feature.layer >= 18
            or feature.position < 1
            or feature.position >= self.token_count
            or feature.feature_id >= 16_384
            for feature in unique
        ):
            raise ScientificInputError(
                "baseline feature set is outside the frozen runtime domain"
            )
        states = self._measurement_backend.measure_states(unique)
        if not isinstance(states, dict) or set(states) != set(unique):
            raise ScientificInputError(
                "loaded runtime returned an incomplete baseline state map"
            )
        for feature, state in states.items():
            if (
                not isinstance(state, MeasuredFeatureState)
                or state.feature != feature
                or state.device != "mps:0"
                or state.dtype != "torch.bfloat16"
            ):
                raise ScientificInputError(
                    "loaded baseline state changed identity, device, or dtype"
                )
        return states

    def measure_point(
        self,
        pair: dict[str, Any],
        plan: AppliedValuePlan,
        *,
        freeze_attention: bool,
        constrained_layers: None,
        stage: str,
    ) -> dict[str, Any]:
        """Apply one source edit and derive target state using the exact loaded gate."""

        if freeze_attention is not True or constrained_layers is not None:
            raise ScientificInputError("canonical intervention regime changed")
        source = _feature(dict(pair["source"]), "source")
        target = _feature(dict(pair["target"]), "target")
        if source.layer >= target.layer or source.position > target.position:
            raise ScientificInputError("intervention pair violates causal order")
        if (
            source.position < 1
            or target.position < 1
            or source.position >= self.token_count
            or target.position >= self.token_count
        ):
            raise ScientificInputError(
                "intervention feature position is outside non-BOS prompt"
            )
        if plan.tensor.device.type != "mps" or plan.tensor.dtype != self.torch.bfloat16:
            raise ScientificInputError("applied source value is not MPS/BF16")
        logits: Any = None
        cache: Any = None
        started = time.perf_counter()
        try:
            next_call_index = self.source_suppression_api_calls + 1
            if self._attempt_recorder is not None:
                self._attempt_recorder(pair, next_call_index)
            self.source_suppression_api_calls += 1
            logits, cache = self.model.feature_intervention(
                self.prompt,
                [(source.layer, source.position, source.feature_id, plan.tensor)],
                constrained_layers=None,
                freeze_attention=True,
                apply_activation_function=False,
                sparse=False,
                return_activations=True,
            )
            for label, tensor in (
                ("intervention logits", logits),
                ("preactivation cache", cache),
            ):
                if (
                    tensor is None
                    or tensor.device.type != "mps"
                    or tensor.dtype != self.torch.bfloat16
                ):
                    raise ScientificInputError(f"{label} is not MPS/BF16")
                if not bool(self.torch.isfinite(tensor).all().item()):
                    raise ScientificInputError(f"{label} is non-finite")
            if cache.ndim != 3 or tuple(cache.shape[:2]) != (18, self.token_count):
                raise ScientificInputError(
                    "intervention preactivation cache shape changed"
                )
            target_z = cache[target.layer, target.position, target.feature_id].reshape(
                ()
            )
            transcoder = self.transcoders[target.layer]
            threshold = transcoder.activation_function.threshold[
                target.feature_id
            ].reshape(())
            from circuit_tracer.transcoder.activation_functions import (  # type: ignore[import-not-found]
                jumprelu,
            )

            activation = jumprelu.apply(
                target_z, threshold, transcoder.activation_function.bandwidth
            ).reshape(())
            strict = target_z > threshold
            reference = target_z * strict
            if not bool(self.torch.equal(activation, reference)):
                raise ScientificInputError(
                    "loaded target gate differs from strict reference"
                )
            z_value = float(target_z.item())
            threshold_value = float(threshold.item())
            activation_value = float(activation.item())
            if not all(
                math.isfinite(value)
                for value in (z_value, threshold_value, activation_value)
            ):
                raise ScientificInputError("selected target state is non-finite")
            target_active = bool(strict.item())
            if target_active != (z_value > threshold_value):
                raise ScientificInputError(
                    "selected target strict gate is inconsistent"
                )
            if (target_active and activation_value != z_value) or (
                not target_active and activation_value != 0.0
            ):
                raise ScientificInputError("selected target activation is inconsistent")
            record = applied_plan_record(plan)
            record.update(
                {
                    "source_suppression_api_call_index": (
                        self.source_suppression_api_calls
                    ),
                    "point_elapsed_seconds": time.perf_counter() - started,
                    "stage": stage,
                    "target_preactivation": z_value,
                    "target_threshold": threshold_value,
                    "target_activation": activation_value,
                    "target_active": target_active,
                    "loaded_gate": "a=z*1[z>tau]",
                    "threshold_equality_activity": "inactive",
                    "source_activation_observation": (
                        "actual_bf16_value_passed_to_absolute_intervention_tuple"
                    ),
                    "freeze_attention": True,
                    "constrained_layers": None,
                    "target_clamped": False,
                    "source_value_device": "mps:0",
                    "source_value_dtype": "torch.bfloat16",
                    "target_value_device": "mps:0",
                    "target_value_dtype": "torch.bfloat16",
                    "finite_checks": {
                        "applied_source": True,
                        "logits": True,
                        "preactivation_cache": True,
                        "target_preactivation": True,
                        "target_threshold": True,
                        "target_activation": True,
                    },
                    "logits_finite": True,
                    "logits_shape": [int(item) for item in logits.shape],
                }
            )
            return record
        finally:
            if cache is not None:
                del cache
            if logits is not None:
                del logits
            gc.collect()
            self.torch.mps.empty_cache()


# Short alias for worker code that is shared structurally with the v1 runner.
Stage1CInterventionBackend = Stage1CVersion3InterventionBackend


def _static_protocol_check(
    backend: Stage1CVersion3InterventionBackend,
) -> Stage1CInterventionBackendProtocol:
    """Give strict type checking a concrete adapter-to-protocol assignment."""

    return backend


__all__ = [
    "AttemptRecorder",
    "Stage1CInterventionBackend",
    "Stage1CInterventionBackendProtocol",
    "Stage1CVersion3InterventionBackend",
]
