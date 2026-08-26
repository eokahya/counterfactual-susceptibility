"""Narrow single-source intervention adapter with selected target-state capture."""

from __future__ import annotations

import gc
import math
from typing import Any

from cfsus.exceptions import ScientificInputError
from cfsus.stage1c.intervention import AppliedValuePlan, applied_plan_record
from cfsus.types import FeatureRef


def _feature(value: dict[str, Any], label: str) -> FeatureRef:
    try:
        result = FeatureRef(
            int(value["layer"]), int(value["position"]), int(value["feature_id"])
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ScientificInputError(f"{label} feature record is invalid") from error
    return result


class Stage1CInterventionBackend:
    """Execute one absolute source edit and return one downstream target scalar."""

    def __init__(self, model: Any, *, prompt: str, torch: Any) -> None:
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
        self.transcoders = getattr(model.transcoders, "_module", model.transcoders)

    def measure_point(
        self,
        pair: dict[str, Any],
        plan: AppliedValuePlan,
        *,
        freeze_attention: bool,
        constrained_layers: None,
        stage: str,
    ) -> dict[str, Any]:
        """Apply one source edit and derive target state by the loaded strict gate."""

        if freeze_attention is not True or constrained_layers is not None:
            raise ScientificInputError("canonical intervention regime changed")
        source = _feature(dict(pair["source"]), "source")
        target = _feature(dict(pair["target"]), "target")
        if source.layer >= target.layer or source.position > target.position:
            raise ScientificInputError("intervention pair violates causal order")
        if plan.tensor.device.type != "mps" or plan.tensor.dtype != self.torch.bfloat16:
            raise ScientificInputError("applied source value is not MPS/BF16")
        logits: Any = None
        cache: Any = None
        try:
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
            if cache.ndim != 3 or tuple(cache.shape[:2]) != (18, 6):
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
                target_z,
                threshold,
                transcoder.activation_function.bandwidth,
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


__all__ = ["Stage1CInterventionBackend"]
