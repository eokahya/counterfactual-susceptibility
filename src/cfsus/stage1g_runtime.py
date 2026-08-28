"""Stage 1G output-sensitivity and multi-edit runtime adapters.

The adapters are deliberately narrow: they preserve the accepted NNsight
MPS/BF16 replacement runtime, expose only scalar behavior evidence, and never
persist a gradient, activation cache, or vocabulary-sized logit tensor.
"""

from __future__ import annotations

import gc
import math
import time
from collections.abc import Callable, Sequence
from typing import Any

from cfsus.exceptions import ScientificInputError
from cfsus.stage1c_v3.intervention_runtime import (
    Stage1CVersion3InterventionBackend,
)
from cfsus.stage1c_v3.runtime import Stage1CVersion3PredictionBackend
from cfsus.types import FeatureRef

AttemptRecorder = Callable[[dict[str, Any], int], None]


def _feature(value: dict[str, Any], label: str) -> FeatureRef:
    if set(value) != {"layer", "position", "feature_id"}:
        raise ScientificInputError(f"{label} feature record is invalid")
    coordinates = (value["layer"], value["position"], value["feature_id"])
    if any(isinstance(item, bool) or not isinstance(item, int) for item in coordinates):
        raise ScientificInputError(f"{label} feature record is invalid")
    return FeatureRef(**value)


class OutputSensitivityVJPContext:
    """Compute bounded graph-independent ``dT/da_i`` contractions."""

    def __init__(self, torch: Any, *, maximum_targets: int) -> None:
        if isinstance(maximum_targets, bool) or maximum_targets < 1:
            raise ScientificInputError("maximum output-sensitivity targets is invalid")
        self.torch = torch
        self.maximum_targets = maximum_targets
        self._final_hidden: Any = None
        self._feature_outputs: list[Any] = []

    def cache(self, model: Any, tracer: Any, barrier: Any = None) -> None:
        """Cache the accepted V3 residual/output points in two invocations."""

        with tracer.invoke():
            self._final_hidden = model.pre_logit_location.output.last_hidden_state
        with tracer.invoke():
            self._feature_outputs.append(model.embed_location.output)
            for feature_output in model.feature_output_locs:
                if barrier:
                    barrier()
                self._feature_outputs.append(feature_output.output)

    def compute(
        self,
        targets: Sequence[FeatureRef],
        *,
        target_decoder_vectors: Any,
        logit_direction: Any,
        final_position: int,
    ) -> Any:
        """Return one scalar VJP per target without constructing graph edges."""

        if (
            not targets
            or len(targets) > self.maximum_targets
            or tuple(targets) != tuple(sorted(set(targets)))
            or self._final_hidden is None
            or len(self._feature_outputs) != 19
        ):
            raise ScientificInputError("output-sensitivity context or targets differ")
        torch = self.torch
        hidden = int(target_decoder_vectors.shape[-1])
        if tuple(target_decoder_vectors.shape) != (len(targets), hidden):
            raise ScientificInputError("target decoder vector shape differs")
        if tuple(logit_direction.shape) != (hidden,):
            raise ScientificInputError("answer-minus-contrast direction shape differs")
        for label, tensor in (
            ("target decoder", target_decoder_vectors),
            ("logit direction", logit_direction),
        ):
            if tensor.device.type != "mps" or tensor.dtype != torch.bfloat16:
                raise ScientificInputError(f"{label} must remain MPS/BF16")
            if not bool(torch.isfinite(tensor).all().item()):
                raise ScientificInputError(f"{label} is non-finite")
        if final_position < 1:
            raise ScientificInputError("behavior position must be a non-BOS token")

        batch_indices = torch.arange(len(targets), device="mps", dtype=torch.long)
        target_layers = torch.tensor(
            [target.layer for target in targets], device="mps", dtype=torch.long
        )
        target_positions = torch.tensor(
            [target.position for target in targets], device="mps", dtype=torch.long
        )
        result = torch.zeros(len(targets), device="mps", dtype=torch.bfloat16)
        final_positions = torch.full(
            (len(targets),), final_position, device="mps", dtype=torch.long
        )
        with self._final_hidden.backward(
            gradient=torch.zeros_like(self._final_hidden), retain_graph=False
        ):
            final_gradients = self._final_hidden.grad.clone()
            final_gradients.index_put_(
                (batch_indices, final_positions),
                logit_direction.expand(len(targets), -1),
                accumulate=False,
            )
            self._final_hidden.grad = final_gradients
            for layer in reversed(range(18)):
                rows = torch.nonzero(target_layers == layer, as_tuple=False).flatten()
                if int(rows.numel()) == 0:
                    continue
                gradients = self._feature_outputs[layer + 1].grad.clone()
                selected = gradients[rows, target_positions[rows]]
                values = (selected * target_decoder_vectors[rows]).sum(dim=-1)
                result.index_put_((rows,), values.to(torch.bfloat16), accumulate=False)
        if result.device.type != "mps" or result.dtype != torch.bfloat16:
            raise ScientificInputError("output-sensitivity result moved off MPS/BF16")
        if not bool(torch.isfinite(result).all().item()):
            raise ScientificInputError("output-sensitivity result is non-finite")
        return result

    def clear(self) -> None:
        self._feature_outputs.clear()
        self._final_hidden = None


class Stage1GPredictionBackend(Stage1CVersion3PredictionBackend):
    """Stage 1C prediction primitives plus independent target-to-output VJPs."""

    def baseline_behavior(
        self, *, answer_token_id: int, contrast_token_id: int
    ) -> dict[str, Any]:
        if (
            isinstance(answer_token_id, bool)
            or isinstance(contrast_token_id, bool)
            or answer_token_id < 0
            or contrast_token_id < 0
            or answer_token_id == contrast_token_id
        ):
            raise ScientificInputError("behavior token IDs are invalid")
        logits: Any = None
        try:
            logits, _ = self.model.feature_intervention(
                self.prompt,
                [],
                constrained_layers=None,
                freeze_attention=True,
                return_activations=False,
            )
            if (
                logits.device.type != "mps"
                or logits.dtype != self.torch.bfloat16
                or logits.ndim != 3
                or not bool(self.torch.isfinite(logits).all().item())
            ):
                raise ScientificInputError("baseline behavior logits are invalid")
            last = logits[0, -1]
            answer = float(last[answer_token_id].item())
            contrast = float(last[contrast_token_id].item())
            top64 = {
                int(item)
                for item in self.torch.topk(last, k=min(64, int(last.numel())))
                .indices.detach()
                .cpu()
                .tolist()
            }
            values = (answer, contrast, answer - contrast)
            if not all(math.isfinite(item) for item in values):
                raise ScientificInputError("baseline behavior scalar is non-finite")
            return {
                "answer_logit": answer,
                "contrast_logit": contrast,
                "behavior_T": answer - contrast,
                "answer_in_top64": answer_token_id in top64,
                "logits_finite": True,
                "logits_shape": [int(item) for item in logits.shape],
            }
        finally:
            if logits is not None:
                del logits
            gc.collect()
            self.torch.mps.empty_cache()

    def output_sensitivities(
        self,
        targets: Sequence[FeatureRef],
        *,
        answer_token_id: int,
        contrast_token_id: int,
        maximum_targets: int,
    ) -> tuple[float, ...]:
        if not targets or len(targets) > maximum_targets:
            raise ScientificInputError("output-sensitivity target batch differs")
        if answer_token_id == contrast_token_id:
            raise ScientificInputError("behavior token IDs must differ")
        torch = self.torch
        transcoders = self._transcoders()
        target_vectors = torch.stack(
            [
                transcoders[target.layer]._get_decoder_vectors(
                    torch.tensor([target.feature_id], device="mps", dtype=torch.long)
                )[0]
                for target in targets
            ]
        )
        logit_direction = (
            self.model.unembed_weight[answer_token_id]
            - self.model.unembed_weight[contrast_token_id]
        )
        context = OutputSensitivityVJPContext(torch, maximum_targets=maximum_targets)
        input_ids = self.model.ensure_tokenized(self.prompt)
        try:
            with self.model.trace() as tracer:
                with tracer.invoke(input_ids.expand(len(targets), -1)):
                    pass
                barrier = tracer.barrier(2)
                self.model.configure_gradient_flow(tracer)
                self.model.configure_skip_connection(tracer, barrier=barrier)
                context.cache(self.model, tracer, barrier=barrier)
            result = context.compute(
                targets,
                target_decoder_vectors=target_vectors,
                logit_direction=logit_direction,
                final_position=int(input_ids.shape[-1]) - 1,
            )
            values = tuple(float(item) for item in result.detach().cpu().tolist())
            if not all(math.isfinite(item) for item in values):
                raise ScientificInputError("output-sensitivity scalar is non-finite")
            return values
        finally:
            context.clear()
            del target_vectors, logit_direction
            if "result" in locals():
                del result
            gc.collect()
            torch.mps.empty_cache()

    def raw_output_edge_sensitivity_reference(
        self,
        targets: Sequence[FeatureRef],
        *,
        answer_token_id: int,
        contrast_token_id: int,
    ) -> tuple[dict[str, float], ...]:
        """Compute an independent signed raw-edge reference for active targets.

        This path deliberately uses the pinned attribution context's raw,
        unnormalized feature-to-logit edge rows.  It never calls, reads, or
        accepts output from :meth:`output_sensitivities` or
        :class:`OutputSensitivityVJPContext`.
        """

        if (
            not targets
            or tuple(targets) != tuple(sorted(set(targets)))
            or answer_token_id == contrast_token_id
        ):
            raise ScientificInputError("raw output-edge reference targets differ")
        torch = self.torch
        input_ids = self.model.ensure_tokenized(self.prompt)
        final_position = int(input_ids.shape[-1]) - 1
        if final_position < 1:
            raise ScientificInputError("raw output-edge behavior position differs")
        context: Any = None
        raw_rows: Any = None
        selected_edges: Any = None
        selected_indices: Any = None
        try:
            context = self.model.setup_attribution(input_ids)
            activation_matrix = context.activation_matrix.coalesce()
            if (
                activation_matrix.device.type != "cpu"
                or activation_matrix.dtype != torch.bfloat16
                or activation_matrix.ndim != 3
            ):
                raise ScientificInputError(
                    "raw output-edge activation metadata differs"
                )
            coordinates = activation_matrix.indices().T.tolist()
            activations = activation_matrix.values().tolist()
            lookup = {
                (int(layer), int(position), int(feature_id)): (
                    index,
                    float(activation),
                )
                for index, ((layer, position, feature_id), activation) in enumerate(
                    zip(coordinates, activations, strict=True)
                )
            }
            matched: list[tuple[int, float]] = []
            for target in targets:
                value = lookup.get((target.layer, target.position, target.feature_id))
                if value is None or value[1] <= 0.0 or not math.isfinite(value[1]):
                    raise ScientificInputError(
                        "raw output-edge target is not baseline active"
                    )
                matched.append(value)
            logit_vectors = torch.stack(
                (
                    self.model.unembed_weight[answer_token_id],
                    self.model.unembed_weight[contrast_token_id],
                )
            )
            if (
                logit_vectors.device.type != "mps"
                or logit_vectors.dtype != torch.bfloat16
                or not bool(torch.isfinite(logit_vectors).all().item())
            ):
                raise ScientificInputError("raw output-edge logit vectors differ")
            with self.model.trace() as tracer:
                with tracer.invoke(input_ids.expand(2, -1)):
                    pass
                barrier = tracer.barrier(2)
                self.model.configure_gradient_flow(tracer)
                self.model.configure_skip_connection(tracer, barrier=barrier)
                context.cache_residual(self.model, tracer, barrier=barrier)
            raw_rows = context.compute_batch(
                layers=torch.full((2,), 18, dtype=torch.long),
                positions=torch.full((2,), final_position, dtype=torch.long),
                inject_values=logit_vectors,
                retain_graph=False,
            )
            if (
                raw_rows.device.type != "mps"
                or raw_rows.dtype != torch.bfloat16
                or tuple(raw_rows.shape[:1]) != (2,)
                or not bool(torch.isfinite(raw_rows).all().item())
            ):
                raise ScientificInputError("raw output-edge rows differ")
            selected_indices = torch.tensor(
                [index for index, _ in matched], device="mps", dtype=torch.long
            )
            selected_edges = raw_rows.index_select(1, selected_indices)
            scalar_edges = selected_edges.detach().cpu().tolist()
            result: list[dict[str, float]] = []
            for column, (_, activation) in enumerate(matched):
                answer_edge = float(scalar_edges[0][column])
                contrast_edge = float(scalar_edges[1][column])
                reference = (answer_edge - contrast_edge) / activation
                if not all(
                    math.isfinite(item)
                    for item in (activation, answer_edge, contrast_edge, reference)
                ):
                    raise ScientificInputError(
                        "raw output-edge reference scalar is non-finite"
                    )
                result.append(
                    {
                        "activation": activation,
                        "raw_answer_logit_edge": answer_edge,
                        "raw_contrast_logit_edge": contrast_edge,
                        "reference_g_i": reference,
                    }
                )
            return tuple(result)
        finally:
            if selected_edges is not None:
                del selected_edges
            if selected_indices is not None:
                del selected_indices
            if raw_rows is not None:
                del raw_rows
            if context is not None:
                del context
            gc.collect()
            torch.mps.empty_cache()


class Stage1GInterventionBackend(Stage1CVersion3InterventionBackend):
    """Apply absolute source/target edits and retain only scalar behavior evidence."""

    def __init__(
        self,
        model: Any,
        *,
        prompt: str,
        torch: Any,
        answer_token_id: int,
        contrast_token_id: int,
        token_count: int | None = None,
        attempt_recorder: AttemptRecorder | None = None,
        prompt_id: str,
        call_index_offset: int = 0,
    ) -> None:
        super().__init__(
            model,
            prompt=prompt,
            torch=torch,
            token_count=token_count,
            attempt_recorder=attempt_recorder,
            prompt_id=prompt_id,
            call_index_offset=call_index_offset,
        )
        if (
            isinstance(answer_token_id, bool)
            or isinstance(contrast_token_id, bool)
            or answer_token_id < 0
            or contrast_token_id < 0
            or answer_token_id == contrast_token_id
        ):
            raise ScientificInputError("behavior token IDs are invalid")
        self.answer_token_id = answer_token_id
        self.contrast_token_id = contrast_token_id

    def measure_condition(
        self,
        pair: dict[str, Any],
        *,
        condition: str,
        desired_source_activation: float | None,
        desired_target_activation: float | None,
        stage: str,
    ) -> dict[str, Any]:
        allowed = {
            "baseline_noop",
            "baseline_repeat",
            "source_full_ablation",
            "source_ablation_target_clamp",
            "target_only_injection",
        }
        if condition not in allowed:
            raise ScientificInputError("Stage 1G intervention condition differs")
        source = _feature(dict(pair["source"]), "source")
        target = _feature(dict(pair["target"]), "target")
        if source.layer >= target.layer or source.position > target.position:
            raise ScientificInputError("Stage 1G pair violates causal order")
        if not stage.strip():
            raise ScientificInputError("Stage 1G intervention stage is empty")
        source_tensor: Any = None
        target_tensor: Any = None
        edits: list[tuple[int, int, int, Any]] = []
        if desired_source_activation is not None:
            if not math.isfinite(desired_source_activation):
                raise ScientificInputError("desired source activation is non-finite")
            source_tensor = self.torch.tensor(
                desired_source_activation, device="mps", dtype=self.torch.bfloat16
            ).reshape(())
            edits.append(
                (source.layer, source.position, source.feature_id, source_tensor)
            )
        if desired_target_activation is not None:
            if not math.isfinite(desired_target_activation):
                raise ScientificInputError("desired target activation is non-finite")
            target_tensor = self.torch.tensor(
                desired_target_activation, device="mps", dtype=self.torch.bfloat16
            ).reshape(())
            edits.append(
                (target.layer, target.position, target.feature_id, target_tensor)
            )
        expected = {
            "baseline_noop": (True, False),
            "baseline_repeat": (True, False),
            "source_full_ablation": (True, False),
            "source_ablation_target_clamp": (True, True),
            "target_only_injection": (False, True),
        }[condition]
        if (source_tensor is not None, target_tensor is not None) != expected:
            raise ScientificInputError("Stage 1G condition edit shape differs")
        if (
            condition in {"source_full_ablation", "source_ablation_target_clamp"}
            and float(source_tensor.item()) != 0.0
        ):
            raise ScientificInputError("source full ablation is not exact zero")
        if (
            condition == "source_ablation_target_clamp"
            and float(target_tensor.item()) != 0.0
        ):
            raise ScientificInputError("target clamp is not exact zero")

        logits: Any = None
        cache: Any = None
        started = time.perf_counter()
        try:
            next_index = self.source_suppression_api_calls + 1
            if self._attempt_recorder is not None:
                self._attempt_recorder(pair, next_index)
            self.source_suppression_api_calls += 1
            logits, cache = self.model.feature_intervention(
                self.prompt,
                edits,
                constrained_layers=None,
                freeze_attention=True,
                apply_activation_function=False,
                sparse=False,
                return_activations=True,
            )
            for label, tensor in (("logits", logits), ("preactivation cache", cache)):
                if (
                    tensor is None
                    or tensor.device.type != "mps"
                    or tensor.dtype != self.torch.bfloat16
                    or not bool(self.torch.isfinite(tensor).all().item())
                ):
                    raise ScientificInputError(f"Stage 1G {label} is invalid")
            if cache.ndim != 3 or tuple(cache.shape[:2]) != (18, self.token_count):
                raise ScientificInputError("Stage 1G preactivation cache shape differs")
            target_z_tensor = cache[
                target.layer, target.position, target.feature_id
            ].reshape(())
            transcoder = self.transcoders[target.layer]
            threshold_tensor = transcoder.activation_function.threshold[
                target.feature_id
            ].reshape(())
            natural_active = target_z_tensor > threshold_tensor
            natural_activation_tensor = target_z_tensor * natural_active
            target_z = float(target_z_tensor.item())
            threshold = float(threshold_tensor.item())
            natural_activation = float(natural_activation_tensor.item())
            target_active = bool(natural_active.item())
            if target_active != (target_z > threshold):
                raise ScientificInputError("Stage 1G strict gate differs")
            last = logits[0, -1]
            answer_logit = float(last[self.answer_token_id].item())
            contrast_logit = float(last[self.contrast_token_id].item())
            behavior = answer_logit - contrast_logit
            scalar_values = (
                target_z,
                threshold,
                natural_activation,
                answer_logit,
                contrast_logit,
                behavior,
            )
            if not all(math.isfinite(item) for item in scalar_values):
                raise ScientificInputError("Stage 1G intervention scalar is non-finite")
            applied_source = (
                None if source_tensor is None else float(source_tensor.item())
            )
            applied_target = (
                None if target_tensor is None else float(target_tensor.item())
            )
            effective_target = (
                natural_activation if applied_target is None else applied_target
            )
            return {
                "source_suppression_api_call_index": self.source_suppression_api_calls,
                "condition": condition,
                "stage": stage,
                "point_elapsed_seconds": time.perf_counter() - started,
                "desired_source_activation": desired_source_activation,
                "actual_bf16_source_activation": applied_source,
                "desired_target_activation": desired_target_activation,
                "actual_bf16_target_activation": applied_target,
                "target_preactivation": target_z,
                "target_threshold": threshold,
                "target_natural_activation": natural_activation,
                "target_effective_activation": effective_target,
                "target_active": target_active,
                "strict_crossing": target_active,
                "loaded_gate": "a=z*1[z>tau]",
                "threshold_equality_activity": "inactive",
                "answer_token_id": self.answer_token_id,
                "contrast_token_id": self.contrast_token_id,
                "answer_logit": answer_logit,
                "contrast_logit": contrast_logit,
                "behavior_T": behavior,
                "freeze_attention": True,
                "constrained_layers": None,
                "target_clamped": condition == "source_ablation_target_clamp",
                "source_value_device": None if source_tensor is None else "mps:0",
                "source_value_dtype": (
                    None if source_tensor is None else "torch.bfloat16"
                ),
                "target_value_device": None if target_tensor is None else "mps:0",
                "target_value_dtype": (
                    None if target_tensor is None else "torch.bfloat16"
                ),
                "logits_finite": True,
                "logits_shape": [int(item) for item in logits.shape],
                "preactivation_cache_persisted": False,
                "full_logits_persisted": False,
            }
        finally:
            if cache is not None:
                del cache
            if logits is not None:
                del logits
            if source_tensor is not None:
                del source_tensor
            if target_tensor is not None:
                del target_tensor
            gc.collect()
            self.torch.mps.empty_cache()


__all__ = [
    "OutputSensitivityVJPContext",
    "Stage1GInterventionBackend",
    "Stage1GPredictionBackend",
]
