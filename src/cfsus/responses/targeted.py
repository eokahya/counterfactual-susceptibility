"""Independent targeted reverse-mode response for the pinned NNsight runtime.

This module intentionally has no graph or adjacency input. It injects a target
encoder direction at the target preactivation and contracts the resulting
source-output gradient with an unscaled source decoder direction.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from cfsus.exceptions import ScientificInputError
from cfsus.types import FeatureRef


class TargetedVJPContext:
    """Cache one frozen NNsight trace and compute only requested VJP scalars."""

    def __init__(self, torch: Any, *, maximum_pairs: int) -> None:
        if isinstance(maximum_pairs, bool) or maximum_pairs < 1:
            raise ScientificInputError("maximum_pairs must be positive")
        self.torch = torch
        self.maximum_pairs = maximum_pairs
        self._residuals: list[Any] = []
        self._feature_outputs: list[Any] = []

    def cache(self, model: Any, tracer: Any, barrier: Any = None) -> None:
        """Record residual and feature-output graph points without graph edges."""

        with tracer.invoke():
            for feature_input in model.feature_input_locs:
                self._residuals.append(feature_input.output)
            self._residuals.append(model.pre_logit_location.output.last_hidden_state)
        with tracer.invoke():
            self._feature_outputs.append(model.embed_location.output)
            for feature_output in model.feature_output_locs:
                if barrier:
                    barrier()
                self._feature_outputs.append(feature_output.output)

    def compute(
        self,
        pairs: Sequence[tuple[FeatureRef, FeatureRef]],
        *,
        target_encoder_vectors: Any,
        source_decoder_vectors: Any,
        retain_graph: bool = False,
    ) -> Any:
        """Return bounded ``partial z_target / partial a_source`` scalars."""

        if not pairs or len(pairs) > self.maximum_pairs:
            raise ScientificInputError("targeted pair batch is empty or too large")
        if len(self._residuals) < 2 or len(self._feature_outputs) < 2:
            raise ScientificInputError("targeted context has not cached a trace")
        for source, target in pairs:
            if not isinstance(source, FeatureRef) or not isinstance(target, FeatureRef):
                raise ScientificInputError(
                    "targeted endpoints must be FeatureRef values"
                )
            if source == target or source.layer >= target.layer:
                raise ScientificInputError("targeted source must be strictly upstream")
            if source.position > target.position:
                raise ScientificInputError("targeted source violates causal ordering")

        torch = self.torch
        pair_count = len(pairs)
        expected_shape = (pair_count, int(target_encoder_vectors.shape[-1]))
        if tuple(target_encoder_vectors.shape) != expected_shape:
            raise ScientificInputError("target encoder vector shape is invalid")
        if tuple(source_decoder_vectors.shape) != expected_shape:
            raise ScientificInputError("source decoder vector shape is invalid")
        for label, tensor in (
            ("target encoder", target_encoder_vectors),
            ("source decoder", source_decoder_vectors),
        ):
            if tensor.device.type != "mps" or tensor.dtype != torch.bfloat16:
                raise ScientificInputError(f"{label} vectors must be MPS/BF16")
            if not bool(torch.isfinite(tensor).all().item()):
                raise ScientificInputError(f"{label} vectors must be finite")

        target_layers = torch.tensor(
            [target.layer for _, target in pairs], device="mps", dtype=torch.long
        )
        target_positions = torch.tensor(
            [target.position for _, target in pairs],
            device="mps",
            dtype=torch.long,
        )
        source_positions = torch.tensor(
            [source.position for source, _ in pairs],
            device="mps",
            dtype=torch.long,
        )
        batch_indices = torch.arange(pair_count, device="mps")
        result = torch.zeros(pair_count, device="mps", dtype=torch.bfloat16)

        def inject(
            gradient_point: Any,
            *,
            rows: Any,
            positions: Any,
            values: Any,
        ) -> None:
            gradients = gradient_point.grad.clone()
            gradients.index_put_((rows, positions), values, accumulate=False)
            gradient_point.grad = gradients

        unique_target_layers = sorted(
            {target.layer for _, target in pairs}, reverse=True
        )
        last_target_layer = max(unique_target_layers)
        with self._residuals[last_target_layer].backward(
            gradient=torch.zeros_like(self._residuals[last_target_layer]),
            retain_graph=retain_graph,
        ):
            for layer in reversed(range(last_target_layer + 1)):
                source_rows = torch.tensor(
                    [
                        index
                        for index, (source, _) in enumerate(pairs)
                        if source.layer == layer
                    ],
                    device="mps",
                    dtype=torch.long,
                )
                if int(source_rows.numel()) > 0:
                    gradients = self._feature_outputs[layer + 1].grad.clone()
                    selected = gradients[source_rows, source_positions[source_rows]]
                    values = (selected * source_decoder_vectors[source_rows]).sum(
                        dim=-1
                    )
                    result.index_put_(
                        (source_rows,),
                        values.to(torch.bfloat16),
                        accumulate=False,
                    )
                mask = target_layers == layer
                if bool(mask.any().item()):
                    inject(
                        self._residuals[layer],
                        rows=batch_indices[mask],
                        positions=target_positions[mask],
                        values=target_encoder_vectors[mask],
                    )
        if result.device.type != "mps" or result.dtype != torch.bfloat16:
            raise ScientificInputError("targeted VJP result moved off MPS/BF16")
        if not bool(torch.isfinite(result).all().item()):
            raise ScientificInputError("targeted VJP produced a non-finite value")
        return result


__all__ = ["TargetedVJPContext"]
