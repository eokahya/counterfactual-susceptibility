"""Bounded target-batch reverse contractions for baseline-only prediction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from cfsus.exceptions import ScientificInputError
from cfsus.types import FeatureRef


class TargetBatchVJPContext:
    """Evaluate a bounded target batch against a deterministic source pool."""

    def __init__(self, torch: Any, *, maximum_targets: int) -> None:
        if isinstance(maximum_targets, bool) or maximum_targets < 1:
            raise ScientificInputError("maximum target batch must be positive")
        self.torch = torch
        self.maximum_targets = maximum_targets
        self._residuals: list[Any] = []
        self._feature_outputs: list[Any] = []

    def cache(self, model: Any, tracer: Any, barrier: Any = None) -> None:
        """Cache the same frozen residual and feature-output points as Stage 1B."""

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
        targets: Sequence[FeatureRef],
        sources: Sequence[FeatureRef],
        *,
        target_encoder_vectors: Any,
        source_decoder_vectors: Any,
    ) -> Any:
        """Return one transient target-by-source response tile on MPS/BF16."""

        if not targets or len(targets) > self.maximum_targets or not sources:
            raise ScientificInputError("target/source response tile bounds are invalid")
        if len(self._residuals) < 2 or len(self._feature_outputs) < 2:
            raise ScientificInputError("target response context is not cached")
        if tuple(targets) != tuple(sorted(set(targets))):
            raise ScientificInputError("targets must be unique canonical order")
        if tuple(sources) != tuple(sorted(set(sources))):
            raise ScientificInputError("sources must be unique canonical order")
        if any(
            not any(
                source.layer < target.layer and source.position <= target.position
                for source in sources
            )
            for target in targets
        ):
            raise ScientificInputError("a target has no causal source")

        torch = self.torch
        hidden = int(target_encoder_vectors.shape[-1])
        if tuple(target_encoder_vectors.shape) != (len(targets), hidden):
            raise ScientificInputError("target encoder tile shape is invalid")
        if tuple(source_decoder_vectors.shape) != (len(sources), hidden):
            raise ScientificInputError("source decoder tile shape is invalid")
        for label, tensor in (
            ("target encoder", target_encoder_vectors),
            ("source decoder", source_decoder_vectors),
        ):
            if tensor.device.type != "mps" or tensor.dtype != torch.bfloat16:
                raise ScientificInputError(f"{label} tile must be MPS/BF16")
            if not bool(torch.isfinite(tensor).all().item()):
                raise ScientificInputError(f"{label} tile is non-finite")

        target_layers = torch.tensor(
            [item.layer for item in targets], device="mps", dtype=torch.long
        )
        target_positions = torch.tensor(
            [item.position for item in targets], device="mps", dtype=torch.long
        )
        source_layers = torch.tensor(
            [item.layer for item in sources], device="mps", dtype=torch.long
        )
        source_positions = torch.tensor(
            [item.position for item in sources], device="mps", dtype=torch.long
        )
        target_rows = torch.arange(len(targets), device="mps")
        result = torch.zeros(
            (len(targets), len(sources)), device="mps", dtype=torch.bfloat16
        )

        def inject(rows: Any, positions: Any, values: Any, point: Any) -> None:
            gradients = point.grad.clone()
            gradients.index_put_((rows, positions), values, accumulate=False)
            point.grad = gradients

        last_target_layer = max(item.layer for item in targets)
        with self._residuals[last_target_layer].backward(
            gradient=torch.zeros_like(self._residuals[last_target_layer]),
            retain_graph=False,
        ):
            for layer in reversed(range(last_target_layer + 1)):
                source_columns = torch.nonzero(
                    source_layers == layer, as_tuple=False
                ).flatten()
                if int(source_columns.numel()) > 0 and layer < last_target_layer:
                    gradients = self._feature_outputs[layer + 1].grad.clone()
                    selected = gradients[:, source_positions[source_columns]]
                    values = (
                        selected * source_decoder_vectors[source_columns].unsqueeze(0)
                    ).sum(dim=-1)
                    rows = (
                        target_rows[:, None].expand(-1, len(source_columns)).reshape(-1)
                    )
                    columns = (
                        source_columns[None, :].expand(len(targets), -1).reshape(-1)
                    )
                    result.index_put_(
                        (rows, columns), values.reshape(-1), accumulate=False
                    )
                target_mask = target_layers == layer
                if bool(target_mask.any().item()):
                    inject(
                        target_rows[target_mask],
                        target_positions[target_mask],
                        target_encoder_vectors[target_mask],
                        self._residuals[layer],
                    )
        if result.device.type != "mps" or result.dtype != torch.bfloat16:
            raise ScientificInputError("target response tile moved off MPS/BF16")
        if not bool(torch.isfinite(result).all().item()):
            raise ScientificInputError("target response tile is non-finite")
        return result

    def clear(self) -> None:
        """Release NNsight proxy references before the next bounded batch."""

        self._residuals.clear()
        self._feature_outputs.clear()


__all__ = ["TargetBatchVJPContext"]
