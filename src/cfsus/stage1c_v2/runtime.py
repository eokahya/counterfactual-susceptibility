"""Stage 1C-v2 baseline backend built on accepted Stage 1B primitives."""

from __future__ import annotations

import gc
from collections.abc import Iterable, Sequence
from typing import Any

from cfsus.backends.nnsight_plt import FEATURE_WIDTH, NNSightPLTMeasurementBackend
from cfsus.exceptions import ScientificInputError
from cfsus.stage1c_v2.vjp import TargetBatchVJPContext
from cfsus.types import FeatureActivity, FeatureRef, MeasuredFeatureState


class Stage1CVersion2PredictionBackend(NNSightPLTMeasurementBackend):
    """Add fresh active-source collection and many-source response tiles."""

    def collect_active_sources(
        self, *, groups: Iterable[tuple[int, int]], chunk_size: int
    ) -> tuple[MeasuredFeatureState, ...]:
        normalized = tuple(groups)
        if normalized != tuple(sorted(set(normalized))) or chunk_size < 1:
            raise ScientificInputError("active-source scan inputs are invalid")
        sources: list[MeasuredFeatureState] = []
        for layer in sorted({item[0] for item in normalized}):
            layer_input = self._capture_layer_input(layer)
            try:
                for _, position in (item for item in normalized if item[0] == layer):
                    for start in range(0, FEATURE_WIDTH, chunk_size):
                        end = min(FEATURE_WIDTH, start + chunk_size)
                        states = self._project_chunk(
                            layer_input=layer_input,
                            layer=layer,
                            position=position,
                            start=start,
                            end=end,
                        )
                        sources.extend(
                            item
                            for item in states
                            if item.activity is FeatureActivity.ACTIVE
                            and item.activation > 0.0
                        )
            finally:
                del layer_input
                gc.collect()
                self.torch.mps.empty_cache()
        result = tuple(sorted(sources, key=lambda item: item.feature))
        if not result or len({item.feature for item in result}) != len(result):
            raise ScientificInputError("active-source scan is empty or duplicated")
        return result

    def _source_decoder_vectors(self, sources: Sequence[MeasuredFeatureState]) -> Any:
        torch = self.torch
        transcoders = self._transcoders()
        vectors: list[Any] = []
        for layer in sorted({item.feature.layer for item in sources}):
            selected = [item for item in sources if item.feature.layer == layer]
            feature_ids = torch.tensor(
                [item.feature.feature_id for item in selected],
                device="mps",
                dtype=torch.long,
            )
            current = transcoders[layer]._get_decoder_vectors(feature_ids)
            if current.device.type != "mps" or current.dtype != torch.bfloat16:
                raise ScientificInputError("source decoder vectors are not MPS/BF16")
            vectors.append(current)
        result = torch.cat(vectors, dim=0)
        if result.shape[0] != len(sources) or not bool(
            torch.isfinite(result).all().item()
        ):
            raise ScientificInputError("source decoder vector pool is invalid")
        return result

    def response_tile(
        self,
        *,
        targets: Sequence[FeatureRef],
        sources: Sequence[MeasuredFeatureState],
        maximum_targets: int,
    ) -> tuple[tuple[float, ...], ...]:
        """Compute one bounded target-by-source tile and scalarize immediately."""

        if not targets or len(targets) > maximum_targets or not sources:
            raise ScientificInputError("response tile endpoint bounds are invalid")
        if tuple(targets) != tuple(sorted(set(targets))):
            raise ScientificInputError("response tile targets are not canonical")
        source_refs = tuple(item.feature for item in sources)
        if source_refs != tuple(sorted(set(source_refs))):
            raise ScientificInputError("response tile sources are not canonical")
        torch = self.torch
        transcoders = self._transcoders()
        target_vectors = torch.stack(
            [transcoders[item.layer].W_enc[item.feature_id] for item in targets]
        )
        source_vectors = self._source_decoder_vectors(sources)
        context = TargetBatchVJPContext(torch, maximum_targets=maximum_targets)
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
                source_refs,
                target_encoder_vectors=target_vectors,
                source_decoder_vectors=source_vectors,
            )
            values = result.detach().cpu().tolist()
            return tuple(tuple(float(value) for value in row) for row in values)
        finally:
            context.clear()
            del target_vectors, source_vectors
            if "result" in locals():
                del result
            gc.collect()
            torch.mps.empty_cache()


Stage1CPredictionBackend = Stage1CVersion2PredictionBackend

__all__ = ["Stage1CPredictionBackend", "Stage1CVersion2PredictionBackend"]
