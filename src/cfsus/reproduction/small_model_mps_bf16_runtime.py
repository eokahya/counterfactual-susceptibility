"""Source-faithful native-MPS/BF16 adapters for pinned circuit-tracer 0.5.2.

The pinned upstream uses ``Tensor.to_sparse()`` in PLT attribution setup and
keeps attribution index tensors on the sparse tensor's device. PyTorch 2.6.0
does not implement dense-to-sparse conversion on MPS. This module subclasses
the pinned public runtime classes instead of mutating them. Scientific dense
values, encoder/decoder vectors, residuals, gradients, and interventions stay
on MPS/BF16; only COO and graph-ranking metadata live on CPU.
"""

# mypy: warn_unused_ignores=False

from __future__ import annotations

from typing import Any

import torch  # type: ignore[import-not-found]
from circuit_tracer.attribution.context_nnsight import (  # type: ignore[import-not-found,import-untyped]
    AttributionContext,
)
from circuit_tracer.attribution.targets import (  # type: ignore[import-not-found,import-untyped]
    AttributionTargets,
)
from circuit_tracer.graph import (  # type: ignore[import-not-found,import-untyped]
    Graph,
    compute_partial_influences,
)
from circuit_tracer.replacement_model.replacement_model_nnsight import (  # type: ignore[import-not-found,import-untyped]
    NNSightReplacementModel,
)
from circuit_tracer.transcoder.single_layer_transcoder import (  # type: ignore[import-not-found,import-untyped]
    TranscoderSet,
)
from nnsight import Envoy, save  # type: ignore[import-not-found,import-untyped]

from cfsus.reproduction.artifacts import ArtifactValidationError
from cfsus.reproduction.small_model_mps_bf16 import (
    dense_to_cpu_sparse_metadata_bf16,
)


def _assert_mps_bf16(value: torch.Tensor, label: str) -> None:
    if value.device.type != "mps" or value.dtype != torch.bfloat16:
        raise ArtifactValidationError(
            f"{label} must be native MPS/BF16, got {value.device}/{value.dtype}"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise ArtifactValidationError(f"{label} is non-finite")


class MPSBF16TranscoderSet(TranscoderSet):  # type: ignore[misc]
    """TranscoderSet with an explicit CPU-COO metadata boundary."""

    adapter_component_calls: int

    def __init__(self, source: TranscoderSet) -> None:
        super().__init__(
            {index: transcoder for index, transcoder in enumerate(source)},
            feature_input_hook=source.feature_input_hook,
            feature_output_hook=source.feature_output_hook,
            scan_name=source.scan_name,
        )
        self.adapter_component_calls = 0

    def compute_attribution_components(
        self,
        mlp_inputs: torch.Tensor,
        zero_positions: slice = slice(0, 1),
    ) -> dict[str, torch.Tensor]:
        """Mirror upstream PLT math without unsupported MPS ``to_sparse``."""

        self.adapter_component_calls += 1
        _assert_mps_bf16(mlp_inputs, "attribution MLP inputs")
        if mlp_inputs.ndim != 3 or len(self) != mlp_inputs.shape[0]:
            raise ArtifactValidationError("attribution MLP input shape is invalid")
        n_layers, n_positions, _ = mlp_inputs.shape
        reconstruction = torch.zeros_like(mlp_inputs)
        encoder_vectors: list[torch.Tensor] = []
        decoder_vectors: list[torch.Tensor] = []
        cpu_indices: list[torch.Tensor] = []
        cpu_values: list[torch.Tensor] = []
        device_locations: list[torch.Tensor] = []

        for layer, transcoder in enumerate(self):
            encoder = transcoder.W_enc
            _assert_mps_bf16(encoder, f"layer {layer} lazy encoder")
            preactivation = torch.nn.functional.linear(
                mlp_inputs[layer], encoder, transcoder.b_enc
            )
            activations = transcoder.activation_function(preactivation)
            activations[zero_positions] = 0
            _assert_mps_bf16(activations, f"layer {layer} activations")
            metadata, indices, values = dense_to_cpu_sparse_metadata_bf16(
                activations, torch
            )
            layer_cpu = torch.full((1, metadata._nnz()), layer, dtype=torch.long)
            cpu_indices.append(torch.cat((layer_cpu, metadata.indices()), dim=0))
            cpu_values.append(metadata.values())

            if values.numel():
                positions = indices[0]
                feature_ids = indices[1]
                active_encoders = encoder[feature_ids]
                active_decoders = transcoder._get_decoder_vectors(
                    feature_ids.detach().cpu()
                )
                _assert_mps_bf16(active_decoders, f"layer {layer} lazy decoder vectors")
                scaled_decoders = active_decoders * values[:, None]
                reconstruction[layer].index_add_(0, positions, scaled_decoders)
                device_locations.append(
                    torch.stack((torch.full_like(positions, layer), positions))
                )
            else:
                active_encoders = torch.empty(
                    (0, transcoder.d_model),
                    device="mps",
                    dtype=torch.bfloat16,
                )
                scaled_decoders = torch.empty_like(active_encoders)
                device_locations.append(
                    torch.empty((2, 0), device="mps", dtype=torch.long)
                )
            if transcoder.W_skip is not None:
                reconstruction[layer] += transcoder.compute_skip(mlp_inputs[layer])
            reconstruction[layer] += transcoder.b_dec
            encoder_vectors.append(active_encoders)
            decoder_vectors.append(scaled_decoders)

        activation_matrix = torch.sparse_coo_tensor(
            torch.cat(cpu_indices, dim=1),
            torch.cat(cpu_values),
            size=(n_layers, n_positions, int(self.d_transcoder)),
            device="cpu",
            dtype=torch.bfloat16,
        ).coalesce()
        locations = torch.cat(device_locations, dim=1)
        encoder_tensor = torch.cat(encoder_vectors)
        decoder_tensor = torch.cat(decoder_vectors)
        active_count = int(activation_matrix._nnz())
        _assert_mps_bf16(reconstruction, "PLT reconstruction")
        _assert_mps_bf16(encoder_tensor, "active encoder vectors")
        _assert_mps_bf16(decoder_tensor, "scaled decoder vectors")
        if active_count < 1 or encoder_tensor.shape != decoder_tensor.shape:
            raise ArtifactValidationError("attribution sparse metadata is inconsistent")
        if locations.shape != (2, active_count):
            raise ArtifactValidationError("decoder locations are inconsistent")
        return {
            "activation_matrix": activation_matrix,
            "reconstruction": reconstruction,
            "encoder_vecs": encoder_tensor,
            "decoder_vecs": decoder_tensor,
            "encoder_to_decoder_map": torch.arange(active_count, device="mps"),
            "decoder_locations": locations,
        }


class MPSBF16AttributionContext(AttributionContext):  # type: ignore[misc]
    """Pinned AttributionContext with scientific indices placed on MPS."""

    adapter_batch_calls: int = 0

    def compute_batch(
        self,
        layers: torch.Tensor,
        positions: torch.Tensor,
        inject_values: torch.Tensor,
        retain_graph: bool = True,
    ) -> torch.Tensor:
        self.adapter_batch_calls += 1
        _assert_mps_bf16(inject_values, "attribution injection")
        layers_mps = layers.to(device="mps", dtype=torch.long)
        positions_mps = positions.to(device="mps", dtype=torch.long)
        batch_size = self._resid_activations[0].shape[0]
        self._batch_buffer = torch.zeros(
            self._row_size,
            batch_size,
            dtype=torch.bfloat16,
            device="mps",
        )
        batch_indices = torch.arange(len(layers_mps), device="mps")

        def inject(
            grad_point: Any,
            *,
            rows: torch.Tensor,
            columns: torch.Tensor,
            values: torch.Tensor,
        ) -> None:
            grads_out = grad_point.grad.clone()
            _assert_mps_bf16(grads_out, "attribution residual gradient")
            grads_out.index_put_((rows, columns), values, accumulate=False)
            grad_point.grad = grads_out

        unique_layers = sorted(
            int(item)
            for item in layers_mps.unique().detach().cpu().tolist()  # type: ignore[no-untyped-call]
        )
        if not unique_layers:
            raise ArtifactValidationError("attribution batch must not be empty")
        last_layer = max(unique_layers)
        with self._resid_activations[last_layer].backward(
            gradient=torch.zeros_like(self._resid_activations[last_layer]),
            retain_graph=retain_graph,
        ):
            for layer in reversed(range(last_layer + 1)):
                if layer != last_layer:
                    gradient = self._feature_output_activations[layer + 1].grad.clone()
                    _assert_mps_bf16(gradient, "feature-output gradient")
                    self.compute_feature_attributions(layer, gradient)
                    self.compute_error_attributions(layer, gradient)
                mask = layers_mps == layer
                if bool(mask.any().item()):
                    inject(
                        self._resid_activations[layer],
                        rows=batch_indices[mask],
                        columns=positions_mps[mask],
                        values=inject_values[mask],
                    )
            token_gradient = self._feature_output_activations[0].grad
            _assert_mps_bf16(token_gradient, "token attribution gradient")
            self.compute_token_attributions(token_gradient)
        buffer = self._batch_buffer
        self._batch_buffer = None  # type: ignore[assignment]
        if buffer is None:
            raise ArtifactValidationError("attribution batch buffer is missing")
        result = buffer.T[: len(layers_mps)]
        _assert_mps_bf16(result, "attribution batch result")
        return result


class MPSBF16ReplacementModel(NNSightReplacementModel):  # type: ignore[misc]
    """Pinned replacement model returning the explicit BF16 context subclass."""

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def setup_attribution(
        self, inputs: str | torch.Tensor
    ) -> MPSBF16AttributionContext:
        tokens = (
            self.ensure_tokenized(inputs)
            if isinstance(inputs, str)
            else inputs.squeeze()
        )
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 1:
            raise ArtifactValidationError("attribution tokens must be a 1D tensor")
        if tokens.device.type != "mps" or tokens.dtype != torch.long:
            raise ArtifactValidationError(
                "attribution token IDs must be integer MPS data"
            )

        with self.trace(tokens):
            mlp_inputs: list[Any] = []
            mlp_outputs: list[Any] = []
            for feature_input, feature_output in zip(
                self.feature_input_locs, self.feature_output_locs, strict=True
            ):
                mlp_inputs.append(feature_input.output)
                output = feature_output.output
                if output.ndim == 2:
                    output = output.unsqueeze(0)
                mlp_outputs.append(output)
            input_cache = save(torch.cat(mlp_inputs, dim=0))
            output_cache = save(torch.cat(mlp_outputs, dim=0))
            logits = save(self.output.logits)

        _assert_mps_bf16(input_cache, "attribution input cache")
        _assert_mps_bf16(output_cache, "attribution output cache")
        _assert_mps_bf16(logits, "attribution logits")
        transcoders = (
            self.transcoders._module
            if isinstance(self.transcoders, Envoy)
            else self.transcoders
        )
        if not isinstance(transcoders, MPSBF16TranscoderSet):
            raise ArtifactValidationError(
                "replacement runtime lost the BF16 PLT adapter"
            )
        attribution_data = transcoders.compute_attribution_components(
            input_cache, self.zero_positions
        )
        error_vectors = output_cache - attribution_data["reconstruction"]
        error_vectors[:, self.zero_positions] = 0
        token_vectors = self.embed_weight[tokens].detach()
        _assert_mps_bf16(error_vectors, "attribution error vectors")
        _assert_mps_bf16(token_vectors, "token vectors")
        return MPSBF16AttributionContext(
            activation_matrix=attribution_data["activation_matrix"],
            logits=logits,
            error_vectors=error_vectors,
            token_vectors=token_vectors,
            decoder_vecs=attribution_data["decoder_vecs"],
            encoder_vecs=attribution_data["encoder_vecs"],
            encoder_to_decoder_map=attribution_data["encoder_to_decoder_map"],
            decoder_locations=attribution_data["decoder_locations"],
        )


def attribute_mps_bf16(
    prompt: str,
    model: MPSBF16ReplacementModel,
    *,
    context: MPSBF16AttributionContext,
    max_n_logits: int,
    desired_logit_probability: float,
    batch_size: int,
    max_feature_nodes: int,
) -> tuple[Graph, dict[str, int]]:
    """Run pinned NNsight attribution with explicit CPU graph metadata only."""

    input_ids = model.ensure_tokenized(prompt)
    activation_matrix = context.activation_matrix
    if (
        activation_matrix.device.type != "cpu"
        or activation_matrix.dtype != torch.bfloat16
    ):
        raise ArtifactValidationError("activation COO metadata identity is invalid")
    if batch_size < 1 or max_feature_nodes < 1:
        raise ArtifactValidationError("attribution bounds must be positive")

    with model.trace() as tracer:
        with tracer.invoke(input_ids.expand(batch_size, -1)):
            pass
        detach_barrier = tracer.barrier(2)
        model.configure_gradient_flow(tracer)
        model.configure_skip_connection(tracer, barrier=detach_barrier)
        context.cache_residual(model, tracer, barrier=detach_barrier)

    feature_layers, feature_positions, _ = activation_matrix.indices()
    n_layers, n_positions, _ = activation_matrix.shape
    total_active = int(activation_matrix._nnz())
    targets = AttributionTargets(
        attribution_targets=None,
        logits=context.logits[0, -1],
        unembed_proj=model.unembed_weight,
        tokenizer=model.tokenizer,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_probability,
    )
    _assert_mps_bf16(targets.logit_vectors, "logit target vectors")
    logit_offset = total_active + (n_layers + 1) * n_positions
    n_logits = len(targets)
    total_nodes = logit_offset + n_logits
    selected_limit = min(max_feature_nodes, total_active)
    edge_matrix = torch.zeros(selected_limit + n_logits, total_nodes)
    row_to_node = torch.zeros(selected_limit + n_logits, dtype=torch.int32)

    start = 0
    for start in range(0, n_logits, batch_size):
        batch = targets.logit_vectors[start : start + batch_size]
        rows = context.compute_batch(
            layers=torch.full((batch.shape[0],), n_layers),
            positions=torch.full((batch.shape[0],), n_positions - 1),
            inject_values=batch,
        )
        edge_matrix[start : start + batch.shape[0], :logit_offset] = rows.cpu()
        row_to_node[start : start + batch.shape[0]] = (
            torch.arange(start, start + batch.shape[0]) + logit_offset
        )

    write_start = n_logits
    visited = torch.zeros(total_active, dtype=torch.bool)
    visited_count = 0
    partial_calls = 0
    while visited_count < selected_limit:
        if selected_limit == total_active:
            pending = torch.arange(total_active)
        else:
            influences = compute_partial_influences(
                edge_matrix[:write_start],
                targets.logit_probabilities.detach().cpu(),
                row_to_node[:write_start],
                device="cpu",
            )
            partial_calls += 1
            ranking = torch.argsort(
                influences[:total_active], descending=True, stable=True
            ).cpu()
            queue_size = min(4 * batch_size, selected_limit - visited_count)
            pending = ranking[~visited[ranking]][:queue_size]
        for index_batch in (
            pending[index : index + batch_size]
            for index in range(0, len(pending), batch_size)
        ):
            visited_count += len(index_batch)
            device_indices = index_batch.to(device="mps", dtype=torch.long)
            rows = context.compute_batch(
                layers=feature_layers[index_batch],
                positions=feature_positions[index_batch],
                inject_values=context.encoder_vecs[device_indices],
                retain_graph=visited_count < selected_limit,
            )
            end = write_start + rows.shape[0]
            edge_matrix[write_start:end, :logit_offset] = rows.cpu()
            row_to_node[write_start:end] = index_batch
            visited[index_batch] = True
            write_start = end

    selected_features = torch.where(visited)[0]
    if selected_limit < total_active:
        non_feature_nodes = torch.arange(total_active, total_nodes)
        columns = torch.cat((selected_features, non_feature_nodes))
        edge_matrix = edge_matrix[:, columns]
    edge_matrix = edge_matrix[row_to_node.argsort(stable=True)]
    final_node_count = edge_matrix.shape[1]
    full_edge_matrix = torch.zeros(final_node_count, final_node_count)
    full_edge_matrix[:selected_limit] = edge_matrix[:selected_limit]
    full_edge_matrix[-n_logits:] = edge_matrix[selected_limit:]
    if not bool(torch.isfinite(full_edge_matrix).all().item()):
        raise ArtifactValidationError("attribution adjacency is non-finite")

    graph = Graph(
        input_string=model.tokenizer.decode(input_ids),
        input_tokens=input_ids,
        logit_targets=targets.logit_targets,
        logit_probabilities=targets.logit_probabilities,
        vocab_size=targets.vocab_size,
        active_features=activation_matrix.indices().T,
        activation_values=activation_matrix.values(),
        selected_features=selected_features,
        adjacency_matrix=full_edge_matrix.detach(),
        cfg=model.config,
        scan_name=model.scan_name,
    )
    transcoders = (
        model.transcoders._module
        if isinstance(model.transcoders, Envoy)
        else model.transcoders
    )
    return graph, {
        "component_calls": int(transcoders.adapter_component_calls),
        "batch_calls": int(context.adapter_batch_calls),
        "partial_ranking_calls": partial_calls,
        "runtime_monkeypatches": 0,
    }


__all__ = [
    "MPSBF16AttributionContext",
    "MPSBF16ReplacementModel",
    "MPSBF16TranscoderSet",
    "attribute_mps_bf16",
]
