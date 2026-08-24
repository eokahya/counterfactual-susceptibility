"""Narrow Stage 1B backend for pinned Gemma 3 PLTs on NNsight/MPS/BF16."""

from __future__ import annotations

import gc
from collections.abc import Iterable, Sequence
from importlib import metadata, util
from typing import Any

from cfsus.exceptions import (
    BackendUnavailableError,
    ScientificInputError,
    UnsupportedBackendOperationError,
)
from cfsus.responses.targeted import TargetedVJPContext
from cfsus.scanning.near_threshold import (
    GroupScanResult,
    ScannerResult,
    scan_feature_group,
)
from cfsus.types import (
    BackendCapabilityReport,
    BackendOperation,
    CapabilityEvidence,
    CapabilityStatus,
    FeatureActivity,
    FeatureRef,
    LocalResponseEstimate,
    MeasuredFeatureState,
    RuntimeIdentity,
)

MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
TRANSCODER_REVISION = "fada11860ac1d337c1e41e9da308798405b94c8e"
UPSTREAM_REVISION = "8f1e2438df612464e229e44c4a00ff637bf9379b"
FEATURE_WIDTH = 16_384
LAYER_COUNT = 18


def _dependency_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


class NNSightPLTMeasurementBackend:
    """Selected-state and scalar-VJP backend; never exposes a full feature cache."""

    def __init__(self, model: Any, *, prompt: str, prompt_id: str, torch: Any) -> None:
        if (
            util.find_spec("circuit_tracer") is None
            or util.find_spec("nnsight") is None
        ):
            raise BackendUnavailableError(
                "pinned circuit-tracer/NNsight is unavailable"
            )
        if not prompt.strip() or not prompt_id.strip():
            raise ScientificInputError("prompt and prompt_id must be non-empty")
        self.model = model
        self.prompt = prompt
        self.prompt_id = prompt_id
        self.torch = torch
        self._measurement_primitives_validated = False
        self.identity = RuntimeIdentity(
            backend="nnsight",
            device="mps:0",
            dtype="torch.bfloat16",
            model_revision=MODEL_REVISION,
            transcoder_revision=TRANSCODER_REVISION,
            upstream_revision=UPSTREAM_REVISION,
            prompt_id=prompt_id,
        )
        if getattr(model, "backend", None) != "nnsight":
            raise ScientificInputError("replacement model backend is not NNsight")
        if str(getattr(model, "device", "")) != "mps:0":
            raise ScientificInputError("replacement model is not on mps:0")
        if getattr(model, "dtype", None) != torch.bfloat16:
            raise ScientificInputError("replacement model is not BF16")
        transcoders = self._transcoders()
        if (
            len(transcoders) != LAYER_COUNT
            or int(transcoders.d_transcoder) != FEATURE_WIDTH
        ):
            raise ScientificInputError(
                "loaded PLT dimensions differ from the frozen runtime"
            )

    def _transcoders(self) -> Any:
        value = self.model.transcoders
        return getattr(value, "_module", value)

    def capability_report(self) -> BackendCapabilityReport:
        """Report only operations implemented for this exact runtime identity."""

        implemented = {
            BackendOperation.THRESHOLD_ACCESS,
            BackendOperation.ACTIVE_ACTIVATION_ACCESS,
            BackendOperation.INACTIVE_PREACTIVATION_ACCESS,
            BackendOperation.TARGETED_PREACTIVATION_ACCESS,
            BackendOperation.LOCAL_JACOBIAN,
            BackendOperation.VECTOR_JACOBIAN_PRODUCT,
            BackendOperation.SELECTED_STATE_CAPTURE,
        }
        evidence = tuple(
            CapabilityEvidence(
                operation=operation,
                status=(
                    (
                        CapabilityStatus.SUPPORTED
                        if self._measurement_primitives_validated
                        else CapabilityStatus.UNVERIFIED
                    )
                    if operation in implemented
                    else CapabilityStatus.UNSUPPORTED
                ),
                detail=(
                    "Scoped to pinned Gemma 3 270M + 18 PLTs + NNsight + "
                    "native MPS/BF16; full-cache, intervention and generalized "
                    "Jacobian claims remain unsupported."
                ),
            )
            for operation in BackendOperation
        )
        return BackendCapabilityReport(
            backend_name="cfsus/nnsight-plt-stage1b",
            dependency_available=True,
            dependency_version=(
                f"circuit-tracer={_dependency_version('circuit-tracer')};"
                f"nnsight={_dependency_version('nnsight')}"
            ),
            evidence=evidence,
        )

    def mark_measurement_primitives_validated(self) -> None:
        """Promote implemented operations only after the hard empirical gates pass."""

        self._measurement_primitives_validated = True

    def require_capability(self, operation: BackendOperation) -> None:
        if (
            self.capability_report().status_for(operation)
            is not CapabilityStatus.SUPPORTED
        ):
            raise UnsupportedBackendOperationError(
                f"{operation.value} is outside the Stage 1B measurement backend"
            )

    def _capture_layer_input(self, layer: int) -> Any:
        if layer < 0 or layer >= LAYER_COUNT:
            raise ScientificInputError("layer is outside the frozen PLT set")
        from nnsight import save  # type: ignore[import-not-found]

        tokens = self.model.ensure_tokenized(self.prompt)
        with self.torch.inference_mode(), self.model.trace(tokens):
            captured = save(self.model.get_feature_input_loc(layer).output)
        if captured.device.type != "mps" or captured.dtype != self.torch.bfloat16:
            raise ScientificInputError("captured PLT input is not MPS/BF16")
        if captured.ndim != 3 or captured.shape[0] != 1:
            raise ScientificInputError("captured PLT input shape is invalid")
        if not bool(self.torch.isfinite(captured).all().item()):
            raise ScientificInputError("captured PLT input is non-finite")
        return captured

    def _project_chunk(
        self,
        *,
        layer_input: Any,
        layer: int,
        position: int,
        start: int,
        end: int,
    ) -> tuple[MeasuredFeatureState, ...]:
        torch = self.torch
        if start < 0 or end <= start or end > FEATURE_WIDTH:
            raise ScientificInputError("feature chunk bounds are invalid")
        if position <= 0 or position >= int(layer_input.shape[1]):
            raise ScientificInputError("scanner position is outside non-BOS tokens")
        transcoder = self._transcoders()[layer]
        encoder = transcoder.W_enc[start:end]
        bias = transcoder.b_enc[start:end]
        threshold = transcoder.activation_function.threshold[start:end]
        for label, tensor in (
            ("encoder", encoder),
            ("encoder bias", bias),
            ("threshold", threshold),
        ):
            if tensor.device.type != "mps" or tensor.dtype != torch.bfloat16:
                raise ScientificInputError(f"{label} chunk is not MPS/BF16")
        preactivation = torch.nn.functional.linear(
            layer_input[0, position], encoder, bias
        )
        from circuit_tracer.transcoder.activation_functions import (  # type: ignore[import-not-found]
            jumprelu,
        )

        activation = jumprelu.apply(
            preactivation,
            threshold,
            transcoder.activation_function.bandwidth,
        )
        loaded_reference = preactivation * (preactivation > threshold)
        if not bool(torch.equal(activation, loaded_reference)):
            raise ScientificInputError("loaded JumpReLU differs from strict reference")
        for label, tensor in (
            ("preactivation", preactivation),
            ("activation", activation),
            ("threshold", threshold),
        ):
            if tensor.device.type != "mps" or tensor.dtype != torch.bfloat16:
                raise ScientificInputError(f"{label} moved off MPS/BF16")
            if not bool(torch.isfinite(tensor).all().item()):
                raise ScientificInputError(f"{label} contains a non-finite value")
        active_mask = preactivation > threshold
        z_values = preactivation.detach().cpu().tolist()
        a_values = activation.detach().cpu().tolist()
        threshold_values = threshold.detach().cpu().tolist()
        activity_values = active_mask.detach().cpu().tolist()
        return tuple(
            MeasuredFeatureState(
                feature=FeatureRef(layer, position, start + offset),
                preactivation=float(z_value),
                activation=float(a_value),
                threshold=float(tau_value),
                activity=(
                    FeatureActivity.ACTIVE
                    if bool(is_active)
                    else FeatureActivity.INACTIVE
                ),
                device="mps:0",
                dtype="torch.bfloat16",
            )
            for offset, (z_value, a_value, tau_value, is_active) in enumerate(
                zip(
                    z_values,
                    a_values,
                    threshold_values,
                    activity_values,
                    strict=True,
                )
            )
        )

    def scan(
        self,
        *,
        groups: Iterable[tuple[int, int]],
        chunk_size: int,
        top_k_per_group: int,
        global_top_k: int,
    ) -> ScannerResult:
        """Scan selected groups with one retained hidden-state layer at a time."""

        normalized = tuple(groups)
        if not normalized or len(normalized) != len(set(normalized)):
            raise ScientificInputError("scanner groups must be non-empty and unique")
        if normalized != tuple(sorted(normalized)):
            raise ScientificInputError("scanner groups must use canonical sorted order")
        group_results: list[GroupScanResult] = []
        global_candidates: list[Any] = []
        for layer in sorted({item[0] for item in normalized}):
            layer_input = self._capture_layer_input(layer)
            try:
                for _, position in (item for item in normalized if item[0] == layer):

                    def project_chunk(
                        start: int,
                        end: int,
                        *,
                        captured_input: Any = layer_input,
                        selected_layer: int = layer,
                        selected_position: int = position,
                    ) -> tuple[MeasuredFeatureState, ...]:
                        return self._project_chunk(
                            layer_input=captured_input,
                            layer=selected_layer,
                            position=selected_position,
                            start=start,
                            end=end,
                        )

                    result = scan_feature_group(
                        layer=layer,
                        position=position,
                        feature_count=FEATURE_WIDTH,
                        chunk_size=chunk_size,
                        top_k=top_k_per_group,
                        project_chunk=project_chunk,
                    )
                    group_results.append(result)
                    global_candidates.extend(result.candidates)
                    global_candidates.sort(key=lambda item: item.sort_key)
                    del global_candidates[global_top_k:]
            finally:
                del layer_input
                gc.collect()
                self.torch.mps.empty_cache()
        return ScannerResult(
            chunk_size=chunk_size,
            top_k_per_group=top_k_per_group,
            global_top_k=global_top_k,
            groups=tuple(group_results),
            global_candidates=tuple(global_candidates),
        )

    def measure_states(
        self, features: Sequence[FeatureRef]
    ) -> dict[FeatureRef, MeasuredFeatureState]:
        """Measure unique selected endpoints without creating a full feature cache."""

        unique = tuple(sorted(set(features)))
        if not unique:
            raise ScientificInputError("selected feature set must not be empty")
        if len(unique) != len(features):
            raise ScientificInputError("selected feature set contains duplicates")
        states: dict[FeatureRef, MeasuredFeatureState] = {}
        for layer in sorted({item.layer for item in unique}):
            layer_input = self._capture_layer_input(layer)
            try:
                for feature in (item for item in unique if item.layer == layer):
                    state = self._project_chunk(
                        layer_input=layer_input,
                        layer=layer,
                        position=feature.position,
                        start=feature.feature_id,
                        end=feature.feature_id + 1,
                    )[0]
                    states[feature] = state
            finally:
                del layer_input
                gc.collect()
                self.torch.mps.empty_cache()
        return states

    def targeted_local_responses(
        self,
        pairs: Sequence[tuple[FeatureRef, FeatureRef]],
        *,
        maximum_pairs: int,
    ) -> tuple[LocalResponseEstimate, ...]:
        """Compute independent targeted VJPs; no graph object is accepted."""

        if not pairs or len(pairs) > maximum_pairs:
            raise ScientificInputError("targeted pair set is empty or too large")
        endpoints = tuple(feature for pair in pairs for feature in pair)
        states = self.measure_states(tuple(dict.fromkeys(endpoints)))
        for source, _ in pairs:
            if states[source].activity is not FeatureActivity.ACTIVE:
                raise ScientificInputError("targeted source is inactive")
            if states[source].activation <= 0.0:
                raise ScientificInputError("targeted source activation is not positive")

        torch = self.torch
        transcoders = self._transcoders()
        target_vectors = torch.stack(
            [transcoders[target.layer].W_enc[target.feature_id] for _, target in pairs]
        )
        source_vectors = torch.stack(
            [
                transcoders[source.layer]._get_decoder_vectors(
                    torch.tensor([source.feature_id], device="mps", dtype=torch.long)
                )[0]
                for source, _ in pairs
            ]
        )
        context = TargetedVJPContext(torch, maximum_pairs=maximum_pairs)
        input_ids = self.model.ensure_tokenized(self.prompt)
        with self.model.trace() as tracer:
            with tracer.invoke(input_ids.expand(len(pairs), -1)):
                pass
            barrier = tracer.barrier(2)
            self.model.configure_gradient_flow(tracer)
            self.model.configure_skip_connection(tracer, barrier=barrier)
            context.cache(self.model, tracer, barrier=barrier)
        responses = context.compute(
            pairs,
            target_encoder_vectors=target_vectors,
            source_decoder_vectors=source_vectors,
            retain_graph=False,
        )
        response_values = responses.detach().cpu().tolist()
        return tuple(
            LocalResponseEstimate(
                source=source,
                target=target,
                source_activation=states[source].activation,
                target_preactivation=states[target].preactivation,
                response=float(value),
                device="mps:0",
                dtype="torch.bfloat16",
                method="target_encoder_reverse_vjp_source_decoder_contraction",
                convention="attribution_matched_target_preactivation_pre_gate",
                graph_edge_used=False,
            )
            for (source, target), value in zip(pairs, response_values, strict=True)
        )


__all__ = ["NNSightPLTMeasurementBackend"]
