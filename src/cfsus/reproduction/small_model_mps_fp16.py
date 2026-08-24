"""Fail-closed helpers for the isolated Stage 1A-S MPS/FP16 pilot.

This module is deliberately independent of the Gemma 2 MPS and T4 runtime
modules. Optional empirical dependencies are imported only inside runtime
functions so the ordinary offline test suite remains lightweight.
"""

from __future__ import annotations

import contextlib
import importlib
import math
import os
import re
import types
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from cfsus.exceptions import ScientificInputError
from cfsus.reproduction.artifacts import ArtifactValidationError, sha256_file
from cfsus.reproduction.runtime_helpers import desired_activation

EXPERIMENT_CLASS = "stage1a_small_model_mps_fp16_pilot"
COMPLETED_STATUS = "completed_small_model_mps_fp16_pilot"
UPSTREAM_REPOSITORY = "https://github.com/decoderesearch/circuit-tracer.git"
UPSTREAM_VERSION = "0.5.2"
UPSTREAM_REVISION = "8f1e2438df612464e229e44c4a00ff637bf9379b"
MODEL_IDENTIFIER = "google/gemma-3-270m"
MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
TRANSCODER_IDENTIFIER = "mwhanna/gemma-scope-2-270m-pt"
TRANSCODER_REVISION = "fada11860ac1d337c1e41e9da308798405b94c8e"
TRANSCODER_SUBFOLDER = "transcoder_all/width_16k_l0_small"
BACKEND = "nnsight"
DEVICE = "mps"
DTYPE = "float16"
LAYER_COUNT = 18
FEATURE_WIDTH = 16_384
HIDDEN_SIZE = 640
FALLBACK_VARIABLE = "PYTORCH_ENABLE_MPS_FALLBACK"

RESULT_DIRECTORY = "results/stage1a_small_model_mps_fp16"
GENERATED_DIRECTORY = "results/generated/stage1a_small_model_mps_fp16"
ENVIRONMENT_LOCK = "environments/stage1a_small_model_mps/requirements-lock.txt"
PROJECTED_MANIFEST = "configs/stage1a_small_model_projected_download.json"

ARTIFACT_ALLOWLIST = frozenset(
    {
        "environment_manifest.json",
        "asset_manifest.json",
        "preflight_summary.json",
        "operator_probe_summary.json",
        "model_forward_summary.json",
        "loaded_semantics_summary.json",
        "attribution_summary.json",
        "intervention_summary.json",
        "memory_timing_summary.json",
        "attempts.json",
        "run_manifest.json",
        "checksums.sha256",
    }
)
ARTIFACT_JSON_ALLOWLIST = ARTIFACT_ALLOWLIST - {"checksums.sha256"}
FORBIDDEN_SUFFIXES = frozenset(
    {".pt", ".pth", ".safetensors", ".bin", ".npy", ".npz", ".parquet"}
)
HEX40_RE = re.compile(r"[0-9a-f]{40}\Z")


@dataclass(frozen=True, slots=True)
class FeatureSelection:
    """One deterministically selected active PLT feature."""

    layer: int
    position: int
    feature: int
    baseline_activation: float
    rule: str
    score: float


@dataclass(frozen=True, slots=True)
class MemoryFeasibility:
    """Conservative pre-download memory projection."""

    model_fp16_bytes: int
    persistent_transcoder_fp16_bytes: int
    one_lazy_matrix_fp16_bytes: int
    full_eager_transcoder_fp16_bytes: int
    fixed_runtime_reserve_bytes: int
    graph_buffer_limit_bytes: int
    total_conservative_bytes: int
    maximum_process_rss_bytes: int
    feasible: bool


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactValidationError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ArtifactValidationError(f"{label} contains a non-string key")
    return cast(Mapping[str, Any], value)


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ArtifactValidationError(f"{label} must be a list")
    return cast(Sequence[Any], value)


def _require_equal(
    mapping: Mapping[str, Any], key: str, expected: Any, label: str
) -> None:
    observed = mapping.get(key)
    if observed != expected:
        raise ArtifactValidationError(
            f"{label}.{key} must be {expected!r}, got {observed!r}"
        )


def load_small_model_config(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the separate Stage 1A-S YAML mapping."""

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - optional runtime boundary
        raise ArtifactValidationError(
            "PyYAML is required to load Stage 1A-S config"
        ) from error
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ArtifactValidationError("Stage 1A-S config must be a regular file")
    with candidate.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    return validate_small_model_config(value)


def validate_small_model_config(value: Any) -> dict[str, Any]:
    """Reject mutable identities and cross-experiment contamination."""

    root = _mapping(value, "config")
    _require_equal(root, "experiment_name", EXPERIMENT_CLASS, "config")
    _require_equal(root, "experiment_class", EXPERIMENT_CLASS, "config")
    _require_equal(root, "completed_status", COMPLETED_STATUS, "config")
    _require_equal(
        root, "claim_class", "local_small_model_runtime_validation", "config"
    )

    upstream = _mapping(root.get("upstream"), "upstream")
    _require_equal(upstream, "repository", UPSTREAM_REPOSITORY, "upstream")
    _require_equal(upstream, "version", UPSTREAM_VERSION, "upstream")
    _require_equal(upstream, "revision", UPSTREAM_REVISION, "upstream")

    model = _mapping(root.get("model"), "model")
    _require_equal(model, "identifier", MODEL_IDENTIFIER, "model")
    _require_equal(model, "revision", MODEL_REVISION, "model")
    _require_equal(model, "architecture", "Gemma3ForCausalLM", "model")
    _require_equal(model, "pretrained_variant", "base", "model")

    transcoder = _mapping(root.get("transcoder"), "transcoder")
    _require_equal(transcoder, "identifier", TRANSCODER_IDENTIFIER, "transcoder")
    _require_equal(transcoder, "revision", TRANSCODER_REVISION, "transcoder")
    _require_equal(transcoder, "subfolder", TRANSCODER_SUBFOLDER, "transcoder")
    _require_equal(transcoder, "model_kind", "transcoder_set", "transcoder")
    _require_equal(transcoder, "feature_input_hook", "mlp.hook_in", "transcoder")
    _require_equal(transcoder, "feature_output_hook", "hook_mlp_out", "transcoder")
    _require_equal(transcoder, "layer_count", LAYER_COUNT, "transcoder")
    _require_equal(transcoder, "feature_width", FEATURE_WIDTH, "transcoder")

    runtime = _mapping(root.get("runtime"), "runtime")
    for runtime_key, runtime_expected in (
        ("backend", BACKEND),
        ("device", DEVICE),
        ("dtype", DTYPE),
        ("python", "3.11"),
        ("platform", "macos-arm64"),
        ("fallback_environment_variable", FALLBACK_VARIABLE),
        ("fallback_allowed", False),
        ("lazy_encoder", True),
        ("lazy_decoder", True),
        ("graph_metadata_device", "cpu"),
        ("scientific_tensor_device", "mps"),
    ):
        _require_equal(runtime, runtime_key, runtime_expected, "runtime")

    accepted = _mapping(root.get("accepted"), "accepted")
    _require_equal(accepted, "max_n_logits", 10, "accepted")
    _require_equal(accepted, "desired_logit_probability", 0.95, "accepted")
    _require_equal(accepted, "max_feature_nodes", 4096, "accepted")
    _require_equal(accepted, "attribution_batch_sizes", [64, 32, 16], "accepted")
    _require_equal(accepted, "intervention_alphas", [0.0, 0.5, 1.0], "accepted")
    _require_equal(accepted, "baseline_repeat", True, "accepted")

    selection = _mapping(root.get("feature_selection"), "feature_selection")
    _require_equal(selection, "manual_selection_allowed", False, "feature_selection")
    _require_equal(
        selection,
        "primary_rule",
        "highest_absolute_direct_contribution_to_baseline_top_logit_at_final_token",
        "feature_selection",
    )
    _require_equal(
        selection,
        "fallback_rule",
        "highest_absolute_active_baseline_activation_at_final_token",
        "feature_selection",
    )

    limits = _mapping(root.get("safety_limits"), "safety_limits")
    for limit_key, limit_expected in (
        ("maximum_mps_driver_bytes", 24 * 1024**3),
        ("maximum_process_rss_bytes", 24 * 1024**3),
        ("maximum_swap_growth_bytes", 4 * 1024**3),
        ("minimum_available_memory_bytes", 4 * 1024**3),
        ("maximum_graph_buffer_bytes", 6 * 1024**3),
        ("maximum_transcoder_download_bytes", 6 * 1024**3),
        ("accepted_thermal_states", ["nominal", "fair"]),
    ):
        _require_equal(limits, limit_key, limit_expected, "safety_limits")

    artifacts = _mapping(root.get("artifacts"), "artifacts")
    for artifact_key, artifact_expected in (
        ("result_directory", RESULT_DIRECTORY),
        ("generated_directory", GENERATED_DIRECTORY),
        ("environment_lock", ENVIRONMENT_LOCK),
        ("projected_download_manifest", PROJECTED_MANIFEST),
    ):
        _require_equal(artifacts, artifact_key, artifact_expected, "artifacts")

    if root.get("prompt") != "The capital of France is":
        raise ArtifactValidationError("accepted prompt is not frozen")
    serialized = repr(dict(root)).casefold()
    forbidden = ("cuda", "t4", "bfloat16", "gemma-2-2b", "transformerlens")
    if any(term in serialized for term in forbidden):
        raise ArtifactValidationError("config contains cross-experiment contamination")
    for revision in (UPSTREAM_REVISION, MODEL_REVISION, TRANSCODER_REVISION):
        if HEX40_RE.fullmatch(revision) is None:
            raise ArtifactValidationError("revision is not an immutable commit SHA")
    return dict(root)


def assert_fallback_disabled(environment: Mapping[str, str] | None = None) -> None:
    """Fail if PyTorch's MPS-to-CPU fallback switch is truthy."""

    source = os.environ if environment is None else environment
    observed = source.get(FALLBACK_VARIABLE)
    if observed is not None and observed.strip().casefold() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }:
        raise ArtifactValidationError(f"{FALLBACK_VARIABLE} must be absent or false")


def validate_projected_manifest(value: Any) -> dict[str, Any]:
    """Validate exact no-download Hugging Face metadata and the allowlists."""

    root = _mapping(value, "projected manifest")
    _require_equal(root, "schema_version", 1, "projected manifest")
    _require_equal(root, "projected_total_bytes", 2_087_816_677, "projected manifest")
    model = _mapping(root.get("model"), "projected model")
    transcoder = _mapping(root.get("transcoder"), "projected transcoder")
    _require_equal(model, "identifier", MODEL_IDENTIFIER, "projected model")
    _require_equal(model, "revision", MODEL_REVISION, "projected model")
    _require_equal(model, "projected_bytes", 575_454_257, "projected model")
    _require_equal(
        transcoder, "identifier", TRANSCODER_IDENTIFIER, "projected transcoder"
    )
    _require_equal(transcoder, "revision", TRANSCODER_REVISION, "projected transcoder")
    _require_equal(
        transcoder, "subfolder", TRANSCODER_SUBFOLDER, "projected transcoder"
    )
    _require_equal(transcoder, "projected_bytes", 1_512_362_420, "projected transcoder")
    model_paths = {
        str(_mapping(item, "model file")["path"])
        for item in _sequence(model.get("files"), "model files")
    }
    transcoder_paths = {
        str(_mapping(item, "transcoder file")["path"])
        for item in _sequence(transcoder.get("files"), "transcoder files")
    }
    if model_paths != {
        "added_tokens.json",
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    }:
        raise ArtifactValidationError("model projected allowlist changed")
    expected_transcoder = {f"{TRANSCODER_SUBFOLDER}/config.yaml"} | {
        f"{TRANSCODER_SUBFOLDER}/layer_{layer}.safetensors"
        for layer in range(LAYER_COUNT)
    }
    if transcoder_paths != expected_transcoder:
        raise ArtifactValidationError("transcoder projected allowlist changed")
    return dict(root)


def conservative_memory_feasibility(
    *, maximum_process_rss_bytes: int = 24 * 1024**3
) -> MemoryFeasibility:
    """Return the preregistered conservative, non-additive-signal projection.

    The total includes model FP16 weights, persistent lazy PLT tensors, one
    encoder and one decoder matrix at once, a 4 GiB NNsight/runtime reserve,
    and the separately capped 6 GiB graph buffer. The 24 GiB MPS and RSS
    counters remain independent safety signals at runtime.
    """

    model_fp16 = 268_098_176 * 2
    persistent = LAYER_COUNT * (FEATURE_WIDTH + FEATURE_WIDTH + HIDDEN_SIZE) * 2
    one_matrix = FEATURE_WIDTH * HIDDEN_SIZE * 2
    full_eager = (
        LAYER_COUNT
        * (2 * FEATURE_WIDTH * HIDDEN_SIZE + 2 * FEATURE_WIDTH + HIDDEN_SIZE)
        * 2
    )
    runtime_reserve = 4 * 1024**3
    graph_limit = 6 * 1024**3
    total = model_fp16 + persistent + 2 * one_matrix + runtime_reserve + graph_limit
    return MemoryFeasibility(
        model_fp16_bytes=model_fp16,
        persistent_transcoder_fp16_bytes=persistent,
        one_lazy_matrix_fp16_bytes=one_matrix,
        full_eager_transcoder_fp16_bytes=full_eager,
        fixed_runtime_reserve_bytes=runtime_reserve,
        graph_buffer_limit_bytes=graph_limit,
        total_conservative_bytes=total,
        maximum_process_rss_bytes=maximum_process_rss_bytes,
        feasible=total < maximum_process_rss_bytes,
    )


def projected_graph_bytes(
    *, active_features: int, selected_features: int, token_count: int, logits: int
) -> int:
    """Upper-bound the two dense CPU adjacency work buffers used upstream."""

    values = (active_features, selected_features, token_count, logits)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in values
    ):
        raise ScientificInputError("graph dimensions must be positive integers")
    non_feature_nodes = (LAYER_COUNT + 1) * token_count + logits
    columns = active_features + non_feature_nodes
    rows = min(selected_features, active_features) + logits
    edge_matrix = rows * columns * 4
    final_columns = min(selected_features, active_features) + non_feature_nodes
    final_matrix = final_columns * final_columns * 4
    return edge_matrix + final_matrix


def _dense_to_cpu_sparse_metadata(dense: Any, torch: Any) -> tuple[Any, Any, Any]:
    """Keep scientific values on MPS while placing only COO metadata on CPU."""

    if dense.device.type != "mps" or dense.dtype != torch.float16:
        raise ArtifactValidationError("sparse adapter input must be MPS FP16")
    if not bool(torch.isfinite(dense).all().item()):
        raise ArtifactValidationError("sparse adapter input is non-finite")
    coordinates = torch.nonzero(dense, as_tuple=False)
    device_indices = coordinates.T.contiguous()
    if coordinates.numel():
        device_values = dense[tuple(device_indices[axis] for axis in range(dense.ndim))]
    else:
        device_values = torch.empty((0,), dtype=dense.dtype, device=dense.device)
    cpu_indices = device_indices.detach().to(device="cpu", dtype=torch.long)
    cpu_values = device_values.detach().to(device="cpu", dtype=torch.float32)
    metadata = torch.sparse_coo_tensor(
        cpu_indices,
        cpu_values,
        size=tuple(int(size) for size in dense.shape),
        device="cpu",
    ).coalesce()
    reconstructed = torch.zeros_like(dense)
    if device_values.numel():
        reconstructed.index_put_(
            tuple(device_indices[axis] for axis in range(dense.ndim)),
            device_values,
            accumulate=False,
        )
    if not bool(torch.equal(reconstructed, dense)):
        raise ArtifactValidationError("CPU sparse metadata boundary is not exact")
    return metadata, device_indices, device_values


def validate_live_sparse_boundary(torch: Any) -> dict[str, Any]:
    """Exercise the exact dense-MPS/CPU-COO compatibility boundary."""

    source = torch.tensor(
        [[0.0, 1.5, 0.0], [-2.0, 0.0, 3.0]],
        device="mps",
        dtype=torch.float16,
    )
    metadata, indices, values = _dense_to_cpu_sparse_metadata(source, torch)
    return {
        "passed": True,
        "execution_deviation": "explicit_cpu_sparse_metadata_only",
        "dense_scientific_tensor_device": str(source.device),
        "metadata_device": str(metadata.device),
        "index_device": str(indices.device),
        "value_device": str(values.device),
        "nonzero_count": int(metadata._nnz()),
        "maximum_absolute_error": 0.0,
    }


def mps_compute_attribution_components(
    transcoder_set: Any,
    mlp_inputs: Any,
    zero_positions: slice = slice(0, 1),
) -> dict[str, Any]:
    """MPS-equivalent replacement for upstream's unsupported ``to_sparse``."""

    torch = importlib.import_module("torch")
    if (
        mlp_inputs.device.type != "mps"
        or mlp_inputs.dtype != torch.float16
        or mlp_inputs.ndim != 3
        or len(transcoder_set) != mlp_inputs.shape[0]
    ):
        raise ArtifactValidationError("attribution inputs must be dense MPS FP16")
    n_layers, n_positions, _ = mlp_inputs.shape
    reconstruction = torch.zeros_like(mlp_inputs)
    encoder_vectors: list[Any] = []
    decoder_vectors: list[Any] = []
    cpu_indices: list[Any] = []
    cpu_values: list[Any] = []
    device_locations: list[Any] = []
    for layer, transcoder in enumerate(transcoder_set):
        encoder = transcoder.W_enc
        preactivation = torch.nn.functional.linear(
            mlp_inputs[layer].to(encoder.dtype), encoder, transcoder.b_enc
        )
        activations = transcoder.activation_function(preactivation)
        activations[zero_positions] = 0
        metadata, indices, values = _dense_to_cpu_sparse_metadata(activations, torch)
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
            if active_decoders.device.type != "mps":
                raise ArtifactValidationError("decoder vectors left MPS")
            scaled_decoders = active_decoders * values[:, None]
            reconstruction[layer].index_add_(0, positions, scaled_decoders)
            device_locations.append(
                torch.stack((torch.full_like(positions, layer), positions))
            )
        else:
            active_encoders = torch.empty(
                (0, transcoder.d_model), device="mps", dtype=torch.float16
            )
            scaled_decoders = torch.empty_like(active_encoders)
            device_locations.append(torch.empty((2, 0), device="mps", dtype=torch.long))
        if transcoder.W_skip is not None:
            reconstruction[layer] += transcoder.compute_skip(mlp_inputs[layer])
        reconstruction[layer] += transcoder.b_dec
        encoder_vectors.append(active_encoders)
        decoder_vectors.append(scaled_decoders)
    activation_matrix = torch.sparse_coo_tensor(
        torch.cat(cpu_indices, dim=1),
        torch.cat(cpu_values),
        size=(n_layers, n_positions, int(transcoder_set.d_transcoder)),
        device="cpu",
    ).coalesce()
    locations = torch.cat(device_locations, dim=1)
    encoder_tensor = torch.cat(encoder_vectors)
    decoder_tensor = torch.cat(decoder_vectors)
    active_count = int(activation_matrix._nnz())
    if active_count < 1 or encoder_tensor.shape != decoder_tensor.shape:
        raise ArtifactValidationError("attribution sparse metadata is inconsistent")
    if locations.shape != (2, active_count):
        raise ArtifactValidationError("attribution device locations are inconsistent")
    return {
        "activation_matrix": activation_matrix,
        "reconstruction": reconstruction,
        "encoder_vecs": encoder_tensor,
        "decoder_vecs": decoder_tensor,
        "encoder_to_decoder_map": torch.arange(active_count, device="mps"),
        "decoder_locations": locations,
    }


def mps_nnsight_compute_batch(
    context: Any,
    layers: Any,
    positions: Any,
    inject_values: Any,
    retain_graph: bool = True,
) -> Any:
    """Pinned NNsight context method with explicit MPS index placement."""

    torch = importlib.import_module("torch")
    if inject_values.device.type != "mps" or inject_values.dtype != torch.float16:
        raise ArtifactValidationError("attribution injection must be MPS FP16")
    layers_mps = layers.to(device="mps", dtype=torch.long)
    positions_mps = positions.to(device="mps", dtype=torch.long)
    batch_size = context._resid_activations[0].shape[0]
    context._batch_buffer = torch.zeros(
        context._row_size,
        batch_size,
        dtype=inject_values.dtype,
        device="mps",
    )
    batch_idx = torch.arange(len(layers_mps), device="mps")

    def inject(grad_point: Any, *, rows: Any, columns: Any, values: Any) -> None:
        grads_out = grad_point.grad.clone()
        grads_out.index_put_((rows, columns), values.to(grads_out.dtype))
        grad_point.grad = grads_out

    unique_layers = sorted(
        int(item) for item in layers_mps.unique().detach().cpu().tolist()
    )
    if not unique_layers:
        raise ArtifactValidationError("attribution batch must not be empty")
    last_layer = max(unique_layers)
    with context._resid_activations[last_layer].backward(
        gradient=torch.zeros_like(context._resid_activations[last_layer]),
        retain_graph=retain_graph,
    ):
        for layer in reversed(range(last_layer + 1)):
            if layer != last_layer:
                grad = context._feature_output_activations[layer + 1].grad.clone()
                context.compute_feature_attributions(layer, grad)
                context.compute_error_attributions(layer, grad)
            mask = layers_mps == layer
            if mask.any():
                inject(
                    context._resid_activations[layer],
                    rows=batch_idx[mask],
                    columns=positions_mps[mask],
                    values=inject_values[mask],
                )
        context.compute_token_attributions(context._feature_output_activations[0].grad)
    buffer, context._batch_buffer = context._batch_buffer, None
    result = buffer.T[: len(layers_mps)]
    if result.device.type != "mps" or not bool(torch.isfinite(result).all().item()):
        raise ArtifactValidationError("attribution batch result is not finite MPS data")
    return result


@contextlib.contextmanager
def mps_nnsight_attribution_adapter(model: Any) -> Iterator[dict[str, int]]:
    """Temporarily install the audited sparse/index boundary for NNsight."""

    attribute_module: Any = importlib.import_module(
        "circuit_tracer.attribution.attribute_nnsight"
    )
    context_module: Any = importlib.import_module(
        "circuit_tracer.attribution.context_nnsight"
    )
    context_class = context_module.AttributionContext
    transcoder_set = model.transcoders
    if hasattr(transcoder_set, "_module"):
        transcoder_set = transcoder_set._module
    original_components = transcoder_set.compute_attribution_components
    original_compute_batch = context_class.compute_batch
    original_partial = attribute_module.compute_partial_influences
    usage = {"component_calls": 0, "batch_calls": 0, "partial_calls": 0}

    def components(bound: Any, inputs: Any, zero_positions: slice = slice(0, 1)) -> Any:
        usage["component_calls"] += 1
        return mps_compute_attribution_components(bound, inputs, zero_positions)

    def compute_batch(
        context: Any,
        layers: Any,
        positions: Any,
        inject_values: Any,
        retain_graph: bool = True,
    ) -> Any:
        usage["batch_calls"] += 1
        return mps_nnsight_compute_batch(
            context, layers, positions, inject_values, retain_graph
        )

    def partial(
        edge_matrix: Any, logit_p: Any, row_map: Any, *args: Any, **kwargs: Any
    ) -> Any:
        usage["partial_calls"] += 1
        kwargs.pop("device", None)
        result = original_partial(
            edge_matrix,
            logit_p.detach().cpu(),
            row_map.detach().cpu(),
            *args,
            device="cpu",
            **kwargs,
        )
        if result.device.type != "cpu":
            raise ArtifactValidationError("graph ranking metadata left CPU boundary")
        return result

    transcoder_set.compute_attribution_components = types.MethodType(
        components, transcoder_set
    )
    context_class.compute_batch = compute_batch
    attribute_module.compute_partial_influences = partial
    try:
        yield usage
    finally:
        transcoder_set.compute_attribution_components = original_components
        context_class.compute_batch = original_compute_batch
        attribute_module.compute_partial_influences = original_partial


def select_feature_from_graph(graph: Any, *, final_position: int) -> FeatureSelection:
    """Apply the frozen direct-contribution rule with deterministic fallback."""

    active = graph.active_features.detach().cpu()
    values = graph.activation_values.detach().cpu()
    selected = graph.selected_features.detach().cpu()
    adjacency = graph.adjacency_matrix.detach().cpu()
    n_logits = len(graph.logit_targets)
    if n_logits < 1 or active.ndim != 2 or active.shape[1] != 3:
        raise ArtifactValidationError("graph feature structure is invalid")
    candidates: list[tuple[float, int, int, int, float]] = []
    top_logit_row = adjacency.shape[0] - n_logits
    for graph_column, active_index_value in enumerate(selected.tolist()):
        active_index = int(active_index_value)
        layer, position, feature = (int(item) for item in active[active_index].tolist())
        if position != final_position:
            continue
        baseline = float(values[active_index].item())
        score = abs(float(adjacency[top_logit_row, graph_column].item()))
        if baseline > 0.0 and math.isfinite(score) and math.isfinite(baseline):
            candidates.append((-score, layer, position, feature, baseline))
    if candidates:
        neg_score, layer, position, feature, baseline = min(candidates)
        return FeatureSelection(
            layer,
            position,
            feature,
            baseline,
            "highest_absolute_direct_contribution_to_baseline_top_logit_at_final_token",
            -neg_score,
        )
    fallback: list[tuple[float, int, int, int, float]] = []
    for active_index in range(active.shape[0]):
        layer, position, feature = (int(item) for item in active[active_index].tolist())
        baseline = float(values[active_index].item())
        if position == final_position and baseline > 0.0 and math.isfinite(baseline):
            fallback.append((-abs(baseline), layer, position, feature, baseline))
    if not fallback:
        raise ArtifactValidationError("no baseline-active final-token feature exists")
    neg_score, layer, position, feature, baseline = min(fallback)
    return FeatureSelection(
        layer,
        position,
        feature,
        baseline,
        "highest_absolute_active_baseline_activation_at_final_token",
        -neg_score,
    )


def intervention_values(
    baseline: float, alphas: Sequence[float]
) -> list[dict[str, float]]:
    """Produce the absolute upstream value for each frozen suppression alpha."""

    return [
        {
            "alpha": float(alpha),
            "baseline_activation": float(baseline),
            "desired_absolute_activation": desired_activation(baseline, alpha),
        }
        for alpha in alphas
    ]


def validate_asset_tree(snapshot: Path, expected_relative_paths: set[str]) -> None:
    """Require only regular allowlisted files under an immutable snapshot."""

    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ArtifactValidationError("asset snapshot is missing or unsafe")
    observed: set[str] = set()
    for candidate in snapshot.rglob("*"):
        relative = candidate.relative_to(snapshot).as_posix()
        if candidate.is_symlink():
            target = candidate.resolve(strict=True)
            if not target.is_file():
                raise ArtifactValidationError(
                    "asset symlink does not resolve to a file"
                )
            observed.add(relative)
        elif candidate.is_file():
            observed.add(relative)
        elif not candidate.is_dir():
            raise ArtifactValidationError("asset snapshot contains a special entry")
    if observed != expected_relative_paths:
        raise ArtifactValidationError("asset snapshot file allowlist mismatch")


def validate_small_artifact_directory(directory: Path) -> None:
    """Reject extra, linked, special, large, or weight-like result entries."""

    if directory.is_symlink() or not directory.is_dir():
        raise ArtifactValidationError("artifact directory is missing or unsafe")
    observed: set[str] = set()
    for entry in directory.iterdir():
        observed.add(entry.name)
        if entry.is_symlink() or not entry.is_file():
            raise ArtifactValidationError("artifact entry must be a regular file")
        metadata = entry.stat()
        if metadata.st_nlink != 1:
            raise ArtifactValidationError("hardlinked artifact entry is forbidden")
        if entry.suffix.casefold() in FORBIDDEN_SUFFIXES:
            raise ArtifactValidationError("weight or raw tensor artifact is forbidden")
        maximum = 4 * 1024**2 if entry.name == "checksums.sha256" else 2 * 1024**2
        if metadata.st_size > maximum:
            raise ArtifactValidationError("artifact entry exceeds its size cap")
    if observed != ARTIFACT_ALLOWLIST:
        raise ArtifactValidationError("artifact allowlist mismatch")


def lock_sha256(repository_root: Path) -> str:
    """Return the exact Stage 1A-S environment lock digest."""

    return sha256_file(repository_root / ENVIRONMENT_LOCK)


__all__ = [
    "ARTIFACT_ALLOWLIST",
    "BACKEND",
    "COMPLETED_STATUS",
    "DEVICE",
    "DTYPE",
    "EXPERIMENT_CLASS",
    "MODEL_IDENTIFIER",
    "MODEL_REVISION",
    "TRANSCODER_IDENTIFIER",
    "TRANSCODER_REVISION",
    "TRANSCODER_SUBFOLDER",
    "UPSTREAM_REVISION",
    "FeatureSelection",
    "MemoryFeasibility",
    "assert_fallback_disabled",
    "conservative_memory_feasibility",
    "intervention_values",
    "load_small_model_config",
    "lock_sha256",
    "mps_compute_attribution_components",
    "mps_nnsight_attribution_adapter",
    "projected_graph_bytes",
    "select_feature_from_graph",
    "validate_asset_tree",
    "validate_live_sparse_boundary",
    "validate_projected_manifest",
    "validate_small_artifact_directory",
    "validate_small_model_config",
]
