"""Reusable native-MPS/BF16 runtime assembly for Stage 1B."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cfsus.exceptions import ScientificInputError
from cfsus.reproduction.small_model_mps_bf16 import (
    assert_module_mps_bf16,
    projected_graph_bytes,
)
from cfsus.stage1b import MODEL_REVISION, TRANSCODER_REVISION, TRANSCODER_SUBFOLDER

MODEL_FP32_BUFFER_EXCEPTIONS = frozenset(
    {"model.rotary_emb.inv_freq", "model.rotary_emb_local.inv_freq"}
)
MODEL_CACHE_ID = "models--google--gemma-3-270m"
TRANSCODER_CACHE_ID = "models--mwhanna--gemma-scope-2-270m-pt"
LAYER_COUNT = 18
FEATURE_WIDTH = 16_384


def resolve_offline_snapshots(cache: Path, repository_root: Path) -> tuple[Path, Path]:
    """Resolve exact existing snapshots without authentication or download."""

    if not cache.is_absolute() or cache.is_symlink() or not cache.is_dir():
        raise ScientificInputError("external immutable cache is missing or unsafe")
    resolved = cache.resolve(strict=True)
    repository = repository_root.resolve(strict=True)
    if resolved == repository or resolved.is_relative_to(repository):
        raise ScientificInputError("asset cache overlaps the repository")
    model = resolved / MODEL_CACHE_ID / "snapshots" / MODEL_REVISION
    transcoder = resolved / TRANSCODER_CACHE_ID / "snapshots" / TRANSCODER_REVISION
    if not model.is_dir() or not transcoder.is_dir():
        raise ScientificInputError("exact offline snapshots are missing")
    return model, transcoder


def load_mps_bf16_transcoders(transcoder_snapshot: Path, torch: Any) -> Any:
    """Load the exact 18-layer PLT subset through the accepted Stage 1A adapter."""

    from circuit_tracer.transcoder.single_layer_transcoder import (  # type: ignore[import-not-found]
        load_transcoder_set,
    )

    from cfsus.reproduction.small_model_mps_bf16_runtime import (
        MPSBF16TranscoderSet,
    )

    root = transcoder_snapshot / TRANSCODER_SUBFOLDER
    paths = {
        layer: str(root / f"layer_{layer}.safetensors") for layer in range(LAYER_COUNT)
    }
    if any(not Path(path).is_file() for path in paths.values()):
        raise ScientificInputError("one or more exact PLT files are missing")
    source = load_transcoder_set(
        paths,
        scan_name=(
            "mwhanna/gemma-scope-2-270m-pt/"
            f"{TRANSCODER_SUBFOLDER}@{TRANSCODER_REVISION}"
        ),
        feature_input_hook="mlp.hook_in",
        feature_output_hook="hook_mlp_out",
        device=torch.device("mps"),
        dtype=torch.bfloat16,
        lazy_encoder=True,
        lazy_decoder=True,
    )
    transcoders = MPSBF16TranscoderSet(source)
    if (
        len(transcoders) != LAYER_COUNT
        or int(transcoders.d_transcoder) != FEATURE_WIDTH
    ):
        raise ScientificInputError("loaded PLT dimensions are invalid")
    return transcoders


def build_mps_bf16_replacement(
    model_snapshot: Path, transcoder_snapshot: Path, torch: Any
) -> tuple[Any, dict[str, Any]]:
    """Construct the exact accepted NNsight replacement model."""

    from cfsus.reproduction.small_model_mps_bf16_runtime import (
        MPSBF16ReplacementModel,
    )

    transcoders = load_mps_bf16_transcoders(transcoder_snapshot, torch)
    model = MPSBF16ReplacementModel.from_pretrained_and_transcoders(
        str(model_snapshot),
        transcoders,
        device=torch.device("mps"),
        dtype=torch.bfloat16,
    )
    suffixes = (
        "model.rotary_emb.inv_freq",
        "model.rotary_emb_local.inv_freq",
    )
    observed = {
        name
        for name, buffer in model.named_buffers()
        if buffer.dtype == torch.float32
        and any(name.endswith(suffix) for suffix in suffixes)
    }
    if observed != MODEL_FP32_BUFFER_EXCEPTIONS:
        raise ScientificInputError("replacement runtime FP32 buffer set changed")
    guard = assert_module_mps_bf16(
        model, torch, allowed_fp32_buffer_names=MODEL_FP32_BUFFER_EXCEPTIONS
    )
    if model.backend != "nnsight" or int(model.cfg.n_layers) != LAYER_COUNT:
        raise ScientificInputError("replacement runtime identity is invalid")
    if torch.is_autocast_enabled():
        raise ScientificInputError("replacement runtime unexpectedly uses autocast")
    return model, guard


def build_raw_reference_graph(
    model: Any,
    *,
    prompt: str,
    graph_config: dict[str, Any],
    maximum_graph_buffer_bytes: int,
    torch: Any,
) -> tuple[Any, dict[str, int]]:
    """Build one fresh in-memory raw graph for independent pair references."""

    from cfsus.reproduction.small_model_mps_bf16_runtime import (
        attribute_mps_bf16,
    )

    input_ids = model.ensure_tokenized(prompt)
    context = model.setup_attribution(input_ids)
    active_count = int(context.activation_matrix._nnz())
    token_count = int(context.activation_matrix.shape[1])
    projected = projected_graph_bytes(
        active_features=active_count,
        selected_features=int(graph_config["max_feature_nodes"]),
        token_count=token_count,
        logits=int(graph_config["max_n_logits"]),
    )
    if projected > maximum_graph_buffer_bytes:
        raise ScientificInputError("projected raw graph exceeds the frozen cap")
    graph, usage = attribute_mps_bf16(
        prompt,
        model,
        context=context,
        max_n_logits=int(graph_config["max_n_logits"]),
        desired_logit_probability=float(graph_config["desired_logit_probability"]),
        batch_size=int(graph_config["attribution_batch_size"]),
        max_feature_nodes=int(graph_config["max_feature_nodes"]),
    )
    adjacency = graph.adjacency_matrix
    if adjacency.numel() == 0 or not bool(torch.isfinite(adjacency).all().item()):
        raise ScientificInputError("raw reference graph is empty or non-finite")
    if int(torch.count_nonzero(adjacency).item()) < 1:
        raise ScientificInputError("raw reference graph has no nonzero edge")
    usage = dict(usage)
    usage["active_feature_count"] = active_count
    usage["token_count"] = token_count
    usage["projected_graph_bytes"] = projected
    usage["nonzero_edge_count"] = int(torch.count_nonzero(adjacency).item())
    return graph, usage


__all__ = [
    "FEATURE_WIDTH",
    "LAYER_COUNT",
    "build_mps_bf16_replacement",
    "build_raw_reference_graph",
    "load_mps_bf16_transcoders",
    "resolve_offline_snapshots",
]
