#!/usr/bin/env python3
"""Fresh-process empirical worker for staged Stage 1A-S MPS execution."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cfsus.reproduction.artifacts import write_json_atomic  # noqa: E402
from cfsus.reproduction.small_model_mps_fp16 import (  # noqa: E402
    BACKEND,
    DEVICE,
    DTYPE,
    FEATURE_WIDTH,
    LAYER_COUNT,
    MODEL_IDENTIFIER,
    MODEL_REVISION,
    TRANSCODER_IDENTIFIER,
    TRANSCODER_REVISION,
    TRANSCODER_SUBFOLDER,
    assert_fallback_disabled,
    intervention_values,
    load_small_model_config,
    mps_nnsight_attribution_adapter,
    projected_graph_bytes,
    select_feature_from_graph,
)

STAGES = (
    "model_forward",
    "loaded_semantics",
    "full_plt",
    "replacement_runtime",
    "smoke",
    "accepted",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/stage1a_small_model_mps_fp16_pilot.yaml",
    )
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _run_text(command: list[str]) -> str:
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
        env={
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        },
    )
    return result.stdout.strip()


def _swap_used_bytes() -> int:
    output = _run_text(["sysctl", "vm.swapusage"])
    match = re.search(r"used = ([0-9.]+)M", output)
    if match is None:
        raise RuntimeError("swap telemetry is unavailable")
    return round(float(match.group(1)) * 1024**2)


def _thermal_state() -> str:
    output = _run_text(["pmset", "-g", "therm"]).casefold()
    if "serious" in output or "critical" in output:
        return "serious_or_critical"
    if "no thermal warning level has been recorded" in output:
        return "nominal"
    if "fair" in output:
        return "fair"
    return "unknown"


@dataclass(frozen=True, slots=True)
class Sample:
    unix_time: float
    stage: str
    mps_current_bytes: int
    mps_driver_bytes: int
    process_rss_bytes: int
    available_memory_bytes: int
    swap_used_bytes: int
    thermal_state: str


class TelemetrySampler:
    """Collect independent safety counters while the worker is active."""

    def __init__(self, torch: Any, limits: Mapping[str, Any]) -> None:
        self.torch = torch
        self.limits = limits
        self.interval = float(limits["sample_interval_seconds"])
        self.started_at = time.time()
        self.swap_start = _swap_used_bytes()
        self._stage = "worker_start"
        self._samples: list[Sample] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        with self._lock:
            self._stage = name
        try:
            yield
        finally:
            self.sample()

    def sample(self) -> None:
        import psutil  # type: ignore[import-untyped]

        with self._lock:
            stage = self._stage
        sample = Sample(
            unix_time=time.time(),
            stage=stage,
            mps_current_bytes=int(self.torch.mps.current_allocated_memory()),
            mps_driver_bytes=int(self.torch.mps.driver_allocated_memory()),
            process_rss_bytes=int(psutil.Process().memory_info().rss),
            available_memory_bytes=int(psutil.virtual_memory().available),
            swap_used_bytes=_swap_used_bytes(),
            thermal_state=_thermal_state(),
        )
        with self._lock:
            self._samples.append(sample)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.sample()
            except Exception:
                continue

    def finish(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=5)
        self.sample()
        with self._lock:
            samples = list(self._samples)
        if not samples:
            raise RuntimeError("telemetry sampler produced no samples")
        metrics = (
            "mps_current_bytes",
            "mps_driver_bytes",
            "process_rss_bytes",
            "swap_used_bytes",
        )
        stage_names = sorted({sample.stage for sample in samples})
        stage_peaks: dict[str, dict[str, int]] = {}
        for stage in stage_names:
            selected = [sample for sample in samples if sample.stage == stage]
            stage_peaks[stage] = {
                metric: max(int(getattr(sample, metric)) for sample in selected)
                for metric in metrics
            }
            stage_peaks[stage]["swap_growth_bytes"] = max(
                0, stage_peaks[stage]["swap_used_bytes"] - self.swap_start
            )
            stage_peaks[stage]["minimum_available_memory_bytes"] = min(
                sample.available_memory_bytes for sample in selected
            )
        attempt_peaks = {
            metric: max(int(getattr(sample, metric)) for sample in samples)
            for metric in metrics
        }
        attempt_peaks["swap_growth_bytes"] = max(
            0, attempt_peaks["swap_used_bytes"] - self.swap_start
        )
        attempt_peaks["minimum_available_memory_bytes"] = min(
            sample.available_memory_bytes for sample in samples
        )
        thermal_states = sorted({sample.thermal_state for sample in samples})
        violations: list[str] = []
        if attempt_peaks["mps_driver_bytes"] > int(
            self.limits["maximum_mps_driver_bytes"]
        ):
            violations.append("maximum_mps_driver_bytes")
        if attempt_peaks["process_rss_bytes"] > int(
            self.limits["maximum_process_rss_bytes"]
        ):
            violations.append("maximum_process_rss_bytes")
        if attempt_peaks["swap_growth_bytes"] > int(
            self.limits["maximum_swap_growth_bytes"]
        ):
            violations.append("maximum_swap_growth_bytes")
        if attempt_peaks["minimum_available_memory_bytes"] < int(
            self.limits["minimum_available_memory_bytes"]
        ):
            violations.append("minimum_available_memory_bytes")
        accepted_thermal = set(self.limits["accepted_thermal_states"])
        if any(state not in accepted_thermal for state in thermal_states):
            violations.append("accepted_thermal_states")
        return {
            "started_at_unix": self.started_at,
            "finished_at_unix": time.time(),
            "sampling_interval_seconds": self.interval,
            "sample_count": len(samples),
            "swap_start_bytes": self.swap_start,
            "attempt_peaks": attempt_peaks,
            "stage_peaks": stage_peaks,
            "thermal_states": thermal_states,
            "violations": violations,
        }


def _resolve_snapshots(config: Mapping[str, Any], cache: Path) -> tuple[Path, Path]:
    if not cache.is_absolute() or cache.is_symlink() or not cache.is_dir():
        raise RuntimeError("external Hugging Face cache is unavailable or unsafe")
    resolved_cache = cache.resolve(strict=True)
    if resolved_cache.is_relative_to(REPOSITORY_ROOT):
        raise RuntimeError("Hugging Face cache overlaps the repository")
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    model = Path(
        snapshot_download(
            MODEL_IDENTIFIER,
            revision=MODEL_REVISION,
            allow_patterns=list(config["model"]["allow_patterns"]),
            cache_dir=resolved_cache,
            local_files_only=True,
        )
    )
    transcoder = Path(
        snapshot_download(
            TRANSCODER_IDENTIFIER,
            revision=TRANSCODER_REVISION,
            allow_patterns=list(config["transcoder"]["allow_patterns"]),
            cache_dir=resolved_cache,
            local_files_only=True,
        )
    )
    if model.name != MODEL_REVISION or transcoder.name != TRANSCODER_REVISION:
        raise RuntimeError("local snapshot identity is mutable or incorrect")
    return model, transcoder


def _assert_model_mps_fp16(model: Any, torch: Any) -> None:
    parameters = list(model.parameters())
    if not parameters:
        raise RuntimeError("model has no parameters")
    if any(parameter.device.type != "mps" for parameter in parameters):
        raise RuntimeError("a model parameter is not on MPS")
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.float16
        for parameter in parameters
    ):
        raise RuntimeError("a floating model parameter is not FP16")
    if any(parameter.device.type == "cuda" for parameter in parameters):
        raise RuntimeError("CUDA contamination detected")


def _load_hf_model(model_snapshot: Path, torch: Any) -> tuple[Any, Any]:
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_snapshot,
        local_files_only=True,
        dtype=torch.float16,
        device_map={"": "mps"},
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    _assert_model_mps_fp16(model, torch)
    return model, tokenizer


def _tokenize(tokenizer: Any, prompt: str, torch: Any) -> Any:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device="mps", dtype=torch.long)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise RuntimeError("tokenizer produced an unexpected input shape")
    return input_ids


def _compact_logits(logits: Any, torch: Any, *, k: int = 10) -> dict[str, Any]:
    last = logits[0, -1]
    if last.device.type != "mps" or not bool(torch.isfinite(last).all().item()):
        raise RuntimeError("logits are not finite MPS tensors")
    probabilities = torch.softmax(last.float(), dim=-1)
    values, indices = torch.topk(probabilities, k)
    selected_logits = last[indices]
    return {
        "shape": [int(size) for size in logits.shape],
        "device": str(logits.device),
        "dtype": str(logits.dtype),
        "nonfinite_count": int((~torch.isfinite(logits)).sum().item()),
        "top_token_ids": [int(item) for item in indices.detach().cpu().tolist()],
        "top_probabilities": [float(item) for item in values.detach().cpu().tolist()],
        "top_logits": [float(item) for item in selected_logits.detach().cpu().tolist()],
    }


def _load_transcoder_set(transcoder_snapshot: Path, torch: Any) -> Any:
    from circuit_tracer.transcoder.single_layer_transcoder import (  # type: ignore[import-not-found]
        load_transcoder_set,
    )

    root = transcoder_snapshot / TRANSCODER_SUBFOLDER
    paths = {
        layer: str(root / f"layer_{layer}.safetensors") for layer in range(LAYER_COUNT)
    }
    if any(not Path(path).is_file() for path in paths.values()):
        raise RuntimeError("one or more exact PLT layer files are absent")
    transcoders = load_transcoder_set(
        paths,
        scan_name=f"{TRANSCODER_IDENTIFIER}/{TRANSCODER_SUBFOLDER}@{TRANSCODER_REVISION}",
        feature_input_hook="mlp.hook_in",
        feature_output_hook="hook_mlp_out",
        device=torch.device("mps"),
        dtype=torch.float16,
        lazy_encoder=True,
        lazy_decoder=True,
    )
    if len(transcoders) != LAYER_COUNT or transcoders.d_transcoder != FEATURE_WIDTH:
        raise RuntimeError("loaded PLT set dimensions are incorrect")
    return transcoders


def _model_forward(
    config: Mapping[str, Any],
    model_snapshot: Path,
    sampler: TelemetrySampler,
    torch: Any,
) -> dict[str, Any]:
    with sampler.stage("model_loading"):
        model, tokenizer = _load_hf_model(model_snapshot, torch)
    with sampler.stage("model_forward"):
        input_ids = _tokenize(tokenizer, str(config["prompt"]), torch)
        with torch.inference_mode():
            logits = model(input_ids=input_ids).logits
        torch.mps.synchronize()
    result = {
        "prompt": config["prompt"],
        "token_ids": [int(item) for item in input_ids[0].detach().cpu().tolist()],
        "decoded_tokens": [
            tokenizer.decode([int(item)])
            for item in input_ids[0].detach().cpu().tolist()
        ],
        "input_shape": [int(size) for size in input_ids.shape],
        "input_device": str(input_ids.device),
        "logits": _compact_logits(logits, torch),
        "all_parameters_device": "mps",
        "all_floating_parameters_dtype": "torch.float16",
    }
    del logits, model
    gc.collect()
    torch.mps.empty_cache()
    return result


def _loaded_semantics(
    config: Mapping[str, Any],
    model_snapshot: Path,
    transcoder_snapshot: Path,
    sampler: TelemetrySampler,
    torch: Any,
) -> dict[str, Any]:
    from circuit_tracer.transcoder.single_layer_transcoder import (
        load_transcoder,
    )

    with sampler.stage("model_loading"):
        model, tokenizer = _load_hf_model(model_snapshot, torch)
    captured: list[Any] = []

    def capture(_module: Any, _inputs: Any, output: Any) -> None:
        captured.append(output.detach())

    hook = model.model.layers[0].pre_feedforward_layernorm.register_forward_hook(
        capture
    )
    try:
        with sampler.stage("model_forward"):
            input_ids = _tokenize(tokenizer, str(config["prompt"]), torch)
            with torch.inference_mode():
                logits = model(input_ids=input_ids).logits
            torch.mps.synchronize()
    finally:
        hook.remove()
    if len(captured) != 1 or captured[0].device.type != "mps":
        raise RuntimeError("Gemma 3 mlp.hook_in location was not captured on MPS")
    path = transcoder_snapshot / TRANSCODER_SUBFOLDER / "layer_0.safetensors"
    with sampler.stage("one_layer_plt_loading"):
        transcoder = load_transcoder(
            str(path),
            layer=0,
            device=torch.device("mps"),
            dtype=torch.float16,
            lazy_encoder=False,
            lazy_decoder=False,
        )
    with sampler.stage("loaded_semantics"):
        mlp_input = captured[0].squeeze(0)
        preactivation = torch.nn.functional.linear(
            mlp_input, transcoder.W_enc, transcoder.b_enc
        )
        activation = transcoder.activation_function(preactivation)
        threshold = transcoder.activation_function.threshold
        expected = preactivation * (preactivation > threshold)
        equality = transcoder.activation_function(threshold)
        discrepancy = float(torch.max(torch.abs(activation - expected)).item())
        final = int(activation.shape[0] - 1)
        final_preactivation = preactivation[final]
        final_activation = activation[final]
        active_ids = torch.nonzero(final_activation > 0, as_tuple=False).flatten()
        inactive_ids = torch.nonzero(final_activation == 0, as_tuple=False).flatten()
        if active_ids.numel() < 1 or inactive_ids.numel() < 1:
            raise RuntimeError("loaded semantics lacks active or inactive evidence")
        active_id = int(active_ids[0].item())
        inactive_id = int(inactive_ids[0].item())
        if discrepancy > float(config["tolerances"]["loaded_semantics_absolute"]):
            raise RuntimeError("loaded JumpReLU semantics exceeded tolerance")
        if not bool(torch.equal(equality, torch.zeros_like(equality))):
            raise RuntimeError("JumpReLU equality is not inactive")
    result = {
        "layer": 0,
        "position": final,
        "encoder_shape": [int(size) for size in transcoder.W_enc.shape],
        "decoder_shape": [int(size) for size in transcoder.W_dec.shape],
        "threshold_shape": [int(size) for size in threshold.shape],
        "tensor_device": str(transcoder.b_enc.device),
        "tensor_dtype": str(transcoder.b_enc.dtype),
        "preactivation_formula": "linear(mlp_hook_in, W_enc, b_enc)",
        "gate_rule": "z if z > threshold else 0",
        "threshold_equality_inactive": True,
        "active_count_final_position": int(active_ids.numel()),
        "inactive_count_final_position": int(inactive_ids.numel()),
        "active_sample": {
            "feature": active_id,
            "preactivation": float(final_preactivation[active_id].item()),
            "threshold": float(threshold[active_id].item()),
            "activation": float(final_activation[active_id].item()),
        },
        "inactive_sample": {
            "feature": inactive_id,
            "preactivation": float(final_preactivation[inactive_id].item()),
            "threshold": float(threshold[inactive_id].item()),
            "activation": float(final_activation[inactive_id].item()),
        },
        "maximum_gate_discrepancy": discrepancy,
        "nonfinite_count": int(
            (~torch.isfinite(preactivation)).sum().item()
            + (~torch.isfinite(threshold)).sum().item()
            + (~torch.isfinite(activation)).sum().item()
        ),
        "model_logits": _compact_logits(logits, torch),
    }
    del logits, model, transcoder
    gc.collect()
    torch.mps.empty_cache()
    return result


def _full_plt(
    transcoder_snapshot: Path, sampler: TelemetrySampler, torch: Any
) -> dict[str, Any]:
    with sampler.stage("full_plt_loading"):
        transcoders = _load_transcoder_set(transcoder_snapshot, torch)
        parameters = list(transcoders.parameters())
        decoder_sample = transcoders._get_decoder_vectors(
            0, torch.tensor([0], device="mps")
        )
        encoder_sample = transcoders[0].W_enc[:1]
        torch.mps.synchronize()
    if any(parameter.device.type != "mps" for parameter in parameters):
        raise RuntimeError("persistent PLT parameter left MPS")
    if decoder_sample.device.type != "mps" or encoder_sample.device.type != "mps":
        raise RuntimeError("lazy PLT matrix sample left MPS")
    if any(
        not transcoder.lazy_encoder or not transcoder.lazy_decoder
        for transcoder in transcoders
    ):
        raise RuntimeError("PLT set is not fully lazy")
    result = {
        "layer_count": len(transcoders),
        "feature_width_per_layer": int(transcoders.d_transcoder),
        "total_feature_width": len(transcoders) * int(transcoders.d_transcoder),
        "lazy_encoder": True,
        "lazy_decoder": True,
        "persistent_parameter_device": "mps",
        "persistent_parameter_dtypes": sorted(
            {str(parameter.dtype) for parameter in parameters}
        ),
        "lazy_encoder_sample_shape": [int(size) for size in encoder_sample.shape],
        "lazy_decoder_sample_shape": [int(size) for size in decoder_sample.shape],
        "source_storage": "project_external_immutable_safetensors",
        "cache_conversion_used": False,
        "nonfinite_count": sum(
            int((~torch.isfinite(parameter)).sum().item())
            for parameter in parameters
            if parameter.is_floating_point()
        ),
    }
    del transcoders
    gc.collect()
    torch.mps.empty_cache()
    return result


def _build_replacement(
    model_snapshot: Path,
    transcoder_snapshot: Path,
    sampler: TelemetrySampler,
    torch: Any,
) -> Any:
    from circuit_tracer.replacement_model.replacement_model_nnsight import (  # type: ignore[import-not-found]
        NNSightReplacementModel,
    )

    with sampler.stage("full_plt_loading"):
        transcoders = _load_transcoder_set(transcoder_snapshot, torch)
    with sampler.stage("replacement_runtime_construction"):
        model = NNSightReplacementModel.from_pretrained_and_transcoders(
            str(model_snapshot),
            transcoders,
            device=torch.device("mps"),
            dtype=torch.float16,
        )
        _assert_model_mps_fp16(model, torch)
        torch.mps.synchronize()
    if model.backend != BACKEND or model.cfg.n_layers != LAYER_COUNT:
        raise RuntimeError("replacement runtime identity is incorrect")
    return model


def _replacement_runtime(
    config: Mapping[str, Any],
    model_snapshot: Path,
    transcoder_snapshot: Path,
    sampler: TelemetrySampler,
    torch: Any,
) -> dict[str, Any]:
    model = _build_replacement(model_snapshot, transcoder_snapshot, sampler, torch)
    with sampler.stage("replacement_feature_access"):
        logits, activations = model.get_activations(
            str(config["prompt"]), sparse=False, apply_activation_function=True
        )
        torch.mps.synchronize()
    if activations.device.type != "mps" or activations.dtype != torch.float16:
        raise RuntimeError("replacement feature cache left MPS FP16")
    if activations.shape[0] != LAYER_COUNT or activations.shape[2] != FEATURE_WIDTH:
        raise RuntimeError("replacement feature cache shape is incorrect")
    result = {
        "backend_class": type(model).__name__,
        "backend": model.backend,
        "device": str(model.device),
        "dtype": str(model.dtype),
        "layer_count": int(model.cfg.n_layers),
        "hidden_size": int(model.cfg.d_model),
        "feature_cache_shape": [int(size) for size in activations.shape],
        "feature_cache_device": str(activations.device),
        "feature_cache_dtype": str(activations.dtype),
        "active_feature_count": int((activations > 0).sum().item()),
        "inactive_feature_count": int((activations == 0).sum().item()),
        "intervention_tuple_semantics": (
            "(layer, position, feature, absolute_post_gate_value)"
        ),
        "model_logits": _compact_logits(logits, torch),
    }
    del activations, logits, model
    gc.collect()
    torch.mps.empty_cache()
    return result


@contextlib.contextmanager
def _cached_attribution_setup(model: Any, context: Any) -> Iterator[None]:
    model_class = type(model)
    original = model_class.setup_attribution

    def cached(self: Any, _inputs: Any) -> Any:
        if self is not model:
            return original(self, _inputs)
        return context

    model_class.setup_attribution = cached
    try:
        yield
    finally:
        model_class.setup_attribution = original


def _attribution_and_intervention(
    config: Mapping[str, Any],
    model_snapshot: Path,
    transcoder_snapshot: Path,
    sampler: TelemetrySampler,
    torch: Any,
    *,
    profile: str,
    batch_size: int,
) -> dict[str, Any]:
    from circuit_tracer.attribution.attribute_nnsight import (  # type: ignore[import-not-found]
        attribute,
    )

    settings = config[profile]
    model = _build_replacement(model_snapshot, transcoder_snapshot, sampler, torch)
    prompt = str(config["prompt"])
    input_ids = model.ensure_tokenized(prompt)
    adapter_usage: dict[str, int]
    with mps_nnsight_attribution_adapter(model) as adapter_usage:
        with sampler.stage("attribution_precompute"):
            context = model.setup_attribution(input_ids)
            active_count = int(context.activation_matrix._nnz())
            token_count = int(context.activation_matrix.shape[1])
            graph_bytes = projected_graph_bytes(
                active_features=active_count,
                selected_features=int(settings["max_feature_nodes"]),
                token_count=token_count,
                logits=int(settings["max_n_logits"]),
            )
            if graph_bytes > int(config["safety_limits"]["maximum_graph_buffer_bytes"]):
                raise RuntimeError("projected attribution graph exceeds the frozen cap")
        with _cached_attribution_setup(model, context), sampler.stage("attribution"):
            graph = attribute(
                prompt,
                model,
                max_n_logits=int(settings["max_n_logits"]),
                desired_logit_prob=float(settings["desired_logit_probability"]),
                batch_size=batch_size,
                max_feature_nodes=int(settings["max_feature_nodes"]),
                offload=None,
                verbose=False,
            )
            torch.mps.synchronize()
    adjacency = graph.adjacency_matrix.detach().cpu()
    if adjacency.numel() == 0 or not bool(torch.isfinite(adjacency).all().item()):
        raise RuntimeError("attribution graph is empty or non-finite")
    selection = select_feature_from_graph(graph, final_position=token_count - 1)
    nonzero_edges = int(torch.count_nonzero(adjacency).item())
    if nonzero_edges < 1:
        raise RuntimeError("attribution graph has no nonzero edges")

    with sampler.stage("loaded_semantics"):
        _, preactivations = model.get_activations(
            prompt, sparse=False, apply_activation_function=False
        )
        _, activations = model.get_activations(
            prompt, sparse=False, apply_activation_function=True
        )
        threshold = model.transcoders._module[
            selection.layer
        ].activation_function.threshold
        expected = preactivations * (preactivations > threshold[None, None, :])
        gate_discrepancy = float(torch.max(torch.abs(expected - activations)).item())
        selected_preactivation = float(
            preactivations[
                selection.layer, selection.position, selection.feature
            ].item()
        )
        selected_threshold = float(threshold[selection.feature].item())
        selected_activation = float(
            activations[selection.layer, selection.position, selection.feature].item()
        )
        inactive_ids = torch.nonzero(
            activations[selection.layer, selection.position] == 0, as_tuple=False
        ).flatten()
        if inactive_ids.numel() < 1:
            raise RuntimeError("selected position has no inactive feature")
        inactive_feature = int(inactive_ids[0].item())
        equality = model.transcoders._module[selection.layer].activation_function(
            threshold
        )
        if not bool(torch.equal(equality, torch.zeros_like(equality))):
            raise RuntimeError("loaded JumpReLU equality is not inactive")
        if gate_discrepancy > float(config["tolerances"]["loaded_semantics_absolute"]):
            raise RuntimeError("full loaded semantics exceeded tolerance")
    if not math.isclose(
        selected_activation,
        selection.baseline_activation,
        abs_tol=float(config["tolerances"]["loaded_semantics_absolute"]),
        rel_tol=float(config["tolerances"]["loaded_semantics_relative"]),
    ):
        raise RuntimeError("graph and loaded baseline activation disagree")

    with sampler.stage("intervention_baseline"):
        baseline_logits, _ = model.feature_intervention(
            prompt, [], freeze_attention=False, return_activations=False
        )
        repeat_logits, _ = model.feature_intervention(
            prompt, [], freeze_attention=False, return_activations=False
        )
    baseline_last = baseline_logits[0, -1]
    repeat_last = repeat_logits[0, -1]
    baseline_repeat_max = float(
        torch.max(torch.abs(baseline_last - repeat_last)).item()
    )
    conditions: list[dict[str, Any]] = []
    alphas = [float(value) for value in settings["intervention_alphas"]]
    for mapping in intervention_values(selected_activation, alphas):
        alpha = mapping["alpha"]
        desired = mapping["desired_absolute_activation"]
        with sampler.stage(f"intervention_alpha_{alpha}"):
            logits, _ = model.feature_intervention(
                prompt,
                [(selection.layer, selection.position, selection.feature, desired)],
                freeze_attention=True,
                return_activations=False,
            )
            torch.mps.synchronize()
        last = logits[0, -1]
        maximum_baseline_difference = float(
            torch.max(torch.abs(baseline_last - last)).item()
        )
        conditions.append(
            {
                **mapping,
                "sent_absolute_activation": desired,
                "observed_post_intervention_activation": None,
                "observed_post_intervention_activation_accessible": False,
                "maximum_absolute_baseline_logit_difference": (
                    maximum_baseline_difference
                ),
                "logits": _compact_logits(logits, torch),
            }
        )
    noop = next(condition for condition in conditions if condition["alpha"] == 0.0)
    noop_max = float(noop["maximum_absolute_baseline_logit_difference"])
    tolerance = float(config["tolerances"]["baseline_noop_logits_absolute"])
    if noop_max > tolerance:
        raise RuntimeError("baseline/no-op consistency exceeded frozen tolerance")
    value_tolerance = float(config["tolerances"]["intervention_value_absolute"])
    for condition in conditions:
        recomputed = (1.0 - float(condition["alpha"])) * selected_activation
        if (
            abs(recomputed - float(condition["sent_absolute_activation"]))
            > value_tolerance
        ):
            raise RuntimeError("absolute suppression mapping is inconsistent")
    result = {
        "profile": profile,
        "batch_size": batch_size,
        "prompt": prompt,
        "token_ids": [int(item) for item in input_ids.detach().cpu().tolist()],
        "token_count": token_count,
        "attribution": {
            "active_feature_count": active_count,
            "selected_feature_count": len(graph.selected_features),
            "adjacency_shape": [int(size) for size in adjacency.shape],
            "nonzero_edge_count": nonzero_edges,
            "logit_node_count": len(graph.logit_targets),
            "error_node_count": LAYER_COUNT * token_count,
            "input_node_count": token_count,
            "nonfinite_count": int((~torch.isfinite(adjacency)).sum().item()),
            "projected_graph_bytes": graph_bytes,
            "adapter_usage": dict(adapter_usage),
            "raw_graph_persisted": False,
        },
        "selection": asdict(selection),
        "loaded_semantics": {
            "selected_preactivation": selected_preactivation,
            "selected_threshold": selected_threshold,
            "selected_activation": selected_activation,
            "inactive_feature": inactive_feature,
            "inactive_preactivation": float(
                preactivations[
                    selection.layer, selection.position, inactive_feature
                ].item()
            ),
            "inactive_threshold": float(threshold[inactive_feature].item()),
            "inactive_activation": float(
                activations[
                    selection.layer, selection.position, inactive_feature
                ].item()
            ),
            "threshold_equality_inactive": True,
            "maximum_gate_discrepancy": gate_discrepancy,
            "nonfinite_count": int(
                (~torch.isfinite(preactivations)).sum().item()
                + (~torch.isfinite(activations)).sum().item()
                + (~torch.isfinite(threshold)).sum().item()
            ),
        },
        "intervention": {
            "baseline": _compact_logits(baseline_logits, torch),
            "baseline_repeat": _compact_logits(repeat_logits, torch),
            "baseline_repeat_maximum_absolute_difference": baseline_repeat_max,
            "baseline_noop_tolerance": tolerance,
            "conditions": conditions,
            "suppression_formula": "desired=(1-alpha)*baseline",
        },
    }
    del graph, adjacency, model, preactivations, activations
    gc.collect()
    torch.mps.empty_cache()
    return result


def _sanitize_error(error: BaseException) -> dict[str, str]:
    text = str(error).replace(str(REPOSITORY_ROOT), "[REPOSITORY]")
    text = re.sub(r"/Users/[^/\s]+/[^\s]*", "[PRIVATE_PATH]", text)
    text = text[:1000]
    return {"type": type(error).__name__, "message": text}


def main() -> int:
    arguments = _parser().parse_args()
    assert_fallback_disabled()
    config = load_small_model_config(arguments.config)
    if platform.machine() != "arm64" or sys.version_info[:2] != (3, 11):
        raise RuntimeError("worker requires native arm64 CPython 3.11")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch  # type: ignore[import-not-found]

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    model_snapshot, transcoder_snapshot = _resolve_snapshots(config, arguments.hf_cache)
    limits = config["safety_limits"]
    sampler = TelemetrySampler(torch, limits)
    outcome: dict[str, Any]
    error: dict[str, str] | None = None
    try:
        if arguments.stage == "model_forward":
            outcome = _model_forward(config, model_snapshot, sampler, torch)
        elif arguments.stage == "loaded_semantics":
            outcome = _loaded_semantics(
                config, model_snapshot, transcoder_snapshot, sampler, torch
            )
        elif arguments.stage == "full_plt":
            outcome = _full_plt(transcoder_snapshot, sampler, torch)
        elif arguments.stage == "replacement_runtime":
            outcome = _replacement_runtime(
                config, model_snapshot, transcoder_snapshot, sampler, torch
            )
        else:
            profile = arguments.stage
            settings = config[profile]
            batch_size = arguments.batch_size
            if batch_size is None:
                if profile == "smoke":
                    batch_size = int(settings["attribution_batch_size"])
                else:
                    batch_size = int(settings["attribution_batch_sizes"][0])
            allowed = (
                {int(settings["attribution_batch_size"])}
                if profile == "smoke"
                else {int(value) for value in settings["attribution_batch_sizes"]}
            )
            if batch_size not in allowed:
                raise RuntimeError(
                    "attribution batch size is outside the frozen policy"
                )
            outcome = _attribution_and_intervention(
                config,
                model_snapshot,
                transcoder_snapshot,
                sampler,
                torch,
                profile=profile,
                batch_size=batch_size,
            )
    except BaseException as caught:
        outcome = {}
        error = _sanitize_error(caught)
    telemetry = sampler.finish()
    status = "passed"
    if error is not None:
        status = "failed"
    if telemetry["violations"]:
        status = "failed_safety"
    record = {
        "schema_version": 1,
        "artifact_type": "stage1a_small_model_mps_worker_attempt",
        "status": status,
        "stage": arguments.stage,
        "runtime": {
            "backend": BACKEND,
            "device": DEVICE,
            "dtype": DTYPE,
            "fallback_enabled": False,
            "offline_execution": True,
        },
        "assets": {
            "model_identifier": MODEL_IDENTIFIER,
            "model_revision": MODEL_REVISION,
            "transcoder_identifier": TRANSCODER_IDENTIFIER,
            "transcoder_revision": TRANSCODER_REVISION,
            "transcoder_subfolder": TRANSCODER_SUBFOLDER,
        },
        "outcome": outcome,
        "telemetry": telemetry,
        "error": error,
    }
    output = arguments.output
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    generated = (REPOSITORY_ROOT / config["artifacts"]["generated_directory"]).resolve()
    if not output.parent.resolve().is_relative_to(generated):
        raise RuntimeError("worker output must remain under generated directory")
    write_json_atomic(output, record)
    print(json.dumps({"status": status, "stage": arguments.stage}, sort_keys=True))
    return 0 if status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
