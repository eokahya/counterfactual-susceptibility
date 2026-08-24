#!/usr/bin/env python3
"""Fresh-process workers for gated Stage 1A-S-BF16 empirical stages."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import platform
import re
import signal
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
from cfsus.reproduction.small_model_mps_bf16 import (  # noqa: E402
    BACKEND,
    CONFIG_PATH,
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
    assert_module_mps_bf16,
    assert_mps_bf16_tensor,
    feature_selection_audit_from_graph,
    intervention_mapping_bf16,
    layerwise_jumprelu_reference,
    load_bf16_config,
    normalized_l2,
    projected_graph_bytes,
    select_feature_from_graph,
    tensor_summary,
    within_bf16_ulps,
)

STAGES = (
    "model_forward",
    "fp32_reference",
    "loaded_semantics",
    "full_plt",
    "replacement_runtime",
    "smoke",
    "accepted",
)
MODEL_FP32_BUFFER_EXCEPTIONS = frozenset(
    {"model.rotary_emb.inv_freq", "model.rotary_emb_local.inv_freq"}
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / CONFIG_PATH)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison-file", type=Path, required=True)
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


def _git_identity() -> dict[str, Any]:
    head = _run_text(["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"])
    branch = _run_text(["git", "-C", str(REPOSITORY_ROOT), "branch", "--show-current"])
    status = _run_text(["git", "-C", str(REPOSITORY_ROOT), "status", "--porcelain"])
    if re.fullmatch(r"[0-9a-f]{40}", head) is None or not branch:
        raise RuntimeError("worker Git identity is invalid")
    return {
        "execution_commit": head,
        "branch": branch,
        "working_tree_clean": not status,
    }


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
    """Collect stage/attempt counters and self-terminate on MPS safety breach."""

    def __init__(
        self,
        torch: Any,
        limits: Mapping[str, Any],
        emergency_path: Path,
        *,
        use_mps: bool,
    ) -> None:
        self.torch = torch
        self.limits = limits
        self.emergency_path = emergency_path
        self.use_mps = use_mps
        self.interval = float(limits["sample_interval_seconds"])
        self.started_at = time.time()
        self.swap_start = _swap_used_bytes()
        self._stage = "worker_start"
        self._samples: list[Sample] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._emergency_sent = False
        self._telemetry_failures = 0
        self.sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        with self._lock:
            self._stage = name
        self.sample()
        try:
            yield
        finally:
            self.sample()

    def _violations(self, sample: Sample) -> list[str]:
        violations: list[str] = []
        if sample.mps_driver_bytes > int(self.limits["maximum_mps_driver_bytes"]):
            violations.append("maximum_mps_driver_bytes")
        if sample.process_rss_bytes > int(self.limits["maximum_process_rss_bytes"]):
            violations.append("maximum_process_rss_bytes")
        if sample.swap_used_bytes - self.swap_start > int(
            self.limits["maximum_swap_growth_bytes"]
        ):
            violations.append("maximum_swap_growth_bytes")
        if sample.available_memory_bytes < int(
            self.limits["minimum_available_memory_bytes"]
        ):
            violations.append("minimum_available_memory_bytes")
        if sample.thermal_state not in set(self.limits["accepted_thermal_states"]):
            violations.append("accepted_thermal_states")
        return violations

    def _emergency(self, violations: list[str], detail: str | None = None) -> None:
        if self._emergency_sent:
            return
        self._emergency_sent = True
        record = {
            "schema_version": 1,
            "artifact_type": "stage1a_small_model_mps_bf16_worker_emergency",
            "violations": violations,
            "detail_class": detail,
            "sample_count": len(self._samples),
            "last_sample": asdict(self._samples[-1]) if self._samples else None,
        }
        try:
            write_json_atomic(self.emergency_path, record)
        finally:
            os.kill(os.getpid(), signal.SIGTERM)

    def sample(self) -> None:
        import psutil

        with self._lock:
            stage = self._stage
        mps_current = 0
        mps_driver = 0
        if self.use_mps:
            mps_current = int(self.torch.mps.current_allocated_memory())
            mps_driver = int(self.torch.mps.driver_allocated_memory())
        sample = Sample(
            unix_time=time.time(),
            stage=stage,
            mps_current_bytes=mps_current,
            mps_driver_bytes=mps_driver,
            process_rss_bytes=int(psutil.Process().memory_info().rss),
            available_memory_bytes=int(psutil.virtual_memory().available),
            swap_used_bytes=_swap_used_bytes(),
            thermal_state=_thermal_state(),
        )
        with self._lock:
            self._samples.append(sample)
        violations = self._violations(sample)
        if violations:
            self._emergency(violations)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.sample()
                self._telemetry_failures = 0
            except BaseException as error:
                self._telemetry_failures += 1
                if self._telemetry_failures >= int(
                    self.limits["telemetry_failure_limit"]
                ):
                    self._emergency(["telemetry_failure_limit"], type(error).__name__)

    def finish(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=5)
        self.sample()
        with self._lock:
            samples = list(self._samples)
        if not samples:
            raise RuntimeError("telemetry sampler produced no samples")
        peak_metrics = (
            "mps_current_bytes",
            "mps_driver_bytes",
            "process_rss_bytes",
            "swap_used_bytes",
        )
        stage_peaks: dict[str, dict[str, int]] = {}
        for stage in sorted({item.stage for item in samples}):
            selected = [item for item in samples if item.stage == stage]
            stage_peaks[stage] = {
                metric: max(int(getattr(item, metric)) for item in selected)
                for metric in peak_metrics
            }
            stage_peaks[stage]["swap_growth_bytes"] = max(
                0, stage_peaks[stage]["swap_used_bytes"] - self.swap_start
            )
            stage_peaks[stage]["minimum_available_memory_bytes"] = min(
                item.available_memory_bytes for item in selected
            )
        attempt_peaks = {
            metric: max(int(getattr(item, metric)) for item in samples)
            for metric in peak_metrics
        }
        attempt_peaks["swap_growth_bytes"] = max(
            0, attempt_peaks["swap_used_bytes"] - self.swap_start
        )
        attempt_peaks["minimum_available_memory_bytes"] = min(
            item.available_memory_bytes for item in samples
        )
        for values in stage_peaks.values():
            for metric in (
                "mps_current_bytes",
                "mps_driver_bytes",
                "process_rss_bytes",
                "swap_used_bytes",
                "swap_growth_bytes",
            ):
                if attempt_peaks[metric] < values[metric]:
                    raise RuntimeError("attempt peak does not dominate stage peak")
            if (
                attempt_peaks["minimum_available_memory_bytes"]
                > values["minimum_available_memory_bytes"]
            ):
                raise RuntimeError("attempt memory minimum does not dominate stage")
        thermal_states = sorted({item.thermal_state for item in samples})
        violations: list[str] = []
        for sample in samples:
            violations.extend(self._violations(sample))
        return {
            "started_at_unix": self.started_at,
            "finished_at_unix": time.time(),
            "sample_count": len(samples),
            "sampling_interval_seconds": self.interval,
            "swap_start_bytes": self.swap_start,
            "attempt_peaks": attempt_peaks,
            "stage_peaks": stage_peaks,
            "thermal_states": thermal_states,
            "violations": sorted(set(violations)),
            "telemetry_failures": self._telemetry_failures,
        }


def _resolve_snapshots(config: Mapping[str, Any], cache: Path) -> tuple[Path, Path]:
    if not cache.is_absolute() or cache.is_symlink() or not cache.is_dir():
        raise RuntimeError("external immutable cache is missing or unsafe")
    cache = cache.resolve(strict=True)
    if cache == REPOSITORY_ROOT or cache.is_relative_to(REPOSITORY_ROOT):
        raise RuntimeError("cache overlaps repository")
    model = cache / "models--google--gemma-3-270m" / "snapshots" / MODEL_REVISION
    transcoder = (
        cache
        / "models--mwhanna--gemma-scope-2-270m-pt"
        / "snapshots"
        / TRANSCODER_REVISION
    )
    for snapshot, revision in (
        (model, MODEL_REVISION),
        (transcoder, TRANSCODER_REVISION),
    ):
        if snapshot.name != revision or snapshot.is_symlink() or not snapshot.is_dir():
            raise RuntimeError("exact immutable snapshot is unavailable")
    for snapshot, expected in (
        (model, set(config["model"]["allow_patterns"])),
        (transcoder, set(config["transcoder"]["allow_patterns"])),
    ):
        observed = {
            candidate.relative_to(snapshot).as_posix()
            for candidate in snapshot.rglob("*")
            if not candidate.is_dir()
        }
        if observed != expected:
            raise RuntimeError("exact immutable snapshot allowlist is incomplete")
        for relative in expected:
            candidate = snapshot / relative
            target = candidate.resolve(strict=True)
            if not target.is_file() or not target.is_relative_to(cache):
                raise RuntimeError("snapshot asset escapes the authorized cache")
    return model, transcoder


def _validate_output_path(path: Path, generated_directory: str) -> Path:
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    if path.is_symlink():
        raise RuntimeError("worker output must not be a symlink")
    generated = (REPOSITORY_ROOT / generated_directory).resolve()
    if not path.parent.resolve().is_relative_to(generated):
        raise RuntimeError("worker output must stay in ignored generated directory")
    return path


def _validate_comparison_path(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise RuntimeError("comparison file must be a non-linked absolute temp path")
    parent = path.parent.resolve(strict=True)
    private_tmp = Path("/private/tmp").resolve(strict=True)
    if not parent.is_relative_to(private_tmp) or not parent.name.startswith(
        "stage1a-bf16-"
    ):
        raise RuntimeError("comparison file must stay in owned private temp directory")
    return path


def _tokenize_prompt(tokenizer: Any, record: Mapping[str, Any], torch: Any) -> Any:
    if record["kind"] == "bos_only":
        if tokenizer.bos_token_id is None:
            raise RuntimeError("tokenizer has no BOS token")
        return torch.tensor(
            [[int(tokenizer.bos_token_id)]], device="mps", dtype=torch.long
        )
    encoded = tokenizer(str(record["text"]), return_tensors="pt")
    return encoded.input_ids.to(device="mps", dtype=torch.long)


def _compact_logits(logits: Any, tokenizer: Any, torch: Any) -> dict[str, Any]:
    assert_mps_bf16_tensor(logits, torch, "model logits")
    last = logits[0, -1]
    probabilities = torch.softmax(last.float(), dim=-1)
    values, indices = torch.topk(probabilities, 10)
    selected_logits = last[indices]
    token_ids = [int(item) for item in indices.detach().cpu().tolist()]
    return {
        "diagnostics": tensor_summary(logits, torch),
        "top_token_ids": token_ids,
        "top_tokens": [tokenizer.decode([item]) for item in token_ids],
        "top_probabilities": [float(item) for item in values.detach().cpu().tolist()],
        "top_logits": [float(item) for item in selected_logits.detach().cpu().tolist()],
    }


def _model_forward(
    config: Mapping[str, Any],
    model_snapshot: Path,
    sampler: TelemetrySampler,
    comparison_path: Path,
    torch: Any,
) -> dict[str, Any]:
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with sampler.stage("model_loading"):
        tokenizer = AutoTokenizer.from_pretrained(model_snapshot, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_snapshot,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": "mps"},
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).eval()
        module_record = assert_module_mps_bf16(
            model,
            torch,
            allowed_fp32_buffer_names=MODEL_FP32_BUFFER_EXCEPTIONS,
        )
        if torch.is_autocast_enabled():
            raise RuntimeError("outer autocast is unexpectedly enabled")
        model.config.use_cache = False

    prompt_results: list[dict[str, Any]] = []
    comparison_arrays: dict[str, Any] = {}
    for prompt_record in config["prompts"]:
        prompt_id = str(prompt_record["id"])
        feedforward_residuals: dict[int, dict[str, Any]] = {}
        feedforward_outputs: dict[int, dict[str, Any]] = {}
        layer_outputs: dict[int, dict[str, Any]] = {}
        handles: list[Any] = []

        for layer_index, layer in enumerate(model.model.layers):

            def capture_ff_input(
                _module: Any,
                inputs: tuple[Any, ...],
                *,
                index: int = layer_index,
                records: dict[int, dict[str, Any]] = feedforward_residuals,
            ) -> None:
                value = inputs[0]
                assert_mps_bf16_tensor(value, torch, f"layer {index} FF residual")
                records[index] = {
                    "summary": tensor_summary(value, torch),
                    "coordinate_0_0_163": float(value[0, 0, 163].item()),
                }

            def capture_ff_output(
                _module: Any,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer_index,
                records: dict[int, dict[str, Any]] = feedforward_outputs,
            ) -> None:
                assert_mps_bf16_tensor(output, torch, f"layer {index} FF output")
                records[index] = {
                    "summary": tensor_summary(output, torch),
                    "coordinate_0_0_163": float(output[0, 0, 163].item()),
                }

            def capture_layer_output(
                _module: Any,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer_index,
                records: dict[int, dict[str, Any]] = layer_outputs,
            ) -> None:
                value = output[0]
                assert_mps_bf16_tensor(value, torch, f"layer {index} output")
                records[index] = {
                    "summary": tensor_summary(value, torch),
                    "coordinate_0_0_163": float(value[0, 0, 163].item()),
                }

            handles.append(
                layer.pre_feedforward_layernorm.register_forward_pre_hook(
                    capture_ff_input
                )
            )
            handles.append(
                layer.post_feedforward_layernorm.register_forward_hook(
                    capture_ff_output
                )
            )
            handles.append(layer.register_forward_hook(capture_layer_output))

        try:
            with sampler.stage(f"model_forward_{prompt_id}"), torch.inference_mode():
                input_ids = _tokenize_prompt(tokenizer, prompt_record, torch)
                outputs = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                torch.mps.synchronize()
        finally:
            for handle in handles:
                handle.remove()
        if len(layer_outputs) != 18 or len(feedforward_outputs) != 18:
            raise RuntimeError("layerwise hook coverage is incomplete")
        hidden_records: list[dict[str, Any]] = []
        for index, hidden in enumerate(outputs.hidden_states):
            assert_mps_bf16_tensor(hidden, torch, f"hidden state {index}")
            hidden_records.append({"index": index, **tensor_summary(hidden, torch)})
        logits = outputs.logits
        compact = _compact_logits(logits, tokenizer, torch)
        layer7_sum = float(feedforward_residuals[7]["coordinate_0_0_163"]) + float(
            feedforward_outputs[7]["coordinate_0_0_163"]
        )
        layer7_observed = float(layer_outputs[7]["coordinate_0_0_163"])
        if not math.isfinite(layer7_sum) or not math.isfinite(layer7_observed):
            raise RuntimeError("prior FP16 failure coordinate remains non-finite")
        prompt_results.append(
            {
                "prompt_id": prompt_id,
                "kind": prompt_record["kind"],
                "text": prompt_record["text"],
                "token_ids": [
                    int(item) for item in input_ids[0].detach().cpu().tolist()
                ],
                "decoded_tokens": [
                    tokenizer.decode([int(item)])
                    for item in input_ids[0].detach().cpu().tolist()
                ],
                "input_device": str(input_ids.device),
                "hidden_states": hidden_records,
                "decoder_layers": [
                    {
                        "layer": layer,
                        "feedforward_residual": feedforward_residuals[layer],
                        "feedforward_post_norm": feedforward_outputs[layer],
                        "output": layer_outputs[layer],
                    }
                    for layer in range(18)
                ],
                "prior_fp16_failure_coordinate": {
                    "layer": 7,
                    "coordinate": [0, 0, 163],
                    "residual_operand": feedforward_residuals[7]["coordinate_0_0_163"],
                    "feedforward_operand": feedforward_outputs[7]["coordinate_0_0_163"],
                    "fp32_scalar_sum_for_diagnostic": layer7_sum,
                    "observed_bf16_layer_output": layer7_observed,
                    "finite": True,
                },
                "logits": compact,
            }
        )
        comparison_arrays[f"token_ids__{prompt_id}"] = (
            input_ids[0].detach().cpu().numpy()
        )
        comparison_arrays[f"logits__{prompt_id}"] = (
            logits[0, -1].detach().cpu().float().numpy()
        )
        comparison_arrays[f"layer7__{prompt_id}"] = (
            outputs.hidden_states[8][0, -1].detach().cpu().float().numpy()
        )
        comparison_arrays[f"final_hidden__{prompt_id}"] = (
            outputs.hidden_states[-1][0, -1].detach().cpu().float().numpy()
        )
        del outputs, logits, input_ids
    np.savez_compressed(comparison_path, **comparison_arrays)
    if not comparison_path.is_file() or comparison_path.stat().st_size <= 0:
        raise RuntimeError("temporary MPS comparison vectors were not written")
    del model
    gc.collect()
    torch.mps.empty_cache()
    return {
        "status": "passed",
        "runtime": {
            "backend": BACKEND,
            "device": DEVICE,
            "dtype": DTYPE,
            "fallback_enabled": False,
            "outer_autocast_enabled": False,
            "source_mandated_internal_fp32": config["runtime"][
                "source_mandated_internal_fp32"
            ],
        },
        "module_guard": module_record,
        "prompts": prompt_results,
        "temporary_comparison_vectors": {
            "location": "project_external_private_temp",
            "persisted_in_repository": False,
            "bytes": comparison_path.stat().st_size,
        },
    }


def _pilot_prompt(config: Mapping[str, Any], profile: str = "smoke") -> str:
    prompt_id = str(config[profile]["prompt_id"])
    matches = [record for record in config["prompts"] if record["id"] == prompt_id]
    if len(matches) != 1 or matches[0]["kind"] != "text":
        raise RuntimeError("pilot prompt identity is invalid")
    return str(matches[0]["text"])


def _tokenize_text(tokenizer: Any, prompt: str, torch: Any) -> Any:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
        device="mps", dtype=torch.long
    )
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise RuntimeError("tokenizer produced an invalid input shape")
    return input_ids


def _loaded_semantics(
    config: Mapping[str, Any],
    model_snapshot: Path,
    transcoder_snapshot: Path,
    sampler: TelemetrySampler,
    torch: Any,
) -> dict[str, Any]:
    from circuit_tracer.transcoder.single_layer_transcoder import load_transcoder
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompt = _pilot_prompt(config)
    with sampler.stage("model_loading"):
        tokenizer = AutoTokenizer.from_pretrained(model_snapshot, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_snapshot,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": "mps"},
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).eval()
        assert_module_mps_bf16(
            model,
            torch,
            allowed_fp32_buffer_names=MODEL_FP32_BUFFER_EXCEPTIONS,
        )
        model.config.use_cache = False

    captured: list[Any] = []

    def capture(
        _module: Any,
        _inputs: Any,
        output: Any,
        records: list[Any] = captured,
    ) -> None:
        records.append(output.detach())

    hook = model.model.layers[0].pre_feedforward_layernorm.register_forward_hook(
        capture
    )
    try:
        with sampler.stage("model_forward"), torch.inference_mode():
            input_ids = _tokenize_text(tokenizer, prompt, torch)
            logits = model(input_ids=input_ids, use_cache=False).logits
            torch.mps.synchronize()
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError("Gemma 3 layer-0 PLT input hook coverage failed")
    mlp_input = captured[0].squeeze(0)
    assert_mps_bf16_tensor(mlp_input, torch, "loaded layer-0 MLP input")

    path = transcoder_snapshot / TRANSCODER_SUBFOLDER / "layer_0.safetensors"
    with sampler.stage("one_layer_plt_loading"):
        transcoder = load_transcoder(
            str(path),
            layer=0,
            device=torch.device("mps"),
            dtype=torch.bfloat16,
            lazy_encoder=False,
            lazy_decoder=False,
        )
        module_guard = assert_module_mps_bf16(transcoder, torch)

    with sampler.stage("loaded_semantics"), torch.inference_mode():
        preactivation = torch.nn.functional.linear(
            mlp_input, transcoder.W_enc, transcoder.b_enc
        )
        activation = transcoder.activation_function(preactivation)
        threshold = transcoder.activation_function.threshold
        expected = preactivation * (preactivation > threshold)
        equality = transcoder.activation_function(threshold)
        reconstruction = transcoder.decode(activation, mlp_input)
        independent_reconstruction = activation @ transcoder.W_dec + transcoder.b_dec
        for label, tensor in (
            ("preactivation", preactivation),
            ("activation", activation),
            ("threshold", threshold),
            ("reconstruction", reconstruction),
        ):
            assert_mps_bf16_tensor(tensor, torch, label)
        if not bool(torch.equal(activation, expected)):
            raise RuntimeError("loaded JumpReLU differs from the strict BF16 reference")
        if not bool(torch.equal(equality, torch.zeros_like(equality))):
            raise RuntimeError("loaded JumpReLU equality is not inactive")
        if not bool(torch.equal(reconstruction, independent_reconstruction)):
            raise RuntimeError("loaded PLT reconstruction differs from its reference")

        final_position = int(activation.shape[0] - 1)
        final_preactivation = preactivation[final_position]
        final_activation = activation[final_position]
        margins = final_preactivation - threshold
        active_ids = torch.nonzero(final_activation > 0, as_tuple=False).flatten()
        inactive_ids = torch.nonzero(final_activation == 0, as_tuple=False).flatten()
        if active_ids.numel() < 1 or inactive_ids.numel() < 1:
            raise RuntimeError("loaded semantics lacks active or inactive evidence")
        active_id = int(active_ids[torch.argmin(torch.abs(margins[active_ids]))].item())
        inactive_id = int(
            inactive_ids[torch.argmin(torch.abs(margins[inactive_ids]))].item()
        )

        shadows: dict[str, dict[str, Any]] = {}
        for label, feature in (
            ("closest_active", active_id),
            ("closest_inactive", inactive_id),
        ):
            shadow = float(
                (
                    torch.dot(
                        mlp_input[final_position].detach().cpu().float(),
                        transcoder.W_enc[feature].detach().cpu().float(),
                    )
                    + transcoder.b_enc[feature].detach().cpu().float()
                ).item()
            )
            observed = float(final_preactivation[feature].item())
            within = within_bf16_ulps(
                observed,
                shadow,
                int(config["tolerances"]["accumulated_value_bf16_ulps"]),
            )
            if not within:
                raise RuntimeError(
                    "loaded preactivation exceeds its frozen FP32 shadow"
                )
            shadows[label] = {
                "feature": feature,
                "bf16_preactivation": observed,
                "fp32_shadow_preactivation": shadow,
                "within_frozen_bf16_ulps": within,
            }
        torch.mps.synchronize()

    result = {
        "status": "passed",
        "layer": 0,
        "position": final_position,
        "prompt": prompt,
        "token_ids": [int(item) for item in input_ids[0].detach().cpu().tolist()],
        "module_guard": module_guard,
        "encoder_shape": [int(size) for size in transcoder.W_enc.shape],
        "decoder_shape": [int(size) for size in transcoder.W_dec.shape],
        "threshold_shape": [int(size) for size in threshold.shape],
        "tensor_device": str(transcoder.b_enc.device),
        "tensor_dtype": str(transcoder.b_enc.dtype),
        "storage_dtype": "torch.float32",
        "runtime_cast": "explicit_load_to_torch.bfloat16",
        "bias_convention": (
            "preactivation=linear(mlp_hook_in,W_enc,b_enc); decode=a@W_dec+b_dec"
        ),
        "gate_rule": "preactivation * (preactivation > threshold)",
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
        "fp32_selected_value_shadows": shadows,
        "maximum_gate_discrepancy": 0.0,
        "maximum_reconstruction_discrepancy": 0.0,
        "nonfinite_count": 0,
        "model_logits": _compact_logits(logits, tokenizer, torch),
    }
    del logits, model, transcoder, captured
    gc.collect()
    torch.mps.empty_cache()
    return result


def _load_transcoder_set(transcoder_snapshot: Path, torch: Any) -> Any:
    from circuit_tracer.transcoder.single_layer_transcoder import load_transcoder_set

    from cfsus.reproduction.small_model_mps_bf16_runtime import (
        MPSBF16TranscoderSet,
    )

    root = transcoder_snapshot / TRANSCODER_SUBFOLDER
    paths = {
        layer: str(root / f"layer_{layer}.safetensors") for layer in range(LAYER_COUNT)
    }
    if any(not Path(path).is_file() for path in paths.values()):
        raise RuntimeError("one or more exact PLT layer files are missing")
    source = load_transcoder_set(
        paths,
        scan_name=(
            f"{TRANSCODER_IDENTIFIER}/{TRANSCODER_SUBFOLDER}@{TRANSCODER_REVISION}"
        ),
        feature_input_hook="mlp.hook_in",
        feature_output_hook="hook_mlp_out",
        device=torch.device("mps"),
        dtype=torch.bfloat16,
        lazy_encoder=True,
        lazy_decoder=True,
    )
    transcoders = MPSBF16TranscoderSet(source)
    if len(transcoders) != LAYER_COUNT or transcoders.d_transcoder != FEATURE_WIDTH:
        raise RuntimeError("loaded PLT set dimensions are incorrect")
    return transcoders


def _full_plt(
    transcoder_snapshot: Path, sampler: TelemetrySampler, torch: Any
) -> dict[str, Any]:
    with sampler.stage("full_plt_loading"):
        started = time.perf_counter()
        transcoders = _load_transcoder_set(transcoder_snapshot, torch)
        load_seconds = time.perf_counter() - started
        module_guard = assert_module_mps_bf16(transcoders, torch)
        decoder_sample = transcoders._get_decoder_vectors(
            0, torch.tensor([0], device="mps", dtype=torch.long)
        )
        encoder_sample = transcoders[0].W_enc[:1]
        threshold_sample = transcoders[0].activation_function.threshold[:1]
        for label, tensor in (
            ("lazy encoder sample", encoder_sample),
            ("lazy decoder sample", decoder_sample),
            ("threshold sample", threshold_sample),
        ):
            assert_mps_bf16_tensor(tensor, torch, label)
        torch.mps.synchronize()
    if any(
        not transcoder.lazy_encoder or not transcoder.lazy_decoder
        for transcoder in transcoders
    ):
        raise RuntimeError("PLT set is not fully lazy")
    result = {
        "status": "passed",
        "layer_count": len(transcoders),
        "feature_width_per_layer": int(transcoders.d_transcoder),
        "total_feature_width": len(transcoders) * int(transcoders.d_transcoder),
        "lazy_encoder": True,
        "lazy_decoder": True,
        "load_seconds": load_seconds,
        "module_guard": module_guard,
        "persistent_parameter_device": "mps",
        "persistent_parameter_dtypes": sorted(
            {str(parameter.dtype) for parameter in transcoders.parameters()}
        ),
        "lazy_encoder_sample_shape": [int(size) for size in encoder_sample.shape],
        "lazy_decoder_sample_shape": [int(size) for size in decoder_sample.shape],
        "source_storage": "project_external_immutable_float32_safetensors",
        "runtime_execution_dtype": "torch.bfloat16",
        "cpu_metadata_boundary_used": False,
        "runtime_monkeypatches": 0,
        "nonfinite_count": 0,
    }
    del transcoders, encoder_sample, decoder_sample, threshold_sample
    gc.collect()
    torch.mps.empty_cache()
    return result


def _replacement_module_guard(model: Any, torch: Any) -> dict[str, Any]:
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
    if len(observed) != 2:
        raise RuntimeError("replacement runtime FP32 source buffers changed")
    return assert_module_mps_bf16(
        model, torch, allowed_fp32_buffer_names=frozenset(observed)
    )


def _build_replacement(
    model_snapshot: Path,
    transcoder_snapshot: Path,
    sampler: TelemetrySampler,
    torch: Any,
) -> Any:
    from cfsus.reproduction.small_model_mps_bf16_runtime import (
        MPSBF16ReplacementModel,
    )

    with sampler.stage("full_plt_loading"):
        transcoders = _load_transcoder_set(transcoder_snapshot, torch)
    with sampler.stage("replacement_runtime_construction"):
        model = MPSBF16ReplacementModel.from_pretrained_and_transcoders(
            str(model_snapshot),
            transcoders,
            device=torch.device("mps"),
            dtype=torch.bfloat16,
        )
        guard = _replacement_module_guard(model, torch)
        if torch.is_autocast_enabled():
            raise RuntimeError("replacement runtime unexpectedly enabled autocast")
        torch.mps.synchronize()
    if model.backend != BACKEND or model.cfg.n_layers != LAYER_COUNT:
        raise RuntimeError("replacement runtime identity is incorrect")
    model._bf16_module_guard = guard
    return model


def _replacement_runtime(
    config: Mapping[str, Any],
    model_snapshot: Path,
    transcoder_snapshot: Path,
    sampler: TelemetrySampler,
    torch: Any,
) -> dict[str, Any]:
    model = _build_replacement(model_snapshot, transcoder_snapshot, sampler, torch)
    prompt = _pilot_prompt(config)
    with sampler.stage("replacement_feature_access"):
        logits, activations = model.get_activations(
            prompt, sparse=False, apply_activation_function=True
        )
        assert_mps_bf16_tensor(logits, torch, "replacement logits")
        assert_mps_bf16_tensor(activations, torch, "replacement activations")
        torch.mps.synchronize()
    if activations.shape[0] != LAYER_COUNT or activations.shape[2] != FEATURE_WIDTH:
        raise RuntimeError("replacement feature cache shape is incorrect")
    active_count = int((activations > 0).sum().item())
    inactive_count = int((activations == 0).sum().item())
    if active_count < 1 or inactive_count < 1:
        raise RuntimeError("replacement runtime lacks active or inactive features")
    result = {
        "status": "passed",
        "backend_class": type(model).__name__,
        "backend": model.backend,
        "device": str(model.device),
        "dtype": str(model.dtype),
        "layer_count": int(model.cfg.n_layers),
        "hidden_size": int(model.cfg.d_model),
        "module_guard": model._bf16_module_guard,
        "feature_cache_shape": [int(size) for size in activations.shape],
        "feature_cache_device": str(activations.device),
        "feature_cache_dtype": str(activations.dtype),
        "active_feature_count": active_count,
        "inactive_feature_count": inactive_count,
        "intervention_tuple_semantics": (
            "(layer, position, feature, absolute_post_gate_value)"
        ),
        "suppression_formula": "desired=(1-alpha)*baseline",
        "runtime_monkeypatches": 0,
        "model_logits": _compact_logits(logits, model.tokenizer, torch),
    }
    del activations, logits, model
    gc.collect()
    torch.mps.empty_cache()
    return result


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
    from cfsus.reproduction.small_model_mps_bf16_runtime import (
        attribute_mps_bf16,
    )

    settings = config[profile]
    prompt = _pilot_prompt(config, profile)
    model = _build_replacement(model_snapshot, transcoder_snapshot, sampler, torch)
    input_ids = model.ensure_tokenized(prompt)
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
    with sampler.stage("attribution"):
        graph, adapter_usage = attribute_mps_bf16(
            prompt,
            model,
            context=context,
            max_n_logits=int(settings["max_n_logits"]),
            desired_logit_probability=float(settings["desired_logit_probability"]),
            batch_size=batch_size,
            max_feature_nodes=int(settings["max_feature_nodes"]),
        )
        torch.mps.synchronize()
    adjacency = graph.adjacency_matrix.detach().cpu()
    if adjacency.numel() == 0 or not bool(torch.isfinite(adjacency).all().item()):
        raise RuntimeError("attribution graph is empty or non-finite")
    nonzero_edges = int(torch.count_nonzero(adjacency).item())
    if nonzero_edges < 1:
        raise RuntimeError("attribution graph has no nonzero edges")
    selection = select_feature_from_graph(graph, final_position=token_count - 1)
    selection_audit = feature_selection_audit_from_graph(
        graph, final_position=token_count - 1, selection=selection
    )

    with sampler.stage("loaded_semantics"):
        raw_logits, preactivations = model.get_activations(
            prompt, sparse=False, apply_activation_function=False
        )
        _, activations = model.get_activations(
            prompt, sparse=False, apply_activation_function=True
        )
        for label, tensor in (
            ("raw replacement logits", raw_logits),
            ("replacement preactivations", preactivations),
            ("replacement activations", activations),
        ):
            assert_mps_bf16_tensor(tensor, torch, label)
        transcoder_set = model.transcoders._module
        thresholds = torch.stack(
            [transcoder.activation_function.threshold for transcoder in transcoder_set]
        )
        assert_mps_bf16_tensor(thresholds, torch, "all loaded thresholds")
        threshold = thresholds[selection.layer]
        assert_mps_bf16_tensor(threshold, torch, "selected loaded threshold")
        expected = layerwise_jumprelu_reference(preactivations, thresholds, torch)
        if not bool(torch.equal(expected, activations)):
            raise RuntimeError("replacement loaded gate differs from strict reference")
        equality = torch.stack(
            [
                transcoder.activation_function(thresholds[layer])
                for layer, transcoder in enumerate(transcoder_set)
            ]
        )
        if not bool(torch.equal(equality, torch.zeros_like(equality))):
            raise RuntimeError("replacement threshold equality is not inactive")
        selected_tensor = activations[
            selection.layer, selection.position, selection.feature
        ].reshape(())
        selected_activation = float(selected_tensor.item())
        if not within_bf16_ulps(
            selected_activation,
            selection.baseline_activation,
            int(config["tolerances"]["gate_value_bf16_ulps"]),
        ):
            raise RuntimeError("graph and loaded selected activation disagree")
        inactive_ids = torch.nonzero(
            activations[selection.layer, selection.position] == 0,
            as_tuple=False,
        ).flatten()
        if inactive_ids.numel() < 1:
            raise RuntimeError("selected position lacks an inactive feature")
        inactive_feature = int(inactive_ids[0].item())

    baseline_mapping = intervention_mapping_bf16(selected_tensor, 0.0, torch)
    baseline_tuple = [
        (
            selection.layer,
            selection.position,
            selection.feature,
            baseline_mapping["tensor"],
        )
    ]
    with sampler.stage("intervention_baseline"):
        baseline_logits, _ = model.feature_intervention(
            prompt,
            baseline_tuple,
            freeze_attention=bool(settings["freeze_attention"]),
            return_activations=False,
        )
        repeat_logits, _ = model.feature_intervention(
            prompt,
            baseline_tuple,
            freeze_attention=bool(settings["freeze_attention"]),
            return_activations=False,
        )
        assert_mps_bf16_tensor(baseline_logits, torch, "intervention baseline logits")
        assert_mps_bf16_tensor(repeat_logits, torch, "baseline repeat logits")

    baseline_last = baseline_logits[0, -1]
    tolerance = float(config["tolerances"]["baseline_noop_normalized_l2_maximum"])
    maximum_absolute_tolerance = float(
        config["tolerances"]["baseline_noop_maximum_absolute_logit_difference"]
    )
    repeat_error = normalized_l2(baseline_last, repeat_logits[0, -1], torch)
    raw_error = normalized_l2(baseline_last, raw_logits[0, -1], torch)
    repeat_difference = torch.abs(repeat_logits[0, -1] - baseline_last)
    raw_difference = torch.abs(raw_logits[0, -1] - baseline_last)
    assert_mps_bf16_tensor(repeat_difference, torch, "baseline repeat difference")
    assert_mps_bf16_tensor(raw_difference, torch, "raw baseline difference")
    repeat_maximum_absolute = float(torch.max(repeat_difference).item())
    raw_maximum_absolute = float(torch.max(raw_difference).item())
    if (
        repeat_error > tolerance
        or raw_error > tolerance
        or repeat_maximum_absolute > maximum_absolute_tolerance
        or raw_maximum_absolute > maximum_absolute_tolerance
    ):
        raise RuntimeError("baseline repeat consistency exceeded frozen tolerance")

    conditions: list[dict[str, Any]] = []
    for alpha_value in settings["intervention_alphas"]:
        mapping = intervention_mapping_bf16(selected_tensor, float(alpha_value), torch)
        sent = mapping["tensor"]
        with sampler.stage(f"intervention_alpha_{mapping['alpha']}"):
            logits, _ = model.feature_intervention(
                prompt,
                [
                    (
                        selection.layer,
                        selection.position,
                        selection.feature,
                        sent,
                    )
                ],
                freeze_attention=bool(settings["freeze_attention"]),
                return_activations=False,
            )
            assert_mps_bf16_tensor(logits, torch, "intervention logits")
            torch.mps.synchronize()
        condition_error = normalized_l2(baseline_last, logits[0, -1], torch)
        condition_difference = torch.abs(logits[0, -1] - baseline_last)
        assert_mps_bf16_tensor(
            condition_difference, torch, "intervention logit difference"
        )
        condition_maximum_absolute = float(torch.max(condition_difference).item())
        conditions.append(
            {
                "alpha": mapping["alpha"],
                "baseline_activation": mapping["baseline_activation"],
                "desired_absolute_activation": mapping["desired_absolute_activation"],
                "sent_absolute_activation": float(sent.item()),
                "sent_device": str(sent.device),
                "sent_dtype": str(sent.dtype),
                "observed_post_intervention_activation": None,
                "observed_post_intervention_activation_accessible": False,
                "normalized_l2_from_baseline": condition_error,
                "maximum_absolute_logit_difference_from_baseline": (
                    condition_maximum_absolute
                ),
                "logits": _compact_logits(logits, model.tokenizer, torch),
            }
        )
    noop = next(condition for condition in conditions if condition["alpha"] == 0.0)
    if float(noop["normalized_l2_from_baseline"]) > tolerance:
        raise RuntimeError("no-op consistency exceeded frozen tolerance")
    if (
        float(noop["maximum_absolute_logit_difference_from_baseline"])
        > maximum_absolute_tolerance
    ):
        raise RuntimeError("no-op maximum absolute difference exceeded tolerance")

    result = {
        "status": "passed",
        "profile": profile,
        "batch_size": batch_size,
        "prompt": prompt,
        "token_ids": [int(item) for item in input_ids.detach().cpu().tolist()],
        "token_count": token_count,
        "module_guard": model._bf16_module_guard,
        "attribution": {
            "active_feature_count": active_count,
            "selected_feature_count": len(graph.selected_features),
            "adjacency_shape": [int(size) for size in adjacency.shape],
            "nonzero_edge_count": nonzero_edges,
            "logit_node_count": len(graph.logit_targets),
            "error_node_count": LAYER_COUNT * token_count,
            "input_node_count": token_count,
            "nonfinite_count": 0,
            "projected_graph_bytes": graph_bytes,
            "adapter_usage": adapter_usage,
            "raw_graph_persisted": False,
            "graph_metadata_device": "cpu",
            "scientific_tensor_device": "mps",
            "scientific_tensor_dtype": "torch.bfloat16",
        },
        "selection": asdict(selection),
        "selection_audit": selection_audit,
        "loaded_semantics": {
            "selected_preactivation": float(
                preactivations[
                    selection.layer, selection.position, selection.feature
                ].item()
            ),
            "selected_threshold": float(threshold[selection.feature].item()),
            "selected_activation": selected_activation,
            "inactive_feature": inactive_feature,
            "inactive_preactivation": float(
                preactivations[
                    selection.layer, selection.position, inactive_feature
                ].item()
            ),
            "inactive_threshold": float(threshold[inactive_feature].item()),
            "inactive_activation": 0.0,
            "threshold_equality_inactive": True,
            "maximum_gate_discrepancy": 0.0,
            "nonfinite_count": 0,
        },
        "intervention": {
            "raw_baseline": _compact_logits(raw_logits, model.tokenizer, torch),
            "baseline": _compact_logits(baseline_logits, model.tokenizer, torch),
            "baseline_repeat": _compact_logits(repeat_logits, model.tokenizer, torch),
            "raw_to_frozen_baseline_normalized_l2": raw_error,
            "baseline_repeat_normalized_l2": repeat_error,
            "raw_to_frozen_baseline_maximum_absolute_logit_difference": (
                raw_maximum_absolute
            ),
            "baseline_repeat_maximum_absolute_logit_difference": (
                repeat_maximum_absolute
            ),
            "baseline_noop_normalized_l2_tolerance": tolerance,
            "baseline_noop_maximum_absolute_logit_difference_tolerance": (
                maximum_absolute_tolerance
            ),
            "conditions": conditions,
            "suppression_formula": "desired=(1-alpha)*baseline",
            "freeze_attention": bool(settings["freeze_attention"]),
            "runtime_monkeypatches": 0,
        },
    }
    del graph, adjacency, model, context, preactivations, activations
    gc.collect()
    torch.mps.empty_cache()
    return result


def _fp32_reference(
    config: Mapping[str, Any],
    model_snapshot: Path,
    sampler: TelemetrySampler,
    comparison_path: Path,
    torch: Any,
) -> dict[str, Any]:
    import numpy as np
    from transformers import AutoModelForCausalLM

    if not comparison_path.is_file() or comparison_path.is_symlink():
        raise RuntimeError("MPS comparison vectors are missing or unsafe")
    vectors = np.load(comparison_path, allow_pickle=False)
    with sampler.stage("fp32_model_loading"):
        model = AutoModelForCausalLM.from_pretrained(
            model_snapshot,
            local_files_only=True,
            dtype=torch.float32,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).eval()
        model.config.use_cache = False
        if any(parameter.device.type != "cpu" for parameter in model.parameters()):
            raise RuntimeError("FP32 diagnostic model left CPU")
        if any(
            parameter.is_floating_point() and parameter.dtype != torch.float32
            for parameter in model.parameters()
        ):
            raise RuntimeError("FP32 diagnostic model is not FP32")
    results: list[dict[str, Any]] = []
    tolerances = config["tolerances"]
    for prompt_record in config["prompts"]:
        prompt_id = str(prompt_record["id"])
        token_ids_np = vectors[f"token_ids__{prompt_id}"]
        mps_logits = torch.from_numpy(vectors[f"logits__{prompt_id}"]).float()
        mps_layer7 = torch.from_numpy(vectors[f"layer7__{prompt_id}"]).float()
        mps_final = torch.from_numpy(vectors[f"final_hidden__{prompt_id}"]).float()
        token_ids = torch.from_numpy(token_ids_np).long().unsqueeze(0)
        captured_layer7: list[Any] = []

        def capture_layer7(
            _module: Any,
            _inputs: Any,
            output: Any,
            records: list[Any] = captured_layer7,
        ) -> None:
            records.append(output[0].detach())

        handle = model.model.layers[7].register_forward_hook(capture_layer7)
        try:
            with sampler.stage(f"fp32_forward_{prompt_id}"), torch.inference_mode():
                outputs = model(
                    input_ids=token_ids,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            handle.remove()
        logits = outputs.logits[0, -1].float()
        if not bool(torch.isfinite(logits).all().item()):
            raise RuntimeError("FP32 diagnostic logits are non-finite")
        if len(captured_layer7) != 1:
            raise RuntimeError("FP32 layer-7 hook coverage failed")
        cpu_layer7 = captured_layer7[0][0, -1].float()
        cpu_final = outputs.hidden_states[-1][0, -1].float()
        cosine = float(
            torch.nn.functional.cosine_similarity(
                mps_logits.double(), logits.double(), dim=0
            ).item()
        )
        l2 = normalized_l2(mps_logits, logits, torch)
        mps_norm = float(torch.linalg.vector_norm(mps_logits.double()).item())
        cpu_norm = float(torch.linalg.vector_norm(logits.double()).item())
        norm_ratio = mps_norm / max(cpu_norm, 1e-30)
        mps_top = torch.topk(mps_logits, 10).indices.tolist()
        cpu_top = torch.topk(logits, 10).indices.tolist()
        overlap = len(set(mps_top) & set(cpu_top))
        top1 = int(mps_top[0]) == int(cpu_top[0])
        layer7_ratio = float(mps_layer7.abs().max().item()) / max(
            float(cpu_layer7.abs().max().item()), 1e-30
        )
        final_ratio = float(mps_final.abs().max().item()) / max(
            float(cpu_final.abs().max().item()), 1e-30
        )
        passed = (
            cosine >= float(tolerances["fp32_reference_cosine_minimum"])
            and l2 <= float(tolerances["fp32_reference_normalized_l2_maximum"])
            and float(tolerances["fp32_reference_norm_ratio_minimum"])
            <= norm_ratio
            <= float(tolerances["fp32_reference_norm_ratio_maximum"])
            and overlap >= int(tolerances["fp32_reference_top10_overlap_minimum"])
            and top1
            and float(tolerances["fp32_reference_magnitude_ratio_minimum"])
            <= layer7_ratio
            <= float(tolerances["fp32_reference_magnitude_ratio_maximum"])
            and float(tolerances["fp32_reference_magnitude_ratio_minimum"])
            <= final_ratio
            <= float(tolerances["fp32_reference_magnitude_ratio_maximum"])
        )
        result = {
            "prompt_id": prompt_id,
            "token_ids": [int(item) for item in token_ids_np.tolist()],
            "mps_bf16_finite": bool(torch.isfinite(mps_logits).all().item()),
            "cpu_fp32_finite": True,
            "final_logit_cosine_similarity": cosine,
            "normalized_l2_error": l2,
            "logit_norm_ratio_mps_over_cpu": norm_ratio,
            "top1_agreement": top1,
            "top10_overlap": overlap,
            "layer7_absmax_ratio_mps_over_cpu": layer7_ratio,
            "final_hidden_absmax_ratio_mps_over_cpu": final_ratio,
            "passed": passed,
        }
        if not passed:
            raise RuntimeError(f"FP32 diagnostic comparison failed for {prompt_id}")
        results.append(result)
        del outputs, logits, token_ids
    vectors.close()
    del model
    gc.collect()
    return {
        "status": "passed",
        "execution_class": "separate_cpu_fp32_diagnostic_only",
        "accepted_execution_fallback": False,
        "ran_after_mps_process_exit": True,
        "prompts": results,
        "frozen_tolerances": dict(tolerances),
    }


def _sanitize_error(error: BaseException) -> dict[str, str]:
    message = str(error).replace(str(REPOSITORY_ROOT), "[REPOSITORY]")
    message = re.sub(r"/Users/[^/\s]+/[^\s]*", "[PRIVATE_PATH]", message)
    return {"type": type(error).__name__, "message": message[:1000]}


def main() -> int:
    arguments = _parser().parse_args()
    assert_fallback_disabled()
    config = load_bf16_config(arguments.config)
    if platform.machine() != "arm64" or sys.version_info[:2] != (3, 11):
        raise RuntimeError("worker requires native arm64 CPython 3.11")
    git_identity = _git_identity()
    if arguments.stage == "accepted" and (
        git_identity["branch"] != "stage-1a-small-model-mps-bf16"
        or not git_identity["working_tree_clean"]
    ):
        raise RuntimeError("accepted worker requires the clean isolated BF16 branch")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    output = _validate_output_path(
        arguments.output, str(config["artifacts"]["generated_directory"])
    )
    comparison_path = _validate_comparison_path(arguments.comparison_file)
    model_snapshot, transcoder_snapshot = _resolve_snapshots(config, arguments.hf_cache)
    import torch

    use_mps = arguments.stage != "fp32_reference"
    if use_mps and (
        not torch.backends.mps.is_built() or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS is unavailable")
    sampler = TelemetrySampler(
        torch,
        config["safety_limits"],
        output.with_suffix(".emergency.json"),
        use_mps=use_mps,
    )
    error: dict[str, str] | None = None
    outcome: dict[str, Any]
    try:
        if arguments.stage == "model_forward":
            outcome = _model_forward(
                config, model_snapshot, sampler, comparison_path, torch
            )
        elif arguments.stage == "fp32_reference":
            outcome = _fp32_reference(
                config, model_snapshot, sampler, comparison_path, torch
            )
        elif arguments.stage == "loaded_semantics":
            outcome = _loaded_semantics(
                config,
                model_snapshot,
                transcoder_snapshot,
                sampler,
                torch,
            )
        elif arguments.stage == "full_plt":
            outcome = _full_plt(transcoder_snapshot, sampler, torch)
        elif arguments.stage == "replacement_runtime":
            outcome = _replacement_runtime(
                config,
                model_snapshot,
                transcoder_snapshot,
                sampler,
                torch,
            )
        else:
            settings = config[arguments.stage]
            allowed_batches = {
                int(value) for value in settings["attribution_batch_sizes"]
            }
            batch_size = arguments.batch_size
            if batch_size is None:
                batch_size = int(settings["attribution_batch_sizes"][0])
            if batch_size not in allowed_batches:
                raise RuntimeError("attribution batch size is outside frozen policy")
            outcome = _attribution_and_intervention(
                config,
                model_snapshot,
                transcoder_snapshot,
                sampler,
                torch,
                profile=arguments.stage,
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
        "artifact_type": "stage1a_small_model_mps_bf16_worker_attempt",
        "status": status,
        "stage": arguments.stage,
        "git": git_identity,
        "runtime": {
            "backend": BACKEND,
            "device": "mps" if use_mps else "cpu",
            "dtype": "bfloat16" if use_mps else "float32",
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
    write_json_atomic(output, record)
    print(json.dumps({"status": status, "stage": arguments.stage}, sort_keys=True))
    return 0 if status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
