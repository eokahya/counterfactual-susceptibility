"""Fail-closed primitives for Stage 1A-S-BF16 on native Apple MPS.

This experiment class is deliberately separate from the protected all-FP16
pilot. Optional empirical dependencies are imported only inside runtime
functions so ordinary offline tests stay lightweight.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import signal
import stat
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from cfsus.exceptions import ScientificInputError
from cfsus.reproduction.artifacts import ArtifactValidationError, sha256_file

EXPERIMENT_CLASS = "stage1a_small_model_mps_bf16_pilot"
COMPLETED_STATUS = "completed_small_model_mps_bf16_pilot"
EXECUTION_BASE_COMMIT = "3baf39a5ac81e172d11d22a6de332dee80a21079"
REQUIRED_BRANCH = "stage-1a-small-model-mps-bf16"

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
DTYPE = "bfloat16"
TORCH_DTYPE = "torch.bfloat16"
FALLBACK_VARIABLE = "PYTORCH_ENABLE_MPS_FALLBACK"
LAYER_COUNT = 18
FEATURE_WIDTH = 16_384
HIDDEN_SIZE = 640
VOCABULARY_SIZE = 262_144

CONFIG_PATH = "configs/stage1a_gemma3_270m_mps_bf16_pilot.yaml"
PROJECTED_MANIFEST = "configs/stage1a_small_model_projected_download.json"
ENVIRONMENT_LOCK = "environments/stage1a_small_model_mps_bf16/requirements-lock.txt"
RESULT_DIRECTORY = "results/stage1a_small_model_mps_bf16"
GENERATED_DIRECTORY = "results/generated/stage1a_small_model_mps_bf16"

SOURCE_MANDATED_INTERNAL_FP32 = (
    "gemma3_rmsnorm_accumulation_then_cast_back",
    "gemma3_rotary_frequency_trigonometry_then_cast_back",
    "gemma3_attention_softmax_then_cast_back",
)

ARTIFACT_ALLOWLIST = frozenset(
    {
        "environment_manifest.json",
        "asset_manifest.json",
        "preflight_summary.json",
        "operator_probe_summary.json",
        "model_forward_summary.json",
        "fp32_reference_summary.json",
        "loaded_semantics_summary.json",
        "attribution_summary.json",
        "intervention_summary.json",
        "memory_timing_summary.json",
        "attempts.json",
        "run_manifest.json",
        "checksums.sha256",
    }
)
FORBIDDEN_ARTIFACT_SUFFIXES = frozenset(
    {
        ".pt",
        ".pth",
        ".bin",
        ".safetensors",
        ".npy",
        ".npz",
        ".parquet",
        ".arrow",
        ".zip",
    }
)
HEX40_RE = re.compile(r"[0-9a-f]{40}\Z")


@dataclass(frozen=True, slots=True)
class MemoryFeasibility:
    """Conservative BF16 projection; MPS and RSS remain separate signals."""

    model_weight_bytes: int
    persistent_lazy_plt_bytes: int
    two_lazy_matrices_bytes: int
    full_eager_plt_bytes: int
    activation_reserve_bytes: int
    runtime_reserve_bytes: int
    graph_buffer_limit_bytes: int
    total_conservative_bytes: int
    maximum_process_rss_bytes: int
    feasible: bool


@dataclass(frozen=True, slots=True)
class FeatureSelection:
    """One deterministic baseline-active, non-error PLT feature."""

    layer: int
    position: int
    feature: int
    baseline_activation: float
    rule: str
    score: float


@dataclass(frozen=True, slots=True)
class SupervisorOutcome:
    """Result of a bounded, process-group-owned child execution."""

    returncode: int
    timed_out: bool
    safety_terminated: bool
    termination_signal: str | None
    telemetry_failures: int
    samples: tuple[dict[str, Any], ...]
    stdout: str
    stderr: str
    started_at_unix: float
    finished_at_unix: float


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactValidationError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ArtifactValidationError(f"{label} contains a non-string key")
    return cast(Mapping[str, Any], value)


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ArtifactValidationError(f"{label} must be a list")
    return value


def _require(mapping: Mapping[str, Any], key: str, expected: Any, label: str) -> None:
    observed = mapping.get(key)
    if observed != expected:
        raise ArtifactValidationError(
            f"{label}.{key} must be {expected!r}, got {observed!r}"
        )


def load_bf16_config(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the isolated BF16 YAML config."""

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ArtifactValidationError("PyYAML is required for BF16 config") from error
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ArtifactValidationError("BF16 config must be a regular file")
    with candidate.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    return validate_bf16_config(value)


def validate_bf16_config(value: Any) -> dict[str, Any]:
    """Reject mutable identity, cross-experiment contamination, and weak gates."""

    root = _mapping(value, "config")
    for root_key, root_expected in (
        ("schema_version", 1),
        ("experiment_name", EXPERIMENT_CLASS),
        ("experiment_class", EXPERIMENT_CLASS),
        ("completed_status", COMPLETED_STATUS),
        ("claim_class", "local_small_model_runtime_validation"),
        ("execution_base_commit", EXECUTION_BASE_COMMIT),
    ):
        _require(root, root_key, root_expected, "config")

    upstream = _mapping(root.get("upstream"), "upstream")
    for upstream_key, upstream_expected in (
        ("repository", UPSTREAM_REPOSITORY),
        ("version", UPSTREAM_VERSION),
        ("revision", UPSTREAM_REVISION),
    ):
        _require(upstream, upstream_key, upstream_expected, "upstream")

    model = _mapping(root.get("model"), "model")
    for model_key, model_expected in (
        ("identifier", MODEL_IDENTIFIER),
        ("revision", MODEL_REVISION),
        ("architecture", "Gemma3ForCausalLM"),
        ("pretrained_variant", "base"),
        ("layer_count", LAYER_COUNT),
        ("hidden_size", HIDDEN_SIZE),
        ("vocabulary_size", VOCABULARY_SIZE),
    ):
        _require(model, model_key, model_expected, "model")

    transcoder = _mapping(root.get("transcoder"), "transcoder")
    for transcoder_key, transcoder_expected in (
        ("identifier", TRANSCODER_IDENTIFIER),
        ("revision", TRANSCODER_REVISION),
        ("subfolder", TRANSCODER_SUBFOLDER),
        ("model_kind", "transcoder_set"),
        ("feature_input_hook", "mlp.hook_in"),
        ("feature_output_hook", "hook_mlp_out"),
        ("layer_count", LAYER_COUNT),
        ("feature_width", FEATURE_WIDTH),
        ("storage_dtype", "float32"),
        ("execution_dtype", DTYPE),
    ):
        _require(transcoder, transcoder_key, transcoder_expected, "transcoder")

    runtime = _mapping(root.get("runtime"), "runtime")
    for runtime_key, runtime_expected in (
        ("backend", BACKEND),
        ("device", DEVICE),
        ("dtype", DTYPE),
        ("torch_dtype", TORCH_DTYPE),
        ("python", "3.11"),
        ("platform", "macos-arm64"),
        ("fallback_environment_variable", FALLBACK_VARIABLE),
        ("fallback_allowed", False),
        ("outer_autocast_allowed", False),
        ("lazy_encoder", True),
        ("lazy_decoder", True),
        ("graph_metadata_device", "cpu"),
        ("scientific_tensor_device", "mps"),
        ("source_mandated_internal_fp32", list(SOURCE_MANDATED_INTERNAL_FP32)),
    ):
        _require(runtime, runtime_key, runtime_expected, "runtime")

    prompt_records = _list(root.get("prompts"), "prompts")
    expected_prompts = [
        {"id": "bos_only", "kind": "bos_only", "text": None},
        {"id": "hello", "kind": "text", "text": "Hello"},
        {
            "id": "pilot",
            "kind": "text",
            "text": "The capital of France is",
        },
    ]
    if prompt_records != expected_prompts:
        raise ArtifactValidationError("three-prompt model gate is not frozen")

    smoke = _mapping(root.get("smoke"), "smoke")
    for smoke_key, smoke_expected in (
        ("max_n_logits", 3),
        ("desired_logit_probability", 0.80),
        ("max_feature_nodes", 512),
        ("attribution_batch_sizes", [16, 8]),
        ("intervention_alphas", [0.0, 0.5, 1.0]),
        ("freeze_attention", True),
        ("constrained_layers", None),
    ):
        _require(smoke, smoke_key, smoke_expected, "smoke")

    accepted = _mapping(root.get("accepted"), "accepted")
    for accepted_key, accepted_expected in (
        ("max_n_logits", 10),
        ("desired_logit_probability", 0.95),
        ("max_feature_nodes", 4096),
        ("attribution_batch_sizes", [64, 32, 16]),
        ("intervention_alphas", [0.0, 0.5, 1.0]),
        ("baseline_repeat", True),
        ("freeze_attention", True),
        ("constrained_layers", None),
    ):
        _require(accepted, accepted_key, accepted_expected, "accepted")

    selection = _mapping(root.get("feature_selection"), "feature_selection")
    _require(selection, "manual_selection_allowed", False, "feature_selection")
    _require(
        selection,
        "primary_rule",
        "highest_absolute_direct_contribution_to_baseline_top_logit_at_final_token",
        "feature_selection",
    )
    _require(
        selection,
        "fallback_rule",
        "highest_absolute_active_baseline_activation_at_final_token",
        "feature_selection",
    )

    tolerances = _mapping(root.get("tolerances"), "tolerances")
    for tolerance_key, tolerance_expected in (
        ("overflow_absolute", 512.0),
        ("overflow_relative", 0.005),
        ("gate_value_bf16_ulps", 2),
        ("accumulated_value_bf16_ulps", 8),
        ("fp32_reference_cosine_minimum", 0.995),
        ("fp32_reference_normalized_l2_maximum", 0.05),
        ("fp32_reference_norm_ratio_minimum", 0.95),
        ("fp32_reference_norm_ratio_maximum", 1.05),
        ("fp32_reference_top10_overlap_minimum", 8),
        ("fp32_reference_top1_agreement_required", True),
        ("fp32_reference_magnitude_ratio_minimum", 0.5),
        ("fp32_reference_magnitude_ratio_maximum", 2.0),
        ("baseline_noop_normalized_l2_maximum", 0.01),
        ("baseline_noop_maximum_absolute_logit_difference", 0.0),
        ("intervention_value_bf16_exact_required", True),
    ):
        _require(tolerances, tolerance_key, tolerance_expected, "tolerances")

    limits = _mapping(root.get("safety_limits"), "safety_limits")
    for limit_key, limit_expected in (
        ("maximum_mps_driver_bytes", 24 * 1024**3),
        ("maximum_process_rss_bytes", 24 * 1024**3),
        ("maximum_swap_growth_bytes", 4 * 1024**3),
        ("minimum_available_memory_bytes", 4 * 1024**3),
        ("maximum_graph_buffer_bytes", 6 * 1024**3),
        ("maximum_transcoder_download_bytes", 6 * 1024**3),
        ("accepted_thermal_states", ["nominal", "fair"]),
        ("telemetry_failure_limit", 3),
    ):
        _require(limits, limit_key, limit_expected, "safety_limits")

    artifacts = _mapping(root.get("artifacts"), "artifacts")
    for artifact_key, artifact_expected in (
        ("result_directory", RESULT_DIRECTORY),
        ("generated_directory", GENERATED_DIRECTORY),
        ("environment_lock", ENVIRONMENT_LOCK),
        ("projected_download_manifest", PROJECTED_MANIFEST),
        ("required_files", sorted(ARTIFACT_ALLOWLIST)),
    ):
        observed = artifacts.get(artifact_key)
        if artifact_key == "required_files":
            observed = sorted(_list(observed, "artifacts.required_files"))
        if observed != artifact_expected:
            raise ArtifactValidationError(
                f"artifacts.{artifact_key} must be {artifact_expected!r}, "
                f"got {observed!r}"
            )

    for revision in (
        EXECUTION_BASE_COMMIT,
        UPSTREAM_REVISION,
        MODEL_REVISION,
        TRANSCODER_REVISION,
    ):
        if HEX40_RE.fullmatch(revision) is None:
            raise ArtifactValidationError("revision is not an immutable SHA")
    serialized = repr(dict(root)).casefold()
    for forbidden in ("cuda", "t4", "gemma-2-2b", "transformerlens"):
        if forbidden in serialized:
            raise ArtifactValidationError("BF16 config contains class contamination")
    return dict(root)


def assert_fallback_disabled(environment: Mapping[str, str] | None = None) -> None:
    """Fail for every truthy or unknown fallback value."""

    source = os.environ if environment is None else environment
    observed = source.get(FALLBACK_VARIABLE)
    if observed is None:
        return
    normalized = observed.strip().casefold()
    if normalized not in {"", "0", "false", "no", "off"}:
        raise ArtifactValidationError(f"{FALLBACK_VARIABLE} must be absent or false")


def supervise_process_group(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    sample_interval_seconds: float,
    sample_host: Any,
    telemetry_failure_limit: int,
    terminate_grace_seconds: float,
    kill_grace_seconds: float,
    environment: Mapping[str, str],
) -> SupervisorOutcome:
    """Run exactly one owned child process group and fail closed on safety/timeout.

    The callback returns a JSON-safe mapping with a `violations` list. Only the
    newly created session/process group is signalled; unrelated processes are
    never targeted.
    """

    if not command or timeout_seconds <= 0 or sample_interval_seconds <= 0:
        raise ScientificInputError("invalid supervisor command or timing")
    if telemetry_failure_limit < 1:
        raise ScientificInputError("telemetry_failure_limit must be positive")
    started = time.time()
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env=dict(environment),
    )
    samples: list[dict[str, Any]] = []
    telemetry_failures = 0
    timed_out = False
    safety_terminated = False
    termination_signal: str | None = None

    def terminate_group(reason: str) -> None:
        nonlocal termination_signal
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            termination_signal = f"SIGTERM:{reason}"
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=terminate_grace_seconds)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
            termination_signal = f"SIGKILL:{reason}"
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=kill_grace_seconds)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                "owned worker process group did not terminate"
            ) from error

    while process.poll() is None:
        elapsed = time.time() - started
        if elapsed > timeout_seconds:
            timed_out = True
            terminate_group("timeout")
            break
        try:
            raw_sample = sample_host(process.pid)
            if not isinstance(raw_sample, Mapping):
                raise TypeError("host telemetry callback must return a mapping")
            sample = dict(raw_sample)
            violations = sample.get("violations")
            if not isinstance(violations, list):
                raise TypeError("host telemetry sample lacks a violations list")
            samples.append(sample)
            telemetry_failures = 0
            if violations:
                safety_terminated = True
                terminate_group("safety")
                break
        except Exception as error:
            telemetry_failures += 1
            samples.append(
                {
                    "sampled_at_unix": time.time(),
                    "telemetry_error_type": type(error).__name__,
                    "violations": [],
                }
            )
            if telemetry_failures >= telemetry_failure_limit:
                safety_terminated = True
                terminate_group("telemetry_failure")
                break
        time.sleep(sample_interval_seconds)
    stdout, stderr = process.communicate(timeout=max(kill_grace_seconds, 1.0))
    return SupervisorOutcome(
        returncode=int(process.returncode),
        timed_out=timed_out,
        safety_terminated=safety_terminated,
        termination_signal=termination_signal,
        telemetry_failures=telemetry_failures,
        samples=tuple(samples),
        stdout=stdout[-20_000:],
        stderr=stderr[-20_000:],
        started_at_unix=started,
        finished_at_unix=time.time(),
    )


def lock_sha256(repository_root: Path) -> str:
    return sha256_file(repository_root / ENVIRONMENT_LOCK)


def normalized_l2(observed: Any, reference: Any, torch: Any) -> float:
    """Return ||observed-reference|| / max(||reference||, tiny) in FP64."""

    # MPS has no float64 dtype: cross the explicit validation-only CPU boundary
    # before promoting for the norm calculation.
    observed64 = observed.detach().to(device="cpu").to(dtype=torch.float64)
    reference64 = reference.detach().to(device="cpu").to(dtype=torch.float64)
    denominator = max(float(torch.linalg.vector_norm(reference64).item()), 1e-30)
    return (
        float(torch.linalg.vector_norm(observed64 - reference64).item()) / denominator
    )


def bf16_ulp(value: float) -> float:
    """Return one IEEE BF16 ULP at the magnitude of a finite scalar."""

    if not math.isfinite(value):
        raise ScientificInputError("BF16 ULP requires a finite value")
    magnitude = abs(value)
    if magnitude == 0.0 or magnitude < 2.0**-126:
        return 2.0**-133
    exponent = math.floor(math.log2(magnitude))
    return 2.0 ** (exponent - 7)


def within_bf16_ulps(observed: float, reference: float, maximum_ulps: int) -> bool:
    if maximum_ulps < 0:
        raise ScientificInputError("maximum_ulps must be nonnegative")
    if not math.isfinite(observed) or not math.isfinite(reference):
        return False
    scale = max(bf16_ulp(observed), bf16_ulp(reference))
    return abs(observed - reference) <= maximum_ulps * scale


def tensor_summary(value: Any, torch: Any) -> dict[str, Any]:
    """Compact device-side finite diagnostics without retaining tensor payloads."""

    if not isinstance(value, torch.Tensor):
        raise ArtifactValidationError("tensor summary requires a tensor")
    floating = bool(value.is_floating_point())
    result: dict[str, Any] = {
        "shape": [int(size) for size in value.shape],
        "device": str(value.device),
        "dtype": str(value.dtype),
        "element_count": int(value.numel()),
    }
    if not floating:
        result["floating"] = False
        return result
    finite = torch.isfinite(value)
    nan_count = int(torch.isnan(value).sum().item())
    positive_inf = int(torch.isposinf(value).sum().item())
    negative_inf = int(torch.isneginf(value).sum().item())
    result.update(
        {
            "floating": True,
            "nan_count": nan_count,
            "positive_infinity_count": positive_inf,
            "negative_infinity_count": negative_inf,
            "nonfinite_count": int((~finite).sum().item()),
        }
    )
    if bool(finite.any().item()):
        finite_values = value[finite]
        result.update(
            {
                "minimum": float(finite_values.min().item()),
                "maximum": float(finite_values.max().item()),
                "absolute_maximum": float(finite_values.abs().max().item()),
            }
        )
    else:
        result.update({"minimum": None, "maximum": None, "absolute_maximum": None})
    return result


def assert_mps_bf16_tensor(value: Any, torch: Any, label: str) -> None:
    """Require a finite floating scientific tensor on native MPS/BF16."""

    if not isinstance(value, torch.Tensor):
        raise ArtifactValidationError(f"{label} is not a tensor")
    if value.device.type != "mps" or value.dtype != torch.bfloat16:
        raise ArtifactValidationError(
            f"{label} must be MPS BF16, got {value.device}/{value.dtype}"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise ArtifactValidationError(f"{label} is non-finite")


def assert_module_mps_bf16(
    module: Any,
    torch: Any,
    *,
    allowed_fp32_buffer_names: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Inspect every parameter and buffer; exceptions are name-exact and MPS-only."""

    parameters = list(module.named_parameters())
    if not parameters:
        raise ArtifactValidationError("module has no parameters")
    fp32_buffers: list[str] = []
    for name, parameter in parameters:
        if parameter.device.type != "mps":
            raise ArtifactValidationError(f"parameter {name} left MPS")
        if parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
            raise ArtifactValidationError(f"parameter {name} is not BF16")
        if parameter.is_floating_point() and not bool(
            torch.isfinite(parameter).all().item()
        ):
            raise ArtifactValidationError(f"parameter {name} is non-finite")
    for name, buffer in module.named_buffers():
        if buffer.device.type != "mps":
            raise ArtifactValidationError(f"buffer {name} left MPS")
        if not buffer.is_floating_point():
            continue
        if buffer.dtype == torch.float32 and name in allowed_fp32_buffer_names:
            fp32_buffers.append(name)
        elif buffer.dtype != torch.bfloat16:
            raise ArtifactValidationError(f"buffer {name} has unapproved dtype")
        if not bool(torch.isfinite(buffer).all().item()):
            raise ArtifactValidationError(f"buffer {name} is non-finite")
    missing = allowed_fp32_buffer_names - set(fp32_buffers)
    if missing:
        raise ArtifactValidationError(
            f"declared FP32 buffers were not observed: {sorted(missing)}"
        )
    return {
        "parameter_count": len(parameters),
        "floating_parameter_dtype": TORCH_DTYPE,
        "parameter_device": "mps",
        "source_mandated_fp32_buffers": sorted(fp32_buffers),
    }


def run_overflow_regression(
    torch: Any, tolerances: Mapping[str, Any]
) -> dict[str, Any]:
    """Reproduce FP16 overflow and require finite real-MPS BF16 recovery."""

    assert_fallback_disabled()
    x_decimal = 55_520.0
    y_decimal = 13_408.0
    reference = float(
        (
            torch.tensor(x_decimal, dtype=torch.float32)
            + torch.tensor(y_decimal, dtype=torch.float32)
        ).item()
    )
    x16 = torch.tensor(x_decimal, device="mps", dtype=torch.float16)
    y16 = torch.tensor(y_decimal, device="mps", dtype=torch.float16)
    fp16_result = x16 + y16
    xbf = torch.tensor(x_decimal, device="mps", dtype=torch.bfloat16)
    ybf = torch.tensor(y_decimal, device="mps", dtype=torch.bfloat16)
    bf16_result = xbf + ybf
    torch.mps.synchronize()
    bf16_value = float(bf16_result.item())
    absolute_error = abs(bf16_value - reference)
    relative_error = absolute_error / abs(reference)
    passed = (
        bool(torch.isinf(fp16_result).item())
        and not bool(torch.isfinite(fp16_result).item())
        and bool(torch.isfinite(bf16_result).item())
        and bf16_value > 0.0
        and absolute_error <= float(tolerances["overflow_absolute"])
        and relative_error <= float(tolerances["overflow_relative"])
        and within_bf16_ulps(bf16_value, reference, 1)
    )
    record = {
        "passed": passed,
        "decimal_operands": [x_decimal, y_decimal],
        "fp16": {
            "operand_representations": [float(x16.item()), float(y16.item())],
            "result": "positive_infinity",
            "finite": bool(torch.isfinite(fp16_result).item()),
            "device": str(fp16_result.device),
            "dtype": str(fp16_result.dtype),
        },
        "bf16": {
            "operand_representations": [float(xbf.item()), float(ybf.item())],
            "result": bf16_value,
            "finite": bool(torch.isfinite(bf16_result).item()),
            "device": str(bf16_result.device),
            "dtype": str(bf16_result.dtype),
            "absolute_error_against_fp32": absolute_error,
            "relative_error_against_fp32": relative_error,
            "ulp_at_result": bf16_ulp(bf16_value),
        },
        "fp32_reference": {"result": reference, "finite": math.isfinite(reference)},
        "frozen_tolerances": {
            "absolute": float(tolerances["overflow_absolute"]),
            "relative": float(tolerances["overflow_relative"]),
            "maximum_bf16_ulps": 1,
        },
    }
    if not passed:
        raise ArtifactValidationError("known overflow recovery regression failed")
    return record


def conservative_memory_feasibility(
    *, maximum_process_rss_bytes: int = 24 * 1024**3
) -> MemoryFeasibility:
    """Return a conservative BF16 runtime projection below independent limits."""

    model_weights = 536_223_056
    persistent_per_layer = (2 * FEATURE_WIDTH + HIDDEN_SIZE) * 2
    persistent = LAYER_COUNT * persistent_per_layer
    one_matrix = FEATURE_WIDTH * HIDDEN_SIZE * 2
    full_eager = (
        LAYER_COUNT
        * (2 * FEATURE_WIDTH * HIDDEN_SIZE + 2 * FEATURE_WIDTH + HIDDEN_SIZE)
        * 2
    )
    activation_reserve = 1024**3
    runtime_reserve = 4 * 1024**3
    graph_limit = 6 * 1024**3
    total = (
        model_weights
        + persistent
        + 2 * one_matrix
        + activation_reserve
        + runtime_reserve
        + graph_limit
    )
    return MemoryFeasibility(
        model_weight_bytes=model_weights,
        persistent_lazy_plt_bytes=persistent,
        two_lazy_matrices_bytes=2 * one_matrix,
        full_eager_plt_bytes=full_eager,
        activation_reserve_bytes=activation_reserve,
        runtime_reserve_bytes=runtime_reserve,
        graph_buffer_limit_bytes=graph_limit,
        total_conservative_bytes=total,
        maximum_process_rss_bytes=maximum_process_rss_bytes,
        feasible=total < maximum_process_rss_bytes,
    )


def projected_graph_bytes(
    *, active_features: int, selected_features: int, token_count: int, logits: int
) -> int:
    values = (active_features, selected_features, token_count, logits)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in values
    ):
        raise ScientificInputError("graph dimensions must be positive integers")
    non_feature_nodes = (LAYER_COUNT + 1) * token_count + logits
    columns = active_features + non_feature_nodes
    rows = min(selected_features, active_features) + logits
    final_columns = min(selected_features, active_features) + non_feature_nodes
    return rows * columns * 4 + final_columns * final_columns * 4


def dense_to_cpu_sparse_metadata_bf16(dense: Any, torch: Any) -> tuple[Any, Any, Any]:
    """Move only COO metadata to CPU while preserving BF16 values bit-exactly."""

    assert_mps_bf16_tensor(dense, torch, "sparse adapter input")
    coordinates = torch.nonzero(dense, as_tuple=False)
    device_indices = coordinates.T.contiguous()
    if coordinates.numel():
        device_values = dense[tuple(device_indices[axis] for axis in range(dense.ndim))]
    else:
        device_values = torch.empty((0,), device="mps", dtype=torch.bfloat16)
    cpu_indices = device_indices.detach().to(device="cpu", dtype=torch.long)
    cpu_values = device_values.detach().to(device="cpu", dtype=torch.bfloat16)
    metadata = torch.sparse_coo_tensor(
        cpu_indices,
        cpu_values,
        size=tuple(int(size) for size in dense.shape),
        device="cpu",
        dtype=torch.bfloat16,
    ).coalesce()
    reconstructed = torch.zeros_like(dense)
    if device_values.numel():
        reconstructed.index_put_(
            tuple(device_indices[axis] for axis in range(dense.ndim)),
            device_values,
            accumulate=False,
        )
    if not bool(torch.equal(reconstructed, dense)):
        raise ArtifactValidationError("BF16 sparse metadata round-trip is not exact")
    if metadata.values().dtype != torch.bfloat16:
        raise ArtifactValidationError("CPU sparse values changed dtype")
    return metadata, device_indices, device_values


def validate_live_sparse_boundary(torch: Any) -> dict[str, Any]:
    source = torch.tensor(
        [[0.0, 1.5, 0.0], [-2.0, 0.0, 3.0]],
        device="mps",
        dtype=torch.bfloat16,
    )
    metadata, indices, values = dense_to_cpu_sparse_metadata_bf16(source, torch)
    restored = metadata.to_dense().to(device="mps", dtype=torch.bfloat16)
    exact = bool(torch.equal(restored, source))
    if not exact:
        raise ArtifactValidationError("CPU COO value round-trip changed BF16 values")
    return {
        "passed": True,
        "execution_deviation": "explicit_cpu_sparse_metadata_only",
        "dense_scientific_tensor_device": str(source.device),
        "dense_scientific_tensor_dtype": str(source.dtype),
        "metadata_device": str(metadata.device),
        "metadata_value_dtype": str(metadata.values().dtype),
        "scientific_index_device": str(indices.device),
        "scientific_value_device": str(values.device),
        "nonzero_count": int(metadata._nnz()),
        "bit_exact_roundtrip": exact,
    }


def intervention_mapping_bf16(
    baseline: Any, alpha: float, torch: Any
) -> dict[str, Any]:
    """Compute and validate the exact BF16 absolute upstream intervention value."""

    assert_mps_bf16_tensor(baseline, torch, "baseline activation")
    if baseline.numel() != 1 or isinstance(alpha, bool) or not 0.0 <= alpha <= 1.0:
        raise ScientificInputError(
            "intervention requires scalar baseline and alpha in [0,1]"
        )
    desired = baseline * torch.tensor(1.0 - alpha, device="mps", dtype=torch.bfloat16)
    assert_mps_bf16_tensor(desired, torch, "desired activation")
    reference = (1.0 - float(alpha)) * float(baseline.item())
    if not within_bf16_ulps(float(desired.item()), reference, 1):
        raise ArtifactValidationError("BF16 suppression mapping is incorrectly rounded")
    return {
        "alpha": float(alpha),
        "baseline_activation": float(baseline.item()),
        "desired_absolute_activation": float(desired.item()),
        "device": str(desired.device),
        "dtype": str(desired.dtype),
        "tensor": desired,
    }


def layerwise_jumprelu_reference(
    preactivations: Any, thresholds: Any, torch: Any
) -> Any:
    """Apply strict JumpReLU with each PLT layer's own loaded thresholds."""

    if (
        not isinstance(preactivations, torch.Tensor)
        or not isinstance(thresholds, torch.Tensor)
        or preactivations.ndim != 3
        or thresholds.ndim != 2
        or preactivations.shape[0] != thresholds.shape[0]
        or preactivations.shape[2] != thresholds.shape[1]
        or preactivations.device != thresholds.device
        or preactivations.dtype != thresholds.dtype
    ):
        raise ScientificInputError("layerwise JumpReLU tensor identity is invalid")
    return preactivations * (preactivations > thresholds[:, None, :])


def select_feature_from_graph(graph: Any, *, final_position: int) -> FeatureSelection:
    """Frozen direct-contribution rule with deterministic capability fallback."""

    active = graph.active_features.detach().cpu()
    values = graph.activation_values.detach().cpu()
    selected = graph.selected_features.detach().cpu()
    adjacency = graph.adjacency_matrix.detach().cpu()
    n_logits = len(graph.logit_targets)
    if n_logits < 1 or active.ndim != 2 or active.shape[1] != 3:
        raise ArtifactValidationError("graph feature structure is invalid")
    candidates: list[tuple[float, int, int, int, float]] = []
    top_logit_row = adjacency.shape[0] - n_logits
    for graph_column, active_index_raw in enumerate(selected.tolist()):
        active_index = int(active_index_raw)
        layer, position, feature = (int(item) for item in active[active_index].tolist())
        if position != final_position:
            continue
        baseline = float(values[active_index].item())
        score = abs(float(adjacency[top_logit_row, graph_column].item()))
        if baseline > 0.0 and math.isfinite(baseline) and math.isfinite(score):
            candidates.append((-score, layer, position, feature, baseline))
    if candidates:
        negative_score, layer, position, feature, baseline = min(candidates)
        return FeatureSelection(
            layer,
            position,
            feature,
            baseline,
            "highest_absolute_direct_contribution_to_baseline_top_logit_at_final_token",
            -negative_score,
        )
    fallback: list[tuple[float, int, int, int, float]] = []
    for index in range(active.shape[0]):
        layer, position, feature = (int(item) for item in active[index].tolist())
        baseline = float(values[index].item())
        if position == final_position and baseline > 0.0 and math.isfinite(baseline):
            fallback.append((-abs(baseline), layer, position, feature, baseline))
    if not fallback:
        raise ArtifactValidationError("no baseline-active final-token feature exists")
    negative_score, layer, position, feature, baseline = min(fallback)
    return FeatureSelection(
        layer,
        position,
        feature,
        baseline,
        "highest_absolute_active_baseline_activation_at_final_token",
        -negative_score,
    )


def feature_selection_audit_from_graph(
    graph: Any, *, final_position: int, selection: FeatureSelection
) -> dict[str, Any]:
    """Emit a small complete candidate table for independent winner validation."""

    active = graph.active_features.detach().cpu()
    values = graph.activation_values.detach().cpu()
    selected = graph.selected_features.detach().cpu()
    adjacency = graph.adjacency_matrix.detach().cpu()
    n_logits = len(graph.logit_targets)
    if n_logits < 1 or active.ndim != 2 or active.shape[1] != 3:
        raise ArtifactValidationError("selection-audit graph structure is invalid")
    primary_rule = (
        "highest_absolute_direct_contribution_to_baseline_top_logit_at_final_token"
    )
    fallback_rule = "highest_absolute_active_baseline_activation_at_final_token"
    records: list[dict[str, int | float]] = []
    excluded = {
        "non_final_position": 0,
        "nonpositive_baseline": 0,
        "nonfinite_baseline": 0,
        "nonfinite_score": 0,
    }
    score_for: Callable[[int, int], float]
    if selection.rule == primary_rule:
        source_indices = [int(item) for item in selected.tolist()]
        top_logit_row = adjacency.shape[0] - n_logits
        score_for = lambda source_offset, active_index: abs(  # noqa: E731
            float(adjacency[top_logit_row, source_offset].item())
        )
        source_class = "selected_graph_features"
    elif selection.rule == fallback_rule:
        source_indices = list(range(active.shape[0]))
        score_for = lambda _source_offset, active_index: abs(  # noqa: E731
            float(values[active_index].item())
        )
        source_class = "all_active_graph_features"
    else:
        raise ArtifactValidationError("selection audit received an unknown rule")

    for source_offset, active_index in enumerate(source_indices):
        if active_index < 0 or active_index >= active.shape[0]:
            raise ArtifactValidationError("selection audit contains an invalid index")
        layer, position, feature = (int(item) for item in active[active_index].tolist())
        baseline = float(values[active_index].item())
        score = score_for(source_offset, active_index)
        if position != final_position:
            excluded["non_final_position"] += 1
        elif not math.isfinite(baseline):
            excluded["nonfinite_baseline"] += 1
        elif baseline <= 0.0:
            excluded["nonpositive_baseline"] += 1
        elif not math.isfinite(score):
            excluded["nonfinite_score"] += 1
        else:
            records.append(
                {
                    "layer": layer,
                    "position": position,
                    "feature": feature,
                    "baseline_activation": baseline,
                    "score": score,
                }
            )
    if not records or len(records) + sum(excluded.values()) != len(source_indices):
        raise ArtifactValidationError("selection audit candidate coverage is invalid")
    records.sort(key=lambda item: (item["layer"], item["position"], item["feature"]))
    winner = min(
        records,
        key=lambda item: (
            -float(item["score"]),
            int(item["layer"]),
            int(item["position"]),
            int(item["feature"]),
        ),
    )
    observed = (
        int(winner["layer"]),
        int(winner["position"]),
        int(winner["feature"]),
        float(winner["baseline_activation"]),
        float(winner["score"]),
    )
    expected = (
        selection.layer,
        selection.position,
        selection.feature,
        selection.baseline_activation,
        selection.score,
    )
    if observed != expected:
        raise ArtifactValidationError("selection audit winner disagrees with selection")
    return {
        "rule": selection.rule,
        "source_class": source_class,
        "source_count": len(source_indices),
        "candidate_count": len(records),
        "excluded_counts": excluded,
        "candidates": records,
        "raw_graph_persisted": False,
    }


def validate_projected_manifest(value: Any) -> dict[str, Any]:
    root = _mapping(value, "projected manifest")
    _require(root, "schema_version", 1, "projected manifest")
    _require(root, "projected_total_bytes", 2_087_816_677, "projected manifest")
    model = _mapping(root.get("model"), "projected model")
    transcoder = _mapping(root.get("transcoder"), "projected transcoder")
    for mapping, label, identity, revision, size in (
        (model, "projected model", MODEL_IDENTIFIER, MODEL_REVISION, 575_454_257),
        (
            transcoder,
            "projected transcoder",
            TRANSCODER_IDENTIFIER,
            TRANSCODER_REVISION,
            1_512_362_420,
        ),
    ):
        _require(mapping, "identifier", identity, label)
        _require(mapping, "revision", revision, label)
        _require(mapping, "projected_bytes", size, label)
    expected_model = {
        "added_tokens.json",
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    }
    model_paths = {
        str(_mapping(item, "model file")["path"])
        for item in _list(model.get("files"), "model files")
    }
    expected_transcoder = {f"{TRANSCODER_SUBFOLDER}/config.yaml"} | {
        f"{TRANSCODER_SUBFOLDER}/layer_{layer}.safetensors"
        for layer in range(LAYER_COUNT)
    }
    transcoder_paths = {
        str(_mapping(item, "transcoder file")["path"])
        for item in _list(transcoder.get("files"), "transcoder files")
    }
    if model_paths != expected_model or transcoder_paths != expected_transcoder:
        raise ArtifactValidationError("projected asset allowlist changed")
    return dict(root)


def validate_snapshot_tree(
    *, snapshot: Path, cache_root: Path, expected_paths: set[str]
) -> list[dict[str, Any]]:
    """Validate canonical cache-contained snapshot entries and return digests."""

    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ArtifactValidationError("snapshot is missing or unsafe")
    resolved_cache = cache_root.resolve(strict=True)
    resolved_snapshot = snapshot.resolve(strict=True)
    if not resolved_snapshot.is_relative_to(resolved_cache):
        raise ArtifactValidationError("snapshot is outside the authorized cache")
    observed: set[str] = set()
    records: list[dict[str, Any]] = []
    for candidate in snapshot.rglob("*"):
        relative = candidate.relative_to(snapshot).as_posix()
        if candidate.is_dir():
            continue
        if candidate.is_symlink():
            target = candidate.resolve(strict=True)
            if not target.is_relative_to(resolved_cache) or not target.is_file():
                raise ArtifactValidationError("snapshot symlink escapes cache")
        else:
            metadata = candidate.stat(follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ArtifactValidationError("snapshot entry is special or hardlinked")
            target = candidate
        size = target.stat().st_size
        if size <= 0:
            raise ArtifactValidationError("snapshot entry is empty")
        observed.add(relative)
        records.append({"path": relative, "bytes": size, "sha256": sha256_file(target)})
    if observed != expected_paths:
        missing = sorted(expected_paths - observed)
        extra = sorted(observed - expected_paths)
        raise ArtifactValidationError(
            f"snapshot allowlist mismatch: missing={missing}, extra={extra}"
        )
    return sorted(records, key=lambda item: str(item["path"]))


def validate_small_artifact_directory(directory: Path) -> None:
    """Structural precheck; the independent validator performs semantic checks."""

    if directory.is_symlink() or not directory.is_dir():
        raise ArtifactValidationError("artifact directory is missing or unsafe")
    observed: set[str] = set()
    for entry in directory.iterdir():
        observed.add(entry.name)
        metadata = entry.stat(follow_symlinks=False)
        if entry.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ArtifactValidationError("artifact entry must be a regular file")
        if metadata.st_nlink != 1:
            raise ArtifactValidationError("hardlinked artifact entry is forbidden")
        if entry.suffix.casefold() in FORBIDDEN_ARTIFACT_SUFFIXES:
            raise ArtifactValidationError("raw tensor/archive artifact is forbidden")
        maximum = 4 * 1024**2 if entry.name == "checksums.sha256" else 2 * 1024**2
        if metadata.st_size > maximum:
            raise ArtifactValidationError("artifact entry exceeds its size cap")
    if observed != ARTIFACT_ALLOWLIST:
        raise ArtifactValidationError("artifact allowlist mismatch")


def sha256_lines_without_comments(path: Path) -> str:
    """Digest the exact normalized freeze entries, excluding documentation comments."""

    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


__all__ = [
    "ARTIFACT_ALLOWLIST",
    "BACKEND",
    "COMPLETED_STATUS",
    "CONFIG_PATH",
    "DEVICE",
    "DTYPE",
    "ENVIRONMENT_LOCK",
    "EXECUTION_BASE_COMMIT",
    "EXPERIMENT_CLASS",
    "FEATURE_WIDTH",
    "LAYER_COUNT",
    "MODEL_IDENTIFIER",
    "MODEL_REVISION",
    "PROJECTED_MANIFEST",
    "SOURCE_MANDATED_INTERNAL_FP32",
    "TRANSCODER_IDENTIFIER",
    "TRANSCODER_REVISION",
    "TRANSCODER_SUBFOLDER",
    "UPSTREAM_REVISION",
    "FeatureSelection",
    "MemoryFeasibility",
    "SupervisorOutcome",
    "assert_fallback_disabled",
    "assert_module_mps_bf16",
    "assert_mps_bf16_tensor",
    "bf16_ulp",
    "conservative_memory_feasibility",
    "dense_to_cpu_sparse_metadata_bf16",
    "feature_selection_audit_from_graph",
    "intervention_mapping_bf16",
    "layerwise_jumprelu_reference",
    "load_bf16_config",
    "lock_sha256",
    "normalized_l2",
    "projected_graph_bytes",
    "run_overflow_regression",
    "select_feature_from_graph",
    "sha256_lines_without_comments",
    "supervise_process_group",
    "tensor_summary",
    "validate_bf16_config",
    "validate_live_sparse_boundary",
    "validate_projected_manifest",
    "validate_small_artifact_directory",
    "validate_snapshot_tree",
    "within_bf16_ulps",
]
