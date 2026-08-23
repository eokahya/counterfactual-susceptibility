"""Strict, dependency-light rules for the Stage 1A MPS/FP16 adaptation.

This module is deliberately separate from :mod:`t4_fp16`.  In particular, an
MPS allocation is a sampled metric (not a CUDA allocator high-water mark),
and an MPS out-of-memory retry must never be inferred from a generic memory
error or from a CUDA diagnostic.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import math
import os
import platform
import re
import resource
import time
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, cast

from cfsus.reproduction.artifacts import (
    ArtifactValidationError,
    assert_publication_safe,
    redact_sensitive,
    validate_json_value,
)
from cfsus.reproduction.config import (
    OFFICIAL_ATTRIBUTION_PROMPT,
    OFFICIAL_INTERVENTION_PROMPT,
    OFFICIAL_MODEL_ID,
    OFFICIAL_MODEL_REVISION,
    OFFICIAL_TRANSCODER_ID,
    OFFICIAL_TRANSCODER_REVISION,
    OFFICIAL_UPSTREAM_REPOSITORY,
    OFFICIAL_UPSTREAM_REVISION,
    Stage1AConfigError,
)
from cfsus.reproduction.runtime_helpers import desired_activation

PROJECT_BASE_COMMIT = "d965e43c34a2ba408b8ae35b13b5651bf269beed"
REPRODUCTION_CLASS = "hardware_adapted_mps_fp16"
EXECUTION_CLASS = "completed_hardware_adapted_mps_fp16"
REFERENCE_DTYPE = "bfloat16"
EXECUTION_DTYPE = "float16"
REFERENCE_STATUS = "pending"
MPS_EXPERIMENT_NAME = "stage1a_gemma2_2b_mps_fp16_hardware_adaptation"
MPS_CLAIM_BOUNDARY = (
    "Apple M2 Max/MPS FP16 hardware-adapted runtime using the pinned "
    "assets; the official native-BF16 reproduction and CUDA/T4 numerical "
    "equivalence remain pending."
)
OOM_BATCH_SEQUENCE = (256, 128, 64)
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MPS_RESULT_DIRECTORY = "results/stage1a_mps_fp16"
MPS_GENERATED_DIRECTORY = "results/generated/stage1a_mps_fp16"
MODEL_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)
MODEL_WEIGHT_FILES = tuple(
    f"model-{index:05d}-of-00003.safetensors" for index in range(1, 4)
)
MODEL_REQUIRED_FILES = MODEL_METADATA_FILES + MODEL_WEIGHT_FILES
TRANSCODER_METADATA_FILES = ("config.yaml",)
TRANSCODER_WEIGHT_FILES = tuple(f"layer_{layer}.safetensors" for layer in range(26))
TRANSCODER_REQUIRED_FILES = TRANSCODER_METADATA_FILES + TRANSCODER_WEIGHT_FILES
MPS_SMALL_FILES = frozenset(
    {
        "preflight_summary.json",
        "feasibility_report.json",
        "environment_manifest.json",
        "asset_manifest.json",
        "attribution_summary.json",
        "intervention_summary.json",
        "semantics_summary.json",
        "memory_summary.json",
        "checksums.sha256",
        "stage1a_mps_run_manifest.json",
    }
)
MPS_SUMMARY_FILES = frozenset(MPS_SMALL_FILES - {"checksums.sha256"})
MAX_BUNDLE_MEMBER_BYTES = 5 * 1024 * 1024
MAX_BUNDLE_TOTAL_BYTES = 25 * 1024 * 1024

_SHA40 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MPS_OOM_MESSAGE = re.compile(
    r"(?:mps(?: backend)?[^\n]{0,80}out of memory|"
    r"out of memory[^\n]{0,80}mps|metal[^\n]{0,80}(?:allocation failed|out of memory))",
    re.IGNORECASE,
)
_OVERCLAIM = re.compile(
    r"(?:official|exact)\s+(?:bf16\s+)?reproduction|"
    r"(?:cuda|t4)\s+(?:numerical\s+)?equivalence",
    re.IGNORECASE,
)
_TRUTHY = frozenset({"1", "true", "yes", "on"})


class MPSRunStatus(StrEnum):
    COMPLETED = EXECUTION_CLASS
    BLOCKED_ACCESS = "blocked_access"
    BLOCKED_RESOURCE = "blocked_resource"
    BLOCKED_ENVIRONMENT = "blocked_environment"
    FAILED_PRECISION = "failed_precision"
    FAILED_RUNTIME = "failed_runtime"
    PREPARED = "prepared_not_executed"


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise Stage1AConfigError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise Stage1AConfigError(
            f"{label} keys are not exact; missing={missing}, unknown={unknown}"
        )


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise Stage1AConfigError(f"{label} must be a non-empty trimmed string")
    return value


def _exact_text(value: object, expected: str, label: str) -> str:
    result = _text(value, label)
    if result != expected:
        raise Stage1AConfigError(f"{label} must equal {expected!r}")
    return result


def _sha(value: object, expected: str, label: str) -> str:
    result = _text(value, label)
    if _SHA40.fullmatch(result) is None or result == "0" * 40:
        raise Stage1AConfigError(
            f"{label} must be a nonzero lowercase 40-character SHA"
        )
    if result != expected:
        raise Stage1AConfigError(f"{label} does not match the immutable pin")
    return result


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Stage1AConfigError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: object, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Stage1AConfigError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise Stage1AConfigError(f"{label} must be finite and >= {minimum}")
    return result


def _bool(value: object, expected: bool, label: str) -> bool:
    if value is not expected:
        raise Stage1AConfigError(f"{label} must be {expected}")
    return expected


def _relative_under(value: object, prefix: str, suffix: str, label: str) -> str:
    text = _text(value, label)
    path = PurePosixPath(text)
    parts = PurePosixPath(prefix).parts
    if (
        path.is_absolute()
        or "\\" in text
        or "." in path.parts
        or ".." in path.parts
        or path.as_posix() != text
        or path.parts[: len(parts)] != parts
        or not text.endswith(suffix)
    ):
        raise Stage1AConfigError(f"{label} must be normalized under {prefix}/")
    return text


@dataclass(frozen=True, slots=True)
class MPSRuntimeConfig:
    backend: str
    device: str
    dtype: str
    execution_class: str
    hardware_family: str
    architecture: str
    official_bf16_reproduction: bool
    t4_fp16_reproduction: bool


@dataclass(frozen=True, slots=True)
class MPSRetryPolicy:
    batch_sizes: tuple[int, ...]
    trigger: str
    fresh_process: bool


@dataclass(frozen=True, slots=True)
class MPSNumericsConfig:
    gate_absolute_tolerance: float
    projection_absolute_tolerance: float
    noop_absolute_tolerance: float
    noop_relative_tolerance: float
    determinism_absolute_tolerance: float
    determinism_relative_tolerance: float
    model_parameter_samples_per_tensor: int


@dataclass(frozen=True, slots=True)
class MPSArtifactPaths:
    preflight_summary: str
    environment_manifest: str
    asset_manifest: str
    attribution_summary: str
    intervention_summary: str
    semantics_summary: str
    memory_summary: str
    checksums: str
    run_manifest: str


@dataclass(frozen=True, slots=True)
class MPSMemoryBudget:
    physical_memory_gib: int
    safety_reserve_gib: int
    telemetry: str
    do_not_disable_guardrails: bool
    high_watermark_ratio_override: None


@dataclass(frozen=True, slots=True)
class Stage1AMPSFP16Config:
    schema_version: int
    experiment_name: str
    reproduction_class: str
    project_base_commit: str
    reference_dtype: str
    execution_dtype: str
    reference_status: str
    claim_boundary: str
    memory_budget: MPSMemoryBudget
    runtime: MPSRuntimeConfig
    oom_retry: MPSRetryPolicy
    numerics: MPSNumericsConfig
    artifacts: MPSArtifactPaths

    @classmethod
    def from_mapping(cls, value: object) -> Stage1AMPSFP16Config:
        data = _mapping(value, "configuration")
        expected_top = {
            "schema_version",
            "experiment_name",
            "reproduction_class",
            "project_base_commit",
            "reference_dtype",
            "execution_dtype",
            "reference_status",
            "claim_boundary",
            "environment",
            "upstream",
            "model",
            "transcoder",
            "runtime",
            "seeds",
            "asset_policy",
            "attribution",
            "intervention",
            "numerics",
            "oom_retry",
            "memory_budget",
            "artifacts",
        }
        _exact_keys(data, expected_top, "configuration")
        if _integer(data["schema_version"], "schema_version") != 1:
            raise Stage1AConfigError("schema_version must equal 1")
        experiment = _exact_text(
            data["experiment_name"], MPS_EXPERIMENT_NAME, "experiment_name"
        )
        boundary = _text(data["claim_boundary"], "claim_boundary")
        if boundary != MPS_CLAIM_BOUNDARY:
            raise Stage1AConfigError("claim boundary is not the MPS boundary")
        upstream = _mapping(data["upstream"], "upstream")
        _exact_keys(upstream, {"repository", "revision", "version"}, "upstream")
        _exact_text(
            upstream["repository"], OFFICIAL_UPSTREAM_REPOSITORY, "upstream.repository"
        )
        _sha(upstream["revision"], OFFICIAL_UPSTREAM_REVISION, "upstream.revision")
        _exact_text(upstream["version"], "0.5.2", "upstream.version")
        for section, identifier, revision, suffix in (
            ("model", OFFICIAL_MODEL_ID, OFFICIAL_MODEL_REVISION, "google-gemma-2-2b"),
            (
                "transcoder",
                OFFICIAL_TRANSCODER_ID,
                OFFICIAL_TRANSCODER_REVISION,
                "mwhanna-gemma-scope-transcoders",
            ),
        ):
            raw = _mapping(data[section], section)
            _exact_keys(raw, {"identifier", "revision", "snapshot_path"}, section)
            _exact_text(raw["identifier"], identifier, f"{section}.identifier")
            _sha(raw["revision"], revision, f"{section}.revision")
            _exact_text(
                raw["snapshot_path"],
                f"{MPS_GENERATED_DIRECTORY}/assets/{suffix}",
                f"{section}.snapshot_path",
            )
        cls._validate_environment(data["environment"])
        cls._validate_seeds(data["seeds"])
        cls._validate_asset_policy(data["asset_policy"])
        cls._validate_attribution(data["attribution"])
        cls._validate_intervention(data["intervention"])
        runtime_data = _mapping(data["runtime"], "runtime")
        runtime_keys = {
            "backend",
            "device",
            "dtype",
            "execution_class",
            "hardware_family",
            "official_bf16_reproduction",
            "t4_fp16_reproduction",
            "fallback_enabled",
            "offload",
        }
        _exact_keys(runtime_data, runtime_keys, "runtime")
        runtime = MPSRuntimeConfig(
            backend=_exact_text(
                runtime_data["backend"], "transformerlens", "runtime.backend"
            ),
            device=_exact_text(runtime_data["device"], "mps", "runtime.device"),
            dtype=_exact_text(runtime_data["dtype"], EXECUTION_DTYPE, "runtime.dtype"),
            execution_class=_exact_text(
                runtime_data["execution_class"],
                EXECUTION_CLASS,
                "runtime.execution_class",
            ),
            hardware_family=_exact_text(
                runtime_data["hardware_family"],
                "Apple M2 Max",
                "runtime.hardware_family",
            ),
            architecture="arm64",
            official_bf16_reproduction=_bool(
                runtime_data["official_bf16_reproduction"],
                False,
                "runtime.official_bf16_reproduction",
            ),
            t4_fp16_reproduction=_bool(
                runtime_data["t4_fp16_reproduction"],
                False,
                "runtime.t4_fp16_reproduction",
            ),
        )
        _bool(runtime_data["fallback_enabled"], False, "runtime.fallback_enabled")
        _exact_text(runtime_data["offload"], "disk", "runtime.offload")
        retry = cls._validate_retry(data["oom_retry"])
        numerics = cls._validate_numerics(data["numerics"])
        memory_budget = cls._validate_memory_budget(data["memory_budget"])
        artifacts = cls._validate_artifacts(data["artifacts"])
        return cls(
            1,
            experiment,
            REPRODUCTION_CLASS,
            _sha(
                data["project_base_commit"], PROJECT_BASE_COMMIT, "project_base_commit"
            ),
            _exact_text(data["reference_dtype"], REFERENCE_DTYPE, "reference_dtype"),
            _exact_text(data["execution_dtype"], EXECUTION_DTYPE, "execution_dtype"),
            _exact_text(data["reference_status"], REFERENCE_STATUS, "reference_status"),
            boundary,
            memory_budget,
            runtime,
            retry,
            numerics,
            artifacts,
        )

    @staticmethod
    def _validate_environment(value: object) -> None:
        data = _mapping(value, "environment")
        _exact_keys(
            data,
            {"python", "pytorch", "platform", "hardware_family", "physical_memory_gib"},
            "environment",
        )
        if not _text(data["python"], "environment.python").startswith("3.11"):
            raise Stage1AConfigError("environment.python must be Python 3.11")
        _text(data["pytorch"], "environment.pytorch")
        _exact_text(data["platform"], "macos-arm64", "environment.platform")
        _exact_text(
            data["hardware_family"], "Apple M2 Max", "environment.hardware_family"
        )
        if (
            _integer(data["physical_memory_gib"], "environment.physical_memory_gib", 1)
            != 32
        ):
            raise Stage1AConfigError("environment.physical_memory_gib must equal 32")

    @staticmethod
    def _validate_seeds(value: object) -> None:
        data = _mapping(value, "seeds")
        _exact_keys(data, {"python", "numpy", "torch"}, "seeds")
        for name in ("python", "numpy", "torch"):
            if _integer(data[name], f"seeds.{name}") != 0:
                raise Stage1AConfigError(f"seeds.{name} must equal 0")

    @staticmethod
    def _validate_asset_policy(value: object) -> None:
        data = _mapping(value, "asset_policy")
        _exact_keys(
            data,
            {
                "allow_download",
                "require_offline_execution",
                "cache_location",
                "immutable_revisions_only",
            },
            "asset_policy",
        )
        _bool(data["allow_download"], True, "asset_policy.allow_download")
        _bool(
            data["require_offline_execution"],
            True,
            "asset_policy.require_offline_execution",
        )
        _exact_text(
            data["cache_location"],
            "project_external_huggingface_cache",
            "asset_policy.cache_location",
        )
        _bool(
            data["immutable_revisions_only"],
            True,
            "asset_policy.immutable_revisions_only",
        )

    @staticmethod
    def _validate_attribution(value: object) -> None:
        data = _mapping(value, "attribution")
        keys = {
            "prompt",
            "max_n_logits",
            "desired_logit_probability",
            "max_feature_nodes",
            "batch_size",
            "offload",
        }
        _exact_keys(data, keys, "attribution")
        _exact_text(data["prompt"], OFFICIAL_ATTRIBUTION_PROMPT, "attribution.prompt")
        for name, expected in (
            ("max_n_logits", 10),
            ("desired_logit_probability", 0.95),
            ("max_feature_nodes", 8192),
            ("batch_size", 256),
        ):
            if isinstance(data[name], bool) or data[name] != expected:
                raise Stage1AConfigError(f"attribution.{name} must equal {expected}")
        _exact_text(data["offload"], "disk", "attribution.offload")

    @staticmethod
    def _validate_intervention(value: object) -> None:
        data = _mapping(value, "intervention")
        _exact_keys(
            data,
            {"prompt", "feature", "alphas", "freeze_attention", "constrained_layers"},
            "intervention",
        )
        _exact_text(data["prompt"], OFFICIAL_INTERVENTION_PROMPT, "intervention.prompt")
        feature = _mapping(data["feature"], "intervention.feature")
        _exact_keys(
            feature, {"layer", "position", "feature_id"}, "intervention.feature"
        )
        if (feature["layer"], feature["position"], feature["feature_id"]) != (
            20,
            -1,
            341,
        ):
            raise Stage1AConfigError("intervention.feature must equal (20, -1, 341)")
        if (
            not isinstance(data["alphas"], Sequence)
            or isinstance(data["alphas"], (str, bytes))
            or tuple(data["alphas"]) != (0.0, 0.5, 1.0)
        ):
            raise Stage1AConfigError("intervention.alphas must equal [0.0, 0.5, 1.0]")
        _bool(data["freeze_attention"], True, "intervention.freeze_attention")
        if data["constrained_layers"] is not None:
            raise Stage1AConfigError("intervention.constrained_layers must be null")

    @staticmethod
    def _validate_retry(value: object) -> MPSRetryPolicy:
        data = _mapping(value, "oom_retry")
        _exact_keys(
            data,
            {
                "batch_sizes",
                "trigger",
                "fresh_process",
                "clear_mps_cache_between_attempts",
                "retry_on_unknown_failure",
            },
            "oom_retry",
        )
        if (
            not isinstance(data["batch_sizes"], Sequence)
            or isinstance(data["batch_sizes"], (str, bytes))
            or tuple(data["batch_sizes"]) != OOM_BATCH_SEQUENCE
        ):
            raise Stage1AConfigError("oom_retry.batch_sizes must equal [256, 128, 64]")
        _bool(
            data["clear_mps_cache_between_attempts"],
            True,
            "oom_retry.clear_mps_cache_between_attempts",
        )
        _bool(
            data["retry_on_unknown_failure"],
            False,
            "oom_retry.retry_on_unknown_failure",
        )
        return MPSRetryPolicy(
            OOM_BATCH_SEQUENCE,
            _exact_text(data["trigger"], "mps_out_of_memory_only", "oom_retry.trigger"),
            _bool(data["fresh_process"], True, "oom_retry.fresh_process"),
        )

    @staticmethod
    def _validate_memory_budget(value: object) -> MPSMemoryBudget:
        data = _mapping(value, "memory_budget")
        _exact_keys(
            data,
            {
                "physical_memory_gib",
                "safety_reserve_gib",
                "telemetry",
                "do_not_disable_guardrails",
                "high_watermark_ratio_override",
            },
            "memory_budget",
        )
        if (
            _integer(
                data["physical_memory_gib"], "memory_budget.physical_memory_gib", 1
            )
            != 32
        ):
            raise Stage1AConfigError("memory_budget.physical_memory_gib must equal 32")
        reserve = _integer(
            data["safety_reserve_gib"], "memory_budget.safety_reserve_gib", 1
        )
        if reserve != 6:
            raise Stage1AConfigError("memory_budget.safety_reserve_gib must equal 6")
        _exact_text(
            data["telemetry"], "sampled_mps_and_process_rss", "memory_budget.telemetry"
        )
        _bool(
            data["do_not_disable_guardrails"],
            True,
            "memory_budget.do_not_disable_guardrails",
        )
        if data["high_watermark_ratio_override"] is not None:
            raise Stage1AConfigError(
                "memory_budget.high_watermark_ratio_override must be null"
            )
        return MPSMemoryBudget(32, reserve, "sampled_mps_and_process_rss", True, None)

    @staticmethod
    def _validate_numerics(value: object) -> MPSNumericsConfig:
        data = _mapping(value, "numerics")
        keys = {
            "gate_absolute_tolerance",
            "projection_absolute_tolerance",
            "noop_absolute_tolerance",
            "noop_relative_tolerance",
            "determinism_absolute_tolerance",
            "determinism_relative_tolerance",
            "model_parameter_samples_per_tensor",
            "preflight_absolute_tolerance",
            "preflight_relative_tolerance",
        }
        _exact_keys(data, keys, "numerics")
        expected = {
            "gate_absolute_tolerance": 0.005,
            "projection_absolute_tolerance": 0.005,
            "noop_absolute_tolerance": 0.02,
            "noop_relative_tolerance": 0.002,
            "determinism_absolute_tolerance": 0.02,
            "determinism_relative_tolerance": 0.002,
        }
        values = {name: _finite(data[name], f"numerics.{name}") for name in expected}
        if values != expected:
            raise Stage1AConfigError(
                "numerical tolerances differ from the preregistration"
            )
        if (
            _finite(
                data["preflight_absolute_tolerance"],
                "numerics.preflight_absolute_tolerance",
            )
            != 0.005
            or _finite(
                data["preflight_relative_tolerance"],
                "numerics.preflight_relative_tolerance",
            )
            != 0.002
        ):
            raise Stage1AConfigError(
                "preflight tolerances differ from the preregistration"
            )
        count = _integer(
            data["model_parameter_samples_per_tensor"],
            "numerics.model_parameter_samples_per_tensor",
            1,
        )
        if count != 16:
            raise Stage1AConfigError(
                "numerics.model_parameter_samples_per_tensor must equal 16"
            )
        return MPSNumericsConfig(**values, model_parameter_samples_per_tensor=count)

    @staticmethod
    def _validate_artifacts(value: object) -> MPSArtifactPaths:
        data = _mapping(value, "artifacts")
        fields = {
            "preflight_summary",
            "environment_manifest",
            "asset_manifest",
            "attribution_summary",
            "intervention_summary",
            "semantics_summary",
            "memory_summary",
            "checksums",
            "run_manifest",
        }
        _exact_keys(data, fields, "artifacts")
        paths: dict[str, str] = {
            "preflight_summary": _exact_text(
                data["preflight_summary"],
                f"{MPS_RESULT_DIRECTORY}/preflight/preflight_summary.json",
                "artifacts.preflight_summary",
            )
        }
        expected = {
            "environment_manifest": "environment_manifest.json",
            "asset_manifest": "asset_manifest.json",
            "attribution_summary": "attribution_summary.json",
            "intervention_summary": "intervention_summary.json",
            "semantics_summary": "semantics_summary.json",
            "memory_summary": "memory_summary.json",
            "checksums": "checksums.sha256",
            "run_manifest": "stage1a_mps_run_manifest.json",
        }
        for name, filename in expected.items():
            paths[name] = _exact_text(
                data[name], f"{MPS_RESULT_DIRECTORY}/{filename}", f"artifacts.{name}"
            )
        if len(set(paths.values())) != len(paths):
            raise Stage1AConfigError("artifact paths must be unique")
        return MPSArtifactPaths(**paths)


def is_mps_fp16_mapping(value: Mapping[str, object] | dict[str, Any]) -> bool:
    return value.get("reproduction_class") == REPRODUCTION_CLASS


def validate_mps_fp16_mapping(value: object) -> Stage1AMPSFP16Config:
    return Stage1AMPSFP16Config.from_mapping(value)


def mps_runtime_identity(
    *, hardware_family: str = "Apple M2 Max", architecture: str = "arm64"
) -> dict[str, object]:
    """Return publication-safe identity fields for an observed MPS runtime."""
    if architecture != "arm64":
        raise ValueError("MPS Stage 1A requires native arm64")
    if not hardware_family.strip() or "apple" not in hardware_family.casefold():
        raise ValueError("hardware_family must identify Apple hardware")
    return {
        "execution_class": EXECUTION_CLASS,
        "backend": "mps",
        "hardware_family": hardware_family,
        "architecture": architecture,
        "precision": EXECUTION_DTYPE,
        "official_bf16_reproduction": False,
        "t4_fp16_reproduction": False,
    }


def validate_mps_runtime_guards(
    *,
    machine: str,
    mps_built: bool,
    mps_available: bool,
    fallback_enabled: bool | None = None,
    high_watermark_ratio: float | None = None,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Fail closed on non-native hosts and memory-safety guard violations."""
    if machine != "arm64":
        raise ArtifactValidationError("MPS runtime must be native arm64")
    if not mps_built or not mps_available:
        raise ArtifactValidationError("MPS must be built and available")
    env = os.environ if environ is None else environ
    fallback_value = env.get("PYTORCH_ENABLE_MPS_FALLBACK", "").strip().casefold()
    fallback_requested = fallback_value in _TRUTHY
    if fallback_enabled is True or fallback_requested:
        raise ArtifactValidationError("PYTORCH_ENABLE_MPS_FALLBACK must be disabled")
    if fallback_value not in {"", "0", "false", "no", "off"}:
        raise ArtifactValidationError(
            "PYTORCH_ENABLE_MPS_FALLBACK has an ambiguous value"
        )
    value = high_watermark_ratio
    if value is None and "PYTORCH_MPS_HIGH_WATERMARK_RATIO" in env:
        try:
            value = float(env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"])
        except ValueError as error:
            raise ArtifactValidationError(
                "MPS high-watermark ratio is not numeric"
            ) from error
    if value is not None and (not math.isfinite(value) or value <= 0.0):
        raise ArtifactValidationError("MPS high-watermark ratio must remain > 0")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _external_hf_cache_root() -> Path:
    configured = os.environ.get("HF_HUB_CACHE")
    if configured:
        root = Path(configured).expanduser().resolve()
    elif os.environ.get("HF_HOME"):
        root = (Path(os.environ["HF_HOME"]).expanduser() / "hub").resolve()
    else:
        root = (REPOSITORY_ROOT.parent / "hf-cache-stage1a-mps" / "hub").resolve()
    if root == REPOSITORY_ROOT or root.is_relative_to(REPOSITORY_ROOT):
        raise RuntimeError("Hugging Face cache must be project-external")
    return root


def _hf_cache_repository_name(identifier: str) -> str:
    parts = identifier.split("/")
    if (
        len(parts) != 2
        or any(not part or part in {".", ".."} for part in parts)
        or any("--" in part or "/" in part or "\\" in part for part in parts)
    ):
        raise RuntimeError("Hugging Face repository identifier is not canonical")
    return f"models--{parts[0]}--{parts[1]}"


def _cache_root_for_override(snapshot: Path, *, identifier: str) -> Path:
    logical = snapshot.expanduser()
    if logical.is_symlink():
        raise RuntimeError("snapshot override cannot be a symlink")
    resolved = logical.resolve()
    expected_repository = _hf_cache_repository_name(identifier)
    if (
        resolved.parent.name != "snapshots"
        or len(resolved.parents) < 3
        or resolved.parents[1].name != expected_repository
    ):
        raise RuntimeError(
            "snapshot override must use the exact repository's canonical "
            "Hugging Face cache layout"
        )
    canonical_root = resolved.parents[2]
    candidates: list[Path] = []
    if os.environ.get("HF_HUB_CACHE"):
        candidates.append(Path(os.environ["HF_HUB_CACHE"]).expanduser().resolve())
    if os.environ.get("HF_HOME"):
        candidates.append((Path(os.environ["HF_HOME"]).expanduser() / "hub").resolve())
    for candidate in candidates:
        if resolved.is_relative_to(candidate):
            return candidate
    return canonical_root


def _validate_required_snapshot(
    snapshot: Path,
    *,
    cache_root: Path,
    identifier: str,
    revision: str,
    required_files: Sequence[str],
    role: str,
) -> list[dict[str, Any]]:
    """Validate and hash exactly the files consumed from one immutable snapshot."""
    logical = snapshot.expanduser()
    if logical.is_symlink() or not logical.is_dir():
        raise RuntimeError(f"{role} snapshot is not a real directory")
    resolved_snapshot = logical.resolve()
    resolved_cache = cache_root.expanduser().resolve()
    if resolved_snapshot == REPOSITORY_ROOT or resolved_snapshot.is_relative_to(
        REPOSITORY_ROOT
    ):
        raise RuntimeError(f"{role} snapshot must be project-external")
    if not resolved_snapshot.is_relative_to(resolved_cache):
        raise RuntimeError(f"{role} snapshot escapes the authorized HF cache")
    if resolved_snapshot.parent.name != "snapshots" or resolved_snapshot.parents[
        1
    ].name != _hf_cache_repository_name(identifier):
        raise RuntimeError(f"{role} snapshot repository identity is invalid")
    if logical.name != revision:
        raise RuntimeError(f"{role} snapshot path does not identify the exact revision")

    expected = set(required_files)
    observed: set[str] = set()
    for entry in logical.rglob("*"):
        relative = entry.relative_to(logical).as_posix()
        if entry.is_dir() and not entry.is_symlink():
            continue
        if entry.is_symlink():
            target = entry.resolve(strict=True)
            if not target.is_file() or not target.is_relative_to(resolved_cache):
                raise RuntimeError(f"{role} snapshot symlink escapes the HF cache")
        elif not entry.is_file():
            raise RuntimeError(f"{role} snapshot contains a special file")
        observed.add(relative)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(
            f"{role} snapshot file set is not exact; missing={missing}, extra={extra}"
        )

    manifest: list[dict[str, Any]] = []
    for name in sorted(expected):
        path = logical / name
        digest = _sha256_file(path)
        if path.suffix == ".safetensors" and path.is_symlink():
            blob_name = path.resolve(strict=True).name
            if _SHA256.fullmatch(blob_name) is None or blob_name != digest:
                raise RuntimeError(
                    f"{role} LFS blob identity does not match its content"
                )
        size = path.stat().st_size
        if size <= 0:
            raise RuntimeError(f"{role} snapshot contains an empty required file")
        manifest.append({"path": name, "size_bytes": size, "sha256": digest})
    return manifest


def _snapshot_phase_complete(
    snapshot: Path,
    *,
    cache_root: Path,
    identifier: str,
    revision: str,
    required_files: Sequence[str],
) -> bool:
    """Return whether an exact download phase is present and safely contained."""

    logical = snapshot.expanduser()
    if logical.is_symlink() or not logical.is_dir():
        return False
    resolved_snapshot = logical.resolve()
    resolved_cache = cache_root.expanduser().resolve()
    if (
        not resolved_snapshot.is_relative_to(resolved_cache)
        or resolved_snapshot.parent.name != "snapshots"
        or resolved_snapshot.parents[1].name != _hf_cache_repository_name(identifier)
        or logical.name != revision
    ):
        raise RuntimeError("snapshot phase has an invalid immutable identity")
    for name in required_files:
        relative = Path(name)
        if (
            relative.is_absolute()
            or len(relative.parts) != 1
            or relative.name != name
            or any(marker in name for marker in ("*", "?", "[", "]"))
        ):
            raise RuntimeError("snapshot phase allowlist is not an exact file list")
        candidate = logical / relative
        if candidate.is_symlink():
            try:
                target = candidate.resolve(strict=True)
            except (OSError, RuntimeError):
                return False
            if not target.is_file() or not target.is_relative_to(resolved_cache):
                raise RuntimeError("snapshot phase symlink escapes the HF cache")
        elif not candidate.is_file():
            return False
        try:
            if candidate.stat().st_size <= 0:
                return False
        except OSError:
            return False
    return True


def _download_snapshot_phase(
    *,
    identifier: str,
    revision: str,
    allow_patterns: Sequence[str],
    cache_root: Path,
    allow_download: bool,
) -> Path:
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("huggingface_hub is unavailable") from error
    arguments = {
        "repo_id": identifier,
        "revision": revision,
        "allow_patterns": list(allow_patterns),
        "cache_dir": str(cache_root),
    }
    local_error: Exception | None = None
    resolved: str | None = None
    try:
        resolved = snapshot_download(**arguments, local_files_only=True)
    except Exception as error:
        local_error = error
    if resolved is not None:
        local_path = Path(resolved)
        if _snapshot_phase_complete(
            local_path,
            cache_root=cache_root,
            identifier=identifier,
            revision=revision,
            required_files=allow_patterns,
        ):
            return local_path
    if not allow_download:
        raise RuntimeError(
            f"exact pinned local {identifier} snapshot phase is unavailable"
        ) from local_error
    try:
        resolved = snapshot_download(**arguments, local_files_only=False)
    except Exception as error:
        raise RuntimeError(
            f"immutable {identifier} snapshot phase failed: {type(error).__name__}"
        ) from error
    result = Path(resolved)
    if not _snapshot_phase_complete(
        result,
        cache_root=cache_root,
        identifier=identifier,
        revision=revision,
        required_files=allow_patterns,
    ):
        raise RuntimeError(f"immutable {identifier} snapshot phase is incomplete")
    return result


@dataclass(slots=True)
class MPSRuntimeBundle:
    model: Any
    torch: Any
    config: dict[str, Any]
    provenance: dict[str, Any]
    device: str
    dtype: str
    asset_manifest: dict[str, Any]
    model_only_forward: dict[str, Any]
    sparse_metadata_boundary: dict[str, Any]

    def close(self) -> None:
        reset = getattr(self.model, "reset_hooks", None)
        if callable(reset):
            reset(including_permanent=True)


def _assert_module_mps_fp16(module: Any, torch: Any, *, label: str) -> None:
    found = False
    for name, tensor in module.named_parameters():
        found = True
        if tensor.device.type != "mps":
            raise RuntimeError(f"{label} parameter {name} is not on MPS")
        if tensor.is_floating_point() and tensor.dtype != torch.float16:
            raise RuntimeError(f"{label} parameter {name} is not FP16")
    for name, tensor in module.named_buffers():
        if tensor.device.type != "mps":
            raise RuntimeError(f"{label} buffer {name} is not on MPS")
        if tensor.is_floating_point() and tensor.dtype not in {
            torch.float16,
            torch.float32,
        }:
            raise RuntimeError(
                f"{label} floating buffer {name} is neither FP16 nor an FP32 "
                "numerical buffer"
            )
    if not found:
        raise RuntimeError(f"{label} has no parameters")


def _build_asset_manifest(
    model_files: Sequence[Mapping[str, Any]],
    transcoder_files: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build publication-safe provenance from verified snapshot entries."""

    def asset(
        identifier: str, revision: str, files: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        records = [dict(item) for item in files]
        return {
            "identifier": identifier,
            "revision": revision,
            "files": records,
            "file_count": len(records),
            "total_bytes": sum(int(item["size_bytes"]) for item in records),
            "complete": True,
            "snapshot_containment_verified": True,
            "offline_ready": True,
        }

    manifest = {
        "verification": "exact_file_content_hashes_matched",
        "cache_policy": "project_external_huggingface_cache",
        "upstream_revision": OFFICIAL_UPSTREAM_REVISION,
        "immutable_revisions_only": True,
        "project_external_cache": True,
        "unmanifested_file_count": 0,
        "model": asset(OFFICIAL_MODEL_ID, OFFICIAL_MODEL_REVISION, model_files),
        "transcoder": asset(
            OFFICIAL_TRANSCODER_ID,
            OFFICIAL_TRANSCODER_REVISION,
            transcoder_files,
        ),
    }
    validate_json_value(manifest)
    assert_publication_safe(manifest)
    return manifest


def load_mps_runtime(
    config: Mapping[str, Any],
    *,
    allow_download: bool,
    model_snapshot: Path | None = None,
    transcoder_snapshot: Path | None = None,
    progressive: bool = True,
) -> MPSRuntimeBundle:
    """Load only exact pinned assets in the required progressive MPS sequence."""
    validate_mps_fp16_mapping(config)
    if not progressive:
        raise ValueError("the MPS runtime requires progressive loading")
    if (model_snapshot is None) != (transcoder_snapshot is None):
        raise ValueError("model and transcoder snapshot overrides must be paired")
    try:
        import random

        import numpy as np  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        import yaml  # type: ignore[import-untyped]
        from circuit_tracer import ReplacementModel  # type: ignore[import-not-found]
        from circuit_tracer.transcoder.single_layer_transcoder import (  # type: ignore[import-not-found]
            load_transcoder_set,
        )
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except ImportError as error:
        raise RuntimeError(
            f"MPS model runtime dependency unavailable: {type(error).__name__}"
        ) from error

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    backend = torch.backends.mps
    validate_mps_runtime_guards(
        machine=platform.machine(),
        mps_built=bool(backend.is_built()),
        mps_available=bool(backend.is_available()),
    )
    torch.mps.manual_seed(0)
    device = torch.device("mps")
    probe = torch.ones((2, 2), device=device, dtype=torch.float16)
    probe_result = probe @ probe
    if probe_result.device.type != "mps" or not bool(
        torch.isfinite(probe_result).all().item()
    ):
        raise RuntimeError("MPS float16 probe produced an invalid result")

    cache_root = _external_hf_cache_root()
    if model_snapshot is None:
        _download_snapshot_phase(
            identifier=OFFICIAL_MODEL_ID,
            revision=OFFICIAL_MODEL_REVISION,
            allow_patterns=MODEL_METADATA_FILES,
            cache_root=cache_root,
            allow_download=allow_download,
        )
        model_path = _download_snapshot_phase(
            identifier=OFFICIAL_MODEL_ID,
            revision=OFFICIAL_MODEL_REVISION,
            allow_patterns=MODEL_WEIGHT_FILES,
            cache_root=cache_root,
            allow_download=allow_download,
        )
        model_cache_root = cache_root
    else:
        model_path = model_snapshot.expanduser()
        model_cache_root = _cache_root_for_override(
            model_path, identifier=OFFICIAL_MODEL_ID
        )
    model_files = _validate_required_snapshot(
        model_path,
        cache_root=model_cache_root,
        identifier=OFFICIAL_MODEL_ID,
        revision=OFFICIAL_MODEL_REVISION,
        required_files=MODEL_REQUIRED_FILES,
        role="model",
    )

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        device_map={"": "mps"},
    )
    hf_model.eval()
    _assert_module_mps_fp16(hf_model, torch, label="Hugging Face model")
    model_only_tokens = tokenizer(OFFICIAL_ATTRIBUTION_PROMPT, return_tensors="pt")[
        "input_ids"
    ].to(device)
    if (
        model_only_tokens.ndim != 2
        or model_only_tokens.shape[0] != 1
        or model_only_tokens.shape[1] <= 0
        or model_only_tokens.device.type != "mps"
    ):
        raise RuntimeError("model-only tokenization is not a nonempty MPS batch")
    with torch.inference_mode():
        model_only_logits = hf_model(input_ids=model_only_tokens).logits
    if (
        model_only_logits.device.type != "mps"
        or model_only_logits.dtype != torch.float16
        or model_only_logits.ndim != 3
        or tuple(model_only_logits.shape[:2]) != (1, int(model_only_tokens.shape[1]))
        or int(model_only_logits.shape[2]) != 256_000
        or not bool(torch.isfinite(model_only_logits).all().item())
    ):
        raise RuntimeError("model-only MPS forward failed shape/device/finite checks")
    model_only_record = {
        "passed": True,
        "prompt": OFFICIAL_ATTRIBUTION_PROMPT,
        "token_count": int(model_only_tokens.shape[1]),
        "logits_shape": [int(value) for value in model_only_logits.shape],
        "tokenizer_revision": OFFICIAL_MODEL_REVISION,
        "device": "mps",
        "dtype": EXECUTION_DTYPE,
        "finite": True,
        "completed_before_transcoder_load": True,
    }
    del model_only_logits

    # Do not resolve or download any transcoder file before the model-only gate.
    if transcoder_snapshot is None:
        _download_snapshot_phase(
            identifier=OFFICIAL_TRANSCODER_ID,
            revision=OFFICIAL_TRANSCODER_REVISION,
            allow_patterns=TRANSCODER_METADATA_FILES,
            cache_root=cache_root,
            allow_download=allow_download,
        )
        transcoder_path = _download_snapshot_phase(
            identifier=OFFICIAL_TRANSCODER_ID,
            revision=OFFICIAL_TRANSCODER_REVISION,
            allow_patterns=TRANSCODER_WEIGHT_FILES,
            cache_root=cache_root,
            allow_download=allow_download,
        )
        transcoder_cache_root = cache_root
    else:
        transcoder_path = transcoder_snapshot.expanduser()
        transcoder_cache_root = _cache_root_for_override(
            transcoder_path, identifier=OFFICIAL_TRANSCODER_ID
        )
    transcoder_files = _validate_required_snapshot(
        transcoder_path,
        cache_root=transcoder_cache_root,
        identifier=OFFICIAL_TRANSCODER_ID,
        revision=OFFICIAL_TRANSCODER_REVISION,
        required_files=TRANSCODER_REQUIRED_FILES,
        role="transcoder",
    )
    # Immutable resolution is now complete.  Every remaining model/science
    # operation is forced through local files only.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    transcoder_config = yaml.safe_load(
        (transcoder_path / "config.yaml").read_text(encoding="utf-8")
    )
    if (
        not isinstance(transcoder_config, dict)
        or transcoder_config.get("model_kind") != "transcoder_set"
        or transcoder_config.get("model_name") not in {None, OFFICIAL_MODEL_ID}
    ):
        raise RuntimeError("pinned MPS transcoder config is invalid")
    layer_files = {
        layer: transcoder_path / f"layer_{layer}.safetensors" for layer in range(26)
    }
    transcoders = load_transcoder_set(
        layer_files,
        scan_name=f"{OFFICIAL_TRANSCODER_ID}@{OFFICIAL_TRANSCODER_REVISION}",
        feature_input_hook=transcoder_config["feature_input_hook"],
        feature_output_hook=transcoder_config["feature_output_hook"],
        activation=transcoder_config.get("activation"),
        k=transcoder_config.get("k"),
        device=device,
        dtype=torch.float16,
        lazy_encoder=False,
        lazy_decoder=True,
    )
    _assert_module_mps_fp16(transcoders, torch, label="transcoder set")
    for layer, transcoder in enumerate(transcoders):
        decoder_sample = transcoder._get_decoder_vectors(
            torch.tensor([0], dtype=torch.long, device="cpu")
        )
        if decoder_sample.device.type != "mps" or decoder_sample.dtype != torch.float16:
            raise RuntimeError(
                f"transcoder layer {layer} decoder sample is not MPS FP16"
            )
        del decoder_sample

    model = ReplacementModel.from_pretrained_and_transcoders(
        model_name=OFFICIAL_MODEL_ID,
        transcoders=transcoders,
        backend="transformerlens",
        device=device,
        dtype=torch.float16,
        hf_model=hf_model,
        tokenizer=tokenizer,
        revision=OFFICIAL_MODEL_REVISION,
        local_files_only=True,
    )
    model.eval()
    del hf_model
    gc.collect()
    torch.mps.empty_cache()
    _assert_module_mps_fp16(model, torch, label="TransformerLens replacement model")
    sparse_boundary = _validate_live_sparse_metadata_boundary(torch)

    asset_manifest = _build_asset_manifest(model_files, transcoder_files)
    provenance = {
        "model_backend": "transformerlens",
        "accelerator_backend": "mps",
        "backend": "mps",
        "device": "mps",
        "dtype": EXECUTION_DTYPE,
        "architecture": "arm64",
        "execution_class": EXECUTION_CLASS,
        "official_bf16_reproduction": False,
        "t4_fp16_reproduction": False,
        "upstream_revision": OFFICIAL_UPSTREAM_REVISION,
        "model_identifier": OFFICIAL_MODEL_ID,
        "model_revision": OFFICIAL_MODEL_REVISION,
        "transcoder_identifier": OFFICIAL_TRANSCODER_ID,
        "transcoder_revision": OFFICIAL_TRANSCODER_REVISION,
        "fallback_enabled": False,
        "offline_execution": True,
        "offload": "disk",
        "seeds": {"python": 0, "numpy": 0, "torch": 0},
        "sparse_metadata_boundary": "explicit_cpu_sparse_metadata_only",
    }
    return MPSRuntimeBundle(
        model=model,
        torch=torch,
        config=dict(config),
        provenance=provenance,
        device="mps",
        dtype=EXECUTION_DTYPE,
        asset_manifest=asset_manifest,
        model_only_forward=model_only_record,
        sparse_metadata_boundary=sparse_boundary,
    )


def model_only_forward_smoke(runtime: Any) -> bool:
    """Run a finite one-prompt forward check on an already loaded bundle."""
    model = runtime.model
    torch = runtime.torch
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("loaded MPS model has no tokenizer")
    input_ids = tokenizer(
        "The capital of state containing Dallas is", return_tensors="pt"
    )["input_ids"].to(runtime.device)
    with torch.inference_mode():
        logits = model(input_ids)
    if hasattr(logits, "logits"):
        logits = logits.logits
    if (
        logits.ndim != 3
        or logits.device.type != "mps"
        or logits.dtype != torch.float16
        or not bool(torch.isfinite(logits).all().item())
    ):
        raise RuntimeError("MPS model-only smoke output is invalid")
    return True


def load_transcoder(runtime: Any) -> Any:
    """Compatibility hook: transcoder loading is completed by the MPS loader."""
    if not hasattr(runtime.model, "transcoders"):
        raise RuntimeError("MPS runtime has no loaded transcoders")
    return runtime


def exception_chain(error: BaseException) -> tuple[BaseException, ...]:
    result: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and all(current is not item for item in result):
        result.append(current)
        current = current.__cause__ or current.__context__
    return tuple(result)


def is_mps_out_of_memory(error: BaseException) -> bool:
    """Recognize only explicit MPS/Metal OOM diagnostics."""
    for item in exception_chain(error):
        module = type(item).__module__.casefold()
        if type(item).__name__ == "OutOfMemoryError" and (
            "mps" in module or "metal" in module
        ):
            return True
        if isinstance(item, RuntimeError) and _MPS_OOM_MESSAGE.search(str(item)):
            return True
    return False


def sanitize_exception_message(error: BaseException, *, limit: int = 240) -> str:
    message = " ".join(str(error).split()) or type(error).__name__
    value = redact_sensitive({"message": message})["message"]
    return value[:limit] if isinstance(value, str) else "[REDACTED]"


def classify_mps_failure(error: BaseException) -> MPSRunStatus:
    if is_mps_out_of_memory(error):
        return MPSRunStatus.BLOCKED_RESOURCE
    combined = " ".join(str(item).casefold() for item in exception_chain(error))
    if any(
        marker in combined
        for marker in (
            "non-finite",
            "nan",
            "infinite",
            "jumprelu",
            "preactivation",
            "no-op",
            "noop",
        )
    ):
        return MPSRunStatus.FAILED_PRECISION
    if any(
        marker in combined
        for marker in ("access", "authentication", "permission", "gated", "401", "403")
    ):
        return MPSRunStatus.BLOCKED_ACCESS
    if "mps" in combined and any(
        marker in combined
        for marker in ("unavailable", "not built", "arm64", "fallback")
    ):
        return MPSRunStatus.BLOCKED_ENVIRONMENT
    return MPSRunStatus.FAILED_RUNTIME


def batch_deviation(selected_batch_size: int) -> str | None:
    if selected_batch_size not in OOM_BATCH_SEQUENCE:
        raise ValueError("selected batch size is outside 256, 128, 64")
    if selected_batch_size == 256:
        return None
    return (
        f"Attribution batch_size reduced from 256 to {selected_batch_size} after a "
        "positively identified MPS OOM; no numerical-equivalence claim is made."
    )


def should_retry_attempt(*, batch_size: int, category: str, failure_stage: str) -> bool:
    if batch_size not in OOM_BATCH_SEQUENCE:
        raise ValueError("attempt batch size is outside 256, 128, 64")
    return (
        category == "mps_out_of_memory"
        and failure_stage == "attribution"
        and batch_size != OOM_BATCH_SEQUENCE[-1]
    )


@dataclass(frozen=True, slots=True)
class MPSTelemetrySample:
    mps_current_allocated_bytes: int
    mps_driver_allocated_bytes: int
    mps_recommended_max_bytes: int | None
    process_rss_bytes: int
    system_memory_pressure: str | None = None
    swap_used_bytes: int | None = None
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "mps_current_allocated_bytes",
            "mps_driver_allocated_bytes",
            "process_rss_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("mps_recommended_max_bytes", "swap_used_bytes"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be non-negative or None")
        if not math.isfinite(float(self.timestamp)):
            raise ValueError("timestamp must be finite")


@dataclass(frozen=True, slots=True)
class MPSTelemetryAggregate:
    sample_count: int
    peak_current_allocated_bytes: int
    peak_driver_allocated_bytes: int
    peak_process_rss_bytes: int
    recommended_max_bytes: int | None
    peak_swap_used_bytes: int | None
    system_memory_pressures: tuple[str, ...]
    sampling_interval_seconds: float | None


def aggregate_mps_telemetry(
    samples: Sequence[MPSTelemetrySample],
) -> MPSTelemetryAggregate:
    if not samples:
        raise ValueError("at least one MPS telemetry sample is required")
    if any(not isinstance(sample, MPSTelemetrySample) for sample in samples):
        raise TypeError("samples must contain MPSTelemetrySample values")
    timestamps = [sample.timestamp for sample in samples]
    interval = (
        None
        if len(samples) < 2
        else max(0.0, max(timestamps) - min(timestamps)) / (len(samples) - 1)
    )
    recommendations = [
        sample.mps_recommended_max_bytes
        for sample in samples
        if sample.mps_recommended_max_bytes is not None
    ]
    swaps = [
        sample.swap_used_bytes
        for sample in samples
        if sample.swap_used_bytes is not None
    ]
    pressures = tuple(
        dict.fromkeys(
            sample.system_memory_pressure
            for sample in samples
            if sample.system_memory_pressure is not None
        )
    )
    return MPSTelemetryAggregate(
        len(samples),
        max(s.mps_current_allocated_bytes for s in samples),
        max(s.mps_driver_allocated_bytes for s in samples),
        max(s.process_rss_bytes for s in samples),
        max(recommendations) if recommendations else None,
        max(swaps) if swaps else None,
        pressures,
        interval,
    )


def aggregate_stage_attempt_telemetry(
    stages: Mapping[str, MPSTelemetryAggregate],
) -> dict[str, object]:
    if not stages or any(not name.strip() for name in stages):
        raise ValueError("at least one named stage aggregate is required")
    attempt_current = max(item.peak_current_allocated_bytes for item in stages.values())
    attempt_driver = max(item.peak_driver_allocated_bytes for item in stages.values())
    attempt_rss = max(item.peak_process_rss_bytes for item in stages.values())
    result: dict[str, object] = {
        "stage_peaks": {
            name: item.peak_driver_allocated_bytes for name, item in stages.items()
        },
        "attempt_peak_current_allocated_bytes": attempt_current,
        "attempt_peak_driver_allocated_bytes": attempt_driver,
        "attempt_peak_process_rss_bytes": attempt_rss,
    }
    validate_attempt_peak_invariants(result)
    return result


def validate_attempt_peak_invariants(record: Mapping[str, object]) -> None:
    stage = record.get("stage_peaks")
    if not isinstance(stage, Mapping) or not stage:
        raise ArtifactValidationError("stage_peaks must be a non-empty mapping")
    attempt = record.get("attempt_peak_driver_allocated_bytes")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise ArtifactValidationError("attempt MPS peak is invalid")
    for name, peak in stage.items():
        if (
            not isinstance(name, str)
            or isinstance(peak, bool)
            or not isinstance(peak, int)
            or peak < 0
        ):
            raise ArtifactValidationError("stage MPS peaks are invalid")
        if attempt < peak:
            raise ArtifactValidationError("attempt peak must be >= every stage peak")


def sample_mps_telemetry(
    torch_module: Any,
    *,
    system_memory_pressure: str | None = None,
    swap_used_bytes: int | None = None,
    timestamp: float | None = None,
) -> MPSTelemetrySample:
    """Sample MPS counters; no CUDA calls or fallback are used."""
    mps = getattr(torch_module, "mps", None)
    if mps is None:
        raise RuntimeError("torch module has no MPS namespace")
    current = int(mps.current_allocated_memory())
    driver = int(mps.driver_allocated_memory())
    recommended_fn = getattr(mps, "recommended_max_memory", None)
    recommended = int(recommended_fn()) if callable(recommended_fn) else None
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports ru_maxrss in bytes; Linux reports KiB.  The host is MPS/macOS,
    # but accepting a synthetic zero makes this sampler easy to test offline.
    if os.uname().sysname != "Darwin":
        rss *= 1024
    return MPSTelemetrySample(
        current,
        driver,
        recommended,
        rss,
        system_memory_pressure,
        swap_used_bytes,
        time.time() if timestamp is None else timestamp,
    )


# Public names used by orchestration code and intentionally backend-specific
# counterparts to the T4 CUDA peak helpers.
MPSMemorySample = MPSTelemetrySample
MPSMemoryAggregate = MPSTelemetryAggregate
aggregate_mps_memory = aggregate_mps_telemetry


@dataclass(frozen=True, slots=True)
class SparseCOOMetadata:
    """CPU-resident COO coordinates/values used for metadata-only operations."""

    indices: tuple[tuple[int, ...], ...]
    values: tuple[float, ...]
    shape: tuple[int, ...]


def make_sparse_coo_metadata(
    indices: Sequence[Sequence[int]], values: Sequence[float], shape: Sequence[int]
) -> SparseCOOMetadata:
    if not isinstance(indices, Sequence):
        raise ValueError("indices must be a coordinate matrix")
    normalized_rows: list[tuple[int, ...]] = []
    for row in indices:
        converted: list[int] = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError("COO indices must be integers")
            converted.append(item)
        normalized_rows.append(tuple(converted))
    normalized = tuple(normalized_rows)
    dimensions_list: list[int] = []
    for size in shape:
        if isinstance(size, bool) or not isinstance(size, int):
            raise ValueError("COO shape must contain integers")
        dimensions_list.append(_integer(size, "shape", 0))
    dimensions = tuple(dimensions_list)
    if len(normalized) != len(dimensions) or any(
        len(row) != len(values) for row in normalized
    ):
        raise ValueError("COO indices, values, and shape are inconsistent")
    for axis, row in enumerate(normalized):
        for coordinate in row:
            if coordinate < 0 or coordinate >= dimensions[axis]:
                raise ValueError("COO index is outside shape")
    numeric_values: list[float] = []
    for raw_value in values:
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError("COO values must be numeric")
        numeric_values.append(float(raw_value))
    numeric = tuple(numeric_values)
    if any(not math.isfinite(item) for item in numeric):
        raise ValueError("COO values must be finite")
    return SparseCOOMetadata(normalized, numeric, dimensions)


def sparse_coo_metadata_to_dense_cpu(metadata: SparseCOOMetadata) -> Any:
    """Materialize metadata on CPU, then let callers explicitly move dense data.

    PyTorch is optional at import time.  The pure-Python fallback is useful for
    deterministic tests and preserves duplicate-COO summation semantics.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - exercised only in minimal installs

        def zeros(shape: tuple[int, ...]) -> Any:
            if not shape:
                return 0.0
            return [zeros(shape[1:]) for _ in range(shape[0])]

        dense: Any = zeros(metadata.shape)
        for entry in range(len(metadata.values)):
            target: Any = dense
            for axis, coordinate in enumerate(row[entry] for row in metadata.indices):
                if axis == len(metadata.indices) - 1:
                    target[coordinate] += metadata.values[entry]
                else:
                    target = target[coordinate]
        return dense
    indices = torch.tensor(metadata.indices, dtype=torch.long, device="cpu")
    values = torch.tensor(metadata.values, dtype=torch.float32, device="cpu")
    return (
        torch.sparse_coo_tensor(indices, values, size=metadata.shape, device="cpu")
        .coalesce()
        .to_dense()
    )


def sparse_coo_to_dense_on_device(
    metadata: SparseCOOMetadata, device: str = "mps"
) -> Any:
    """Use CPU COO metadata while keeping the resulting scientific tensor explicit."""
    dense = sparse_coo_metadata_to_dense_cpu(metadata)
    if device == "cpu":
        return dense
    return dense.to(device=device)


def sparse_metadata_cpu_equivalent(
    indices: Sequence[Sequence[int]], values: Sequence[float], shape: Sequence[int]
) -> Any:
    """Convenience adapter used by the pinned upstream SparseCOO compatibility path."""
    return sparse_coo_metadata_to_dense_cpu(
        make_sparse_coo_metadata(indices, values, shape)
    )


# Descriptive aliases kept intentionally small so a runner can name the
# compatibility boundary without importing an implementation detail.  They
# do not create an MPS sparse tensor: only coordinates and values are held on
# CPU, and the dense scientific result is moved explicitly by the caller.
MPSCOOMetadata = SparseCOOMetadata
cpu_sparse_coo_metadata_to_dense = sparse_coo_metadata_to_dense_cpu
materialize_sparse_metadata_on_device = sparse_coo_to_dense_on_device


def _dense_to_cpu_sparse_metadata(dense: Any, torch: Any) -> tuple[Any, Any, Any]:
    """Extract only COO coordinates/values to CPU and verify an MPS dense round trip."""
    if dense.ndim < 1 or not bool(torch.isfinite(dense).all().item()):
        raise ArtifactValidationError("dense sparse-adapter input is invalid")
    coordinates = torch.nonzero(dense, as_tuple=False)
    if coordinates.numel():
        device_indices = coordinates.T.contiguous()
        device_values = dense[tuple(device_indices[axis] for axis in range(dense.ndim))]
    else:
        device_indices = torch.empty(
            (dense.ndim, 0), dtype=torch.long, device=dense.device
        )
        device_values = torch.empty((0,), dtype=dense.dtype, device=dense.device)
    cpu_indices = device_indices.detach().to(device="cpu", dtype=torch.long)
    cpu_values = device_values.detach().float().to(device="cpu")
    metadata = torch.sparse_coo_tensor(
        cpu_indices,
        cpu_values,
        size=tuple(int(value) for value in dense.shape),
        device="cpu",
    ).coalesce()
    reconstructed = torch.zeros_like(dense)
    if device_values.numel():
        # ``torch.nonzero`` produces unique coordinates, so replacement is
        # exactly equivalent here and avoids unsupported MPS FP16 accumulation.
        reconstructed.index_put_(
            tuple(device_indices[axis] for axis in range(dense.ndim)),
            device_values,
            accumulate=False,
        )
    if not bool(torch.allclose(reconstructed, dense, atol=5e-3, rtol=2e-3)):
        raise ArtifactValidationError(
            "explicit CPU sparse metadata boundary failed numerical equivalence"
        )
    return metadata, device_indices, device_values


def _validate_live_sparse_metadata_boundary(torch: Any) -> dict[str, Any]:
    source = torch.tensor(
        [[0.0, 1.5, 0.0], [-2.0, 0.0, 3.0]],
        device="mps",
        dtype=torch.float16,
    )
    metadata, indices, values = _dense_to_cpu_sparse_metadata(source, torch)
    if metadata.device.type != "cpu" or indices.device.type != "mps":
        raise ArtifactValidationError(
            "sparse metadata boundary used an unexpected device"
        )
    dense = torch.zeros_like(source)
    # The extracted COO coordinates are unique; no reduction is required.
    dense.index_put_((indices[0], indices[1]), values, accumulate=False)
    maximum_error = float(torch.max(torch.abs(dense - source)).item())
    if maximum_error > 5e-3:
        raise ArtifactValidationError(
            "live sparse metadata boundary exceeded tolerance"
        )
    return {
        "passed": True,
        "execution_deviation": "explicit_cpu_sparse_metadata_only",
        "dense_scientific_tensor_device": "mps",
        "metadata_device": "cpu",
        "nonzero_count": int(metadata._nnz()),
        "maximum_absolute_error": maximum_error,
        "absolute_tolerance": 5e-3,
        "relative_tolerance": 2e-3,
    }


def _mps_compute_attribution_components(
    transcoder_set: Any,
    mlp_inputs: Any,
    zero_positions: slice = slice(0, 1),
) -> dict[str, Any]:
    """Scientifically equivalent dense-MPS/CPU-metadata replacement for SparseMPS."""
    torch = __import__("torch")
    if (
        mlp_inputs.device.type != "mps"
        or mlp_inputs.dtype != torch.float16
        or mlp_inputs.ndim != 3
        or len(transcoder_set) != mlp_inputs.shape[0]
        or not bool(torch.isfinite(mlp_inputs).all().item())
    ):
        raise ArtifactValidationError(
            "attribution activations must remain finite dense MPS FP16"
        )
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
        layer_cpu = torch.full(
            (1, metadata._nnz()), layer, dtype=torch.long, device="cpu"
        )
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
            layer_device = torch.full_like(positions, layer)
            device_locations.append(torch.stack((layer_device, positions)))
        else:
            active_encoders = torch.empty(
                (0, transcoder.d_model),
                device=mlp_inputs.device,
                dtype=mlp_inputs.dtype,
            )
            scaled_decoders = torch.empty_like(active_encoders)
            device_locations.append(
                torch.empty((2, 0), device=mlp_inputs.device, dtype=torch.long)
            )
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
    active_count = int(activation_matrix._nnz())
    encoder_tensor = torch.cat(encoder_vectors, dim=0)
    decoder_tensor = torch.cat(decoder_vectors, dim=0)
    if (
        active_count <= 0
        or locations.shape[1] != active_count
        or encoder_tensor.shape != decoder_tensor.shape
        or encoder_tensor.shape[0] != active_count
        or encoder_tensor.shape[1] != mlp_inputs.shape[2]
    ):
        raise ArtifactValidationError("sparse metadata ordering/count is inconsistent")
    dense_outputs = (reconstruction, encoder_tensor, decoder_tensor, locations)
    if any(value.device.type != "mps" for value in dense_outputs) or not all(
        bool(torch.isfinite(value).all().item())
        for value in (reconstruction, encoder_tensor, decoder_tensor)
    ):
        raise ArtifactValidationError("sparse adapter reconstruction is non-finite")
    return {
        "activation_matrix": activation_matrix,
        "reconstruction": reconstruction,
        "encoder_vecs": encoder_tensor,
        "decoder_vecs": decoder_tensor,
        "encoder_to_decoder_map": torch.arange(active_count, device=mlp_inputs.device),
        "decoder_locations": locations,
    }


def _mps_context_compute_batch(
    context: Any,
    layers: Any,
    positions: Any,
    inject_values: Any,
    retain_graph: bool = True,
) -> Any:
    """Pinned upstream context operation with all dense indices made explicit MPS."""
    torch = __import__("torch")
    device = inject_values.device
    if (
        device.type != "mps"
        or inject_values.ndim != 2
        or layers.ndim != 1
        or positions.ndim != 1
        or len(layers) == 0
        or len(layers) != len(positions)
        or len(layers) != inject_values.shape[0]
        or not bool(torch.isfinite(inject_values).all().item())
    ):
        raise ArtifactValidationError("attribution batch input is not finite dense MPS")
    layers_device = layers.to(device=device, dtype=torch.long)
    positions_device = positions.to(device=device, dtype=torch.long)
    batch_size = int(context._resid_activations[0].shape[0])
    if any(
        activation is None or activation.device.type != "mps"
        for activation in context._resid_activations
    ):
        raise ArtifactValidationError("attribution residual cache left MPS")
    context._batch_buffer = torch.zeros(
        context._row_size,
        batch_size,
        dtype=inject_values.dtype,
        device=device,
    )
    batch_indices = torch.arange(len(layers_device), device=device)

    def inject(grads: Any, *, rows: Any, columns: Any, values: Any) -> Any:
        output = grads.clone().to(values.dtype)
        output.index_put_((rows, columns), values)
        return output.to(grads.dtype)

    handles: list[Any] = []
    unique_layers = [
        int(value)
        for value in layers_device.unique().detach().to(device="cpu").tolist()
    ]
    if unique_layers[0] < 0 or unique_layers[-1] >= len(context._resid_activations):
        raise ArtifactValidationError("attribution batch layer index is invalid")
    for layer in unique_layers:
        mask = layers_device == layer
        callback = lambda grads, mask=mask: inject(  # noqa: E731
            grads,
            rows=batch_indices[mask],
            columns=positions_device[mask],
            values=inject_values[mask],
        )
        handles.append(context._resid_activations[layer].register_hook(callback))
    try:
        last_layer = max(unique_layers)
        context._resid_activations[last_layer].backward(
            gradient=torch.zeros_like(context._resid_activations[last_layer]),
            retain_graph=retain_graph,
        )
    finally:
        for handle in handles:
            handle.remove()
    buffer, context._batch_buffer = context._batch_buffer, None
    result = buffer.T[: len(layers_device)]
    if result.device.type != "mps" or not bool(torch.isfinite(result).all().item()):
        raise ArtifactValidationError("attribution batch result left finite dense MPS")
    return result


@contextlib.contextmanager
def _mps_sparse_attribution_adapter(model: Any) -> Any:
    """Temporarily install the audited project-local MPS compatibility boundary."""
    from circuit_tracer.attribution import (  # type: ignore[import-not-found]
        attribute_transformerlens as attribute_module,
    )
    from circuit_tracer.attribution.context_transformerlens import (  # type: ignore[import-not-found]
        AttributionContext,
    )

    transcoder_set = model.transcoders
    original_components = transcoder_set.compute_attribution_components
    original_compute_batch = AttributionContext.compute_batch
    original_partial = attribute_module.compute_partial_influences
    usage = {"component_calls": 0, "batch_calls": 0, "partial_calls": 0}

    def components(bound: Any, inputs: Any, zero_positions: slice = slice(0, 1)) -> Any:
        usage["component_calls"] += 1
        return _mps_compute_attribution_components(bound, inputs, zero_positions)

    def compute_batch(
        context: Any,
        layers: Any,
        positions: Any,
        inject_values: Any,
        retain_graph: bool = True,
    ) -> Any:
        usage["batch_calls"] += 1
        return _mps_context_compute_batch(
            context,
            layers,
            positions,
            inject_values,
            retain_graph=retain_graph,
        )

    def partial(
        edge_matrix: Any, logit_p: Any, row_map: Any, *args: Any, **kwargs: Any
    ) -> Any:
        usage["partial_calls"] += 1
        kwargs.pop("device", None)
        torch = __import__("torch")
        if edge_matrix.device.type != "cpu":
            raise ArtifactValidationError("graph edge metadata unexpectedly left CPU")
        result = original_partial(
            edge_matrix,
            logit_p.detach().to(device="cpu"),
            row_map.detach().to(device="cpu"),
            *args,
            device="cpu",
            **kwargs,
        )
        if result.device.type != "cpu" or not bool(torch.isfinite(result).all().item()):
            raise ArtifactValidationError("CPU graph influence metadata is invalid")
        return result

    transcoder_set.compute_attribution_components = types.MethodType(
        components, transcoder_set
    )
    AttributionContext.compute_batch = compute_batch
    attribute_module.compute_partial_influences = partial
    try:
        yield usage
    finally:
        transcoder_set.compute_attribution_components = original_components
        AttributionContext.compute_batch = original_compute_batch
        attribute_module.compute_partial_influences = original_partial


def _write_mps_summary(path: Path | None, record: Mapping[str, Any]) -> dict[str, Any]:
    """Write only a small publication-safe MPS summary when requested."""
    result = dict(record)
    validate_json_value(result)
    assert_publication_safe(result)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            __import__("json").dumps(result, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return result


def _science_timing(torch: Any, started: float) -> dict[str, Any]:
    synchronize = getattr(getattr(torch, "mps", None), "synchronize", None)
    if callable(synchronize):
        synchronize()
    sample = sample_mps_telemetry(torch)
    current = sample.mps_current_allocated_bytes
    driver = sample.mps_driver_allocated_bytes
    rss = sample.process_rss_bytes
    if current <= 0 or driver <= 0 or rss <= 0:
        raise ArtifactValidationError("loaded MPS timing counters are not positive")
    result: dict[str, Any] = {
        "wall_seconds": time.perf_counter() - started,
        "mps_current_allocated_bytes": current,
        "mps_driver_allocated_bytes": driver,
        "process_peak_rss_bytes": rss,
    }
    if sample.mps_recommended_max_bytes is not None:
        result["mps_recommended_max_bytes"] = sample.mps_recommended_max_bytes
    return result


def _require_loaded_mps_bundle(bundle: Any) -> tuple[Any, Any]:
    if str(bundle.device) != "mps" or str(bundle.dtype) != EXECUTION_DTYPE:
        raise ArtifactValidationError("loaded bundle is not the MPS FP16 runtime")
    config = getattr(bundle, "config", None)
    if not isinstance(config, Mapping):
        raise ArtifactValidationError("loaded MPS bundle has no pinned configuration")
    validate_mps_fp16_mapping(config)
    torch = bundle.torch
    model = bundle.model
    if not hasattr(model, "transcoders") or len(model.transcoders) != 26:
        raise ArtifactValidationError("loaded MPS model must have 26 transcoders")
    if (
        int(model.cfg.n_layers) != 26
        or int(model.cfg.d_model) != 2304
        or str(model.cfg.device) != "mps"
        or model.cfg.dtype != torch.float16
        or int(model.transcoders.d_transcoder) != 16384
    ):
        raise ArtifactValidationError("loaded MPS model dimensions/device are invalid")
    provenance = getattr(bundle, "provenance", None)
    expected = {
        "model_backend": "transformerlens",
        "accelerator_backend": "mps",
        "device": "mps",
        "dtype": EXECUTION_DTYPE,
        "upstream_revision": OFFICIAL_UPSTREAM_REVISION,
        "model_identifier": OFFICIAL_MODEL_ID,
        "model_revision": OFFICIAL_MODEL_REVISION,
        "transcoder_identifier": OFFICIAL_TRANSCODER_ID,
        "transcoder_revision": OFFICIAL_TRANSCODER_REVISION,
        "fallback_enabled": False,
        "offline_execution": True,
        "offload": "disk",
        "seeds": {"python": 0, "numpy": 0, "torch": 0},
    }
    if not isinstance(provenance, Mapping) or any(
        provenance.get(name) != value for name, value in expected.items()
    ):
        raise ArtifactValidationError("loaded MPS asset/runtime provenance is invalid")
    model_only = getattr(bundle, "model_only_forward", None)
    model_only_shape = (
        model_only.get("logits_shape") if isinstance(model_only, Mapping) else None
    )
    model_only_tokens = (
        model_only.get("token_count") if isinstance(model_only, Mapping) else None
    )
    if (
        not isinstance(model_only, Mapping)
        or model_only.get("passed") is not True
        or model_only.get("prompt") != OFFICIAL_ATTRIBUTION_PROMPT
        or isinstance(model_only_tokens, bool)
        or not isinstance(model_only_tokens, int)
        or model_only_tokens <= 0
        or not isinstance(model_only_shape, list)
        or len(model_only_shape) != 3
        or model_only_shape[0] != 1
        or model_only_shape[1] != model_only_tokens
        or model_only_shape[2] != 256_000
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in model_only_shape
        )
        or model_only.get("tokenizer_revision") != OFFICIAL_MODEL_REVISION
        or model_only.get("device") != "mps"
        or model_only.get("dtype") != EXECUTION_DTYPE
        or model_only.get("finite") is not True
        or model_only.get("completed_before_transcoder_load") is not True
    ):
        raise ArtifactValidationError("progressive model-only MPS gate is invalid")
    validate_mps_runtime_guards(
        machine=platform.machine(),
        mps_built=bool(torch.backends.mps.is_built()),
        mps_available=bool(torch.backends.mps.is_available()),
    )
    _assert_module_mps_fp16(model, torch, label="loaded replacement model")
    return torch, model


def _manual_loaded_preactivations(bundle: Any, prompt: str) -> Any:
    torch, model = _require_loaded_mps_bundle(bundle)

    def names_filter(name: str) -> bool:
        return name.endswith(model.feature_input_hook)

    with torch.inference_mode():
        _, cache = model.run_with_cache(prompt, names_filter=names_filter)
    projected = []
    for layer in range(model.cfg.n_layers):
        name = f"blocks.{layer}.{model.feature_input_hook}"
        if name not in cache:
            raise ArtifactValidationError(
                f"missing feature-input hook for layer {layer}"
            )
        transcoder = model.transcoders[layer]
        inputs = cache[name]
        if inputs.device.type != "mps":
            raise ArtifactValidationError("feature-input cache left MPS")
        manual = torch.nn.functional.linear(
            inputs.to(transcoder.W_enc.dtype), transcoder.W_enc, transcoder.b_enc
        ).squeeze(0)
        manual[model.zero_positions] = 0
        projected.append(manual)
    return torch.stack(projected)


def _gate_sample(
    preactivation: Any, threshold: Any, activation: Any, feature_id: int
) -> dict[str, Any]:
    z = float(preactivation[20, -1, feature_id].item())
    tau = float(threshold[20, feature_id].item())
    value = float(activation[20, -1, feature_id].item())
    return {
        "layer": 20,
        "position": -1,
        "feature_id": feature_id,
        "preactivation": z,
        "threshold": tau,
        "post_gate_activation": value,
        "active": z > tau,
        "signed_margin": z - tau,
    }


def verify_mps_runtime_semantics(
    bundle: Any, summary_output: Path | None = None
) -> dict[str, Any]:
    """Verify loaded preactivation, threshold, strict gate, and feature semantics."""
    started = time.perf_counter()
    torch, model = _require_loaded_mps_bundle(bundle)
    numerics = _mapping(bundle.config.get("numerics", {}), "numerics")
    gate_atol = _finite(
        numerics.get("gate_absolute_tolerance", 5e-3), "gate_absolute_tolerance"
    )
    projection_atol = _finite(
        numerics.get("projection_absolute_tolerance", 5e-3),
        "projection_absolute_tolerance",
    )
    with torch.inference_mode():
        _, preactivation = model.get_activations(
            OFFICIAL_INTERVENTION_PROMPT,
            sparse=False,
            apply_activation_function=False,
        )
        _, activation = model.get_activations(
            OFFICIAL_INTERVENTION_PROMPT,
            sparse=False,
            apply_activation_function=True,
        )
        _, activation_repeat = model.get_activations(
            OFFICIAL_INTERVENTION_PROMPT,
            sparse=False,
            apply_activation_function=True,
        )
    if (
        preactivation.ndim != 3
        or preactivation.shape != activation.shape
        or preactivation.shape[0] != 26
        or preactivation.shape[2] != 16384
    ):
        raise ArtifactValidationError("loaded activation cache has an invalid shape")
    if (
        preactivation.device.type != "mps"
        or activation.device.type != "mps"
        or activation_repeat.device.type != "mps"
        or preactivation.dtype != torch.float16
        or activation.dtype != torch.float16
        or activation_repeat.dtype != torch.float16
    ):
        raise ArtifactValidationError("loaded activation caches are not MPS FP16")
    if (
        not bool(torch.isfinite(preactivation).all().item())
        or not bool(torch.isfinite(activation).all().item())
        or not bool(torch.isfinite(activation_repeat).all().item())
    ):
        raise ArtifactValidationError("loaded activation cache is non-finite")

    thresholds = []
    for layer in range(26):
        transcoder = model.transcoders[layer]
        if type(transcoder.activation_function).__name__ != "JumpReLU":
            raise ArtifactValidationError(
                "loaded transcoder activation is not JumpReLU"
            )
        threshold = transcoder.activation_function.threshold.detach()
        if (
            threshold.device.type != "mps"
            or threshold.dtype != torch.float16
            or tuple(threshold.shape) != (16384,)
            or tuple(transcoder.W_enc.shape) != (16384, 2304)
            or tuple(transcoder.b_enc.shape) != (16384,)
            or tuple(transcoder.b_dec.shape) != (2304,)
        ):
            raise ArtifactValidationError(
                "loaded transcoder tensor identity is invalid"
            )
        thresholds.append(threshold)
    threshold_tensor = torch.stack(thresholds)
    if not bool(torch.isfinite(threshold_tensor).all().item()):
        raise ArtifactValidationError("loaded JumpReLU thresholds are non-finite")

    expected = torch.where(
        preactivation > threshold_tensor[:, None, :],
        preactivation,
        torch.zeros_like(preactivation),
    )
    gate_error = float(torch.max(torch.abs(expected - activation)).item())
    if gate_error > gate_atol:
        raise ArtifactValidationError("loaded strict JumpReLU cache exceeds tolerance")
    equality_error = 0.0
    for layer in range(26):
        observed = model.transcoders[layer].activation_function(threshold_tensor[layer])
        equality_error = max(
            equality_error, float(torch.max(torch.abs(observed)).item())
        )
    if equality_error != 0.0:
        raise ArtifactValidationError("loaded JumpReLU activates threshold equality")

    manual = _manual_loaded_preactivations(bundle, OFFICIAL_INTERVENTION_PROMPT)
    projection_error = float(torch.max(torch.abs(manual - preactivation)).item())
    if projection_error > projection_atol:
        raise ArtifactValidationError(
            "loaded preactivation projection exceeds tolerance"
        )
    active_ids = torch.nonzero(
        preactivation[20, -1] > threshold_tensor[20], as_tuple=False
    ).flatten()
    inactive_ids = torch.nonzero(
        preactivation[20, -1] <= threshold_tensor[20], as_tuple=False
    ).flatten()
    if active_ids.numel() == 0 or inactive_ids.numel() == 0:
        raise ArtifactValidationError("loaded gate lacks active or inactive examples")
    active_id = int(active_ids[0].item())
    inactive_id = int(inactive_ids[0].item())
    baseline_activation = float(activation[20, -1, 341].item())
    repeated_baseline_activation = float(activation_repeat[20, -1, 341].item())
    baseline_repeat_error = abs(repeated_baseline_activation - baseline_activation)
    if not math.isfinite(baseline_activation) or baseline_activation <= 0.0:
        raise ArtifactValidationError("official intervention feature is inactive")
    if baseline_repeat_error > 2e-2:
        raise ArtifactValidationError(
            "official intervention feature baseline is not repeatable"
        )
    desired_values = [
        desired_activation(baseline_activation, alpha) for alpha in (0.0, 0.5, 1.0)
    ]
    if desired_values != [baseline_activation, 0.5 * baseline_activation, 0.0]:
        raise ArtifactValidationError("desired intervention mapping is inconsistent")

    payload = {
        "loaded_runtime": {
            "passed": True,
            "model_loaded": True,
            "transcoder_loaded": True,
            "model_device": "mps",
            "transcoder_device": "mps",
            "model_dtype": EXECUTION_DTYPE,
            "transcoder_dtype": EXECUTION_DTYPE,
            "model_only_forward_passed": bool(bundle.model_only_forward["passed"]),
            "model_only_forward": dict(bundle.model_only_forward),
            "output_finite": True,
            "fallback_used": False,
        },
        "preactivation": {
            "verified": True,
            "definition": "F.linear(feature_input, W_enc, b_enc)",
            "bias_convention": "b_enc is included; b_dec is excluded",
            "threshold_retrieved": True,
            "cache_shape": [int(value) for value in preactivation.shape],
            "projection_absolute_tolerance": projection_atol,
        },
        "gate_check": {
            "rule": "z if z > threshold else 0",
            "strict_greater_than": True,
            "equality_inactive": True,
            "equality_probe_maximum_absolute_output": equality_error,
            "absolute_tolerance": gate_atol,
            "active_example": _gate_sample(
                preactivation, threshold_tensor, activation, active_id
            ),
            "inactive_example": _gate_sample(
                preactivation, threshold_tensor, activation, inactive_id
            ),
            "official_intervention_source": _gate_sample(
                preactivation, threshold_tensor, activation, 341
            ),
        },
        "intervention_value_check": {
            "passed": True,
            "formula": "(1-alpha)*baseline_activation",
            "alphas": [0.0, 0.5, 1.0],
        },
        "feature": {
            "layer": 20,
            "position": -1,
            "feature_id": 341,
            "baseline_activation": baseline_activation,
        },
        "baseline_repeat_error": baseline_repeat_error,
        "projection_discrepancy": projection_error,
        "gate_discrepancy": max(gate_error, equality_error),
        "timing": _science_timing(torch, started),
        "nonfinite_count": 0,
    }
    return _write_mps_summary(summary_output, payload)


def _next_token_vector(logits: Any) -> Any:
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ArtifactValidationError(
            "intervention logits must be [1, position, vocab]"
        )
    return logits[0, -1].float()


def _maximum_relative_error(reference: Any, observed: Any, torch: Any) -> float:
    """Return the conventional maximum elementwise relative logit error."""
    difference = torch.abs(observed - reference)
    denominator = torch.clamp(torch.abs(reference), min=1e-12)
    value = float(torch.max(difference / denominator).item())
    if not math.isfinite(value):
        raise ArtifactValidationError("relative logit error is non-finite")
    return value


def _maximum_combined_tolerance_ratio(
    reference: Any,
    observed: Any,
    torch: Any,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> float:
    """Measure the exact allclose inequality without storing raw logits."""

    if absolute_tolerance <= 0.0 or relative_tolerance < 0.0:
        raise ArtifactValidationError("combined logit tolerances are invalid")
    difference = torch.abs(observed - reference)
    allowance = absolute_tolerance + relative_tolerance * torch.abs(reference)
    value = float(torch.max(difference / allowance).item())
    if not math.isfinite(value):
        raise ArtifactValidationError("combined logit tolerance ratio is non-finite")
    return value


def _intervention_feature_value(cache: Any) -> float:
    if cache is None:
        raise ArtifactValidationError("intervention activation cache is missing")
    if cache.ndim == 4:
        value = cache[20, 0, -1, 341]
    elif cache.ndim == 3:
        value = cache[20, -1, 341]
    else:
        raise ArtifactValidationError("intervention activation cache has invalid shape")
    return float(value.item())


@contextlib.contextmanager
def _capture_feature_intervention_write(model: Any, *, layer: int) -> Any:
    """Capture the dense residual write around the real upstream intervention hook."""
    original_builder = model._get_feature_intervention_hooks
    captured: dict[str, Any] = {}

    def builder(bound: Any, *args: Any, **kwargs: Any) -> Any:
        hooks, cached_logits, activation_cache = original_builder(*args, **kwargs)
        target_name = f"blocks.{layer}.{bound.feature_output_hook}"
        matches = []
        for index, (name, hook) in enumerate(hooks):
            function = getattr(hook, "func", hook)
            if (
                name == target_name
                and getattr(function, "__name__", "") == "intervention_hook"
            ):
                matches.append(index)
        if len(matches) != 1:
            raise ArtifactValidationError(
                "could not isolate the loaded feature-intervention write hook"
            )
        index = matches[0]

        def capture_before(activations: Any, hook: Any) -> Any:
            del hook
            captured["before"] = activations.detach().clone()
            return activations

        def capture_after(activations: Any, hook: Any) -> Any:
            del hook
            captured["after"] = activations.detach().clone()
            return activations

        instrumented = [
            *hooks[:index],
            (target_name, capture_before),
            hooks[index],
            (target_name, capture_after),
            *hooks[index + 1 :],
        ]
        return instrumented, cached_logits, activation_cache

    model._get_feature_intervention_hooks = types.MethodType(builder, model)
    try:
        yield captured
    finally:
        model._get_feature_intervention_hooks = original_builder


def _observed_intervention_coordinate(
    *,
    model: Any,
    torch: Any,
    captured: Mapping[str, Any],
    baseline_activation: float,
    layer: int,
    position: int,
    feature_id: int,
) -> tuple[float, float]:
    """Recover the actual scalar decoder write and its vector residual."""
    before = captured.get("before")
    after = captured.get("after")
    if before is None or after is None or before.shape != after.shape:
        raise ArtifactValidationError("feature-intervention write was not captured")
    if before.device.type != "mps" or after.device.type != "mps":
        raise ArtifactValidationError("feature-intervention write left MPS")
    delta = (after[0, position] - before[0, position]).float()
    feature_index = torch.tensor([feature_id], dtype=torch.long, device=delta.device)
    decoder = model.transcoders._get_decoder_vectors(layer, feature_index)
    if decoder.ndim != 2 or decoder.shape[0] != 1 or decoder.device.type != "mps":
        raise ArtifactValidationError("loaded decoder vector identity is invalid")
    direction = decoder[0].float()
    squared_norm = torch.sum(direction * direction)
    if not bool(torch.isfinite(squared_norm).item()) or float(squared_norm.item()) <= 0:
        raise ArtifactValidationError("loaded decoder vector has invalid norm")
    coefficient = torch.sum(delta * direction) / squared_norm
    residual = float(torch.max(torch.abs(delta - coefficient * direction)).item())
    observed = baseline_activation + float(coefficient.item())
    if not math.isfinite(observed) or not math.isfinite(residual):
        raise ArtifactValidationError("observed intervention coordinate is non-finite")
    return observed, residual


def reproduce_mps_intervention(
    bundle: Any, summary_output: Path | None = None
) -> dict[str, Any]:
    """Run baseline repeat and the exact no-op/half/full loaded interventions."""
    started = time.perf_counter()
    torch, model = _require_loaded_mps_bundle(bundle)
    numerics = _mapping(bundle.config.get("numerics", {}), "numerics")
    atol = _finite(numerics.get("noop_absolute_tolerance", 2e-2), "noop_atol")
    rtol = _finite(numerics.get("noop_relative_tolerance", 2e-3), "noop_rtol")
    determinant_atol = _finite(
        numerics.get("determinism_absolute_tolerance", atol), "determinism_atol"
    )
    determinant_rtol = _finite(
        numerics.get("determinism_relative_tolerance", rtol), "determinism_rtol"
    )
    gate_atol = _finite(numerics.get("gate_absolute_tolerance", 5e-3), "gate_atol")

    with torch.inference_mode():
        baseline_raw, baseline_cache = model.feature_intervention(
            OFFICIAL_INTERVENTION_PROMPT,
            [],
            freeze_attention=True,
            constrained_layers=None,
            apply_activation_function=True,
            sparse=False,
            return_activations=True,
        )
        repeat_raw, repeat_cache = model.feature_intervention(
            OFFICIAL_INTERVENTION_PROMPT,
            [],
            freeze_attention=True,
            constrained_layers=None,
            apply_activation_function=True,
            sparse=False,
            return_activations=True,
        )
    baseline_activation = _intervention_feature_value(baseline_cache)
    repeat_activation = _intervention_feature_value(repeat_cache)
    if not math.isfinite(baseline_activation) or baseline_activation <= 0.0:
        raise ArtifactValidationError("official intervention feature is inactive")
    desired = [
        desired_activation(baseline_activation, alpha) for alpha in (0.0, 0.5, 1.0)
    ]
    condition_records: list[tuple[Any, Any, float, float]] = []
    with torch.inference_mode():
        for value in desired:
            with _capture_feature_intervention_write(model, layer=20) as captured:
                raw, cache = model.feature_intervention(
                    OFFICIAL_INTERVENTION_PROMPT,
                    [(20, -1, 341, value)],
                    freeze_attention=True,
                    constrained_layers=None,
                    apply_activation_function=True,
                    sparse=False,
                    return_activations=True,
                )
            cached_baseline = _intervention_feature_value(cache)
            if abs(cached_baseline - baseline_activation) > gate_atol:
                raise ArtifactValidationError(
                    "intervention source cache changed before its decoder write"
                )
            observed, write_residual = _observed_intervention_coordinate(
                model=model,
                torch=torch,
                captured=captured,
                baseline_activation=cached_baseline,
                layer=20,
                position=-1,
                feature_id=341,
            )
            condition_records.append((raw, cache, observed, write_residual))

    baseline = _next_token_vector(baseline_raw)
    repeat = _next_token_vector(repeat_raw)
    conditions = [_next_token_vector(item[0]) for item in condition_records]
    vectors = [baseline, repeat, *conditions]
    if any(value.device.type != "mps" for value in vectors) or not all(
        bool(torch.isfinite(value).all().item()) for value in vectors
    ):
        raise ArtifactValidationError("intervention logits left MPS or are non-finite")
    probabilities = [torch.softmax(value, dim=-1) for value in vectors]
    if not all(bool(torch.isfinite(value).all().item()) for value in probabilities):
        raise ArtifactValidationError("intervention probabilities are non-finite")

    baseline_repeat_error = float(torch.max(torch.abs(repeat - baseline)).item())
    noop_error = float(torch.max(torch.abs(conditions[0] - baseline)).item())
    noop_relative_error = _maximum_relative_error(baseline, conditions[0], torch)
    baseline_repeat_ratio = _maximum_combined_tolerance_ratio(
        baseline,
        repeat,
        torch,
        absolute_tolerance=determinant_atol,
        relative_tolerance=determinant_rtol,
    )
    noop_combined_ratio = _maximum_combined_tolerance_ratio(
        baseline,
        conditions[0],
        torch,
        absolute_tolerance=atol,
        relative_tolerance=rtol,
    )
    if baseline_repeat_ratio > 1.0:
        raise ArtifactValidationError("baseline repeat exceeds determinism tolerance")
    if noop_combined_ratio > 1.0:
        raise ArtifactValidationError("no-op intervention exceeds tolerance")
    achieved = [item[2] for item in condition_records]
    write_residuals = [item[3] for item in condition_records]
    achieved_errors = [
        abs(observed - expected)
        for observed, expected in zip(achieved, desired, strict=True)
    ]
    if any(error > gate_atol for error in achieved_errors):
        raise ArtifactValidationError(
            "intervention did not achieve the requested activation"
        )
    if any(error > gate_atol for error in write_residuals):
        raise ArtifactValidationError(
            "intervention residual write is not the requested decoder coordinate"
        )
    if abs(repeat_activation - baseline_activation) > gate_atol:
        raise ArtifactValidationError("baseline feature activation is not repeatable")

    desired_summaries = []
    for alpha, expected, observed, _logits, probability in zip(
        (0.0, 0.5, 1.0), desired, achieved, conditions, probabilities[2:], strict=True
    ):
        finite = bool(torch.isfinite(probability).all().item())
        error = abs(observed - expected)
        desired_summaries.append(
            {
                "alpha": alpha,
                "expected_activation": expected,
                "observed_activation": observed,
                "absolute_error": error,
                "within_tolerance": error <= gate_atol,
                "output_finite": finite,
            }
        )
    payload = {
        "parameters": {
            "prompt": OFFICIAL_INTERVENTION_PROMPT,
            "feature": {"layer": 20, "position": -1, "feature_id": 341},
            "alphas": [0.0, 0.5, 1.0],
            "freeze_attention": True,
            "constrained_layers": None,
        },
        "baseline_activation_captured": True,
        "baseline_activation": baseline_activation,
        "baseline_repeat_error": baseline_repeat_error,
        "baseline_repeat_max_combined_tolerance_ratio": baseline_repeat_ratio,
        "baseline_noop_comparison": {
            "within_tolerance": True,
            "max_abs_error": noop_error,
            "max_rel_error": noop_relative_error,
            "max_combined_tolerance_ratio": noop_combined_ratio,
            "absolute_tolerance": atol,
            "relative_tolerance": rtol,
        },
        "desired_values": desired_summaries,
        "outputs_finite": True,
        "same_assets_and_runtime": True,
        "timing": _science_timing(torch, started),
        "nonfinite_count": 0,
    }
    return _write_mps_summary(summary_output, payload)


def _summarize_graph(
    graph: Any, torch: Any, *, require_ten_logits: bool
) -> dict[str, Any]:
    adjacency = graph.adjacency_matrix
    selected_features = graph.selected_features
    active_features = graph.active_features
    activation_values = graph.activation_values
    if (
        selected_features.ndim != 1
        or active_features.ndim != 2
        or activation_values.ndim != 1
    ):
        raise ArtifactValidationError("attribution feature metadata has invalid rank")
    if active_features.shape[1] != 3:
        raise ArtifactValidationError("attribution feature coordinates are invalid")
    selected_count = int(selected_features.numel())
    total_active_count = int(active_features.shape[0])
    if int(activation_values.numel()) != total_active_count:
        raise ArtifactValidationError(
            "attribution feature activation metadata is inconsistent"
        )
    logit_count = len(graph.logit_targets)
    probabilities = graph.logit_probabilities.detach()
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ArtifactValidationError("attribution adjacency is not square")
    if selected_count <= 0:
        raise ArtifactValidationError(
            "attribution graph has no active selected features"
        )
    if selected_count > 8192 or total_active_count < selected_count:
        raise ArtifactValidationError("attribution feature counts are inconsistent")
    if int(torch.unique(selected_features).numel()) != selected_count:
        raise ArtifactValidationError("attribution selected features are duplicated")
    if (
        int(selected_features.min().item()) < 0
        or int(selected_features.max().item()) >= total_active_count
    ):
        raise ArtifactValidationError("attribution selected feature index is invalid")
    if require_ten_logits and logit_count != 10:
        raise ArtifactValidationError(
            "full attribution graph does not contain 10 logits"
        )
    if logit_count <= 0 or len(probabilities) != logit_count:
        raise ArtifactValidationError("attribution logit metadata is inconsistent")
    if (
        not bool(torch.isfinite(adjacency).all().item())
        or not bool(torch.isfinite(probabilities).all().item())
        or not bool(torch.isfinite(activation_values).all().item())
    ):
        raise ArtifactValidationError("attribution graph contains non-finite values")
    edge_count = int(torch.count_nonzero(adjacency).item())
    if edge_count <= 0:
        raise ArtifactValidationError("attribution graph has no nonzero edges")
    n_positions = int(graph.n_pos)
    n_layers = int(graph.cfg.n_layers)
    error_count = n_layers * n_positions
    input_count = n_positions
    expected_nodes = selected_count + error_count + input_count + logit_count
    if tuple(adjacency.shape) != (expected_nodes, expected_nodes):
        raise ArtifactValidationError(
            "attribution adjacency dimensions are inconsistent"
        )
    return {
        "finite": True,
        "node_count": expected_nodes,
        "adjacency_shape": [int(value) for value in adjacency.shape],
        "active_feature_count": total_active_count,
        "selected_feature_count": selected_count,
        "error_node_count": error_count,
        "input_node_count": input_count,
        "logit_node_count": logit_count,
        "edge_count": edge_count,
    }


def reproduce_mps_attribution(
    bundle: Any,
    *,
    batch_size: int = 256,
    summary_output: Path | None = None,
    raw_graph_output: Path | None = None,
) -> dict[str, Any]:
    """Run an engineering smoke, then the exact full attribution with MPS adapter."""
    if batch_size not in OOM_BATCH_SEQUENCE:
        raise ValueError("MPS attribution batch must be 256, 128, or 64")
    started = time.perf_counter()
    torch, model = _require_loaded_mps_bundle(bundle)
    try:
        import circuit_tracer
        from circuit_tracer.graph import Graph  # type: ignore[import-not-found]
    except ImportError as error:
        raise ArtifactValidationError("circuit-tracer is unavailable") from error

    with _mps_sparse_attribution_adapter(model) as adapter_usage:
        smoke = circuit_tracer.attribute(
            prompt=OFFICIAL_ATTRIBUTION_PROMPT,
            model=model,
            max_n_logits=2,
            desired_logit_prob=0.95,
            max_feature_nodes=16,
            batch_size=16,
            offload="disk",
            verbose=False,
        )
        _summarize_graph(smoke, torch, require_ten_logits=False)
        graph = circuit_tracer.attribute(
            prompt=OFFICIAL_ATTRIBUTION_PROMPT,
            model=model,
            max_n_logits=10,
            desired_logit_prob=0.95,
            max_feature_nodes=8192,
            batch_size=batch_size,
            offload="disk",
            verbose=False,
        )
    if (
        adapter_usage["component_calls"] != 2
        or adapter_usage["batch_calls"] <= 0
        or adapter_usage["partial_calls"] <= 0
    ):
        raise ArtifactValidationError(
            "MPS sparse attribution compatibility adapter was not exercised"
        )
    graph_summary = _summarize_graph(graph, torch, require_ten_logits=True)

    raw_validation: dict[str, Any] = {
        "passed": True,
        "raw_graph_committed": False,
    }
    if raw_graph_output is not None:
        allowed = (REPOSITORY_ROOT / MPS_GENERATED_DIRECTORY).resolve()
        candidate = raw_graph_output.expanduser().resolve()
        if not candidate.is_relative_to(allowed) or candidate.is_symlink():
            raise ArtifactValidationError("raw MPS graph path is not allowlisted")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        try:
            graph.to_pt(str(candidate))
            reloaded = Graph.from_pt(str(candidate), map_location="cpu")
            reloaded_summary = _summarize_graph(
                reloaded, torch, require_ten_logits=True
            )
            if reloaded_summary != graph_summary:
                raise ArtifactValidationError(
                    "reloaded raw MPS graph changed its summary"
                )
        finally:
            candidate.unlink(missing_ok=True)
        if candidate.exists():
            raise ArtifactValidationError("validated raw MPS graph was retained")

    payload = {
        "parameters": {
            "prompt": OFFICIAL_ATTRIBUTION_PROMPT,
            "max_n_logits": 10,
            "desired_logit_probability": 0.95,
            "max_feature_nodes": 8192,
            "offload": "disk",
        },
        "accepted_batch_size": batch_size,
        "graph": graph_summary,
        "raw_validation": raw_validation,
        "timing": _science_timing(torch, started),
        "nonfinite_count": 0,
    }
    return _write_mps_summary(summary_output, payload)


__all__ = [name for name in globals() if not name.startswith("_")]
