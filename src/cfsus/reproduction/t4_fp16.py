"""Strict, dependency-light rules for the Stage 1A T4/FP16 adaptation.

This module deliberately does not relax the official BF16 configuration model.
It defines a second, explicitly hardware-adapted configuration and the small
terminal run manifest used by the Colab orchestrator.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
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

PROJECT_BASE_COMMIT = "13b42a5debe38def14f173530bcbc81ca3f8440e"
REPRODUCTION_CLASS = "hardware_adapted_fp16"
REFERENCE_DTYPE = "bfloat16"
EXECUTION_DTYPE = "float16"
REFERENCE_STATUS = "pending"
T4_EXPERIMENT_NAME = "stage1a_gemma2_2b_t4_fp16_hardware_adaptation"
T4_CLAIM_BOUNDARY = (
    "T4/FP16 hardware-adapted runtime/API reproduction using the pinned assets; "
    "native-BF16 reference reproduction remains pending."
)
OOM_BATCH_SEQUENCE = (256, 128, 64)
MAX_BUNDLE_MEMBER_BYTES = 5 * 1024 * 1024
MAX_BUNDLE_TOTAL_BYTES = 20 * 1024 * 1024
T4_RESULT_DIRECTORY = "results/stage1a_t4_fp16"
T4_GENERATED_DIRECTORY = "results/generated/stage1a_t4_fp16"
T4_SMALL_FILES = frozenset(
    {
        "environment_manifest.json",
        "asset_manifest.json",
        "attribution_summary.json",
        "intervention_summary.json",
        "semantics_summary.json",
        "checksums.sha256",
        "stage1a_t4_fp16_run_manifest.json",
    }
)
T4_SUMMARY_FILES = frozenset(
    T4_SMALL_FILES - {"checksums.sha256", "stage1a_t4_fp16_run_manifest.json"}
)

_SHA40 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CUDA_OOM_MESSAGE = re.compile(
    r"(?:cuda(?: error)?:?\s*out of memory|cuda allocation[^\n]*out of memory|"
    r"cublas_status_alloc_failed)",
    re.IGNORECASE,
)
_OVERCLAIM = re.compile(r"(?:official|exact)\s+(?:bf16\s+)?reproduction", re.I)


class T4RunStatus(StrEnum):
    """Only terminal statuses allowed for the T4/FP16 run manifest."""

    COMPLETED = "completed_hardware_adapted_fp16"
    BLOCKED_ACCESS = "blocked_access"
    BLOCKED_RESOURCE = "blocked_resource"
    FAILED_PRECISION = "failed_precision"
    FAILED_RUNTIME = "failed_runtime"
    PREPARED = "prepared_not_executed"


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
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
            f"{label} must be an exact lowercase nonzero 40-character SHA"
        )
    if result != expected:
        raise Stage1AConfigError(f"{label} does not match the immutable pin")
    return result


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Stage1AConfigError(
            f"{label} must be an integer greater than or equal to {minimum}"
        )
    return value


def _finite_number(value: object, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Stage1AConfigError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise Stage1AConfigError(f"{label} must be finite and at least {minimum}")
    return result


def _boolean(value: object, expected: bool, label: str) -> bool:
    if value is not expected:
        raise Stage1AConfigError(f"{label} must be {expected}")
    return expected


def _relative_under(value: object, prefix: str, suffix: str, label: str) -> str:
    text = _text(value, label)
    path = PurePosixPath(text)
    prefix_parts = PurePosixPath(prefix).parts
    if (
        path.is_absolute()
        or "\\" in text
        or ".." in path.parts
        or "." in path.parts
        or path.as_posix() != text
        or path.parts[: len(prefix_parts)] != prefix_parts
        or not text.endswith(suffix)
    ):
        raise Stage1AConfigError(
            f"{label} must be a normalized path under {prefix}/ ending in {suffix}"
        )
    return text


@dataclass(frozen=True, slots=True)
class T4RuntimeConfig:
    backend: str
    device: str
    dtype: str


@dataclass(frozen=True, slots=True)
class T4RetryPolicy:
    batch_sizes: tuple[int, ...]
    trigger: str
    fresh_process: bool
    clear_cuda_cache_between_attempts: bool


@dataclass(frozen=True, slots=True)
class T4NumericsConfig:
    gate_absolute_tolerance: float
    projection_absolute_tolerance: float
    noop_absolute_tolerance: float
    noop_relative_tolerance: float
    determinism_absolute_tolerance: float
    determinism_relative_tolerance: float
    model_parameter_samples_per_tensor: int


@dataclass(frozen=True, slots=True)
class T4ArtifactPaths:
    raw_graph: str
    environment_manifest: str
    asset_manifest: str
    attribution_summary: str
    intervention_summary: str
    semantics_summary: str
    checksums: str
    run_manifest: str


@dataclass(frozen=True, slots=True)
class Stage1AT4FP16Config:
    """Validated identity and execution policy for the separate T4 path."""

    schema_version: int
    experiment_name: str
    reproduction_class: str
    project_base_commit: str
    reference_dtype: str
    execution_dtype: str
    reference_status: str
    runtime: T4RuntimeConfig
    oom_retry: T4RetryPolicy
    numerics: T4NumericsConfig
    artifacts: T4ArtifactPaths

    @classmethod
    def from_mapping(cls, value: object) -> Stage1AT4FP16Config:
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
            "artifacts",
        }
        _exact_keys(data, expected_top, "configuration")
        if _integer(data["schema_version"], "schema_version") != 1:
            raise Stage1AConfigError("schema_version must equal 1")
        experiment_name = _exact_text(
            data["experiment_name"], T4_EXPERIMENT_NAME, "experiment_name"
        )
        claim_boundary = _exact_text(
            data["claim_boundary"], T4_CLAIM_BOUNDARY, "claim_boundary"
        )
        if _OVERCLAIM.search(experiment_name) or _OVERCLAIM.search(claim_boundary):
            raise Stage1AConfigError(
                "T4 labels must not claim exact/official reproduction"
            )

        upstream = _mapping(data["upstream"], "upstream")
        _exact_keys(upstream, {"repository", "revision"}, "upstream")
        _exact_text(
            upstream["repository"], OFFICIAL_UPSTREAM_REPOSITORY, "upstream.repository"
        )
        _sha(upstream["revision"], OFFICIAL_UPSTREAM_REVISION, "upstream.revision")
        cls._validate_asset(
            data["model"],
            label="model",
            identifier=OFFICIAL_MODEL_ID,
            revision=OFFICIAL_MODEL_REVISION,
            path_suffix="google-gemma-2-2b",
        )
        cls._validate_asset(
            data["transcoder"],
            label="transcoder",
            identifier=OFFICIAL_TRANSCODER_ID,
            revision=OFFICIAL_TRANSCODER_REVISION,
            path_suffix="mwhanna-gemma-scope-transcoders",
        )
        cls._validate_environment(data["environment"])
        cls._validate_seeds(data["seeds"])
        cls._validate_asset_policy(data["asset_policy"])
        cls._validate_attribution(data["attribution"])
        cls._validate_intervention(data["intervention"])

        runtime_data = _mapping(data["runtime"], "runtime")
        _exact_keys(runtime_data, {"backend", "device", "dtype"}, "runtime")
        runtime = T4RuntimeConfig(
            backend=_exact_text(
                runtime_data["backend"], "transformerlens", "runtime.backend"
            ),
            device=_exact_text(runtime_data["device"], "cuda", "runtime.device"),
            dtype=_exact_text(runtime_data["dtype"], EXECUTION_DTYPE, "runtime.dtype"),
        )
        retry = cls._validate_retry(data["oom_retry"])
        numerics = cls._validate_numerics(data["numerics"])
        artifacts = cls._validate_artifacts(data["artifacts"])
        return cls(
            schema_version=1,
            experiment_name=experiment_name,
            reproduction_class=_exact_text(
                data["reproduction_class"],
                REPRODUCTION_CLASS,
                "reproduction_class",
            ),
            project_base_commit=_sha(
                data["project_base_commit"],
                PROJECT_BASE_COMMIT,
                "project_base_commit",
            ),
            reference_dtype=_exact_text(
                data["reference_dtype"], REFERENCE_DTYPE, "reference_dtype"
            ),
            execution_dtype=_exact_text(
                data["execution_dtype"], EXECUTION_DTYPE, "execution_dtype"
            ),
            reference_status=_exact_text(
                data["reference_status"], REFERENCE_STATUS, "reference_status"
            ),
            runtime=runtime,
            oom_retry=retry,
            numerics=numerics,
            artifacts=artifacts,
        )

    @staticmethod
    def _validate_asset(
        value: object,
        *,
        label: str,
        identifier: str,
        revision: str,
        path_suffix: str,
    ) -> None:
        data = _mapping(value, label)
        _exact_keys(data, {"identifier", "revision", "snapshot_path"}, label)
        _exact_text(data["identifier"], identifier, f"{label}.identifier")
        _sha(data["revision"], revision, f"{label}.revision")
        expected_path = f"{T4_GENERATED_DIRECTORY}/assets/{path_suffix}"
        _exact_text(data["snapshot_path"], expected_path, f"{label}.snapshot_path")

    @staticmethod
    def _validate_environment(value: object) -> None:
        data = _mapping(value, "environment")
        _exact_keys(data, {"python", "pytorch", "cuda_wheel"}, "environment")
        _exact_text(data["python"], "3.11", "environment.python")
        _exact_text(data["pytorch"], "2.6.0", "environment.pytorch")
        _exact_text(data["cuda_wheel"], "cu124", "environment.cuda_wheel")

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
            data, {"allow_download", "require_offline_execution"}, "asset_policy"
        )
        _boolean(data["allow_download"], True, "asset_policy.allow_download")
        _boolean(
            data["require_offline_execution"],
            True,
            "asset_policy.require_offline_execution",
        )

    @staticmethod
    def _validate_attribution(value: object) -> None:
        data = _mapping(value, "attribution")
        _exact_keys(
            data,
            {
                "prompt",
                "max_n_logits",
                "desired_logit_probability",
                "max_feature_nodes",
                "batch_size",
                "offload",
            },
            "attribution",
        )
        _exact_text(data["prompt"], OFFICIAL_ATTRIBUTION_PROMPT, "attribution.prompt")
        expected = {
            "max_n_logits": 10,
            "desired_logit_probability": 0.95,
            "max_feature_nodes": 8192,
            "batch_size": 256,
        }
        for name, required in expected.items():
            if data[name] != required or isinstance(data[name], bool):
                raise Stage1AConfigError(f"attribution.{name} must equal {required}")
        _exact_text(data["offload"], "disk", "attribution.offload")

    @staticmethod
    def _validate_intervention(value: object) -> None:
        data = _mapping(value, "intervention")
        _exact_keys(
            data,
            {
                "prompt",
                "feature",
                "alphas",
                "freeze_attention",
                "constrained_layers",
            },
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
            raise Stage1AConfigError(
                "intervention.feature must equal the pinned coordinates (20, -1, 341)"
            )
        alphas = data["alphas"]
        if (
            isinstance(alphas, (str, bytes))
            or not isinstance(alphas, Sequence)
            or tuple(alphas) != (0.0, 0.5, 1.0)
        ):
            raise Stage1AConfigError("intervention.alphas must equal [0.0, 0.5, 1.0]")
        _boolean(data["freeze_attention"], True, "intervention.freeze_attention")
        if data["constrained_layers"] is not None:
            raise Stage1AConfigError("intervention.constrained_layers must be null")

    @staticmethod
    def _validate_retry(value: object) -> T4RetryPolicy:
        data = _mapping(value, "oom_retry")
        _exact_keys(
            data,
            {
                "batch_sizes",
                "trigger",
                "fresh_process",
                "clear_cuda_cache_between_attempts",
            },
            "oom_retry",
        )
        batches = data["batch_sizes"]
        if (
            isinstance(batches, (str, bytes))
            or not isinstance(batches, Sequence)
            or tuple(batches) != OOM_BATCH_SEQUENCE
        ):
            raise Stage1AConfigError("oom_retry.batch_sizes must equal [256, 128, 64]")
        return T4RetryPolicy(
            batch_sizes=OOM_BATCH_SEQUENCE,
            trigger=_exact_text(
                data["trigger"], "cuda_out_of_memory_only", "oom_retry.trigger"
            ),
            fresh_process=_boolean(
                data["fresh_process"], True, "oom_retry.fresh_process"
            ),
            clear_cuda_cache_between_attempts=_boolean(
                data["clear_cuda_cache_between_attempts"],
                True,
                "oom_retry.clear_cuda_cache_between_attempts",
            ),
        )

    @staticmethod
    def _validate_numerics(value: object) -> T4NumericsConfig:
        data = _mapping(value, "numerics")
        fields = {
            "gate_absolute_tolerance",
            "projection_absolute_tolerance",
            "noop_absolute_tolerance",
            "noop_relative_tolerance",
            "determinism_absolute_tolerance",
            "determinism_relative_tolerance",
            "model_parameter_samples_per_tensor",
        }
        _exact_keys(data, fields, "numerics")
        expected = {
            "gate_absolute_tolerance": 5.0e-3,
            "projection_absolute_tolerance": 5.0e-3,
            "noop_absolute_tolerance": 2.0e-2,
            "noop_relative_tolerance": 2.0e-3,
            "determinism_absolute_tolerance": 2.0e-2,
            "determinism_relative_tolerance": 2.0e-3,
        }
        parsed = {
            name: _finite_number(data[name], f"numerics.{name}") for name in expected
        }
        if parsed != expected:
            raise Stage1AConfigError(
                "numerical tolerances differ from the preregistration"
            )
        sample_count = _integer(
            data["model_parameter_samples_per_tensor"],
            "numerics.model_parameter_samples_per_tensor",
            minimum=1,
        )
        if sample_count != 16:
            raise Stage1AConfigError(
                "numerics.model_parameter_samples_per_tensor must equal 16"
            )
        return T4NumericsConfig(
            **parsed,
            model_parameter_samples_per_tensor=sample_count,
        )

    @staticmethod
    def _validate_artifacts(value: object) -> T4ArtifactPaths:
        data = _mapping(value, "artifacts")
        fields = {
            "raw_graph",
            "environment_manifest",
            "asset_manifest",
            "attribution_summary",
            "intervention_summary",
            "semantics_summary",
            "checksums",
            "run_manifest",
        }
        _exact_keys(data, fields, "artifacts")
        paths: dict[str, str] = {}
        paths["raw_graph"] = _relative_under(
            data["raw_graph"], T4_GENERATED_DIRECTORY, ".pt", "artifacts.raw_graph"
        )
        expected_names = {
            "environment_manifest": "environment_manifest.json",
            "asset_manifest": "asset_manifest.json",
            "attribution_summary": "attribution_summary.json",
            "intervention_summary": "intervention_summary.json",
            "semantics_summary": "semantics_summary.json",
            "checksums": "checksums.sha256",
            "run_manifest": "stage1a_t4_fp16_run_manifest.json",
        }
        for name, filename in expected_names.items():
            paths[name] = _exact_text(
                data[name], f"{T4_RESULT_DIRECTORY}/{filename}", f"artifacts.{name}"
            )
        if len(set(paths.values())) != len(paths):
            raise Stage1AConfigError("artifact paths must be unique")
        return T4ArtifactPaths(**paths)


def is_t4_fp16_mapping(value: Mapping[str, object] | dict[str, Any]) -> bool:
    """Return whether a raw mapping explicitly declares the separate T4 class."""

    return value.get("reproduction_class") == REPRODUCTION_CLASS


def validate_t4_fp16_mapping(value: object) -> Stage1AT4FP16Config:
    """Validate and return a typed T4/FP16 configuration."""

    return Stage1AT4FP16Config.from_mapping(value)


def exception_chain(error: BaseException) -> tuple[BaseException, ...]:
    """Return a cycle-safe explicit/implicit exception chain."""

    result: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and all(current is not item for item in result):
        result.append(current)
        current = current.__cause__ or current.__context__
    return tuple(result)


def is_cuda_out_of_memory(error: BaseException) -> bool:
    """Accept only CUDA-specific OOM types or CUDA-specific OOM messages."""

    for item in exception_chain(error):
        type_name = type(item).__name__
        module_name = type(item).__module__.casefold()
        if type_name == "OutOfMemoryError" and (
            module_name.startswith("torch") or "cuda" in module_name
        ):
            return True
        if isinstance(item, RuntimeError) and _CUDA_OOM_MESSAGE.search(str(item)):
            return True
    return False


def sanitize_exception_message(error: BaseException, *, limit: int = 240) -> str:
    """Return a one-line, publication-safe diagnostic without credential leakage."""

    message = " ".join(str(error).split()) or type(error).__name__
    redacted = redact_sensitive({"message": message})["message"]
    if not isinstance(redacted, str):
        return "[REDACTED]"
    return redacted[:limit]


def classify_t4_failure(error: BaseException) -> T4RunStatus:
    """Map a runtime failure to a truthful terminal T4 status."""

    if is_cuda_out_of_memory(error):
        return T4RunStatus.BLOCKED_RESOURCE
    combined = " ".join(str(item).casefold() for item in exception_chain(error))
    precision_markers = (
        "non-finite",
        "nan",
        "infinite",
        "no-op",
        "noop",
        "not deterministic",
        "within tolerance",
        "jumprelu",
        "preactivation",
    )
    if any(marker in combined for marker in precision_markers):
        return T4RunStatus.FAILED_PRECISION
    access_markers = (
        "access",
        "authentication",
        "permission",
        "gated",
        "401",
        "403",
    )
    if any(marker in combined for marker in access_markers):
        return T4RunStatus.BLOCKED_ACCESS
    return T4RunStatus.FAILED_RUNTIME


def batch_deviation(selected_batch_size: int) -> str | None:
    """Describe a reduced-batch engineering deviation without equivalence claims."""

    if selected_batch_size not in OOM_BATCH_SEQUENCE:
        raise ValueError("selected batch size is outside the preregistered sequence")
    if selected_batch_size == OOM_BATCH_SEQUENCE[0]:
        return None
    return (
        f"Attribution batch_size reduced from 256 to {selected_batch_size} after "
        "a positively identified CUDA OOM; no bitwise-equivalence claim is made."
    )


def should_retry_attempt(*, batch_size: int, category: str, failure_stage: str) -> bool:
    """Return whether the next fresh-process attribution batch is authorized."""

    if batch_size not in OOM_BATCH_SEQUENCE:
        raise ValueError("attempt batch size is outside the preregistered sequence")
    return (
        category == "cuda_out_of_memory"
        and failure_stage == "attribution"
        and batch_size != OOM_BATCH_SEQUENCE[-1]
    )


def _manifest_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{label} must be a JSON object")
    return value


def _manifest_bool(value: object, label: str, expected: bool | None = None) -> bool:
    if not isinstance(value, bool):
        raise ArtifactValidationError(f"{label} must be a boolean")
    if expected is not None and value is not expected:
        raise ArtifactValidationError(f"{label} must be {expected}")
    return value


def validate_t4_run_manifest(value: object) -> None:
    """Validate terminal status, provenance, retries, finiteness, and readiness."""

    validate_json_value(value)
    assert_publication_safe(value)
    record = _manifest_mapping(value, "T4 run manifest")
    required = {
        "schema_version",
        "status",
        "reproduction_class",
        "claim_boundary",
        "project",
        "upstream",
        "model",
        "transcoder",
        "runtime",
        "attribution",
        "retry_history",
        "timings",
        "checks",
        "artifacts",
        "readiness",
        "bf16_reference",
    }
    if set(record) != required:
        raise ArtifactValidationError("T4 run manifest keys are not exact")
    if record["schema_version"] != 1 or isinstance(record["schema_version"], bool):
        raise ArtifactValidationError("T4 run manifest schema_version must be 1")
    try:
        status = T4RunStatus(record["status"])
    except (TypeError, ValueError) as error:
        raise ArtifactValidationError("invalid T4 terminal status") from error
    if record["reproduction_class"] != REPRODUCTION_CLASS:
        raise ArtifactValidationError("T4 reproduction_class is invalid")
    if record["claim_boundary"] != T4_CLAIM_BOUNDARY:
        raise ArtifactValidationError("T4 claim boundary is invalid")

    project = _manifest_mapping(record["project"], "project")
    if project.get("base_commit") != PROJECT_BASE_COMMIT:
        raise ArtifactValidationError("project base commit is invalid")
    commit = project.get("execution_commit")
    if not isinstance(commit, str) or _SHA40.fullmatch(commit) is None:
        raise ArtifactValidationError("project execution commit must be a SHA")
    _manifest_bool(project.get("dirty"), "project.dirty", expected=False)

    pins = (
        ("upstream", record["upstream"], "revision", OFFICIAL_UPSTREAM_REVISION),
        ("model", record["model"], "revision", OFFICIAL_MODEL_REVISION),
        (
            "transcoder",
            record["transcoder"],
            "revision",
            OFFICIAL_TRANSCODER_REVISION,
        ),
    )
    for label, raw, key, expected in pins:
        section = _manifest_mapping(raw, label)
        if section.get(key) != expected:
            raise ArtifactValidationError(f"{label} immutable revision is invalid")
    if _manifest_mapping(record["upstream"], "upstream").get("repository") != (
        OFFICIAL_UPSTREAM_REPOSITORY
    ):
        raise ArtifactValidationError("upstream repository is invalid")
    if (
        _manifest_mapping(record["model"], "model").get("identifier")
        != OFFICIAL_MODEL_ID
    ):
        raise ArtifactValidationError("model identifier is invalid")
    if (
        _manifest_mapping(record["transcoder"], "transcoder").get("identifier")
        != OFFICIAL_TRANSCODER_ID
    ):
        raise ArtifactValidationError("transcoder identifier is invalid")

    runtime = _manifest_mapping(record["runtime"], "runtime")
    required_runtime = {
        "backend",
        "device",
        "gpu_name",
        "compute_capability",
        "torch_version",
        "torch_cuda_version",
        "reference_dtype",
        "execution_dtype",
        "bf16_supported",
    }
    if not required_runtime.issubset(runtime):
        raise ArtifactValidationError("runtime provenance is incomplete")
    if (
        runtime["backend"] != "transformerlens"
        or runtime["device"] != "cuda"
        or runtime["reference_dtype"] != REFERENCE_DTYPE
        or runtime["execution_dtype"] != EXECUTION_DTYPE
    ):
        raise ArtifactValidationError("runtime dtype/backend provenance is invalid")
    for name in ("gpu_name", "torch_version", "torch_cuda_version"):
        if not isinstance(runtime[name], str) or not runtime[name].strip():
            raise ArtifactValidationError(f"runtime.{name} must be non-empty text")
    if "T4" not in runtime["gpu_name"]:
        raise ArtifactValidationError("T4 run manifest must identify a T4 GPU")
    capability = runtime["compute_capability"]
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) for item in capability
        )
    ):
        raise ArtifactValidationError("runtime compute capability is invalid")
    if capability != [7, 5]:
        raise ArtifactValidationError("T4 compute capability must equal [7, 5]")
    _manifest_bool(runtime["bf16_supported"], "runtime.bf16_supported", expected=False)

    attribution = _manifest_mapping(record["attribution"], "attribution")
    attempted = attribution.get("attempted_batch_sizes")
    if not isinstance(attempted, list):
        raise ArtifactValidationError("attempted batches must be a JSON array")
    if status is T4RunStatus.PREPARED:
        if attempted:
            raise ArtifactValidationError("prepared manifest cannot contain attempts")
    elif attempted and tuple(attempted) != OOM_BATCH_SEQUENCE[: len(attempted)]:
        raise ArtifactValidationError(
            "attempted batches must be a prefix of 256,128,64"
        )
    elif status is T4RunStatus.COMPLETED and not attempted:
        raise ArtifactValidationError("completed manifest requires an attempt")
    selected = attribution.get("selected_batch_size")
    if selected is not None and selected not in attempted:
        raise ArtifactValidationError("selected batch was not attempted")
    if selected is not None and selected != attempted[-1]:
        raise ArtifactValidationError("selected batch must be the final attempt")
    deviation = attribution.get("batch_deviation")
    if selected in (128, 64) and deviation != batch_deviation(selected):
        raise ArtifactValidationError("reduced batch must record a deviation")
    if selected == 256 and deviation is not None:
        raise ArtifactValidationError("batch 256 must not record a deviation")

    history = record["retry_history"]
    if not isinstance(history, list) or len(history) != len(attempted):
        raise ArtifactValidationError("retry history must align with attempted batches")
    for index, attempt in enumerate(history):
        item = _manifest_mapping(attempt, f"retry_history[{index}]")
        expected_attempt_keys = {
            "batch_size",
            "outcome",
            "category",
            "exception_type",
            "message",
            "failure_stage",
            "peak_memory_bytes",
            "wall_seconds",
            "cleanup_succeeded",
        }
        if set(item) != expected_attempt_keys:
            raise ArtifactValidationError("retry history attempt keys are not exact")
        if item.get("batch_size") != attempted[index]:
            raise ArtifactValidationError("retry history batch order is invalid")
        if index + 1 < len(history) and (
            item.get("category") != "cuda_out_of_memory"
            or item.get("failure_stage") != "attribution"
        ):
            raise ArtifactValidationError(
                "only attribution-stage CUDA OOM may lead to a retry"
            )
        message = item.get("message")
        if not isinstance(message, str) or not message:
            raise ArtifactValidationError("retry message must be sanitized text")
        if item.get("outcome") not in {"completed", "failed"}:
            raise ArtifactValidationError("retry outcome is invalid")
        for name in ("exception_type", "failure_stage"):
            if item.get(name) is not None and not isinstance(item[name], str):
                raise ArtifactValidationError(f"retry {name} is invalid")
        peak_memory = item.get("peak_memory_bytes")
        if peak_memory is not None and (
            isinstance(peak_memory, bool)
            or not isinstance(peak_memory, int)
            or peak_memory < 0
        ):
            raise ArtifactValidationError("retry peak memory is invalid")
        wall_seconds = item.get("wall_seconds")
        if (
            isinstance(wall_seconds, bool)
            or not isinstance(wall_seconds, (int, float))
            or wall_seconds < 0
        ):
            raise ArtifactValidationError("retry wall time is invalid")
        _manifest_bool(item.get("cleanup_succeeded"), "retry cleanup_succeeded")

    timings = _manifest_mapping(record["timings"], "timings")
    if not {"attempt_wall_seconds", "attempt_peak_memory_bytes"}.issubset(timings):
        raise ArtifactValidationError(
            "attempt timing and peak-memory lists are required"
        )
    if timings["attempt_wall_seconds"] != [
        item["wall_seconds"] for item in history
    ] or timings["attempt_peak_memory_bytes"] != [
        item["peak_memory_bytes"] for item in history
    ]:
        raise ArtifactValidationError("attempt timings must align with retry history")

    checks = _manifest_mapping(record["checks"], "checks")
    required_checks = {
        "immutable_assets_loaded",
        "model_parameter_samples_finite",
        "thresholds_finite",
        "baseline_logits_finite",
        "cached_values_finite",
        "attribution_values_finite",
        "intervention_values_finite",
        "nonfinite_count",
        "baseline_repeat_within_tolerance",
        "noop_within_tolerance",
        "jumprelu_semantics_passed",
        "desired_value_mapping_passed",
        "artifact_validation_passed",
        "attribution_graph_nonempty",
        "intervention_completed",
        "semantics_completed",
    }
    if set(checks) != required_checks:
        raise ArtifactValidationError("T4 run checks are not exact")
    nonfinite = checks["nonfinite_count"]
    if isinstance(nonfinite, bool) or not isinstance(nonfinite, int) or nonfinite < 0:
        raise ArtifactValidationError("checks.nonfinite_count must be non-negative")

    readiness = _manifest_mapping(record["readiness"], "readiness")
    if set(readiness) != {
        "stage1b_engineering_readiness",
        "stage1b_empirical_claim_readiness",
    }:
        raise ArtifactValidationError("readiness flags are not exact")
    engineering = _manifest_bool(
        readiness["stage1b_engineering_readiness"],
        "stage1b_engineering_readiness",
    )
    _manifest_bool(
        readiness["stage1b_empirical_claim_readiness"],
        "stage1b_empirical_claim_readiness",
        expected=False,
    )
    bf16 = _manifest_mapping(record["bf16_reference"], "bf16_reference")
    if bf16 != {
        "dtype": REFERENCE_DTYPE,
        "status": REFERENCE_STATUS,
        "statement": "Native-BF16 reference reproduction remains pending.",
    }:
        raise ArtifactValidationError("BF16 pending statement is invalid")

    if status is T4RunStatus.COMPLETED:
        if selected is None or nonfinite != 0:
            raise ArtifactValidationError("completed run lacks a finite selected batch")
        if not all(
            checks[name] is True for name in required_checks - {"nonfinite_count"}
        ):
            raise ArtifactValidationError("completed run has a failed required check")
        if not engineering:
            raise ArtifactValidationError(
                "completed run must enable engineering readiness"
            )
        if history[-1]["outcome"] != "completed":
            raise ArtifactValidationError(
                "completed run must end in a completed attempt"
            )
    elif status is T4RunStatus.PREPARED and (
        nonfinite != 0
        or any(
            checks[name] is not False for name in required_checks - {"nonfinite_count"}
        )
    ):
        raise ArtifactValidationError("prepared manifest cannot claim empirical checks")
    elif engineering:
        raise ArtifactValidationError("non-completed run cannot enable readiness")
    elif selected is not None:
        raise ArtifactValidationError("non-completed run cannot select a batch")
    elif (
        status is not T4RunStatus.PREPARED
        and history
        and history[-1]["outcome"] != "failed"
    ):
        raise ArtifactValidationError("non-completed run must end in a failed attempt")

    artifacts = _manifest_mapping(record["artifacts"], "artifacts")
    if status is T4RunStatus.COMPLETED and set(artifacts) != T4_SMALL_FILES - {
        "stage1a_t4_fp16_run_manifest.json"
    }:
        raise ArtifactValidationError(
            "completed run must record all six prior artifacts"
        )
    for name, metadata in artifacts.items():
        if name not in T4_SMALL_FILES - {"stage1a_t4_fp16_run_manifest.json"}:
            raise ArtifactValidationError(
                "run manifest records an unsupported artifact"
            )
        item = _manifest_mapping(metadata, f"artifacts.{name}")
        if set(item) != {"size_bytes", "sha256"}:
            raise ArtifactValidationError("artifact metadata keys are not exact")
        size = item["size_bytes"]
        digest = item["sha256"]
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 < size <= (MAX_BUNDLE_MEMBER_BYTES)
        ):
            raise ArtifactValidationError("artifact size is invalid")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ArtifactValidationError("artifact SHA-256 is invalid")
