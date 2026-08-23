#!/usr/bin/env python3
"""Validate the canonical, publication-safe Stage 1A MPS/FP16 bundle.

Every JSON artifact except the terminal run manifest uses the shared Stage 1A
artifact envelope. The terminal manifest is raw JSON because its status is the
immutable execution-class label rather than a generic envelope status.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cfsus.reproduction.artifacts import (  # noqa: E402
    ArtifactValidationError,
    assert_publication_safe,
    build_checksum_manifest,
    validate_artifact_envelope,
    validate_json_value,
    verify_checksum_manifest,
    write_checksum_manifest_atomic,
)
from cfsus.reproduction.config import (  # noqa: E402
    OFFICIAL_MODEL_ID,
    OFFICIAL_MODEL_REVISION,
    OFFICIAL_TRANSCODER_ID,
    OFFICIAL_TRANSCODER_REVISION,
    OFFICIAL_UPSTREAM_REPOSITORY,
    OFFICIAL_UPSTREAM_REVISION,
)

PROJECT_BASE_COMMIT = "d965e43c34a2ba408b8ae35b13b5651bf269beed"
REPRODUCTION_CLASS = "hardware_adapted_mps_fp16"
EXECUTION_CLASS = "completed_hardware_adapted_mps_fp16"
COMPLETED_STATUS = EXECUTION_CLASS
EXECUTION_DTYPE = "float16"
REFERENCE_DTYPE = "bfloat16"
HARDWARE_FAMILY = "Apple M2 Max"
EXPECTED_PYTHON = "3.11.13"
EXPECTED_TORCH = "2.6.0"
EXPECTED_TRANSFORMER_LENS = "3.2.1"
EXPECTED_CIRCUIT_TRACER = "0.5.2"
EXPECTED_CIRCUIT_TRACER_VCS_URL = "https://github.com/decoderesearch/circuit-tracer.git"
EXPECTED_LOCK_SHA256 = (
    "9adfd17bf39b20552af73eff90e659fb29c0a40adb06b8967fe7d47f853637fd"
)
CLAIM_BOUNDARY = (
    "Apple M2 Max/MPS FP16 hardware-adapted runtime using the pinned assets; "
    "the official native-BF16 reproduction and CUDA/T4 numerical equivalence "
    "remain pending."
)
CANONICAL_EXECUTION_DEVIATION = (
    "Explicit CPU sparse COO metadata adapter is required because native MPS "
    "sparse COO is unsupported; dense scientific tensors remain on MPS and "
    "scientific parameters are unchanged."
)
CANONICAL_DEVIATIONS = [CANONICAL_EXECUTION_DEVIATION]

MPS_RESULT_DIRECTORY = "results/stage1a_mps_fp16"
RUN_MANIFEST_NAME = "stage1a_mps_run_manifest.json"
CHECKSUM_NAME = "checksums.sha256"
PREFLIGHT_DIRECTORY = "preflight"
PREFLIGHT_NAME = "preflight_summary.json"
ROOT_JSON_FILES = frozenset(
    {
        "feasibility_report.json",
        "environment_manifest.json",
        "asset_manifest.json",
        "attribution_summary.json",
        "intervention_summary.json",
        "semantics_summary.json",
        "memory_summary.json",
        RUN_MANIFEST_NAME,
        CHECKSUM_NAME,
    }
)
MPS_SMALL_FILES = frozenset(
    ROOT_JSON_FILES | {f"{PREFLIGHT_DIRECTORY}/{PREFLIGHT_NAME}"}
)
MPS_JSON_FILES = frozenset(MPS_SMALL_FILES - {CHECKSUM_NAME})
MAX_ARTIFACT_BYTES = 5 * 1024 * 1024
MAX_CHECKSUM_BYTES = 256 * 1024
MAX_TOTAL_BYTES = 25 * 1024 * 1024
MAX_LIST_LENGTH = 10_000
MAX_JSON_DEPTH = 48
MAX_JSON_KEY_LENGTH = 256
MAX_JSON_STRING_LENGTH = 65_536

# Trusted content pins come from the Hugging Face revision metadata for the two
# immutable revisions. The gated model's five small Git blobs were SHA-256'd
# from Fedir-Ilina/gemma-2-2b@0fdb89f81d28150d4179a66facaebc44312e6dfa
# after matching every Git blob object ID to the primary pinned revision.
MODEL_FILE_PINS: dict[str, tuple[int, str]] = {
    "config.json": (
        818,
        "56ee4e528fedfe70b2fabb0704eec9f2ef94461de936928220a3d6e7fac5ef98",
    ),
    "generation_config.json": (
        168,
        "4c8b8739d9583289c138eb8f412bdd4854b6fed6ba65ba28a323c0721c9a5973",
    ),
    "model-00001-of-00003.safetensors": (
        4_992_576_136,
        "1425aa066ec77e3eb79aac14a5bdea3ebcec46aa5c96cd40608c5c1fd70d193d",
    ),
    "model-00002-of-00003.safetensors": (
        4_983_443_424,
        "96c111d3dcdbde9271595e463b5d9f7fc4810ad8b79e736309c0a1833e6c0d35",
    ),
    "model-00003-of-00003.safetensors": (
        481_381_384,
        "4e08abc64d1767fdacd2c94da7f2ec4b8c65b25b19a53e87d19dc432901b5f02",
    ),
    "model.safetensors.index.json": (
        24_224,
        "e5c6f26fbe40fc3712f4df57da9b89c58b033714316a4547d052b37f43217e8e",
    ),
    "special_tokens_map.json": (
        636,
        "baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351",
    ),
    "tokenizer.json": (
        17_525_357,
        "3f289bc05132635a8bc7aca7aa21255efd5e18f3710f43e3cdb96bcd41be4922",
    ),
    "tokenizer.model": (
        4_241_003,
        "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
    ),
    "tokenizer_config.json": (
        46_379,
        "bb8245b8d39a065fdf869ee4fce48db9cc491473eb7ff1b41d321fb4caf540d1",
    ),
}
_TRANSCODER_LAYER_SHA256 = (
    "6ca4641a649c68095f91e2201553e7510eea69e0a6202224a5629efacfee4b1a",
    "efa96e216de43623c1eba9e48f51b4c0d50fa335abd89a56d483428b0b70d346",
    "8ef681fe2f0a7c8dbe7afe1c353d1f5bf6e0c845ad066b8141517ab2c863d343",
    "216cbefe65bb2e82b6dcd7a7a8858ec099d32d003c10628cd3c9c072bca7bacb",
    "32ff2262cfddea0e9d84bcc2282495a9f77afc86889282067bcb8832e113cf04",
    "b8f63ffd554ecfc771de134aaa9819b332336ad67565e054067e79e3ec788d9f",
    "3ff91ec09aaf404c0e13f4774d4074e1c214ea82601d9b1230523b789230c5c7",
    "faf812c8cf2e4a114d5e339ec9ebfd59bc0aba3fce40925931f497a11deff8f8",
    "32fd4179cb8194585adb255da8707754eb48142a1b01349eb3b27feeed264c43",
    "171a4f7ebd4849aaee6d817dd0a21605e691a8f557400096fc27b105cf3a7645",
    "8b068b65d573050b6b962da2ae9f60b6b23d2dc6b60215014d87d6af9753694a",
    "012b4fe0b5c61ca23ff44dfdf0057e395a00f68a6783d306415d142b1b33ddcc",
    "7e8fb79ea2f00e5eff654a13e44c23b651146b3f35597821ebfd604b28171600",
    "30d48b1766acf22d946d3634417ddf8a25633b6c3c40a90ee586791235e00bea",
    "0012141eeb5f75acd5b6a4df01fabc8ac74d0677df505de5326c31bbd9f6fff3",
    "6bb3b196b7a6fda68bb1260aa064e77795a0e9054614e9e3ab80c5f4ae7f2169",
    "4afb2a89adb951b15a89a6d7a209122d18e26eef0402809601f4a0c412241ae8",
    "36bbaebc916ff9d05413d72deb366a52852ddf8ca987eb256d8bf33c2248eaf3",
    "90f83eb90b4c5286248aafd96e20f32bd6c7bbd3672b495bfc575f007f5c658e",
    "5bf2413355d4edd9dd8af0a748a75da2afde879e0477ccc8baeadbc80fab2e38",
    "c5afa214dff8f33708a2cef28cdf752f38b25427343610ebc55895e0b4bf2385",
    "450216b326e1c77b8e4fdfa27b73e5a079c67471da34b79d4d4fb54d204fe0db",
    "eb8ab5f01b9199e8af001b1386a8dc588b9b7cccbe18eb8540850eb0e788073f",
    "42d081487b5f422fe52ba248ec5036eb160709c47bf43582b78fd40c1a8b8806",
    "72a540935c4ed6169e563a63a0b64622f49ee26809ab4d8657f16a5eeaed3aee",
    "ac09b1edc0d1bf0e38a00601a65be18fcefe62ab3663180268434e5475b1c231",
)
TRANSCODER_FILE_PINS: dict[str, tuple[int, str]] = {
    "config.yaml": (
        202,
        "1ba7a34f5caa7a8f62789b20b3b2c3d97070a09daa4316af51ddc74f4eb12084",
    ),
    **{
        f"layer_{layer}.safetensors": (302_130_600, digest)
        for layer, digest in enumerate(_TRANSCODER_LAYER_SHA256)
    },
}
MODEL_REQUIRED_FILES = frozenset(MODEL_FILE_PINS)
TRANSCODER_REQUIRED_FILES = frozenset(TRANSCODER_FILE_PINS)
MODEL_REQUIRED_BYTES = sum(size for size, _digest in MODEL_FILE_PINS.values())
TRANSCODER_REQUIRED_BYTES = sum(size for size, _digest in TRANSCODER_FILE_PINS.values())
EXPECTED_OPERATOR_NAMES = frozenset(
    {
        "matmul",
        "einsum",
        "softmax",
        "layernorm",
        "topk",
        "gather",
        "scatter",
        "index_add",
        "index_put",
        "where",
        "sort",
        "unique",
        "searchsorted",
    }
)
EXPECTED_PREFLIGHT_OPERATIONS = frozenset(
    {
        "operators",
        "transfers",
        "autograd_hooks",
        "strict_jumprelu",
        "tiny_hooked_transformer",
        "sparse_metadata_boundary",
        "bounded_graph_construction",
        "disk_offload_safetensors",
    }
)
EXPECTED_PREFLIGHT_CHECKS = frozenset(
    {
        "python_3_11",
        "darwin",
        "native_arm64",
        "torch_2_6_0",
        "fallback_disabled",
        "memory_guardrail_preserved",
        "torch_importable",
        "mps_built",
        "mps_available",
        *EXPECTED_PREFLIGHT_OPERATIONS,
    }
)
PEAK_KEYS = (
    "mps_current_allocated_peak_bytes",
    "mps_driver_allocated_peak_bytes",
    "process_rss_peak_bytes",
    "swap_used_peak_bytes",
)
EXPECTED_ATTEMPT_STAGES = frozenset(
    {
        "runtime_loading",
        "model_only_forward",
        "semantics",
        "intervention",
        "attribution",
        "cleanup",
    }
)
TIMING_ALIASES = {
    "attribution": "attribution_summary.json",
    "intervention": "intervention_summary.json",
    "semantics": "semantics_summary.json",
    "memory": "memory_summary.json",
}
_EXPECTED_TYPES = {
    "feasibility_report.json": "feasibility_report",
    "environment_manifest.json": "environment_manifest",
    "asset_manifest.json": "asset_manifest",
    "attribution_summary.json": "attribution_summary",
    "intervention_summary.json": "intervention_summary",
    "semantics_summary.json": "semantics_summary",
    "memory_summary.json": "memory_summary",
    RUN_MANIFEST_NAME: "run_manifest",
    f"{PREFLIGHT_DIRECTORY}/{PREFLIGHT_NAME}": "preflight_summary",
}
_SHA40 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SENSITIVE_KEY_PARTS = (
    "access_key",
    "api_key",
    "apikey",
    "api_token",
    "authorization",
    "bearer",
    "client_secret",
    "cookie",
    "credential",
    "password",
    "passwd",
    "private_key",
    "secret",
    "secret_key",
)
_FORBIDDEN_RAW_KEYS = {
    "activations",
    "adjacency_matrix",
    "cache_path",
    "logits_array",
    "model_weights",
    "raw_adjacency",
    "raw_graph",
    "state_dict",
    "tensor_values",
    "weight_array",
}
_SENSITIVE_TOKEN_KEYS = frozenset(
    {
        "token",
        "access_token",
        "api_token",
        "auth_token",
        "bearer_token",
        "github_token",
        "hf_token",
        "id_token",
        "private_token",
        "refresh_token",
        "token_credential",
        "token_secret",
        "token_value",
    }
)
_SAFE_RAW_KEYS = frozenset({"raw_validation", "raw_graph_committed"})
_CAMEL_BOUNDARY_1 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_BOUNDARY_2 = re.compile(r"([a-z0-9])([A-Z])")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ArtifactValidationError(f"non-finite JSON constant {value!r}")


def _bounded_json(value: Any, *, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise ArtifactValidationError("JSON nesting exceeds the artifact limit")
    if isinstance(value, str):
        if len(value) > MAX_JSON_STRING_LENGTH:
            raise ArtifactValidationError("JSON string exceeds the artifact limit")
    elif isinstance(value, list):
        if len(value) > MAX_LIST_LENGTH:
            raise ArtifactValidationError("JSON list exceeds the artifact limit")
        for item in value:
            _bounded_json(item, depth=depth + 1)
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ArtifactValidationError("JSON object key is not text")
            if len(key) > MAX_JSON_KEY_LENGTH:
                raise ArtifactValidationError(
                    "JSON object key exceeds the artifact limit"
                )
            _bounded_json(item, depth=depth + 1)


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ArtifactValidationError(f"artifact must be a regular file: {path}")
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ArtifactValidationError(f"artifact exceeds size limit: {path.name}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except ArtifactValidationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"invalid JSON artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"JSON artifact must be an object: {path.name}")
    _bounded_json(value)
    validate_json_value(value)
    assert_publication_safe(value)
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ArtifactValidationError(
            f"{label} keys are not exact; missing={missing}, unknown={unknown}"
        )


def _require_keys(value: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ArtifactValidationError(f"{label} is missing keys: {', '.join(missing)}")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactValidationError(f"{label} must be non-empty text")
    return value


def _sha40(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA40.fullmatch(value) is None:
        raise ArtifactValidationError(f"{label} must be a lowercase 40-character SHA")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ArtifactValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactValidationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ArtifactValidationError(f"{label} is outside the finite range")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ArtifactValidationError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _false(value: Any, label: str) -> None:
    if value is not False:
        raise ArtifactValidationError(f"{label} must be false")


def _true(value: Any, label: str) -> None:
    if value is not True:
        raise ArtifactValidationError(f"{label} must be true")


def _normalized_relative_file(value: Any, label: str) -> str:
    text = _text(value, label)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or "\\" in text
        or "." in path.parts
        or ".." in path.parts
        or path.as_posix() != text
        or len(path.parts) != 1
    ):
        raise ArtifactValidationError(f"{label} is not a safe snapshot-relative file")
    return text


def _normalized_key(key: str) -> str:
    value = _CAMEL_BOUNDARY_1.sub(r"\1_\2", key)
    value = _CAMEL_BOUNDARY_2.sub(r"\1_\2", value)
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _raw_key_forbidden(normalized: str) -> bool:
    if normalized in _FORBIDDEN_RAW_KEYS or normalized in {
        "edges",
        "logits",
        "nodes",
        "tensors",
        "weights",
    }:
        return True
    if normalized not in _SAFE_RAW_KEYS and (
        normalized.startswith("raw_") or normalized.endswith("_raw")
    ):
        return True
    if "adjacency" in normalized and normalized != "adjacency_shape":
        return True
    data_terms = (
        "array",
        "cache",
        "data",
        "matrix",
        "object",
        "payload",
        "serialized",
        "tensor",
        "values",
    )
    science_terms = (
        "activation",
        "dense",
        "feature",
        "graph",
        "logit",
        "model",
        "parameter",
        "transcoder",
        "weight",
    )
    return any(term in normalized for term in science_terms) and any(
        term in normalized for term in data_terms
    )


def _unsafe_overclaim(value: str) -> bool:
    if value == CLAIM_BOUNDARY:
        return False
    text = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    accelerator = re.search(r"\b(?:bf16|bfloat16|cuda|t4)\b", text) is not None
    positive_claim = re.search(
        r"\b(?:achiev(?:e|ed|ement)|complet(?:e|ed|ion)|confirm(?:ed|ation)|"
        r"done|equivalen(?:t|ce)|establish(?:ed|ment)|identical|match(?:ed|es|ing)?|"
        r"parity|pass(?:ed|es|ing)?|reproduc(?:e|ed|tion)|succeed(?:ed|s)?|"
        r"success(?:ful|fully)?|validat(?:e|ed|ion)|verif(?:y|ied|ication))\b",
        text,
    )
    # Any non-canonical positive accelerator statement is an overclaim. This
    # deliberately does not let a distant "pending" qualifier hide a second,
    # positive CUDA/T4 or BF16 clause.
    return accelerator and positive_claim is not None


def _walk_safety(value: Any, *, path: tuple[str | int, ...] = ()) -> None:
    """Fail closed on secrets, fallback, CPU tensors, and raw scientific data."""

    if isinstance(value, dict):
        for key, item in value.items():
            normalized = _normalized_key(key)
            location = ".".join(map(str, (*path, key))).casefold()
            token_like = normalized == "token" or any(
                f"_{sensitive}_" in f"_{normalized}_"
                for sensitive in _SENSITIVE_TOKEN_KEYS - {"token"}
            )
            if normalized == "secure_hf_token_present" and isinstance(item, bool):
                token_like = False
            if token_like or any(part in normalized for part in _SENSITIVE_KEY_PARTS):
                raise ArtifactValidationError(
                    f"sensitive field name is forbidden even when redacted: {key}"
                )
            if _raw_key_forbidden(normalized):
                raise ArtifactValidationError(
                    f"raw/large scientific field is forbidden: {key}"
                )
            if (
                normalized == "cuda"
                or normalized.startswith("cuda_")
                or normalized.startswith("nvidia_")
                or normalized
                in {
                    "compute_capability",
                    "torch_cuda_version",
                    "t4",
                }
            ):
                raise ArtifactValidationError(f"CUDA/T4-only field is forbidden: {key}")
            fallback_negative_assertion = (
                normalized.startswith("no_") or normalized.endswith("_disabled")
            ) and item is True
            if (
                "fallback" in normalized
                and item is not False
                and not fallback_negative_assertion
            ):
                raise ArtifactValidationError(
                    "hidden or enabled MPS fallback is forbidden"
                )
            if "high_watermark" in normalized and item is not None:
                raise ArtifactValidationError(
                    "MPS high-watermark override is forbidden"
                )
            if (
                normalized
                in {
                    "official_bf16_reproduction",
                    "official_cuda_reproduction",
                    "t4_fp16_reproduction",
                }
                and item is not False
            ):
                raise ArtifactValidationError(
                    "BF16/CUDA/T4 completion claims are forbidden"
                )
            if (
                any(marker in normalized for marker in ("bf16", "cuda", "t4"))
                and any(
                    marker in normalized
                    for marker in ("complete", "equivalence", "equivalent", "passed")
                )
                and item is not False
            ):
                raise ArtifactValidationError(
                    "accelerator-equivalence/completion claim field is forbidden"
                )
            if isinstance(item, str) and re.fullmatch(
                r"cpu(?:\s*:\s*\d+)?", item.casefold().strip()
            ):
                metadata_boundary = (
                    "sparse" in location or "coo" in location or "graph" in location
                ) and ("metadata" in location or "storage" in location)
                cpu_reference = "cpu_reference" in location
                if not metadata_boundary and not cpu_reference:
                    raise ArtifactValidationError(
                        "scientific tensor/device cannot be CPU"
                    )
            _walk_safety(item, path=(*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _walk_safety(item, path=(*path, index))
    elif isinstance(value, str) and _unsafe_overclaim(value):
        raise ArtifactValidationError("artifact text crosses the MPS claim boundary")


def _validate_deviations(record: dict[str, Any]) -> None:
    if record.get("deviations") != CANONICAL_DEVIATIONS:
        raise ArtifactValidationError(
            "execution deviations must be the exact CPU sparse-metadata boundary"
        )
    if record.get("warnings") != []:
        raise ArtifactValidationError(
            "completed canonical artifacts cannot have warnings"
        )


def _validate_envelope(path: Path, expected_type: str) -> dict[str, Any]:
    record = _load_json(path)
    if expected_type == "run_manifest":
        _walk_safety(record)
        return record
    validate_artifact_envelope(record, expected_type=expected_type)
    _walk_safety(record)
    _validate_deviations(record)
    return record


def _validate_runtime_provenance(record: dict[str, Any], label: str) -> str:
    provenance = _mapping(record.get("provenance"), f"{label} provenance")
    expected = {
        "base_commit": PROJECT_BASE_COMMIT,
        "upstream_repository": OFFICIAL_UPSTREAM_REPOSITORY,
        "upstream_revision": OFFICIAL_UPSTREAM_REVISION,
        "model_identifier": OFFICIAL_MODEL_ID,
        "model_revision": OFFICIAL_MODEL_REVISION,
        "transcoder_identifier": OFFICIAL_TRANSCODER_ID,
        "transcoder_revision": OFFICIAL_TRANSCODER_REVISION,
        "backend": "transformerlens",
        "accelerator_backend": "mps",
        "device": "mps",
        "dtype": EXECUTION_DTYPE,
        "execution_dtype": EXECUTION_DTYPE,
        "reference_dtype": REFERENCE_DTYPE,
        "reproduction_class": REPRODUCTION_CLASS,
        "execution_class": EXECUTION_CLASS,
        "architecture": "arm64",
        "hardware_family": HARDWARE_FAMILY,
        "offload": "disk",
        "fallback_enabled": False,
        "fallback_used": False,
        "official_bf16_reproduction": False,
        "t4_fp16_reproduction": False,
    }
    _exact_keys(provenance, set(expected) | {"project_commit"}, f"{label} provenance")
    for key, expected_value in expected.items():
        if provenance.get(key) != expected_value:
            raise ArtifactValidationError(
                f"{label} provenance field {key!r} is invalid"
            )
    return _sha40(provenance.get("project_commit"), f"{label} project commit")


def _validate_feasibility(record: dict[str, Any], execution_commit: str) -> None:
    if record.get("status") != "resolved":
        raise ArtifactValidationError("feasibility report status must be resolved")
    if _validate_runtime_provenance(record, "feasibility") != execution_commit:
        raise ArtifactValidationError("feasibility execution commit mismatch")
    payload = _mapping(record.get("payload"), "feasibility payload")
    required = {
        "status",
        "passed",
        "downloads_authorized",
        "physical_memory_bytes",
        "system_memory_pressure",
        "swap_used_bytes",
        "snapshot_sizes",
        "estimate_components",
        "conservative_budget_bytes",
        "estimated_peak_bytes",
        "safety_reserve_bytes",
        "scientific_parameters_changed",
    }
    _exact_keys(payload, required, "feasibility payload")
    gate = payload.get("status")
    if gate != "feasible_with_explicit_execution_deviation":
        raise ArtifactValidationError("32 GB feasibility gate did not pass")
    _true(payload.get("passed"), "feasibility passed")
    _true(payload.get("downloads_authorized"), "feasibility downloads_authorized")
    _false(
        payload.get("scientific_parameters_changed"),
        "feasibility scientific_parameters_changed",
    )
    physical = _integer(
        payload.get("physical_memory_bytes"), "physical memory", minimum=1
    )
    if physical < 32 * 1024**3:
        raise ArtifactValidationError("physical memory is below the 32 GiB gate")
    pressure = payload.get("system_memory_pressure")
    if not isinstance(pressure, str) or pressure.casefold() != "normal":
        raise ArtifactValidationError("pre-run memory pressure is unsafe")
    swap = _integer(payload.get("swap_used_bytes"), "swap usage")
    if swap > 4 * 1024**3:
        raise ArtifactValidationError("pre-run swap usage exceeds the safety gate")
    sizes = _mapping(payload.get("snapshot_sizes"), "snapshot sizes")
    if sizes != {
        "model_bytes": MODEL_REQUIRED_BYTES,
        "transcoder_bytes": TRANSCODER_REQUIRED_BYTES,
    }:
        raise ArtifactValidationError("immutable snapshot size plan is invalid")
    components = _mapping(payload.get("estimate_components"), "estimate components")
    _exact_keys(
        components,
        {
            "model_resident_estimate_bytes",
            "transcoder_resident_estimate_bytes",
            "temporary_and_system_headroom_bytes",
        },
        "estimate components",
    )
    model_resident = _integer(
        components["model_resident_estimate_bytes"],
        "model resident estimate",
        minimum=MODEL_REQUIRED_BYTES // 2,
    )
    transcoder_resident = _integer(
        components["transcoder_resident_estimate_bytes"],
        "transcoder resident estimate",
        minimum=TRANSCODER_REQUIRED_BYTES,
    )
    temporary_headroom = _integer(
        components["temporary_and_system_headroom_bytes"],
        "temporary/system headroom",
        minimum=6 * 1024**3,
    )
    budget = _integer(
        payload.get("conservative_budget_bytes"), "memory budget", minimum=1
    )
    estimate = _integer(
        payload.get("estimated_peak_bytes"), "estimated peak", minimum=1
    )
    reserve = _integer(payload.get("safety_reserve_bytes"), "safety reserve", minimum=1)
    if estimate != model_resident + transcoder_resident + temporary_headroom:
        raise ArtifactValidationError("feasibility component ledger does not sum")
    if (
        estimate > budget
        or reserve < 6 * 1024**3
        or budget + reserve > physical
        or estimate + reserve > physical
    ):
        raise ArtifactValidationError("conservative memory budget is not safe")


def _validate_environment(record: dict[str, Any], execution_commit: str) -> None:
    if record.get("status") != "observed":
        raise ArtifactValidationError("environment manifest status must be observed")
    provenance_commit = _validate_runtime_provenance(record, "environment")
    if provenance_commit != execution_commit:
        raise ArtifactValidationError("environment execution commit mismatch")
    payload = _mapping(record.get("payload"), "environment payload")
    required = {
        "platform",
        "python",
        "packages",
        "runtime",
        "mps",
        "fallback_enabled",
        "fallback_used",
        "fallback_env_value_present",
        "high_watermark_override",
        "memory_guardrails_preserved",
        "pip_check",
        "lock_path",
        "lock_sha256",
        "lock_match",
    }
    _exact_keys(payload, required, "environment payload")
    platform_data = _mapping(payload.get("platform"), "environment platform")
    _exact_keys(
        platform_data,
        {"system", "machine", "hardware_family", "physical_memory_bytes"},
        "environment platform",
    )
    if (
        platform_data.get("system") != "Darwin"
        or platform_data.get("machine") != "arm64"
        or platform_data.get("hardware_family") != HARDWARE_FAMILY
        or _integer(
            platform_data.get("physical_memory_bytes"),
            "environment physical memory",
            minimum=1,
        )
        < 32 * 1024**3
    ):
        raise ArtifactValidationError(
            "environment is not the required Apple arm64 host"
        )
    python = _mapping(payload.get("python"), "environment Python")
    _exact_keys(python, {"version", "architecture"}, "environment Python")
    if (
        python.get("version") != EXPECTED_PYTHON
        or python.get("architecture") != "arm64"
    ):
        raise ArtifactValidationError("environment Python is not native 3.11.13 arm64")
    packages = _mapping(payload.get("packages"), "environment packages")
    expected_packages = {
        "torch": EXPECTED_TORCH,
        "transformer_lens": EXPECTED_TRANSFORMER_LENS,
        "circuit_tracer": EXPECTED_CIRCUIT_TRACER,
        "circuit_tracer_revision": OFFICIAL_UPSTREAM_REVISION,
        "circuit_tracer_vcs_url": EXPECTED_CIRCUIT_TRACER_VCS_URL,
    }
    _exact_keys(
        packages,
        set(expected_packages) | {"circuit_tracer_record_hashes_verified"},
        "environment packages",
    )
    for key, expected in expected_packages.items():
        if packages.get(key) != expected:
            raise ArtifactValidationError(f"environment package {key!r} is invalid")
    _integer(
        packages.get("circuit_tracer_record_hashes_verified"),
        "Circuit Tracer verified RECORD hash count",
        minimum=1,
    )
    runtime = _mapping(payload.get("runtime"), "environment runtime")
    expected_runtime = {
        "backend": "transformerlens",
        "accelerator_backend": "mps",
        "device": "mps",
        "dtype": EXECUTION_DTYPE,
        "execution_class": EXECUTION_CLASS,
        "offload": "disk",
    }
    _exact_keys(runtime, set(expected_runtime), "environment runtime")
    for key, expected in expected_runtime.items():
        if runtime.get(key) != expected:
            raise ArtifactValidationError(f"environment runtime {key!r} is invalid")
    mps = _mapping(payload.get("mps"), "environment MPS")
    _exact_keys(mps, {"built", "available", "allocation_probe"}, "environment MPS")
    if mps.get("built") is not True or mps.get("available") is not True:
        raise ArtifactValidationError("MPS is not built and available")
    allocation = _mapping(mps.get("allocation_probe"), "MPS allocation probe")
    _exact_keys(
        allocation,
        {"success", "device", "dtype", "finite"},
        "MPS allocation probe",
    )
    if (
        allocation.get("success") is not True
        or allocation.get("device") != "mps"
        or allocation.get("dtype") != EXECUTION_DTYPE
        or allocation.get("finite") is not True
    ):
        raise ArtifactValidationError("MPS FP16 allocation probe failed")
    _false(payload.get("fallback_enabled"), "environment fallback_enabled")
    _false(payload.get("fallback_used"), "environment fallback_used")
    _false(
        payload.get("fallback_env_value_present"),
        "environment fallback_env_value_present",
    )
    if payload.get("high_watermark_override") is not None:
        raise ArtifactValidationError("MPS high-watermark override must be absent")
    _true(
        payload.get("memory_guardrails_preserved"),
        "environment memory_guardrails_preserved",
    )
    if payload.get("pip_check") != "passed":
        raise ArtifactValidationError("pip check did not pass")
    if payload.get("lock_path") != "environments/stage1a_mps/requirements-lock.txt":
        raise ArtifactValidationError("environment lock path is invalid")
    if (
        _sha256(payload.get("lock_sha256"), "environment lock SHA-256")
        != EXPECTED_LOCK_SHA256
    ):
        raise ArtifactValidationError("environment lock SHA-256 is not canonical")
    if payload.get("lock_match") != "exact":
        raise ArtifactValidationError(
            "installed environment does not exactly match lock"
        )


def _validate_asset_files(
    value: Any,
    *,
    label: str,
    identifier: str,
    revision: str,
    expected_files: Mapping[str, tuple[int, str]],
) -> None:
    asset = _mapping(value, label)
    _exact_keys(
        asset,
        {
            "identifier",
            "revision",
            "files",
            "file_count",
            "total_bytes",
            "complete",
            "snapshot_containment_verified",
            "offline_ready",
        },
        label,
    )
    if asset.get("identifier") != identifier or asset.get("revision") != revision:
        raise ArtifactValidationError(f"{label} immutable identity is invalid")
    _true(asset.get("complete"), f"{label} complete")
    _true(
        asset.get("snapshot_containment_verified"),
        f"{label} snapshot containment",
    )
    _true(asset.get("offline_ready"), f"{label} offline_ready")
    files = asset.get("files")
    if not isinstance(files, list) or not files:
        raise ArtifactValidationError(f"{label} file manifest is empty")
    observed: dict[str, tuple[int, str]] = {}
    for index, item_raw in enumerate(files):
        item = _mapping(item_raw, f"{label} files[{index}]")
        _exact_keys(item, {"path", "size_bytes", "sha256"}, f"{label} files[{index}]")
        path = _normalized_relative_file(item.get("path"), f"{label} file path")
        if path in observed:
            raise ArtifactValidationError(f"{label} contains a duplicate file")
        size = _integer(item.get("size_bytes"), f"{label} file size", minimum=1)
        digest = _sha256(item.get("sha256"), f"{label} file SHA-256")
        observed[path] = (size, digest)
    if set(observed) != set(expected_files):
        raise ArtifactValidationError(
            f"{label} required file set is incomplete or extra"
        )
    if observed != dict(expected_files):
        raise ArtifactValidationError(f"{label} size/content digest pin is invalid")
    if asset.get("file_count") != len(expected_files):
        raise ArtifactValidationError(f"{label} file count is invalid")
    expected_total = sum(size for size, _digest in expected_files.values())
    if (
        asset.get("total_bytes") != expected_total
        or sum(size for size, _digest in observed.values()) != expected_total
    ):
        raise ArtifactValidationError(f"{label} total bytes are invalid")


def _validate_assets(record: dict[str, Any], execution_commit: str) -> None:
    if record.get("status") != "resolved":
        raise ArtifactValidationError("asset manifest status must be resolved")
    if _validate_runtime_provenance(record, "asset") != execution_commit:
        raise ArtifactValidationError("asset execution commit mismatch")
    payload = _mapping(record.get("payload"), "asset payload")
    _exact_keys(
        payload,
        {
            "verification",
            "immutable_revisions_only",
            "project_external_cache",
            "unmanifested_file_count",
            "assets",
        },
        "asset payload",
    )
    if payload.get("verification") != "exact_file_content_hashes_matched":
        raise ArtifactValidationError("asset content hashes were not verified")
    _true(payload.get("immutable_revisions_only"), "immutable revisions")
    _true(payload.get("project_external_cache"), "project-external cache")
    if payload.get("unmanifested_file_count") != 0:
        raise ArtifactValidationError("asset snapshot contains unmanifested files")
    assets = _mapping(payload.get("assets"), "asset records")
    _exact_keys(assets, {"model", "transcoder"}, "asset records")
    _validate_asset_files(
        assets["model"],
        label="model asset",
        identifier=OFFICIAL_MODEL_ID,
        revision=OFFICIAL_MODEL_REVISION,
        expected_files=MODEL_FILE_PINS,
    )
    _validate_asset_files(
        assets["transcoder"],
        label="transcoder asset",
        identifier=OFFICIAL_TRANSCODER_ID,
        revision=OFFICIAL_TRANSCODER_REVISION,
        expected_files=TRANSCODER_FILE_PINS,
    )


def _validate_preflight(record: dict[str, Any], execution_commit: str) -> None:
    if record.get("status") != "observed":
        raise ArtifactValidationError("MPS preflight status must be observed")
    if _validate_runtime_provenance(record, "preflight") != execution_commit:
        raise ArtifactValidationError("preflight execution commit mismatch")
    payload = _mapping(record.get("payload"), "preflight payload")
    required = {
        "probe_status",
        "environment",
        "checks",
        "operations",
        "tolerances",
        "large_assets_downloaded",
        "scientific_model_result",
    }
    _exact_keys(payload, required, "preflight payload")
    if payload.get("probe_status") != "passed":
        raise ArtifactValidationError("MPS preflight did not pass")
    _false(payload.get("large_assets_downloaded"), "preflight large assets")
    _false(payload.get("scientific_model_result"), "preflight scientific result")
    environment = _mapping(payload.get("environment"), "preflight environment")
    expected_environment = {
        "python": EXPECTED_PYTHON,
        "system": "Darwin",
        "architecture": "arm64",
        "torch_version": EXPECTED_TORCH,
        "mps_built": True,
        "mps_available": True,
        "fallback_enabled": False,
        "fallback_used": False,
        "fallback_env_value_present": False,
        "high_watermark_override": None,
    }
    _exact_keys(environment, set(expected_environment), "preflight environment")
    for key, expected in expected_environment.items():
        if environment.get(key) != expected:
            raise ArtifactValidationError(f"preflight environment {key!r} is invalid")
    checks = _mapping(payload.get("checks"), "preflight checks")
    if set(checks) != set(EXPECTED_PREFLIGHT_CHECKS) or any(
        value is not True for value in checks.values()
    ):
        raise ArtifactValidationError("preflight checks are incomplete or failed")
    tolerances = _mapping(payload.get("tolerances"), "preflight tolerances")
    if tolerances != {
        "absolute": 0.005,
        "relative": 0.002,
        "finite_required": True,
    }:
        raise ArtifactValidationError("preflight tolerances are invalid")
    operations = _mapping(payload.get("operations"), "preflight operations")
    if set(operations) != set(EXPECTED_PREFLIGHT_OPERATIONS):
        raise ArtifactValidationError("preflight operation categories are incomplete")
    for name, operation_raw in operations.items():
        operation = _mapping(operation_raw, f"preflight operation {name}")
        if (
            operation.get("attempted") is not True
            or operation.get("passed") is not True
        ):
            raise ArtifactValidationError(f"preflight operation {name} did not pass")
        if (
            operation.get("device") != "mps"
            or operation.get("dtype") != EXECUTION_DTYPE
        ):
            raise ArtifactValidationError(
                f"preflight operation {name} changed device/dtype"
            )
        if operation.get("cpu_reference_passed") is not True:
            raise ArtifactValidationError(
                f"preflight operation {name} lacks CPU comparison"
            )
    operator_records = _mapping(operations["operators"].get("operations"), "operators")
    if set(operator_records) != set(EXPECTED_OPERATOR_NAMES):
        raise ArtifactValidationError("preflight primitive operator set is incomplete")
    for name, item_raw in operator_records.items():
        item = _mapping(item_raw, f"operator {name}")
        if (
            item.get("attempted") is not True
            or item.get("passed") is not True
            or item.get("device") != "mps"
            or item.get("dtype") != EXECUTION_DTYPE
            or item.get("cpu_reference_passed") is not True
        ):
            raise ArtifactValidationError(f"operator {name} failed its MPS/CPU probe")
    jump = operations["strict_jumprelu"]
    if (
        jump.get("strict_gate_equal") is not True
        or jump.get("equality_inactive") is not True
    ):
        raise ArtifactValidationError("strict JumpReLU preflight failed")
    sparse = operations["sparse_metadata_boundary"]
    if (
        sparse.get("cpu_metadata_explicit") is not True
        or sparse.get("replacement_boundary_passed") is not True
        or sparse.get("dense_scientific_device") != "mps"
    ):
        raise ArtifactValidationError("explicit sparse metadata boundary failed")
    disk = operations["disk_offload_safetensors"]
    if disk.get("upstream_disk_offload_helper_tested") is not True:
        raise ArtifactValidationError("selected disk-offload helper was not tested")


def _validate_timing(value: Any, label: str) -> dict[str, Any]:
    timing = _mapping(value, f"{label} timing")
    required = {
        "started_at_unix",
        "finished_at_unix",
        "wall_seconds",
        "sampling_method",
        "sampling_interval_seconds",
        "sample_count",
        "mps_current_allocated_peak_bytes",
        "mps_driver_allocated_peak_bytes",
        "mps_recommended_max_bytes",
        "process_rss_peak_bytes",
        "memory_pressure_states",
        "swap_used_peak_bytes",
    }
    _exact_keys(timing, required, f"{label} timing")
    started = _number(timing["started_at_unix"], f"{label} start", minimum=0)
    finished = _number(timing["finished_at_unix"], f"{label} finish", minimum=0)
    wall = _number(timing["wall_seconds"], f"{label} wall time", minimum=0)
    if finished < started or abs((finished - started) - wall) > max(0.5, wall * 0.05):
        raise ArtifactValidationError(f"{label} timing interval is inconsistent")
    method = _text(timing["sampling_method"], f"{label} sampling method").casefold()
    if "mps" not in method or "sample" not in method or "cuda" in method:
        raise ArtifactValidationError(f"{label} telemetry method is not sampled MPS")
    interval = _number(
        timing["sampling_interval_seconds"],
        f"{label} sampling interval",
        minimum=1e-6,
    )
    sample_count = _integer(timing["sample_count"], f"{label} sample count", minimum=2)
    boundary_samples = 2 * len(EXPECTED_ATTEMPT_STAGES) if label == "memory" else 2
    if max(0, sample_count - boundary_samples) * interval > wall + max(
        0.5, interval
    ) or wall > max(1.0, interval * sample_count * 3):
        raise ArtifactValidationError(
            f"{label} sample count/interval cannot cover its timing window"
        )
    current = _integer(
        timing["mps_current_allocated_peak_bytes"],
        f"{label} current MPS peak",
    )
    driver = _integer(
        timing["mps_driver_allocated_peak_bytes"],
        f"{label} driver MPS peak",
        minimum=1,
    )
    recommended = _integer(
        timing["mps_recommended_max_bytes"],
        f"{label} recommended MPS maximum",
        minimum=1,
    )
    if current > driver or driver > recommended:
        raise ArtifactValidationError(
            f"{label} MPS allocator counters are internally impossible"
        )
    _integer(
        timing["process_rss_peak_bytes"],
        f"{label} process RSS peak",
        minimum=1,
    )
    _integer(timing["swap_used_peak_bytes"], f"{label} swap peak")
    pressure = timing["memory_pressure_states"]
    if (
        not isinstance(pressure, list)
        or not pressure
        or any(
            not isinstance(item, str) or item.casefold() != "normal"
            for item in pressure
        )
    ):
        raise ArtifactValidationError(f"{label} memory pressure telemetry is unsafe")
    return timing


def _validate_science_common(
    record: dict[str, Any], label: str, execution_commit: str
) -> dict[str, Any]:
    if record.get("status") != "completed":
        raise ArtifactValidationError(f"{label} did not complete")
    if _validate_runtime_provenance(record, label) != execution_commit:
        raise ArtifactValidationError(f"{label} execution commit mismatch")
    payload = _mapping(record.get("payload"), f"{label} payload")
    if payload.get("nonfinite_count") != 0:
        raise ArtifactValidationError(f"{label} contains non-finite values")
    _validate_timing(payload.get("timing"), label)
    return payload


def _validate_attribution(payload: dict[str, Any]) -> None:
    _exact_keys(
        payload,
        {
            "parameters",
            "accepted_batch_size",
            "graph",
            "raw_validation",
            "nonfinite_count",
            "timing",
        },
        "attribution payload",
    )
    parameters = _mapping(payload["parameters"], "attribution parameters")
    if parameters != {
        "prompt": "The capital of state containing Dallas is",
        "max_n_logits": 10,
        "desired_logit_probability": 0.95,
        "max_feature_nodes": 8192,
        "offload": "disk",
    }:
        raise ArtifactValidationError("attribution scientific parameters are invalid")
    if payload["accepted_batch_size"] not in {256, 128, 64}:
        raise ArtifactValidationError("attribution accepted batch is invalid")
    graph = _mapping(payload["graph"], "attribution graph")
    _exact_keys(
        graph,
        {
            "finite",
            "node_count",
            "selected_feature_count",
            "active_feature_count",
            "edge_count",
            "logit_node_count",
            "input_node_count",
            "error_node_count",
            "adjacency_shape",
        },
        "attribution graph",
    )
    _true(graph["finite"], "attribution graph finite")
    nodes = _integer(graph["node_count"], "attribution node count", minimum=1)
    selected = _integer(
        graph["selected_feature_count"], "selected feature count", minimum=1
    )
    active = _integer(graph["active_feature_count"], "active feature count", minimum=1)
    if selected > 8192 or selected > active:
        raise ArtifactValidationError("attribution feature counts are invalid")
    edges = _integer(graph["edge_count"], "attribution edge count", minimum=1)
    if graph["logit_node_count"] != 10:
        raise ArtifactValidationError("attribution must contain 10 logit nodes")
    inputs = _integer(graph["input_node_count"], "input node count", minimum=1)
    errors = _integer(graph["error_node_count"], "error node count")
    if (
        nodes != selected + 10 + inputs + errors
        or edges > nodes * nodes
        or graph["adjacency_shape"] != [nodes, nodes]
    ):
        raise ArtifactValidationError("attribution adjacency dimensions are invalid")
    raw_validation = _mapping(payload["raw_validation"], "raw graph validation")
    if raw_validation != {"passed": True, "raw_graph_committed": False}:
        raise ArtifactValidationError("raw graph validation/retention boundary failed")


def _validate_intervention(payload: dict[str, Any]) -> None:
    _exact_keys(
        payload,
        {
            "parameters",
            "baseline_activation_captured",
            "baseline_activation",
            "baseline_repeat_error",
            "baseline_repeat_max_combined_tolerance_ratio",
            "baseline_noop_comparison",
            "desired_values",
            "outputs_finite",
            "same_assets_and_runtime",
            "nonfinite_count",
            "timing",
        },
        "intervention payload",
    )
    parameters = _mapping(payload["parameters"], "intervention parameters")
    if parameters != {
        "prompt": "Hecho: Michael Jordan juega al",
        "feature": {"layer": 20, "position": -1, "feature_id": 341},
        "alphas": [0.0, 0.5, 1.0],
        "freeze_attention": True,
        "constrained_layers": None,
    }:
        raise ArtifactValidationError("intervention scientific parameters are invalid")
    _true(payload["baseline_activation_captured"], "baseline activation captured")
    baseline = _number(payload["baseline_activation"], "baseline activation")
    if baseline <= 0.0:
        raise ArtifactValidationError("official intervention feature is inactive")
    _number(payload["baseline_repeat_error"], "baseline repeat error", minimum=0)
    repeat_ratio = _number(
        payload["baseline_repeat_max_combined_tolerance_ratio"],
        "baseline repeat combined tolerance ratio",
        minimum=0,
    )
    if repeat_ratio > 1.0:
        raise ArtifactValidationError("baseline repeat exceeds combined tolerance")
    comparison = _mapping(
        payload["baseline_noop_comparison"], "baseline/no-op comparison"
    )
    _exact_keys(
        comparison,
        {
            "within_tolerance",
            "max_abs_error",
            "max_rel_error",
            "max_combined_tolerance_ratio",
            "absolute_tolerance",
            "relative_tolerance",
        },
        "baseline/no-op comparison",
    )
    _number(comparison["max_abs_error"], "no-op absolute error", minimum=0)
    _number(comparison["max_rel_error"], "no-op relative error", minimum=0)
    combined_ratio = _number(
        comparison["max_combined_tolerance_ratio"],
        "no-op combined tolerance ratio",
        minimum=0,
    )
    if (
        comparison["within_tolerance"] is not True
        or comparison["absolute_tolerance"] != 0.02
        or comparison["relative_tolerance"] != 0.002
        or combined_ratio > 1.0
    ):
        raise ArtifactValidationError("baseline/no-op intervention check failed")
    values = payload["desired_values"]
    if not isinstance(values, list) or len(values) != 3:
        raise ArtifactValidationError("intervention desired values are incomplete")
    for item_raw, alpha in zip(values, (0.0, 0.5, 1.0), strict=True):
        item = _mapping(item_raw, f"intervention alpha {alpha}")
        _exact_keys(
            item,
            {
                "alpha",
                "expected_activation",
                "observed_activation",
                "absolute_error",
                "within_tolerance",
                "output_finite",
            },
            f"intervention alpha {alpha}",
        )
        if item["alpha"] != alpha:
            raise ArtifactValidationError("intervention alpha ordering is invalid")
        expected = _number(item["expected_activation"], "expected activation")
        observed = _number(item["observed_activation"], "observed activation")
        error = _number(item["absolute_error"], "activation error", minimum=0)
        formula_value = (1.0 - alpha) * baseline
        if (
            abs(expected - formula_value) > 0.005
            or abs(error - abs(observed - expected)) > 1e-9
            or error > 0.005
            or item["within_tolerance"] is not True
            or item["output_finite"] is not True
        ):
            raise ArtifactValidationError("intervention desired mapping failed")
    _true(payload["outputs_finite"], "intervention outputs finite")
    _true(payload["same_assets_and_runtime"], "intervention asset/runtime identity")


def _validate_gate_sample(
    value: Any,
    label: str,
    *,
    expected_feature_id: int | None = None,
    expected_active: bool | None = None,
) -> tuple[int, float]:
    sample = _mapping(value, label)
    _exact_keys(
        sample,
        {
            "layer",
            "position",
            "feature_id",
            "preactivation",
            "threshold",
            "post_gate_activation",
            "active",
            "signed_margin",
        },
        label,
    )
    if sample["layer"] != 20 or sample["position"] != -1:
        raise ArtifactValidationError(f"{label} targets the wrong layer/position")
    feature_id = _integer(sample["feature_id"], f"{label} feature id")
    if feature_id >= 16_384 or (
        expected_feature_id is not None and feature_id != expected_feature_id
    ):
        raise ArtifactValidationError(f"{label} feature id is invalid")
    preactivation = _number(sample["preactivation"], f"{label} preactivation")
    threshold = _number(sample["threshold"], f"{label} threshold")
    post_gate = _number(sample["post_gate_activation"], f"{label} activation")
    signed_margin = _number(sample["signed_margin"], f"{label} signed margin")
    active = sample["active"]
    if not isinstance(active, bool) or (
        expected_active is not None and active is not expected_active
    ):
        raise ArtifactValidationError(f"{label} active flag is invalid")
    if abs(signed_margin - (preactivation - threshold)) > 1e-9:
        raise ArtifactValidationError(f"{label} signed margin is inconsistent")
    expected_activation = preactivation if active else 0.0
    if (
        active != (preactivation > threshold)
        or abs(post_gate - expected_activation) > 0.005
    ):
        raise ArtifactValidationError(f"{label} violates strict JumpReLU semantics")
    return feature_id, post_gate


def _validate_semantics(payload: dict[str, Any]) -> None:
    _exact_keys(
        payload,
        {
            "loaded_runtime",
            "preactivation",
            "gate_check",
            "intervention_value_check",
            "feature",
            "baseline_repeat_error",
            "projection_discrepancy",
            "gate_discrepancy",
            "nonfinite_count",
            "timing",
        },
        "semantics payload",
    )
    loaded = _mapping(payload["loaded_runtime"], "loaded runtime semantics")
    expected_loaded = {
        "passed": True,
        "model_loaded": True,
        "transcoder_loaded": True,
        "model_device": "mps",
        "transcoder_device": "mps",
        "model_dtype": EXECUTION_DTYPE,
        "transcoder_dtype": EXECUTION_DTYPE,
        "model_only_forward_passed": True,
        "output_finite": True,
        "fallback_used": False,
    }
    _exact_keys(
        loaded,
        set(expected_loaded) | {"model_only_forward"},
        "loaded runtime semantics",
    )
    if any(loaded.get(key) != expected for key, expected in expected_loaded.items()):
        raise ArtifactValidationError("loaded runtime semantics are missing or invalid")
    model_only = _mapping(
        loaded["model_only_forward"], "progressive model-only forward"
    )
    _exact_keys(
        model_only,
        {
            "passed",
            "prompt",
            "token_count",
            "logits_shape",
            "tokenizer_revision",
            "device",
            "dtype",
            "finite",
            "completed_before_transcoder_load",
        },
        "progressive model-only forward",
    )
    token_count = _integer(
        model_only["token_count"], "model-only token count", minimum=1
    )
    if (
        model_only["passed"] is not True
        or model_only["prompt"] != "The capital of state containing Dallas is"
        or model_only["logits_shape"] != [1, token_count, 256_000]
        or model_only["tokenizer_revision"] != OFFICIAL_MODEL_REVISION
        or model_only["device"] != "mps"
        or model_only["dtype"] != EXECUTION_DTYPE
        or model_only["finite"] is not True
        or model_only["completed_before_transcoder_load"] is not True
    ):
        raise ArtifactValidationError(
            "progressive model-only forward evidence is invalid"
        )
    preactivation = _mapping(payload["preactivation"], "preactivation semantics")
    _exact_keys(
        preactivation,
        {
            "verified",
            "threshold_retrieved",
            "definition",
            "bias_convention",
            "cache_shape",
            "projection_absolute_tolerance",
        },
        "preactivation semantics",
    )
    if (
        preactivation.get("verified") is not True
        or preactivation.get("threshold_retrieved") is not True
        or preactivation.get("definition") != "F.linear(feature_input, W_enc, b_enc)"
        or preactivation.get("bias_convention")
        != "b_enc is included; b_dec is excluded"
        or preactivation.get("projection_absolute_tolerance") != 0.005
    ):
        raise ArtifactValidationError("preactivation/bias semantics are incomplete")
    cache_shape = preactivation["cache_shape"]
    if (
        not isinstance(cache_shape, list)
        or len(cache_shape) != 3
        or cache_shape[0] != 26
        or _integer(cache_shape[1], "semantic cache positions", minimum=1) < 1
        or cache_shape[2] != 16_384
    ):
        raise ArtifactValidationError("loaded preactivation cache shape is invalid")
    gate = _mapping(payload["gate_check"], "JumpReLU gate check")
    _exact_keys(
        gate,
        {
            "rule",
            "strict_greater_than",
            "equality_inactive",
            "equality_probe_maximum_absolute_output",
            "absolute_tolerance",
            "active_example",
            "inactive_example",
            "official_intervention_source",
        },
        "JumpReLU gate check",
    )
    if (
        gate.get("rule") != "z if z > threshold else 0"
        or gate.get("strict_greater_than") is not True
        or gate.get("equality_inactive") is not True
        or gate.get("absolute_tolerance") != 0.005
        or _number(
            gate.get("equality_probe_maximum_absolute_output"),
            "gate equality output",
            minimum=0,
        )
        != 0.0
    ):
        raise ArtifactValidationError("JumpReLU strict semantics failed")
    active_id, _active_value = _validate_gate_sample(
        gate["active_example"], "active gate example", expected_active=True
    )
    inactive_id, _inactive_value = _validate_gate_sample(
        gate["inactive_example"], "inactive gate example", expected_active=False
    )
    official_id, official_value = _validate_gate_sample(
        gate["official_intervention_source"],
        "official intervention gate source",
        expected_feature_id=341,
        expected_active=True,
    )
    if len({active_id, inactive_id}) != 2 or official_id != 341:
        raise ArtifactValidationError("loaded gate examples are not distinct/official")
    value_check = _mapping(
        payload["intervention_value_check"], "intervention value semantics"
    )
    _exact_keys(
        value_check,
        {"passed", "formula", "alphas"},
        "intervention value semantics",
    )
    if (
        value_check.get("passed") is not True
        or value_check.get("formula") != "(1-alpha)*baseline_activation"
        or value_check.get("alphas") != [0.0, 0.5, 1.0]
    ):
        raise ArtifactValidationError("loaded desired intervention mapping failed")
    feature = _mapping(payload["feature"], "loaded semantic feature")
    _exact_keys(
        feature,
        {"layer", "position", "feature_id", "baseline_activation"},
        "loaded semantic feature",
    )
    if {key: feature.get(key) for key in ("layer", "position", "feature_id")} != {
        "layer": 20,
        "position": -1,
        "feature_id": 341,
    }:
        raise ArtifactValidationError("loaded semantic feature is invalid")
    baseline = _number(
        feature.get("baseline_activation"), "semantic baseline activation"
    )
    if baseline <= 0.0 or abs(baseline - official_value) > 0.005:
        raise ArtifactValidationError("official loaded feature baseline is invalid")
    if (
        _number(payload["baseline_repeat_error"], "semantic repeat error", minimum=0)
        > 0.02
        or _number(
            payload["projection_discrepancy"], "projection discrepancy", minimum=0
        )
        > 0.005
        or _number(payload["gate_discrepancy"], "gate discrepancy", minimum=0) > 0.005
    ):
        raise ArtifactValidationError("loaded runtime semantic tolerance failed")


def _validate_peak_mapping(value: Any, label: str) -> dict[str, int]:
    mapping = _mapping(value, label)
    _exact_keys(mapping, set(PEAK_KEYS), label)
    result: dict[str, int] = {}
    for key in PEAK_KEYS:
        minimum = (
            1
            if key in {"mps_driver_allocated_peak_bytes", "process_rss_peak_bytes"}
            else 0
        )
        result[key] = _integer(mapping[key], f"{label} {key}", minimum=minimum)
    if (
        result["mps_current_allocated_peak_bytes"]
        > result["mps_driver_allocated_peak_bytes"]
    ):
        raise ArtifactValidationError(f"{label} MPS current peak exceeds driver peak")
    return result


def _validate_attempts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    attempts = payload.get("attempts")
    if not isinstance(attempts, list) or not attempts or len(attempts) > 3:
        raise ArtifactValidationError("memory attempt history is missing or too long")
    expected_batches = [256, 128, 64][: len(attempts)]
    observed_batches: list[int] = []
    for index, item_raw in enumerate(attempts):
        item = _mapping(item_raw, f"memory attempts[{index}]")
        _exact_keys(
            item,
            {
                "batch_size",
                "outcome",
                "category",
                "failure_stage",
                "fresh_process",
                "cleanup_succeeded",
                "process_exit_code",
                "sample_count",
                "oom_classifier_match",
                "exception_type",
                "diagnostic_redacted",
                "stage_peaks",
                "attempt_peaks",
            },
            f"memory attempts[{index}]",
        )
        observed_batches.append(
            _integer(item["batch_size"], "attempt batch", minimum=1)
        )
        _true(item["fresh_process"], "attempt fresh_process")
        _true(item["cleanup_succeeded"], "attempt cleanup_succeeded")
        _true(item["diagnostic_redacted"], "attempt diagnostic_redacted")
        _integer(item["sample_count"], "attempt sample count", minimum=2)
        stages = _mapping(item["stage_peaks"], "attempt stage peaks")
        if set(stages) != set(EXPECTED_ATTEMPT_STAGES):
            raise ArtifactValidationError("attempt stage peak set is incomplete")
        stage_peaks = {
            name: _validate_peak_mapping(stage, f"attempt stage {name}")
            for name, stage in stages.items()
        }
        attempt_peaks = _validate_peak_mapping(item["attempt_peaks"], "attempt peaks")
        for metric in PEAK_KEYS:
            if attempt_peaks[metric] < max(
                stage[metric] for stage in stage_peaks.values()
            ):
                raise ArtifactValidationError(
                    f"attempt {metric} is below a like-for-like stage peak"
                )
        if index + 1 < len(attempts):
            if (
                item["outcome"] != "failed"
                or item["category"] != "mps_out_of_memory"
                or item["failure_stage"] != "attribution"
                or not isinstance(item["process_exit_code"], int)
                or item["process_exit_code"] == 0
                or item["oom_classifier_match"] is not True
                or not isinstance(item["exception_type"], str)
                or not item["exception_type"].strip()
            ):
                raise ArtifactValidationError(
                    "only attribution-stage MPS OOM may retry"
                )
        else:
            if (
                item["outcome"] != "completed"
                or item["category"] != "completed"
                or item["failure_stage"] is not None
                or item["process_exit_code"] != 0
                or item["oom_classifier_match"] is not False
                or item["exception_type"] is not None
            ):
                raise ArtifactValidationError("accepted MPS attempt did not complete")
            if not {"attribution", "intervention", "semantics"}.issubset(stages):
                raise ArtifactValidationError(
                    "accepted attempt lacks science-stage peaks"
                )
    if observed_batches != expected_batches:
        raise ArtifactValidationError("attempt batches are not the 256->128->64 prefix")
    accepted_index = payload.get("accepted_attempt_index")
    if accepted_index != len(attempts) - 1:
        raise ArtifactValidationError("accepted attempt index is invalid")
    if payload.get("accepted_batch_size") != observed_batches[-1]:
        raise ArtifactValidationError("accepted batch does not match attempt history")
    return attempts


def _validate_memory(payload: dict[str, Any]) -> list[dict[str, Any]]:
    _exact_keys(
        payload,
        {
            "nonfinite_count",
            "timing",
            "telemetry_method",
            "sampling_interval_seconds",
            "accepted_attempt_index",
            "accepted_batch_size",
            "attempts",
        },
        "memory payload",
    )
    method = _text(payload["telemetry_method"], "memory telemetry method")
    timing = _mapping(payload["timing"], "memory timing")
    if method != timing.get("sampling_method"):
        raise ArtifactValidationError("memory telemetry method is inconsistent")
    interval = _number(
        payload["sampling_interval_seconds"], "memory sampling interval", minimum=1e-6
    )
    timing_interval = _number(
        timing.get("sampling_interval_seconds"),
        "memory timing sampling interval",
        minimum=1e-6,
    )
    if interval != timing_interval:
        raise ArtifactValidationError("memory sampling interval is inconsistent")
    return _validate_attempts(payload)


def _validate_science_summaries(
    records: dict[str, dict[str, Any]], execution_commit: str
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    payloads: dict[str, dict[str, Any]] = {}
    for name, label in (
        ("attribution_summary.json", "attribution"),
        ("intervention_summary.json", "intervention"),
        ("semantics_summary.json", "semantics"),
        ("memory_summary.json", "memory"),
    ):
        payloads[name] = _validate_science_common(
            records[name], label, execution_commit
        )
    _validate_attribution(payloads["attribution_summary.json"])
    _validate_intervention(payloads["intervention_summary.json"])
    _validate_semantics(payloads["semantics_summary.json"])
    attempts = _validate_memory(payloads["memory_summary.json"])
    return payloads, attempts


def _validate_run_manifest(
    record: dict[str, Any], *, require_complete: bool
) -> tuple[str, str]:
    required = {
        "schema_version",
        "run_id",
        "status",
        "reproduction_class",
        "claim_boundary",
        "project",
        "upstream",
        "model",
        "transcoder",
        "runtime",
        "timings",
        "retry_history",
        "checks",
        "artifacts",
        "readiness",
    }
    _exact_keys(record, required, "MPS run manifest")
    if record.get("schema_version") != 1 or isinstance(
        record.get("schema_version"), bool
    ):
        raise ArtifactValidationError("MPS run manifest schema_version is invalid")
    run_id = _text(record.get("run_id"), "MPS run id")
    if record.get("reproduction_class") != REPRODUCTION_CLASS:
        raise ArtifactValidationError("MPS reproduction class is invalid")
    if record.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ArtifactValidationError("MPS claim boundary is not exact")
    status = record.get("status")
    if status != COMPLETED_STATUS:
        raise ArtifactValidationError(
            "the full canonical artifact contract only accepts a completed run"
        )
    if not isinstance(require_complete, bool):
        raise ArtifactValidationError("require_complete must be boolean")
    project = _mapping(record.get("project"), "run project")
    _exact_keys(
        project,
        {
            "base_commit",
            "execution_commit",
            "source_clean_excluding_preserved_t4",
            "preserved_t4_untracked",
        },
        "run project",
    )
    if project["base_commit"] != PROJECT_BASE_COMMIT:
        raise ArtifactValidationError("run project base commit is invalid")
    execution_commit = _sha40(project["execution_commit"], "run execution commit")
    _true(
        project["source_clean_excluding_preserved_t4"],
        "source_clean_excluding_preserved_t4",
    )
    _true(project["preserved_t4_untracked"], "preserved_t4_untracked")
    for label, section, identifier, revision in (
        (
            "upstream",
            record["upstream"],
            OFFICIAL_UPSTREAM_REPOSITORY,
            OFFICIAL_UPSTREAM_REVISION,
        ),
        ("model", record["model"], OFFICIAL_MODEL_ID, OFFICIAL_MODEL_REVISION),
        (
            "transcoder",
            record["transcoder"],
            OFFICIAL_TRANSCODER_ID,
            OFFICIAL_TRANSCODER_REVISION,
        ),
    ):
        item = _mapping(section, label)
        _exact_keys(item, {"identifier", "revision"}, label)
        if item["identifier"] != identifier or item["revision"] != revision:
            raise ArtifactValidationError(f"{label} immutable pin is invalid")
    runtime = _mapping(record.get("runtime"), "run runtime")
    expected_runtime = {
        "backend": "transformerlens",
        "accelerator_backend": "mps",
        "device": "mps",
        "architecture": "arm64",
        "hardware_family": HARDWARE_FAMILY,
        "reference_dtype": REFERENCE_DTYPE,
        "execution_dtype": EXECUTION_DTYPE,
        "execution_class": EXECUTION_CLASS,
        "official_bf16_reproduction": False,
        "t4_fp16_reproduction": False,
        "fallback_enabled": False,
        "fallback_used": False,
        "offload": "disk",
        "execution_deviations": CANONICAL_DEVIATIONS,
    }
    _exact_keys(
        runtime,
        set(expected_runtime) | {"accepted_batch_size", "retry_occurred"},
        "run runtime",
    )
    for key, expected in expected_runtime.items():
        if runtime.get(key) != expected:
            raise ArtifactValidationError(f"run runtime field {key!r} is invalid")
    if runtime["accepted_batch_size"] not in {256, 128, 64}:
        raise ArtifactValidationError("run accepted batch is invalid")
    if not isinstance(runtime["retry_occurred"], bool):
        raise ArtifactValidationError("run retry flag is invalid")
    timings = _mapping(record.get("timings"), "run timings")
    _exact_keys(timings, set(TIMING_ALIASES), "run timings")
    retry_history = record.get("retry_history")
    if not isinstance(retry_history, list) or not retry_history:
        raise ArtifactValidationError("run retry history is missing")
    checks = _mapping(record.get("checks"), "run checks")
    required_checks = {
        "preflight_passed",
        "feasibility_passed",
        "assets_verified",
        "model_only_forward_passed",
        "loaded_runtime_semantics_passed",
        "attribution_passed",
        "intervention_passed",
        "telemetry_passed",
        "no_hidden_fallback",
        "nonfinite_count",
    }
    _exact_keys(checks, required_checks, "run checks")
    if checks["nonfinite_count"] != 0 or any(
        checks[key] is not True for key in required_checks - {"nonfinite_count"}
    ):
        raise ArtifactValidationError("completed MPS run has failed checks")
    artifacts = _mapping(record.get("artifacts"), "run artifacts")
    expected_artifacts = {
        "preflight": f"{PREFLIGHT_DIRECTORY}/{PREFLIGHT_NAME}",
        "feasibility": "feasibility_report.json",
        "environment": "environment_manifest.json",
        "assets": "asset_manifest.json",
        "attribution": "attribution_summary.json",
        "intervention": "intervention_summary.json",
        "semantics": "semantics_summary.json",
        "memory": "memory_summary.json",
        "checksums": CHECKSUM_NAME,
    }
    if artifacts != expected_artifacts:
        raise ArtifactValidationError("run artifact map is not canonical")
    readiness = _mapping(record.get("readiness"), "run readiness")
    if readiness != {
        "stage1b_engineering_readiness": status == COMPLETED_STATUS,
        "stage1b_empirical_claim_readiness": False,
    }:
        raise ArtifactValidationError("MPS readiness flags are invalid")
    return execution_commit, run_id


def _validate_cross_file(
    records: dict[str, dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    attempts: list[dict[str, Any]],
    manifest: dict[str, Any],
    execution_commit: str,
    run_id: str,
) -> None:
    for name, record in records.items():
        if name == RUN_MANIFEST_NAME:
            continue
        if record.get("run_id") != run_id:
            raise ArtifactValidationError("artifact run_id values are inconsistent")
        if record.get("provenance", {}).get("project_commit") != execution_commit:
            raise ArtifactValidationError("artifact execution commits are inconsistent")
    timings = _mapping(manifest["timings"], "run timings")
    for alias, name in TIMING_ALIASES.items():
        if timings.get(alias) != payloads[name]["timing"]:
            raise ArtifactValidationError("summary timing metadata mismatch")
    if manifest["retry_history"] != attempts:
        raise ArtifactValidationError("run retry history differs from memory summary")
    feasibility_memory = records["feasibility_report.json"]["payload"][
        "physical_memory_bytes"
    ]
    environment_memory = records["environment_manifest.json"]["payload"]["platform"][
        "physical_memory_bytes"
    ]
    if feasibility_memory != environment_memory:
        raise ArtifactValidationError(
            "physical-memory observations differ across files"
        )
    accepted_batch = payloads["memory_summary.json"]["accepted_batch_size"]
    if (
        payloads["attribution_summary.json"]["accepted_batch_size"] != accepted_batch
        or manifest["runtime"]["accepted_batch_size"] != accepted_batch
        or manifest["runtime"]["retry_occurred"] != (len(attempts) > 1)
    ):
        raise ArtifactValidationError("accepted batch/retry state is inconsistent")
    accepted = attempts[-1]
    memory_timing = payloads["memory_summary.json"]["timing"]
    if accepted["sample_count"] != memory_timing["sample_count"]:
        raise ArtifactValidationError(
            "accepted attempt sample count differs from memory timing"
        )
    stage_peaks = accepted["stage_peaks"]
    for alias in ("attribution", "intervention", "semantics"):
        stage = stage_peaks[alias]
        timing = payloads[f"{alias}_summary.json"]["timing"]
        for metric in PEAK_KEYS:
            if stage[metric] != timing[metric]:
                raise ArtifactValidationError("stage telemetry differs across files")
    for metric in PEAK_KEYS:
        if accepted["attempt_peaks"][metric] != memory_timing[metric]:
            raise ArtifactValidationError(
                "accepted attempt telemetry differs from memory summary"
            )
    memory_start = float(memory_timing["started_at_unix"])
    memory_finish = float(memory_timing["finished_at_unix"])
    science_timings = [
        payloads[f"{alias}_summary.json"]["timing"]
        for alias in ("semantics", "intervention", "attribution")
    ]
    for timing in science_timings:
        if (
            timing["sampling_method"] != memory_timing["sampling_method"]
            or timing["sampling_interval_seconds"]
            != memory_timing["sampling_interval_seconds"]
        ):
            raise ArtifactValidationError(
                "science and attempt telemetry sampling methods differ"
            )
    for earlier, later in pairwise(science_timings):
        if earlier["finished_at_unix"] > later["started_at_unix"]:
            raise ArtifactValidationError("science stages overlap or are out of order")
    for alias in ("attribution", "intervention", "semantics"):
        timing = payloads[f"{alias}_summary.json"]["timing"]
        if (
            timing["started_at_unix"] < memory_start
            or timing["finished_at_unix"] > memory_finish
        ):
            raise ArtifactValidationError(
                "science timing lies outside accepted attempt"
            )
    semantic_baseline = payloads["semantics_summary.json"]["feature"][
        "baseline_activation"
    ]
    intervention_baseline = payloads["intervention_summary.json"]["baseline_activation"]
    baseline_difference = abs(semantic_baseline - intervention_baseline)
    baseline_scale = max(abs(semantic_baseline), abs(intervention_baseline), 1.0)
    if baseline_difference > 0.02 or baseline_difference / baseline_scale > 0.002:
        raise ArtifactValidationError(
            "semantics/intervention feature baselines are inconsistent"
        )


def checksum_targets(directory: Path) -> tuple[Path, ...]:
    return tuple(directory / name for name in sorted(MPS_JSON_FILES))


def write_mps_checksums(directory: Path) -> str:
    """Write deterministic checksums for every canonical JSON artifact."""

    if directory.is_symlink() or not directory.is_dir():
        raise ArtifactValidationError("MPS artifact directory is missing or unsafe")
    for path in checksum_targets(directory):
        if path.is_symlink() or not path.is_file():
            raise ArtifactValidationError(f"missing checksum target: {path}")
    # Secret-bearing or otherwise unsafe content must never be hashed, even if
    # validation would reject it immediately after checksum generation.
    for path in checksum_targets(directory):
        record = _load_json(path)
        _walk_safety(record)
    return write_checksum_manifest_atomic(
        directory / CHECKSUM_NAME, checksum_targets(directory), root=directory
    )


def _validate_checksums(directory: Path) -> None:
    manifest = directory / CHECKSUM_NAME
    if manifest.is_symlink() or not manifest.is_file():
        raise ArtifactValidationError("checksum manifest is missing or unsafe")
    if manifest.stat().st_size > MAX_CHECKSUM_BYTES:
        raise ArtifactValidationError("checksum manifest exceeds size limit")
    verified = verify_checksum_manifest(manifest, root=directory)
    expected = build_checksum_manifest(checksum_targets(directory), root=directory)
    if (
        manifest.read_text(encoding="utf-8") != expected
        or set(verified) != MPS_JSON_FILES
    ):
        raise ArtifactValidationError(
            "MPS checksums are incomplete, unsorted, or mismatched"
        )


def validate_mps_artifact_directory(
    directory: Path, *, require_complete: bool = True
) -> tuple[str, ...]:
    """Validate one canonical Stage 1A MPS artifact directory."""

    if directory.is_symlink() or not directory.is_dir():
        raise ArtifactValidationError("MPS artifact directory is missing or unsafe")
    entries = tuple(sorted(directory.iterdir(), key=lambda path: path.name))
    names = {path.name for path in entries}
    if names != set(ROOT_JSON_FILES) | {PREFLIGHT_DIRECTORY}:
        raise ArtifactValidationError("MPS artifact root allowlist is not exact")
    for path in entries:
        if path.is_symlink() or (
            path.name != PREFLIGHT_DIRECTORY and not path.is_file()
        ):
            raise ArtifactValidationError(
                "MPS artifact directory contains an unsafe entry"
            )
    preflight_dir = directory / PREFLIGHT_DIRECTORY
    if preflight_dir.is_symlink() or not preflight_dir.is_dir():
        raise ArtifactValidationError("MPS preflight directory is unsafe")
    preflight_entries = tuple(preflight_dir.iterdir())
    if (
        len(preflight_entries) != 1
        or preflight_entries[0].name != PREFLIGHT_NAME
        or preflight_entries[0].is_symlink()
        or not preflight_entries[0].is_file()
    ):
        raise ArtifactValidationError("MPS preflight allowlist is not exact")
    total = sum(
        path.stat().st_size for path in (*entries, *preflight_entries) if path.is_file()
    )
    if total > MAX_TOTAL_BYTES:
        raise ArtifactValidationError("MPS artifacts exceed total size limit")

    records = {
        name: _validate_envelope(directory / name, expected_type)
        for name, expected_type in _EXPECTED_TYPES.items()
    }
    execution_commit, run_id = _validate_run_manifest(
        records[RUN_MANIFEST_NAME], require_complete=require_complete
    )
    _validate_feasibility(records["feasibility_report.json"], execution_commit)
    _validate_environment(records["environment_manifest.json"], execution_commit)
    _validate_assets(records["asset_manifest.json"], execution_commit)
    _validate_preflight(
        records[f"{PREFLIGHT_DIRECTORY}/{PREFLIGHT_NAME}"], execution_commit
    )
    payloads, attempts = _validate_science_summaries(records, execution_commit)
    _validate_cross_file(
        records,
        payloads,
        attempts,
        records[RUN_MANIFEST_NAME],
        execution_commit,
        run_id,
    )
    _validate_checksums(directory)
    return tuple(
        sorted(
            path.relative_to(directory).as_posix()
            for path in (*entries, *preflight_entries)
        )
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir", type=Path, default=REPOSITORY_ROOT / MPS_RESULT_DIRECTORY
    )
    parser.add_argument("--write-checksums", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report: dict[str, Any] = {"schema_version": 1, "valid": False, "errors": []}
    try:
        directory = args.artifact_dir.expanduser()
        if args.write_checksums:
            write_mps_checksums(directory)
        report["files"] = list(
            validate_mps_artifact_directory(
                directory, require_complete=not args.allow_incomplete
            )
        )
        report["valid"] = True
    except (OSError, ArtifactValidationError) as exc:
        message = str(exc).replace(str(REPOSITORY_ROOT), ".")
        message = re.sub(r"(?:file://)?/Users/[^/\s\"']+", "<HOME>", message)
        report["errors"] = [message]
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
