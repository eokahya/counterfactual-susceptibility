#!/usr/bin/env python3
"""Independent, fail-closed validator for the Stage 1C-v3 artifact bundle.

This module deliberately imports only the Python standard library. It does not
import the project prediction/intervention implementation: scores, pair
identities, schedules, observations, analyses, and aggregates are recomputed
from serialized JSON at this boundary.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import re
import stat
import struct
import sys
import zipfile
from itertools import pairwise
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any, NoReturn

ROOT = Path(__file__).resolve().parents[2]

SCHEMA_VERSION = 3
EXPERIMENT_CLASS = "stage1c_v3_preregistered_prospective_prediction"
BRANCH = "stage-1c-v3-preregistered-prospective-prediction"
BASE_COMMIT = "ee9cc944fbdabaa6437b7be3c997725fce5de0a6"
PROMPT_ID = "capital_norway_preregistered_v3"
PROMPT_TEXT = "The capital of Norway is"
EXPECTED_TOKEN_IDS = [2, 818, 5279, 529, 32649, 563]
PAIR_ID_SEED = "stage1c-v3-preregistered-prospective-prediction"
RUNTIME_FINGERPRINT = (
    "gemma3-270m@9b0cfec892e2/plt@fada11860ac1/"
    "circuit-tracer@8f1e2438df61/nnsight/mps/bf16/stage1c-v3"
)
PAIR_ID_DOMAIN = f"{EXPERIMENT_CLASS}:{PROMPT_ID}"
PROMPT_POOL = [
    "The capital of Italy is",
    "The capital of Spain is",
    "The capital of Japan is",
    "The capital of Canada is",
    "The capital of Brazil is",
    "The capital of India is",
    "The capital of Egypt is",
    "The capital of Norway is",
    "The capital of Poland is",
    "The capital of Chile is",
    "The capital of Kenya is",
    "The capital of Peru is",
]
PROMPT_SALT = "stage1c-v3-prompt-v1"
PROMPT_SHA256 = "66e7d4281197efefdbc83bf369d9d317faa7641990c27fa1c3842de99c358e41"
PROMPT_INDEX = 7
HISTORICAL_SOURCE = Path(
    "results/stage1c_first_prospective_prediction/prediction_manifest.json"
)
HISTORICAL_SOURCE_SHA256 = (
    "43cf17f3f87ff97f9fa2aa6b827c84416add5dced2824b69c057d99a5f2b882a"
)
HISTORICAL_SOURCE_BLOB_SHA1 = "847e9c3389097b529c0ac2861b3d519afe18d050"
HISTORICAL_FREEZE_COMMIT = "6ec950d93fe1215fdcfee68c87e1f58a23a78ae8"
DENYLIST_PATH = Path("configs/stage1c_v3_v1_exact_pair_denylist.json")
DENYLIST_SHA256 = "ee31e29e2eb2be5aa5cbf72b95d75ea275098592eb54106f20aa7b3ba87405ad"
OVERLAP_CATEGORIES = (
    "neither_endpoint_seen_in_v1",
    "source_endpoint_seen_in_v1_only",
    "target_endpoint_seen_in_v1_only",
    "both_endpoints_seen_but_not_as_exact_v1_pair",
)
PROTOCOL_PATHS = (
    "configs/stage1c_v3_preregistered_prospective_prediction.yaml",
    "configs/stage1c_v3_preregistered_prospective_prediction_artifact_schema.json",
    "configs/stage1c_v3_v1_exact_pair_denylist.json",
    "src/cfsus/stage1c_v3/__init__.py",
    "src/cfsus/stage1c_v3/config.py",
    "src/cfsus/stage1c_v3/historical.py",
    "src/cfsus/stage1c_v3/serialization.py",
    "src/cfsus/stage1c_v3/worker_result.py",
    "src/cfsus/stage1c_v3/prediction.py",
    "src/cfsus/stage1c_v3/vjp.py",
    "src/cfsus/stage1c_v3/runtime.py",
    "src/cfsus/stage1c_v3/intervention.py",
    "src/cfsus/stage1c_v3/intervention_runtime.py",
    "src/cfsus/stage1c_v3/analysis.py",
    "scripts/stage1c_v3/preflight_stage1c_v3.py",
    "scripts/stage1c_v3/run_stage1c_v3.py",
    "scripts/stage1c_v3/run_stage1c_v3_prediction_worker.py",
    "scripts/stage1c_v3/run_stage1c_v3_intervention_worker.py",
    "scripts/stage1c_v3/assemble_stage1c_prediction.py",
    "scripts/stage1c_v3/assemble_stage1c_artifacts.py",
    "scripts/stage1c_v3/validate_stage1c_v3_artifacts.py",
    "scripts/stage1c_v3/validate_stage1c_v3_denylist.py",
)
FORBIDDEN_PREDICTION_FIELDS = frozenset(
    {
        "actual_bf16_value_passed",
        "intervention_sweeps",
        "observed",
        "observed_alpha",
        "observed_crossing",
        "observed_outcome",
        "realized_suppression",
        "scientific_outcome",
        "target_activation_after_intervention",
    }
)

PREDICTION_TYPE = "stage1c_v3_prediction_manifest"
WORKER_TYPE = "stage1c_v3_intervention_worker"
FINAL_BUNDLE_TYPE = "stage1c_v3_final_bundle"
ALLOWLIST = (
    "run_manifest.json",
    "asset_manifest.json",
    "environment_manifest.json",
    "prediction_manifest.json",
    "intervention_sweeps.json",
    "crossing_summary.json",
    "local_linearity_summary.json",
    "memory_timing_summary.json",
    "attempts.json",
    "checksums.sha256",
)
JSON_NAMES = frozenset(ALLOWLIST[:-1])
MAX_FILE = 2 * 1024 * 1024
MAX_BUNDLE = 5 * 1024 * 1024
SHA40 = re.compile(r"\A[0-9a-f]{40}\Z")
SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")

ARTIFACT_TYPES = {
    "run_manifest.json": "stage1c_v3_final_bundle_run_manifest",
    "asset_manifest.json": "stage1c_v3_asset_manifest",
    "environment_manifest.json": "stage1c_v3_environment_manifest",
    "prediction_manifest.json": PREDICTION_TYPE,
    "intervention_sweeps.json": "stage1c_v3_intervention_sweeps",
    "crossing_summary.json": "stage1c_v3_crossing_summary",
    "local_linearity_summary.json": "stage1c_v3_local_linearity_summary",
    "memory_timing_summary.json": "stage1c_v3_memory_timing_summary",
    "attempts.json": "stage1c_v3_attempts",
}
FORBIDDEN_KEYS = frozenset(
    {
        "model_weight",
        "transcoder_weight",
        "cache",
        "raw_graph",
        "raw_adjacency",
        "adjacency",
        "full_derivative_matrix",
        "dense_preactivation_tensor",
        "dense_activation_tensor",
        "gradient_tensor",
        "tokenizer_payload",
    }
)
SENSITIVE_EXACT = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "bearer_token",
        "cookie",
        "credential",
        "credentials",
        "github_token",
        "hf_token",
        "id_token",
        "password",
        "passwd",
        "private_key",
        "private_path",
        "private_absolute_path",
        "refresh_token",
        "secret",
        "secret_key",
        "token",
    }
)
SENSITIVE_SUFFIXES = (
    "_api_key",
    "_authorization",
    "_cookie",
    "_credential",
    "_password",
    "_private_key",
    "_secret",
    "_token",
)
SECRET_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    re.compile(r"(?i)https?://[^/@\s:]+:[^/@\s]+@"),
)
PRIVATE_PATH_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9_])(?:file://)?/Users/[^\s\"']+"),
    re.compile(r"(?<![A-Za-z0-9_])(?:file://)?/home/[^\s\"']+"),
    re.compile(r"(?<![A-Za-z0-9_])/(?:private/)?var/(?:folders|tmp)/[^\s\"']+"),
    re.compile(r"(?<![A-Za-z0-9_])(?:\$HOME|~)/(?:\.cache|Library/Caches)[^\s\"']*"),
    re.compile(r"(?i)(?<![A-Za-z0-9_])(?:file:///)?[A-Z]:\\Users\\[^\\\s\"']+"),
)


class ValidationError(ValueError):
    """Raised when an artifact is unsafe or fails the v3 contract."""


def fail(message: str) -> NoReturn:
    raise ValidationError(message)


def _sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
    return normalized in SENSITIVE_EXACT or normalized.endswith(SENSITIVE_SUFFIXES)


def _scan_string(value: str, path: str) -> None:
    if any(ord(char) < 0x20 or 0x7F <= ord(char) <= 0x9F for char in value):
        fail(f"control character at {path}")
    if any(pattern.search(value) for pattern in SECRET_PATTERNS):
        fail(f"secret or credential-like content at {path}")
    if any(pattern.search(value) for pattern in PRIVATE_PATH_PATTERNS):
        fail(f"private path at {path}")


def scan_value(value: Any, path: str = "$") -> None:
    """Reject non-finite, sensitive, forbidden, or non-JSON values."""
    if type(value) is float:
        if not math.isfinite(value):
            fail(f"non-finite value at {path}")
        return
    if type(value) is str:
        _scan_string(value, path)
        return
    if value is None or type(value) is bool or type(value) is int:
        return
    if type(value) is list:
        for index, item in enumerate(value):
            scan_value(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                fail(f"non-string key at {path}")
            if _sensitive_key(key):
                fail(f"sensitive object key at {path}.{key}")
            normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold())
            if normalized in FORBIDDEN_KEYS:
                fail(f"forbidden payload key at {path}.{key}")
            scan_value(item, f"{path}.{key}")
        return
    fail(f"unsupported JSON value at {path}")


def _canonical_json_bytes(value: dict[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ValidationError("value cannot be encoded as canonical JSON") from error


def strict_json(raw: bytes, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> NoReturn:
        fail(f"non-finite JSON constant in {name}: {value}")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                fail(f"duplicate JSON key in {name}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_pairs,
        )
    except ValidationError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValidationError(f"invalid strict UTF-8 JSON: {name}") from error
    if type(value) is not dict:
        fail(f"{name} root must be an object")
    try:
        scan_value(value)
    except RecursionError as error:
        raise ValidationError(f"JSON nesting exceeds safe depth in {name}") from error
    if raw != _canonical_json_bytes(value):
        fail(f"{name} is not canonical deterministic JSON")
    return value


def _assert_no_symlink_ancestors(path: Path) -> Path:
    candidate = path if path.is_absolute() else Path.cwd() / path
    for parent in candidate.parents:
        try:
            info = parent.lstat()
        except OSError as error:
            raise ValidationError(
                f"cannot inspect artifact parent: {parent}"
            ) from error
        if stat.S_ISLNK(info.st_mode):
            fail(f"symlinked artifact ancestor: {parent}")
    return candidate


def _open_regular(path: Path, *, maximum: int, scan_content: bool = True) -> bytes:
    path = _assert_no_symlink_ancestors(path)
    try:
        initial = path.lstat()
    except OSError as error:
        raise ValidationError(f"cannot inspect artifact: {path}") from error
    if (
        stat.S_ISLNK(initial.st_mode)
        or not stat.S_ISREG(initial.st_mode)
        or initial.st_nlink != 1
    ):
        fail(f"not a single-link regular file: {path.name}")
    if initial.st_size > maximum:
        fail(f"file exceeds size cap: {path.name}")
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        current = os.fstat(descriptor)
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or current.st_size != initial.st_size
        ):
            fail(f"artifact changed or is not a single-link regular file: {path.name}")
        if current.st_size > maximum:
            fail(f"file exceeds size cap: {path.name}")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            raw = stream.read()
        if len(raw) != current.st_size or b"\0" in raw:
            fail(f"unstable or binary artifact: {path.name}")
        text = raw.decode("utf-8")
        if scan_content and any(
            pattern.search(text)
            for pattern in (*SECRET_PATTERNS, *PRIVATE_PATH_PATTERNS)
        ):
            fail(f"secret or private path in {path.name}")
        return raw
    except UnicodeError as error:
        raise ValidationError(f"artifact is not UTF-8 text: {path.name}") from error
    except ValidationError:
        raise
    except (OSError, ValueError) as error:
        raise ValidationError(f"failed to read artifact: {path.name}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _digest(value: Any, label: str) -> str:
    if type(value) is not str or SHA256.fullmatch(value) is None:
        fail(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        fail(f"{label} must be an integer")
    return value


def number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        fail(f"{label} must be finite numeric")
    return float(value)


def feature_key(
    value: Any, label: str = "feature", *, token_count: int | None = None
) -> tuple[int, int, int]:
    if type(value) is not dict or set(value) != {"layer", "position", "feature_id"}:
        fail(f"{label} is not an exact feature reference")
    result = tuple(value[key] for key in ("layer", "position", "feature_id"))
    if any(type(item) is not int or item < 0 for item in result):
        fail(f"{label} has invalid coordinates")
    layer, position, feature_id = result
    if layer >= 18 or position < 1 or feature_id >= 16_384:
        fail(f"{label} is outside the frozen PLT domain")
    if token_count is not None and position >= token_count:
        fail(f"{label} is outside the frozen prompt token domain")
    return result


def _feature_record(value: tuple[int, int, int]) -> dict[str, int]:
    return {
        "layer": value[0],
        "position": value[1],
        "feature_id": value[2],
    }


def _pair_record(
    pair: tuple[tuple[int, int, int], tuple[int, int, int]],
) -> dict[str, dict[str, int]]:
    return {"source": _feature_record(pair[0]), "target": _feature_record(pair[1])}


def _scan_prediction_outcomes(value: Any, path: str = "$") -> None:
    if type(value) is dict:
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
            if normalized in FORBIDDEN_PREDICTION_FIELDS:
                fail(f"intervention outcome field in prediction: {path}.{key}")
            _scan_prediction_outcomes(item, f"{path}.{key}")
    elif type(value) is list:
        for index, item in enumerate(value):
            _scan_prediction_outcomes(item, f"{path}[{index}]")


@functools.lru_cache(maxsize=1)
def _historical_metadata() -> tuple[
    frozenset[tuple[tuple[int, int, int], tuple[int, int, int]]],
    frozenset[tuple[int, int, int]],
]:
    source_raw = _open_regular(ROOT / HISTORICAL_SOURCE, maximum=MAX_FILE)
    if hashlib.sha256(source_raw).hexdigest() != HISTORICAL_SOURCE_SHA256:
        fail("historical source SHA-256 differs")
    source_blob = hashlib.sha1(
        f"blob {len(source_raw)}\0".encode("ascii") + source_raw
    ).hexdigest()
    if source_blob != HISTORICAL_SOURCE_BLOB_SHA1:
        fail("historical source Git blob identity differs")
    source = strict_json(source_raw, HISTORICAL_SOURCE.name)
    if any(
        source.get(key) != expected
        for key, expected in {
            "schema_version": 1,
            "artifact_type": "stage1c_prediction_manifest",
            "experiment_class": "stage1c_first_prospective_prediction",
            "branch": "stage-1c-first-prospective-prediction",
            "base_commit": "efbf70a7e462e640a0e1819a93f3b92727bbd193",
        }.items()
    ):
        fail("historical source identity differs")
    guards = source.get("prediction_only_guards")
    if (
        type(guards) is not dict
        or guards.get("source_suppression_api_calls") != 0
        or guards.get("prior_inactive_target_outcome_read") is not False
    ):
        fail("historical source is not baseline-only")
    groups = source.get("selected_groups")
    expected_counts = {"primary": 12, "near_boundary": 8, "directional": 8}
    if type(groups) is not dict or set(groups) != set(expected_counts):
        fail("historical selected groups differ")
    pairs: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    for group in ("primary", "near_boundary", "directional"):
        rows = groups[group]
        if type(rows) is not list or len(rows) != expected_counts[group]:
            fail(f"historical {group} count differs")
        for row in rows:
            if type(row) is not dict:
                fail("historical selected row is not an object")
            pairs.append(
                (feature_key(row.get("source")), feature_key(row.get("target")))
            )
    pairs.sort()
    if len(pairs) != 28 or len(set(pairs)) != 28:
        fail("historical exact-pair count or uniqueness differs")

    denylist_raw = _open_regular(ROOT / DENYLIST_PATH, maximum=MAX_FILE)
    if hashlib.sha256(denylist_raw).hexdigest() != DENYLIST_SHA256:
        fail("sanitized denylist SHA-256 differs")
    denylist = strict_json(denylist_raw, DENYLIST_PATH.name)
    expected_denylist = {
        "artifact_type": "stage1c_v3_historical_exact_pair_denylist",
        "pair_count": 28,
        "pairs": [_pair_record(pair) for pair in pairs],
        "schema_version": 1,
        "source_manifest": {
            "artifact_type": "stage1c_prediction_manifest",
            "branch": "stage-1c-first-prospective-prediction",
            "experiment_class": "stage1c_first_prospective_prediction",
            "freeze_commit": HISTORICAL_FREEZE_COMMIT,
            "git_blob_sha1": HISTORICAL_SOURCE_BLOB_SHA1,
            "path": HISTORICAL_SOURCE.as_posix(),
            "schema_version": 1,
            "sha256": HISTORICAL_SOURCE_SHA256,
        },
    }
    if denylist != expected_denylist:
        fail("sanitized denylist differs from independent source extraction")
    exact = frozenset(pairs)
    endpoints = frozenset(endpoint for pair in pairs for endpoint in pair)
    return exact, endpoints


def _overlap_category(
    source: tuple[int, int, int],
    target: tuple[int, int, int],
    exact_pairs: frozenset[tuple[tuple[int, int, int], tuple[int, int, int]]],
    endpoints: frozenset[tuple[int, int, int]],
) -> str:
    if (source, target) in exact_pairs:
        fail("historical exact pair entered the v3 prediction")
    source_seen = source in endpoints
    target_seen = target in endpoints
    if source_seen and target_seen:
        return OVERLAP_CATEGORIES[3]
    if source_seen:
        return OVERLAP_CATEGORIES[1]
    if target_seen:
        return OVERLAP_CATEGORIES[2]
    return OVERLAP_CATEGORIES[0]


def canonical_pair_id(
    *, source: tuple[int, int, int], target: tuple[int, int, int]
) -> str:
    payload = {
        "experiment_class": EXPERIMENT_CLASS,
        "prompt_id": PROMPT_ID,
        "runtime_fingerprint": RUNTIME_FINGERPRINT,
        "seed": PAIR_ID_SEED,
        "source": list(source),
        "target": list(target),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _require_identity(value: dict[str, Any], name: str, artifact_type: str) -> None:
    if value.get("schema_version") != SCHEMA_VERSION:
        fail(f"{name} schema version is not v3")
    if value.get("artifact_type") != artifact_type:
        fail(f"{name} artifact type is not the v3 type")
    if value.get("experiment_class") != EXPERIMENT_CLASS:
        fail(f"{name} experiment class differs")


def _expected_schedule(
    pair: dict[str, Any], protocol: dict[str, Any]
) -> tuple[float, ...]:
    schedule = protocol.get("schedule")
    if type(schedule) is not dict:
        fail("prediction schedule is missing")
    coarse = schedule.get("coarse_alphas")
    if type(coarse) is not list or not coarse:
        fail("coarse schedule is malformed")
    values = {number(item, "coarse alpha") for item in coarse}
    offset = number(schedule.get("alpha_hat_offset"), "alpha-hat offset")
    alpha = pair.get("predicted_alpha_star")
    if alpha is not None:
        alpha_value = number(alpha, "predicted alpha")
        if 0.0 <= alpha_value <= 1.0:
            values.update(
                {
                    max(0.0, min(1.0, alpha_value - offset)),
                    alpha_value,
                    max(0.0, min(1.0, alpha_value + offset)),
                }
            )
    if any(not 0.0 <= item <= 1.0 for item in values):
        fail("requested schedule leaves [0,1]")
    return tuple(sorted(values))


def _pair_order_key(
    row: dict[str, Any],
    group: str,
    primary_targets: set[tuple[int, int, int]],
    used_targets: set[tuple[int, int, int]],
) -> tuple[Any, ...]:
    target = feature_key(row["target"])
    source = feature_key(row["source"])
    if group == "primary":
        return (
            -number(row["susceptibility"], "susceptibility"),
            number(row["predicted_alpha_star"], "predicted alpha"),
            target,
            source,
        )
    if group == "near_boundary":
        return (
            target in primary_targets,
            number(row["predicted_alpha_star"], "predicted alpha") - 1.0,
            target,
            source,
        )
    return (
        target in used_targets,
        -abs(number(row["q"], "q")) / (number(row["margin"], "margin") + 1.0e-12),
        target,
        source,
    )


def scan_prediction(value: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and independently recompute the frozen v3 prediction rows."""
    _scan_prediction_outcomes(value)
    if set(value) != {
        "artifact_schema_sha256",
        "artifact_type",
        "base_commit",
        "baseline_pools",
        "branch",
        "claim_boundary",
        "config_sha256",
        "experiment_class",
        "historical_independence",
        "pair_id_domain",
        "prediction_only_guards",
        "prompt",
        "prompt_derivation",
        "protocol",
        "protocol_file_sha256",
        "schema_version",
        "selected_groups",
        "selection_audit",
        "status",
        "runtime_identity",
    }:
        fail("prediction manifest has missing or unknown top-level fields")
    _require_identity(value, "prediction manifest", PREDICTION_TYPE)
    if value.get("status") != "prediction_frozen_ready_for_commit":
        fail("prediction manifest is not frozen-ready")
    if value.get("base_commit") != BASE_COMMIT or value.get("branch") != BRANCH:
        fail("prediction manifest Git identity differs")
    _digest(value.get("config_sha256"), "prediction config_sha256")
    _digest(value.get("artifact_schema_sha256"), "prediction artifact_schema_sha256")
    if value.get("pair_id_domain") != PAIR_ID_DOMAIN:
        fail("prediction pair-ID domain differs")
    prompt = value.get("prompt")
    if (
        type(prompt) is not dict
        or prompt.get("id") != PROMPT_ID
        or prompt.get("text") != PROMPT_TEXT
    ):
        fail("preregistered prompt identity differs")
    token_ids = prompt.get("token_ids")
    if token_ids != EXPECTED_TOKEN_IDS:
        fail("frozen prompt token IDs differ from immutable tokenizer preflight")
    message = f"{BASE_COMMIT}|{PROMPT_SALT}"
    digest = hashlib.sha256(message.encode("utf-8")).hexdigest()
    index = int(digest[:16], 16) % len(PROMPT_POOL)
    if (
        value.get("prompt_derivation")
        != {
            "algorithm": "sha256_prefix16_mod_pool_length",
            "base_commit": BASE_COMMIT,
            "salt": PROMPT_SALT,
            "message": message,
            "sha256_hex": PROMPT_SHA256,
            "index": PROMPT_INDEX,
            "prompt": PROMPT_TEXT,
            "prompt_id": PROMPT_ID,
            "pool": PROMPT_POOL,
        }
        or digest != PROMPT_SHA256
        or index != PROMPT_INDEX
    ):
        fail("preregistered prompt derivation differs")
    historical_pairs, historical_endpoints = _historical_metadata()
    if value.get("historical_independence") != {
        "source_manifest_path": HISTORICAL_SOURCE.as_posix(),
        "source_manifest_sha256": HISTORICAL_SOURCE_SHA256,
        "source_manifest_git_blob_sha1": HISTORICAL_SOURCE_BLOB_SHA1,
        "source_manifest_freeze_commit": HISTORICAL_FREEZE_COMMIT,
        "denylist_path": DENYLIST_PATH.as_posix(),
        "denylist_sha256": DENYLIST_SHA256,
        "exact_pair_count": 28,
        "historical_endpoint_count": len(historical_endpoints),
        "mask_applied_before_ranking": True,
        "endpoint_overlap_policy": "audit_only",
        "historical_intervention_outcome_read": False,
        "v2_temporary_baseline_artifact_read": False,
    }:
        fail("historical-independence provenance differs")
    runtime = value.get("runtime_identity")
    required_runtime = {
        "backend": "nnsight",
        "device": "mps:0",
        "dtype": "torch.bfloat16",
        "model_identifier": "google/gemma-3-270m",
        "model_revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
        "transcoder_identifier": "mwhanna/gemma-scope-2-270m-pt",
        "transcoder_revision": "fada11860ac1d337c1e41e9da308798405b94c8e",
        "transcoder_subfolder": "transcoder_all/width_16k_l0_small",
        "layer_count": 18,
        "feature_width": 16_384,
        "upstream_revision": "8f1e2438df612464e229e44c4a00ff637bf9379b",
    }
    if type(runtime) is not dict or any(
        runtime.get(key) != expected for key, expected in required_runtime.items()
    ):
        fail("prediction immutable runtime identity differs")
    protocol = value.get("protocol")
    if type(protocol) is not dict or set(protocol) != {
        "analysis",
        "engineering_calibration",
        "intervention_regime",
        "responses",
        "scanner",
        "schedule",
        "scoring",
        "selection",
        "source_pool",
    }:
        fail("prediction protocol is missing")
    scanner = protocol.get("scanner")
    expected_scanner = {
        "selected_layers": list(range(18)),
        "selected_positions": list(range(1, len(token_ids))),
        "feature_width": 16_384,
        "canonical_chunk_size": 1_024,
        "top_k_per_group": 8,
        "global_top_k": 128,
        "dense_oracle_chunk_size": 16_384,
        "tie_break": ["margin", "layer", "position", "feature_id"],
        "dense_oracle_lifecycle": "one_group_ephemeral",
    }
    if scanner != expected_scanner:
        fail("prediction scanner protocol differs")
    if protocol.get("source_pool") != {
        "selection": "all_exact_loaded_active_features_with_causal_target",
        "ordering": ["layer", "position", "feature_id"],
        "require_positive_activation": True,
        "require_strictly_earlier_layer": True,
        "require_causal_position": True,
        "raw_graph_input": "forbidden",
        "maximum_active_sources": 10_000,
    }:
        fail("prediction source-pool protocol differs")
    if protocol.get("engineering_calibration") != {
        "endpoint_class": (
            "baseline_active_source_active_target_disjoint_from_inactive_pool"
        ),
        "pair_count": 4,
        "reference_method": "stage1b_independent_pairwise_targeted_vjp",
        "comparison": "exact_bf16_identity",
        "inactive_target_intervention_calls": 0,
    }:
        fail("prediction engineering-calibration protocol differs")
    responses = protocol.get("responses")
    if type(responses) is not dict or any(
        responses.get(key) != expected
        for key, expected in {
            "method": "target_encoder_reverse_vjp_many_source_contraction",
            "convention": "attribution_matched_target_preactivation_pre_gate",
            "graph_edge_input": "forbidden",
            "target_batch_size": 8,
            "maximum_eligible_pairs": 500_000,
        }.items()
    ):
        fail("prediction response protocol differs")
    if protocol.get("scoring") != {
        "epsilon": 1.0e-12,
        "crossing_tolerance": 1.0e-9,
        "pair_seed": PAIR_ID_SEED,
    }:
        fail("prediction scoring protocol differs")
    if protocol.get("selection") != {
        "primary_maximum": 12,
        "near_boundary_maximum": 8,
        "directional_maximum": 8,
        "maximum_per_target": 1,
        "maximum_primary_per_source": 2,
        "primary_order": ["susceptibility_desc", "alpha_hat_asc", "target", "source"],
        "near_order": ["distance_above_one_asc", "target", "source"],
        "directional_order": ["movement_over_margin_desc", "target", "source"],
        "prefer_unused_control_targets": True,
        "control_overlap_fallback": "deterministic_after_unique_exhausted",
    }:
        fail("prediction selection protocol differs")
    if protocol.get("schedule") != {
        "coarse_alphas": [0.0, 0.25, 0.5, 0.75, 1.0],
        "alpha_hat_offset": 0.015625,
        "maximum_bisection_steps": 8,
        "deduplicate_applied_bf16": True,
    }:
        fail("prediction schedule differs")
    if protocol.get("intervention_regime") != {
        "source_count": 1,
        "mapping": "desired=(1-alpha)*baseline",
        "freeze_attention": True,
        "constrained_layers": None,
        "target_clamp_allowed": False,
        "canonical_attempts": 1,
        "scientific_retries": 0,
    }:
        fail("prediction intervention regime differs")
    if protocol.get("analysis") != {
        "minimum_nonzero_points": 3,
        "movement_sign_agreement_minimum": 0.80,
        "median_movement_sne_maximum": 0.50,
        "p95_movement_sne_maximum": 1.00,
        "critical_bracket_distance_maximum": 0.125,
        "undefined_metric_policy": "null_with_reason",
    }:
        fail("prediction analysis protocol differs")
    pools = value.get("baseline_pools")
    if type(pools) is not dict or set(pools) != {
        "complete_derivative_matrix_persisted",
        "dense_scanner_arrays_persisted",
        "eligible_pair_count_after_historical_mask",
        "eligible_pair_count_before_historical_mask",
        "eligible_source_count",
        "eligible_target_count",
        "excluded_exact_historical_pair_count",
        "excluded_no_causal_source_target_count",
        "many_source_vjp_engineering_calibration",
        "pair_score_sha256_after_historical_mask",
        "pair_score_sha256_before_historical_mask",
        "predicted_status_counts",
        "q_sign_counts",
        "raw_active_source_count",
        "scanner_dense_oracle_validation",
        "scanner_candidate_count",
        "source_pool_sha256",
        "target_pool_sha256",
    }:
        fail("prediction baseline-pool schema differs")
    before = _integer(
        pools.get("eligible_pair_count_before_historical_mask"),
        "eligible pair count before mask",
        minimum=0,
    )
    excluded = _integer(
        pools.get("excluded_exact_historical_pair_count"),
        "excluded historical exact-pair count",
        minimum=0,
    )
    after = _integer(
        pools.get("eligible_pair_count_after_historical_mask"),
        "eligible pair count after mask",
        minimum=0,
    )
    if before - excluded != after:
        fail("historical exact-pair mask counts are inconsistent")
    scanner_count = _integer(
        pools.get("scanner_candidate_count"), "scanner candidate count", minimum=0
    )
    target_count = _integer(
        pools.get("eligible_target_count"), "eligible target count", minimum=0
    )
    excluded_targets = _integer(
        pools.get("excluded_no_causal_source_target_count"),
        "excluded target count",
        minimum=0,
    )
    raw_sources = _integer(
        pools.get("raw_active_source_count"), "raw active-source count", minimum=0
    )
    source_count = _integer(
        pools.get("eligible_source_count"), "eligible source count", minimum=0
    )
    if (
        scanner_count - excluded_targets != target_count
        or source_count > raw_sources
        or source_count > 10_000
        or before > 500_000
        or excluded > 28
    ):
        fail("prediction baseline-pool counts violate frozen bounds")
    calibration = pools.get("many_source_vjp_engineering_calibration")
    if calibration != {
        "endpoint_class": (
            "baseline_active_source_active_target_disjoint_from_inactive_pool"
        ),
        "pair_count": 4,
        "reference_method": "stage1b_independent_pairwise_targeted_vjp",
        "comparison": "exact_bf16_identity",
        "passed": True,
        "inactive_target_intervention_calls": 0,
    }:
        fail("prediction VJP calibration evidence differs")
    if pools.get("scanner_dense_oracle_validation") != {
        "group_count": 18 * (len(token_ids) - 1),
        "exact_dense_oracle_identity_and_order": True,
        "bounded_oracle_recall": 1.0,
        "dense_oracle_persisted": False,
    }:
        fail("prediction scanner dense-oracle evidence differs")
    for key in (
        "pair_score_sha256_before_historical_mask",
        "pair_score_sha256_after_historical_mask",
        "source_pool_sha256",
        "target_pool_sha256",
    ):
        _digest(pools.get(key), key)
    q_counts = pools.get("q_sign_counts")
    status_counts = pools.get("predicted_status_counts")
    if (
        type(q_counts) is not dict
        or set(q_counts) != {"positive", "zero", "negative"}
        or sum(_integer(item, "q count", minimum=0) for item in q_counts.values())
        != after
        or type(status_counts) is not dict
        or set(status_counts)
        != {"boundary_ambiguous", "definitely_crossing", "not_crossing"}
        or sum(
            _integer(item, "predicted status count", minimum=0)
            for item in status_counts.values()
        )
        != after
        or pools.get("complete_derivative_matrix_persisted") is not False
        or pools.get("dense_scanner_arrays_persisted") is not False
    ):
        fail("prediction masked-pool aggregate counts differ")
    groups = value.get("selected_groups")
    if type(groups) is not dict or set(groups) != {
        "primary",
        "near_boundary",
        "directional",
    }:
        fail("prediction selected groups are malformed")
    expected_rows: list[dict[str, Any]] = []
    primary_targets: set[tuple[int, int, int]] = set()
    all_ids: set[str] = set()
    group_rows: dict[str, list[dict[str, Any]]] = {}
    for group in ("primary", "near_boundary", "directional"):
        rows = groups[group]
        if type(rows) is not list:
            fail(f"prediction group {group} is not a list")
        group_rows[group] = rows
        seen_targets: set[tuple[int, int, int]] = set()
        for row in rows:
            if type(row) is not dict or row.get("group") != group:
                fail("prediction pair row/group is malformed")
            if set(row) != {
                "endpoint_overlap_category",
                "exact_pair_key",
                "group",
                "margin",
                "pair_id",
                "predicted_alpha_star",
                "predicted_status",
                "q",
                "requested_alphas",
                "source",
                "source_activation",
                "susceptibility",
                "target",
                "target_preactivation",
                "target_threshold",
                "targeted_response",
            }:
                fail("prediction pair row has missing or unknown fields")
            source = feature_key(
                row.get("source"), f"{group}.source", token_count=len(token_ids)
            )
            target = feature_key(
                row.get("target"), f"{group}.target", token_count=len(token_ids)
            )
            pair_key = source, target
            if row.get("exact_pair_key") != _pair_record(pair_key):
                fail("prediction exact-pair record differs")
            category = _overlap_category(
                source,
                target,
                historical_pairs,
                historical_endpoints,
            )
            if row.get("endpoint_overlap_category") != category:
                fail("prediction endpoint-overlap category differs")
            if source[0] >= target[0] or source[1] > target[1]:
                fail("prediction pair violates causal ordering")
            if target in seen_targets:
                fail(f"prediction {group} repeats a target")
            seen_targets.add(target)
            if group == "primary":
                primary_targets.add(target)
            pair_id = row.get("pair_id")
            if (
                type(pair_id) is not str
                or SHA256.fullmatch(pair_id) is None
                or pair_id in all_ids
            ):
                fail("prediction pair IDs are malformed or duplicated")
            all_ids.add(pair_id)
            activation = number(row.get("source_activation"), "source activation")
            z = number(row.get("target_preactivation"), "target preactivation")
            tau = number(row.get("target_threshold"), "target threshold")
            margin = number(row.get("margin"), "margin")
            response = number(row.get("targeted_response"), "targeted response")
            q = -activation * response
            if activation <= 0.0 or z > tau or margin < 0.0 or margin != tau - z:
                fail("prediction baseline source/target state is invalid")
            if number(row.get("q"), "q") != q:
                fail("prediction q differs from recomputation")
            epsilon = number(protocol["scoring"]["epsilon"], "epsilon")
            susceptibility = q / (margin + epsilon)
            if number(row.get("susceptibility"), "susceptibility") != susceptibility:
                fail("prediction susceptibility differs")
            alpha = row.get("predicted_alpha_star")
            if q > 0.0:
                if alpha is None or number(alpha, "predicted alpha") != margin / q:
                    fail("prediction alpha differs")
            elif alpha is not None:
                fail("non-positive q has a predicted alpha")
            tolerance = number(
                protocol["scoring"]["crossing_tolerance"], "crossing tolerance"
            )
            if abs(margin) <= tolerance:
                expected_status = "boundary_ambiguous"
            elif q <= 0.0:
                expected_status = "not_crossing"
            elif q - margin > tolerance:
                expected_status = "definitely_crossing"
            elif abs(q - margin) <= tolerance:
                expected_status = "boundary_ambiguous"
            else:
                expected_status = "not_crossing"
            if row.get("predicted_status") != expected_status:
                fail("prediction status differs")
            if pair_id != canonical_pair_id(source=source, target=target):
                fail("prediction pair ID differs from v3 recomputation")
            if group == "primary" and not (
                expected_status == "definitely_crossing"
                and alpha is not None
                and 0.0 < float(alpha) < 1.0
            ):
                fail("primary row is not eligible")
            if group == "near_boundary" and not (
                q > 0.0
                and expected_status == "not_crossing"
                and alpha is not None
                and float(alpha) > 1.0
                and margin - q > tolerance
            ):
                fail("near-boundary row is not eligible")
            if group == "directional" and not (
                q <= 0.0 and expected_status == "not_crossing" and margin > tolerance
            ):
                fail("directional row is not eligible")
            requested = row.get("requested_alphas")
            if type(requested) is not list:
                fail("prediction requested schedule is missing")
            requested_values = tuple(
                number(item, "requested alpha") for item in requested
            )
            if requested_values != tuple(
                sorted(set(requested_values))
            ) or requested_values != _expected_schedule(row, protocol):
                fail("prediction requested schedule differs")
            expected_rows.append(row)
    if (
        len(group_rows["primary"]) > 12
        or len(group_rows["near_boundary"]) > 8
        or len(group_rows["directional"]) > 8
    ):
        fail("prediction group exceeds frozen cap")
    if not group_rows["primary"] and (
        group_rows["near_boundary"] or group_rows["directional"]
    ):
        fail("controls cannot be selected without a primary pair")
    if any(
        sum(1 for row in group_rows["primary"] if feature_key(row["source"]) == source)
        > 2
        for source in {feature_key(row["source"]) for row in group_rows["primary"]}
    ):
        fail("prediction primary source cap is violated")
    used_targets = primary_targets | {
        feature_key(row["target"]) for row in group_rows["near_boundary"]
    }
    for group, rows in group_rows.items():
        keys = [
            _pair_order_key(row, group, primary_targets, used_targets) for row in rows
        ]
        if keys != sorted(keys):
            fail(f"prediction {group} order differs")
    category_counts = {name: 0 for name in OVERLAP_CATEGORIES}
    for row in expected_rows:
        category_counts[str(row["endpoint_overlap_category"])] += 1
    expected_audit = {
        "primary_count": len(group_rows["primary"]),
        "near_boundary_count": len(group_rows["near_boundary"]),
        "directional_count": len(group_rows["directional"]),
        "near_overlap_fallback_count": sum(
            feature_key(row["target"]) in primary_targets
            for row in group_rows["near_boundary"]
        ),
        "directional_overlap_fallback_count": sum(
            feature_key(row["target"]) in used_targets
            for row in group_rows["directional"]
        ),
        "groups_disjoint": True,
        "primary_target_unique": True,
        "primary_source_cap": 2,
        "exact_pair_mask_applied_before_ranking": True,
        "selected_exact_pair_overlap_count": 0,
        "endpoint_overlap_category_counts": category_counts,
        "endpoint_overlap_used_for_ranking_or_quota": False,
    }
    if value.get("selection_audit") != expected_audit:
        fail("prediction selection audit differs")
    if value.get("prediction_only_guards") != {
        "source_suppression_api_calls": 0,
        "prior_inactive_target_outcome_read": False,
        "historical_intervention_outcome_read": False,
        "v2_temporary_baseline_artifact_read": False,
        "intervention_worker_imported": False,
        "raw_graph_read": False,
        "raw_adjacency_read": False,
    }:
        fail("prediction-only guard record is invalid")
    if value.get("claim_boundary") != {
        "behavioral_importance_result": "none",
        "mediation_result": "none",
        "official_bf16_reproduction": "pending",
        "reference_clt_reproduction": "pending",
        "paper_results_readiness": False,
    }:
        fail("prediction claim boundary differs")
    hashes = value.get("protocol_file_sha256")
    if type(hashes) is not dict or set(hashes) != set(PROTOCOL_PATHS):
        fail("prediction protocol hash map differs from the exact allowlist")
    for name in PROTOCOL_PATHS:
        protocol_digest = hashes.get(name)
        if (
            type(protocol_digest) is not str
            or SHA256.fullmatch(protocol_digest) is None
        ):
            fail("prediction protocol hash is malformed")
        raw = _open_regular(ROOT / name, maximum=MAX_FILE, scan_content=False)
        if hashlib.sha256(raw).hexdigest() != protocol_digest:
            fail(f"prediction protocol hash differs: {name}")
    if (
        value["config_sha256"]
        != hashes["configs/stage1c_v3_preregistered_prospective_prediction.yaml"]
        or value["artifact_schema_sha256"]
        != hashes[
            "configs/stage1c_v3_preregistered_prospective_prediction_artifact_schema.json"
        ]
    ):
        fail("prediction config/schema digests differ from protocol hashes")
    return expected_rows


def bf16_round(value: float) -> float:
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    lower = bits & 0xFFFF
    upper = bits >> 16
    if lower > 0x8000 or (lower == 0x8000 and upper & 1):
        upper += 1
    return float(struct.unpack(">f", struct.pack(">I", upper << 16))[0])


def _first_bracket(
    points: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    for left, right in pairwise(points):
        if left["target_active"] is False and right["target_active"] is True:
            return left, right
    return None


def _validate_point(
    point: dict[str, Any], pair: dict[str, Any], *, is_bisection: bool
) -> None:
    required = {
        "baseline_source_activation",
        "baseline_target_preactivation",
        "endpoint_overlap_category",
        "exact_pair_key",
        "finite_checks",
        "finite_value_checks_passed",
        "logits_finite",
        "logits_shape",
        "memory_stage_identity",
        "point_elapsed_seconds",
        "predicted_alpha_star",
        "predicted_status",
        "q",
        "source_suppression_api_call_index",
        "target_value_device",
        "target_value_dtype",
        "targeted_response",
    }
    if (
        any(
            key not in point
            for key in ("pair_id", "group", "source", "target", *required)
        )
        or point["pair_id"] != pair["pair_id"]
        or point["group"] != pair["group"]
        or point["source"] != pair["source"]
        or point["target"] != pair["target"]
        or point["exact_pair_key"] != pair["exact_pair_key"]
        or point["endpoint_overlap_category"] != pair["endpoint_overlap_category"]
        or point["predicted_alpha_star"] != pair.get("predicted_alpha_star")
        or point["predicted_status"] != pair["predicted_status"]
    ):
        fail("point pair identity differs from frozen pair")
    feature_key(point["source"], "point.source")
    feature_key(point["target"], "point.target")
    baseline = number(pair["source_activation"], "baseline source activation")
    baseline_z = number(pair["target_preactivation"], "baseline target preactivation")
    response = number(pair["targeted_response"], "targeted response")
    if (
        number(point["baseline_source_activation"], "point baseline source") != baseline
        or number(point["baseline_target_preactivation"], "point baseline target")
        != baseline_z
        or number(point["targeted_response"], "point targeted response") != response
        or number(point["q"], "point q") != -baseline * response
        or number(point["q"], "point q") != number(pair["q"], "frozen q")
    ):
        fail("point baseline or targeted-response evidence differs")
    requested = number(point.get("requested_alpha"), "requested alpha")
    desired = number(point.get("desired_high_precision"), "desired activation")
    applied = number(point.get("actual_bf16_value_passed"), "actual BF16 source value")
    realized = number(point.get("realized_suppression"), "realized suppression")
    if (
        not 0.0 <= requested <= 1.0
        or desired != (1.0 - requested) * baseline
        or applied != bf16_round(desired)
        or realized != 1.0 - applied / baseline
        or not 0.0 <= realized <= 1.0
    ):
        fail("point source suppression mapping differs")
    if (
        point.get("source_value_device") != "mps:0"
        or point.get("source_value_dtype") != "torch.bfloat16"
        or point.get("target_value_device") != "mps:0"
        or point.get("target_value_dtype") != "torch.bfloat16"
    ):
        fail("point value runtime identity differs")
    call_index = point.get("source_suppression_api_call_index")
    elapsed = number(point.get("point_elapsed_seconds"), "point elapsed seconds")
    stage_identity = point.get("memory_stage_identity")
    finite_checks = point.get("finite_checks")
    if (
        type(call_index) is not int
        or call_index < 1
        or elapsed <= 0.0
        or type(stage_identity) is not str
        or not stage_identity
        or point.get("finite_value_checks_passed") is not True
        or type(finite_checks) is not dict
        or set(finite_checks)
        != {
            "applied_source",
            "logits",
            "preactivation_cache",
            "target_activation",
            "target_preactivation",
            "target_threshold",
        }
        or any(value is not True for value in finite_checks.values())
        or point.get("logits_finite") is not True
        or type(point.get("logits_shape")) is not list
        or not point["logits_shape"]
        or any(type(item) is not int or item < 1 for item in point["logits_shape"])
        or point.get("source_activation_observation")
        != "actual_bf16_value_passed_to_absolute_intervention_tuple"
    ):
        fail("point runtime, finite, timing, or memory evidence differs")
    requests = point.get("requested_mappings")
    if (
        type(requests) is not list
        or not requests
        or (is_bisection and len(requests) != 1)
    ):
        fail("point requested mappings are missing or invalid")
    request_alphas: list[float] = []
    for request in requests:
        if type(request) is not dict:
            fail("point request mapping is malformed")
        alpha = number(request.get("requested_alpha"), "mapped requested alpha")
        request_desired = number(
            request.get("desired_high_precision"), "mapped desired activation"
        )
        request_applied = number(
            request.get("actual_bf16_value_passed"), "mapped BF16 value"
        )
        request_realized = number(
            request.get("realized_suppression"), "mapped realized suppression"
        )
        if (
            not 0.0 <= alpha <= 1.0
            or request_desired != (1.0 - alpha) * baseline
            or request_applied != bf16_round(request_desired)
            or request_realized != 1.0 - request_applied / baseline
            or request_applied != applied
            or request_realized != realized
        ):
            fail("point requested mapping differs")
        request_alphas.append(alpha)
    if (
        request_alphas != sorted(set(request_alphas))
        or number(
            point.get("representative_requested_alpha"),
            "representative requested alpha",
        )
        != request_alphas[0]
        or requested != request_alphas[0]
        or point.get("collapsed_request_count") != len(requests)
    ):
        fail("point representative/collapsed request metadata differs")
    z = number(point.get("target_preactivation"), "target preactivation")
    tau = number(point.get("target_threshold"), "target threshold")
    if tau != number(pair["target_threshold"], "frozen target threshold"):
        fail("point target threshold differs")
    active = point.get("target_active")
    if type(active) is not bool or active != (z > tau):
        fail("point strict gate differs")
    if (
        number(point.get("target_activation"), "target activation")
        != (z if active else 0.0)
        or point.get("loaded_gate") != "a=z*1[z>tau]"
        or point.get("threshold_equality_activity") != "inactive"
        or point.get("target_clamped") is not False
        or point.get("freeze_attention") is not True
        or point.get("constrained_layers") is not None
    ):
        fail("point loaded gate/intervention regime differs")
    predicted_z = number(
        point.get("predicted_target_preactivation"), "predicted target preactivation"
    )
    expected_predicted_z = baseline_z + realized * number(pair["q"], "q")
    if predicted_z != expected_predicted_z:
        fail("point local prediction differs")
    predicted_active = point.get("predicted_target_active")
    if (
        type(predicted_active) is not bool
        or predicted_active != (predicted_z > tau)
        or number(
            point.get("predicted_target_activation"), "predicted target activation"
        )
        != (predicted_z if predicted_active else 0.0)
    ):
        fail("point predicted gate differs")
    error = number(
        point.get("target_preactivation_absolute_error"), "absolute prediction error"
    )
    denominator = abs(z) + abs(predicted_z)
    expected_sne = (
        0.0 if denominator == 0.0 else 2.0 * abs(z - predicted_z) / denominator
    )
    if (
        error != abs(z - predicted_z)
        or number(
            point.get("target_preactivation_symmetric_normalized_error"),
            "symmetric prediction error",
        )
        != expected_sne
    ):
        fail("point prediction error differs")
    if (
        type(point.get("stage")) is not str
        or not point["stage"]
        or (is_bisection and type(point.get("bisection_step")) is not int)
        or (not is_bisection and point.get("bisection_step") is not None)
    ):
        fail("point stage/bisection metadata is malformed")


def _analysis(
    pair: dict[str, Any], points: list[dict[str, Any]], analysis: dict[str, Any]
) -> dict[str, Any]:
    if (
        not points
        or points[0]["realized_suppression"] != 0.0
        or points[0]["target_active"] is not False
        or points[-1]["realized_suppression"] != 1.0
    ):
        fail("sweep lacks exact inactive zero and full suppression endpoints")
    baseline_z = number(pair["target_preactivation"], "baseline target preactivation")
    q = number(pair["q"], "q")
    errors: list[float] = []
    signs: list[bool] = []
    for point in points[1:]:
        delta = (
            number(point["target_preactivation"], "target preactivation") - baseline_z
        )
        expected = number(point["realized_suppression"], "realized suppression") * q
        denominator = abs(expected) + abs(delta)
        errors.append(
            0.0 if denominator == 0.0 else 2.0 * abs(expected - delta) / denominator
        )
        signs.append(
            delta > 0.0
            if expected > 0.0
            else delta < 0.0
            if expected < 0.0
            else delta == 0.0
        )
    first = _first_bracket(points)
    bracket = (
        None
        if first is None
        else {
            "lower_realized_suppression": first[0]["realized_suppression"],
            "upper_realized_suppression": first[1]["realized_suppression"],
        }
    )
    alpha = pair.get("predicted_alpha_star")
    distance = None
    if alpha is not None and bracket is not None:
        alpha_value = number(alpha, "predicted alpha")
        distance = (
            0.0
            if bracket["lower_realized_suppression"]
            <= alpha_value
            <= bracket["upper_realized_suppression"]
            else min(
                abs(alpha_value - bracket["lower_realized_suppression"]),
                abs(alpha_value - bracket["upper_realized_suppression"]),
            )
        )
    sign_agreement = sum(signs) / len(signs) if signs else None
    median_error = median(errors) if errors else None
    p95_error = (
        sorted(errors)[max(1, math.ceil(0.95 * len(errors))) - 1] if errors else None
    )
    local_pass = (
        len(errors) >= int(analysis["minimum_nonzero_points"])
        and sign_agreement is not None
        and sign_agreement >= float(analysis["movement_sign_agreement_minimum"])
        and median_error is not None
        and median_error <= float(analysis["median_movement_sne_maximum"])
        and p95_error is not None
        and p95_error <= float(analysis["p95_movement_sne_maximum"])
        and distance is not None
        and distance <= float(analysis["critical_bracket_distance_maximum"])
    )
    active_seen = False
    nonmonotonic = False
    for point in points:
        if point["target_active"]:
            active_seen = True
        elif active_seen:
            nonmonotonic = True
    full_delta = (
        number(points[-1]["target_preactivation"], "full-ablation target preactivation")
        - baseline_z
    )
    group = pair["group"]
    return {
        "pair_id": pair["pair_id"],
        "group": group,
        "exact_pair_key": pair["exact_pair_key"],
        "endpoint_overlap_category": pair["endpoint_overlap_category"],
        "point_count": len(points),
        "nonzero_point_count": len(errors),
        "predicted_full_ablation_crossing": pair["predicted_status"]
        == "definitely_crossing",
        "observed_full_ablation_crossing": bool(points[-1]["target_active"]),
        "predicted_alpha_star": alpha,
        "observed_critical_bracket": bracket,
        "critical_bracket_distance": distance,
        "movement_sign_agreement": sign_agreement,
        "median_movement_symmetric_normalized_error": median_error,
        "p95_movement_symmetric_normalized_error": p95_error,
        "full_ablation_observed_movement": full_delta,
        "local_calibration_passed": local_pass,
        "supporting_primary": group == "primary"
        and bool(points[-1]["target_active"])
        and full_delta > 0.0
        and local_pass,
        "directional_control_violation": group == "directional" and full_delta > 0.0,
        "near_boundary_control_crossing": group == "near_boundary"
        and any(point["target_active"] for point in points),
        "nonmonotonic_gate": nonmonotonic,
    }


def scientific_outcome(rows: list[dict[str, Any]]) -> str:
    primary = [row for row in rows if row["group"] == "primary"]
    if not primary:
        return "no_eligible_pairs"
    supporting = sum(bool(row["supporting_primary"]) for row in primary)
    if supporting == 0:
        return "not_supported"
    discrepancy = supporting != len(primary) or any(
        row["directional_control_violation"]
        or row["near_boundary_control_crossing"]
        or row["nonmonotonic_gate"]
        for row in rows
    )
    return "mixed" if discrepancy else "supported"


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [row for row in rows if row["group"] == "primary"]
    near = [row for row in rows if row["group"] == "near_boundary"]
    directional = [row for row in rows if row["group"] == "directional"]
    brackets = [
        float(row["critical_bracket_distance"])
        for row in primary
        if row["critical_bracket_distance"] is not None
    ]
    medians = [
        float(row["median_movement_symmetric_normalized_error"])
        for row in primary
        if row["median_movement_symmetric_normalized_error"] is not None
    ]
    p95s = [
        float(row["p95_movement_symmetric_normalized_error"])
        for row in primary
        if row["p95_movement_symmetric_normalized_error"] is not None
    ]
    predicted = [
        float(row["predicted_alpha_star"])
        for row in primary
        if row["predicted_alpha_star"] is not None
        and row["observed_critical_bracket"] is not None
    ]
    observed = [
        float(
            (
                row["observed_critical_bracket"]["lower_realized_suppression"]
                + row["observed_critical_bracket"]["upper_realized_suppression"]
            )
            / 2.0
        )
        for row in primary
        if row["predicted_alpha_star"] is not None
        and row["observed_critical_bracket"] is not None
    ]
    correlation = None
    if len(predicted) >= 2 and len(set(predicted)) > 1 and len(set(observed)) > 1:

        def ranks(values: list[float]) -> list[float]:
            order = sorted(range(len(values)), key=lambda index: (values[index], index))
            result = [0.0] * len(values)
            start = 0
            while start < len(order):
                end = start + 1
                while end < len(order) and values[order[end]] == values[order[start]]:
                    end += 1
                rank = (start + 1 + end) / 2.0
                for index in range(start, end):
                    result[order[index]] = rank
                start = end
            return result

        left, right = ranks(predicted), ranks(observed)
        lm, rm = sum(left) / len(left), sum(right) / len(right)
        denominator = math.sqrt(
            sum((x - lm) ** 2 for x in left) * sum((y - rm) ** 2 for y in right)
        )
        correlation = (
            None
            if denominator == 0.0
            else sum((x - lm) * (y - rm) for x, y in zip(left, right, strict=True))
            / denominator
        )

    def nearest(values: list[float]) -> float | None:
        return (
            sorted(values)[max(1, math.ceil(0.95 * len(values))) - 1]
            if values
            else None
        )

    primary_crossings = sum(
        bool(row["observed_full_ablation_crossing"]) for row in primary
    )
    near_crossings = sum(bool(row["near_boundary_control_crossing"]) for row in near)
    directional_violations = sum(
        bool(row["directional_control_violation"]) for row in directional
    )
    return {
        "primary_pair_count": len(primary),
        "primary_full_ablation_crossing_count": primary_crossings,
        "primary_full_ablation_crossing_precision": primary_crossings / len(primary)
        if primary
        else None,
        "primary_precision_undefined_reason": None if primary else "no_primary_pairs",
        "supporting_primary_count": sum(
            bool(row["supporting_primary"]) for row in primary
        ),
        "near_boundary_pair_count": len(near),
        "near_boundary_crossing_count": near_crossings,
        "near_boundary_crossing_fraction": near_crossings / len(near) if near else None,
        "near_boundary_fraction_undefined_reason": None
        if near
        else "no_near_boundary_controls",
        "directional_pair_count": len(directional),
        "directional_violation_count": directional_violations,
        "directional_violation_fraction": directional_violations / len(directional)
        if directional
        else None,
        "directional_fraction_undefined_reason": None
        if directional
        else "no_directional_controls",
        "critical_suppression_spearman": correlation,
        "critical_suppression_spearman_pair_count": len(predicted),
        "critical_suppression_spearman_undefined_reason": None
        if correlation is not None
        else "fewer_than_two_nonconstant_observed_crossings",
        "primary_bracket_distance_count": len(brackets),
        "primary_bracket_distance_median": median(brackets) if brackets else None,
        "primary_bracket_distance_p95": nearest(brackets),
        "primary_bracket_distance_undefined_reason": None
        if brackets
        else "no_observed_primary_crossing_brackets",
        "primary_pair_median_movement_sne_median": median(medians) if medians else None,
        "primary_pair_median_movement_sne_p95": nearest(medians),
        "primary_pair_movement_error_undefined_reason": None
        if medians
        else "no_primary_movement_errors",
        "primary_pair_p95_movement_sne_median": median(p95s) if p95s else None,
    }


def validate_sweeps(
    prediction: dict[str, Any], sweeps: dict[str, Any], crossing: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = sweeps.get("pairs")
    if type(rows) is not list:
        fail("intervention sweeps.pairs is missing")
    selected = scan_prediction(prediction)
    selected_by_id = {row["pair_id"]: row for row in selected}
    primary_count = len(prediction["selected_groups"]["primary"])
    expected_ids = [row["pair_id"] for row in selected] if primary_count else []
    if [item.get("pair_id") for item in rows] != expected_ids:
        fail("sweep pair IDs or order differ from prediction")
    analyses: list[dict[str, Any]] = []
    api_call_indices: list[int] = []
    memory_stage_identities: list[str] = []
    for item in rows:
        if type(item) is not dict or item.get("pair_id") not in selected_by_id:
            fail("sweep row is malformed")
        pair = selected_by_id[item["pair_id"]]
        fields = {
            "baseline_source_activation",
            "bisection_step_count",
            "endpoint_overlap_category",
            "exact_pair_key",
            "group",
            "pair_id",
            "point_count",
            "points",
            "predicted_alpha_star",
            "predicted_status",
            "q",
            "source",
            "source_activation",
            "target",
            "target_preactivation",
            "target_threshold",
            "targeted_response",
        }
        if set(item) != fields:
            fail("sweep row has missing or unknown fields")
        if (
            item["group"] != pair["group"]
            or item["source"] != pair["source"]
            or item["target"] != pair["target"]
            or item["exact_pair_key"] != pair["exact_pair_key"]
            or item["endpoint_overlap_category"] != pair["endpoint_overlap_category"]
            or number(item["source_activation"], "sweep source activation")
            != number(pair["source_activation"], "pair source activation")
            or number(item["targeted_response"], "sweep targeted response")
            != number(pair["targeted_response"], "pair targeted response")
            or number(item["target_preactivation"], "sweep baseline z")
            != number(pair["target_preactivation"], "pair baseline z")
            or number(item["target_threshold"], "sweep baseline threshold")
            != number(pair["target_threshold"], "pair threshold")
            or number(item["q"], "sweep q") != number(pair["q"], "pair q")
            or item["predicted_alpha_star"] != pair.get("predicted_alpha_star")
            or item["predicted_status"] != pair["predicted_status"]
            or number(item["baseline_source_activation"], "sweep baseline source")
            != number(pair["source_activation"], "pair source activation")
        ):
            fail("sweep wrapper differs from frozen pair")
        points = item["points"]
        if (
            type(points) is not list
            or not points
            or type(item["point_count"]) is not int
            or item["point_count"] != len(points)
            or type(item["bisection_step_count"]) is not int
            or item["bisection_step_count"] < 0
        ):
            fail("sweep point counts are invalid")
        realized_values: list[float] = []
        bisection_points: list[dict[str, Any]] = []
        frozen_schedule = set(_expected_schedule(pair, prediction["protocol"]))
        seen_frozen: set[float] = set()
        for point in points:
            if type(point) is not dict:
                fail("sweep point is not an object")
            is_bisection = point.get("bisection_step") is not None
            _validate_point(point, pair, is_bisection=is_bisection)
            api_call_indices.append(
                _integer(
                    point["source_suppression_api_call_index"],
                    "source-suppression API-call index",
                    minimum=1,
                )
            )
            memory_stage_identities.append(str(point["memory_stage_identity"]))
            realized = number(point["realized_suppression"], "realized suppression")
            if realized_values and realized <= realized_values[-1]:
                fail("serialized points are not strictly realized-suppression ordered")
            realized_values.append(realized)
            request_alphas = {
                number(request["requested_alpha"], "requested alpha")
                for request in point["requested_mappings"]
            }
            if is_bisection:
                bisection_points.append(point)
            else:
                if not request_alphas.issubset(frozen_schedule):
                    fail("non-bisection point contains an unfrozen request")
                seen_frozen.update(request_alphas)
        if (
            seen_frozen != frozen_schedule
            or len(bisection_points) != item["bisection_step_count"]
        ):
            fail("point schedule coverage or bisection count differs")
        if realized_values[0] != 0.0 or realized_values[-1] != 1.0:
            fail("sweep lacks exact zero/full realized suppression")
        zero, full = points[0], points[-1]
        if (
            number(zero["target_preactivation"], "zero-suppression target")
            != number(pair["target_preactivation"], "frozen target preactivation")
            or number(zero["actual_bf16_value_passed"], "zero source value")
            != number(pair["source_activation"], "frozen source activation")
            or 0.0
            not in {
                number(request["requested_alpha"], "zero requested alpha")
                for request in zero["requested_mappings"]
            }
            or number(full["actual_bf16_value_passed"], "full source value") != 0.0
            or 1.0
            not in {
                number(request["requested_alpha"], "full requested alpha")
                for request in full["requested_mappings"]
            }
        ):
            fail("zero/full suppression point differs from frozen endpoints")
        refined = [point for point in points if point.get("bisection_step") is None]
        for step, point in enumerate(
            sorted(bisection_points, key=lambda value: value["bisection_step"])
        ):
            if (
                type(point["bisection_step"]) is not int
                or point["bisection_step"] < 0
                or point["bisection_step"] != step
            ):
                fail("bisection steps are not contiguous")
            bracket = _first_bracket(refined)
            if bracket is None:
                fail("bisection point has no inactive-active bracket")
            midpoint = (
                bracket[0]["realized_suppression"] + bracket[1]["realized_suppression"]
            ) / 2.0
            if (
                point["requested_alpha"] != midpoint
                or not bracket[0]["realized_suppression"]
                < point["realized_suppression"]
                < bracket[1]["realized_suppression"]
            ):
                fail("bisection midpoint is not deterministic")
            refined.append(point)
        if len(bisection_points) > int(
            prediction["protocol"]["schedule"]["maximum_bisection_steps"]
        ):
            fail("bisection exceeds frozen cap")
        analyses.append(_analysis(pair, points, prediction["protocol"]["analysis"]))
    analysis_rows = crossing.get("pairs")
    if (
        type(analysis_rows) is not list
        or [row.get("pair_id") for row in analysis_rows] != expected_ids
    ):
        fail("crossing analysis pair IDs or order differ")
    for observed, expected in zip(analysis_rows, analyses, strict=True):
        if observed != expected:
            fail("crossing analysis differs from independent recomputation")
    if crossing.get("aggregate_metrics") != _aggregate(analyses):
        fail("crossing aggregate differs from independent recomputation")
    if crossing.get("scientific_outcome") != scientific_outcome(analyses):
        fail("scientific outcome differs from independent recomputation")
    point_count = sum(len(item["points"]) for item in rows)
    if sorted(api_call_indices) != list(range(1, point_count + 1)):
        fail("point-level suppression API-call indices are not exact and contiguous")
    if len(memory_stage_identities) != len(set(memory_stage_identities)):
        fail("point-level memory stage identities are not unique")
    return analyses


def load_bundle(directory: Path) -> dict[str, dict[str, Any]]:
    directory = _assert_no_symlink_ancestors(directory)
    try:
        info = directory.lstat()
    except OSError as error:
        raise ValidationError("bundle directory is unreadable") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        fail("bundle must be a real directory")
    if {item.name for item in directory.iterdir()} != set(ALLOWLIST):
        fail("bundle differs from exact allowlist")
    payload: dict[str, bytes] = {}
    total = 0
    for name in ALLOWLIST:
        raw = _open_regular(
            directory / name,
            maximum=64 * 1024 if name == "checksums.sha256" else MAX_FILE,
        )
        payload[name] = raw
        total += len(raw)
    if total > MAX_BUNDLE:
        fail("bundle exceeds total size cap")
    try:
        checksum_text = payload["checksums.sha256"].decode("utf-8")
    except UnicodeError as error:
        raise ValidationError("checksum manifest is not UTF-8") from error
    expected_lines = [
        f"{hashlib.sha256(payload[name]).hexdigest()}  {name}\n"
        for name in sorted(JSON_NAMES)
    ]
    if checksum_text != "".join(expected_lines):
        fail("checksum manifest is not exact sorted canonical coverage")
    return {name: strict_json(payload[name], name) for name in JSON_NAMES}


def _validate_file_identities(artifacts: dict[str, dict[str, Any]]) -> None:
    for name, artifact_type in ARTIFACT_TYPES.items():
        _require_identity(artifacts[name], name, artifact_type)


def _validate_telemetry(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "attempt_peaks",
        "finished_at_unix",
        "sample_count",
        "sampling_interval_seconds",
        "stage_peaks",
        "started_at_unix",
        "telemetry_failures",
        "thermal_states",
        "violations",
    }:
        fail(f"{label} telemetry schema differs")
    started = number(value["started_at_unix"], f"{label} telemetry start")
    finished = number(value["finished_at_unix"], f"{label} telemetry finish")
    if (
        finished < started
        or _integer(value["sample_count"], f"{label} sample count", minimum=1) < 1
        or number(value["sampling_interval_seconds"], f"{label} sample interval") != 1.0
        or value["telemetry_failures"] != 0
        or value["violations"] != []
        or type(value["thermal_states"]) is not list
        or not value["thermal_states"]
        or value["thermal_states"] != sorted(set(value["thermal_states"]))
        or any(item not in {"nominal", "fair"} for item in value["thermal_states"])
    ):
        fail(f"{label} telemetry safety record differs")

    peak_keys = {
        "available_memory_bytes",
        "minimum_available_memory_bytes",
        "mps_current_bytes",
        "mps_driver_bytes",
        "process_rss_bytes",
        "swap_growth_bytes",
        "swap_used_bytes",
    }

    def validate_peaks(peaks: Any, peak_label: str) -> None:
        # Older sampler records never contain ``available_memory_bytes`` at the
        # aggregate boundary, so require the exact six persisted peak fields.
        expected = peak_keys - {"available_memory_bytes"}
        if type(peaks) is not dict or set(peaks) != expected:
            fail(f"{peak_label} peak schema differs")
        numeric = {
            key: _integer(peaks[key], f"{peak_label}.{key}", minimum=0)
            for key in expected
        }
        if (
            numeric["mps_driver_bytes"] > 25_769_803_776
            or numeric["process_rss_bytes"] > 25_769_803_776
            or numeric["swap_growth_bytes"] > 4_294_967_296
            or numeric["minimum_available_memory_bytes"] < 4_294_967_296
        ):
            fail(f"{peak_label} exceeds the frozen memory limits")

    validate_peaks(value["attempt_peaks"], f"{label}.attempt")
    stages = value["stage_peaks"]
    if type(stages) is not dict or not stages:
        fail(f"{label} stage telemetry is empty")
    for stage, peaks in stages.items():
        if type(stage) is not str or not stage:
            fail(f"{label} telemetry stage name is invalid")
        validate_peaks(peaks, f"{label}.stage.{stage}")
    return value


def _validate_supervisor(value: Any, label: str) -> None:
    if type(value) is not dict or set(value) != {
        "finished_at_unix",
        "minimum_available_memory_bytes",
        "peak_process_group_rss_bytes",
        "peak_swap_growth_bytes",
        "returncode",
        "safety_terminated",
        "sample_count",
        "started_at_unix",
        "telemetry_failures",
        "termination_signal",
        "thermal_states",
        "timed_out",
    }:
        fail(f"{label} supervisor schema differs")
    started = number(value["started_at_unix"], f"{label} supervisor start")
    finished = number(value["finished_at_unix"], f"{label} supervisor finish")
    if (
        value["returncode"] != 0
        or value["timed_out"] is not False
        or value["safety_terminated"] is not False
        or value["termination_signal"] is not None
        or value["telemetry_failures"] != 0
        or _integer(value["sample_count"], f"{label} sample count", minimum=1) < 1
        or _integer(
            value["peak_process_group_rss_bytes"], f"{label} peak RSS", minimum=0
        )
        > 25_769_803_776
        or _integer(
            value["minimum_available_memory_bytes"],
            f"{label} available memory",
            minimum=0,
        )
        < 4_294_967_296
        or _integer(value["peak_swap_growth_bytes"], f"{label} swap growth", minimum=0)
        > 4_294_967_296
        or type(value["thermal_states"]) is not list
        or not value["thermal_states"]
        or any(item not in {"nominal", "fair"} for item in value["thermal_states"])
        or finished < started
    ):
        fail(f"{label} supervisor safety record differs")


def _validate_artifacts(
    artifacts: dict[str, dict[str, Any]], execution_commit: str | None = None
) -> dict[str, Any]:
    if execution_commit is not None and SHA40.fullmatch(execution_commit) is None:
        fail("execution commit is not a SHA-1")
    _validate_file_identities(artifacts)
    prediction = artifacts["prediction_manifest.json"]
    scan_prediction(prediction)
    run = artifacts["run_manifest.json"]
    if execution_commit is not None and run.get("execution_commit") != execution_commit:
        fail("run execution commit differs")
    if (
        run.get("final_bundle_type") != FINAL_BUNDLE_TYPE
        or run.get("branch") != BRANCH
        or run.get("base_commit") != BASE_COMMIT
    ):
        fail("run final-bundle identity differs")
    if run.get("pre_intervention_commit") != run.get("execution_commit"):
        fail("pre-intervention and execution commits differ")
    if (
        run.get("prediction_manifest_sha256")
        != hashlib.sha256(_canonical_json_bytes(prediction)).hexdigest()
    ):
        fail("prediction-manifest digest differs")
    analyses = validate_sweeps(
        prediction,
        artifacts["intervention_sweeps.json"],
        artifacts["crossing_summary.json"],
    )
    sweep_rows = artifacts["intervention_sweeps.json"]["pairs"]
    point_count = sum(len(item["points"]) for item in sweep_rows)
    local = artifacts["local_linearity_summary.json"]
    if (
        local.get("pairs") != analyses
        or local.get("aggregate_metrics") != _aggregate(analyses)
        or local.get("scientific_outcome") != scientific_outcome(analyses)
    ):
        fail("local-linearity pair analyses differ from point recomputation")
    errors = [
        number(point["target_preactivation_symmetric_normalized_error"], "point error")
        for item in sweep_rows
        for point in item["points"]
    ]
    expected_local = {
        "point_count": point_count,
        "median_symmetric_normalized_error": median(errors) if errors else None,
        "p95_symmetric_normalized_error": sorted(errors)[
            max(1, math.ceil(0.95 * len(errors))) - 1
        ]
        if errors
        else None,
        "undefined_metric_reason": None if errors else "no_intervention_points",
    }
    if any(local.get(key) != value for key, value in expected_local.items()):
        fail("local-linearity summary differs")
    if (
        type(run.get("execution_commit")) is not str
        or SHA40.fullmatch(run["execution_commit"]) is None
    ):
        fail("run execution commit is not a SHA-1")
    calls = _integer(
        run.get("canonical_source_suppression_api_calls"),
        "run API-call count",
        minimum=0,
    )
    attempts = artifacts["attempts.json"]
    instrumented_calls = _integer(
        run.get("instrumented_source_suppression_api_calls"),
        "run instrumented API-call count",
        minimum=0,
    )
    if (
        attempts.get("canonical_source_suppression_api_calls") != calls
        or attempts.get("instrumented_source_suppression_api_calls")
        != instrumented_calls
        or attempts.get("prediction_attempts") != 1
        or attempts.get("scientific_retry_count") != 0
    ):
        fail("attempt metadata differs")
    primary_count = len(prediction["selected_groups"]["primary"])
    expected_intervention_attempts = 0 if primary_count == 0 else 1
    if attempts.get(
        "intervention_attempts"
    ) != expected_intervention_attempts or attempts.get(
        "intervention_required"
    ) is not (primary_count > 0):
        fail("attempt count violates primary/no-primary policy")
    if calls != point_count or instrumented_calls != point_count:
        fail("API-call count does not equal serialized point count")
    if primary_count == 0 and (
        sweep_rows
        or calls != 0
        or artifacts["crossing_summary.json"]["scientific_outcome"]
        != "no_eligible_pairs"
    ):
        fail("no-eligible-pairs result is not zero-call/zero-sweep")
    if run.get("scientific_outcome") != artifacts["crossing_summary.json"][
        "scientific_outcome"
    ] or run.get("verdict") != run.get("status"):
        fail("run outcome/verdict differs")
    if run.get("status") != "completed_stage1c_v3_preregistered_prospective_prediction":
        fail("final bundle status is not completed v3")
    if run.get("scientific_retry_count") != 0:
        fail("run scientific retry count is nonzero")
    if run.get("claim_boundary") != {
        "behavioral_importance_result": "none",
        "mediation_result": "none",
        "official_bf16_reproduction": "pending",
        "reference_clt_reproduction": "pending",
        "paper_results_readiness": False,
    }:
        fail("claim boundary differs")
    expected_readiness = {
        "stage1b_measurement_primitives": "completed",
        "stage1c_v3_prospective_prediction": "completed",
        "stage1c_v3_scientific_outcome": run["scientific_outcome"],
        **run["claim_boundary"],
    }
    if run.get("readiness") != expected_readiness:
        fail("run readiness boundary differs")
    if run.get("fresh_canonical_run") is not (primary_count > 0) or run.get(
        "intervention_required"
    ) is not (primary_count > 0):
        fail("run canonical execution policy differs")
    asset = artifacts["asset_manifest.json"]
    model = asset.get("model")
    transcoder = asset.get("transcoder")
    if (
        set(asset)
        != {
            "actual_total_bytes",
            "artifact_type",
            "authentication_used",
            "authentication_value_recorded",
            "download_performed",
            "exact_allowlist_hashes_verified",
            "experiment_class",
            "model",
            "network_accessed",
            "schema_version",
            "status",
            "transcoder",
        }
        or asset.get("status") != "verified"
        or asset.get("download_performed") is not False
        or asset.get("network_accessed") is not False
        or asset.get("authentication_used") is not False
        or asset.get("authentication_value_recorded") is not False
        or asset.get("exact_allowlist_hashes_verified") is not True
        or asset.get("actual_total_bytes") != 2_087_816_677
        or type(model) is not dict
        or model.get("identifier") != "google/gemma-3-270m"
        or model.get("revision") != "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
        or model.get("total_bytes") != 575_454_257
        or type(transcoder) is not dict
        or transcoder.get("identifier") != "mwhanna/gemma-scope-2-270m-pt"
        or transcoder.get("revision") != "fada11860ac1d337c1e41e9da308798405b94c8e"
        or transcoder.get("subfolder") != "transcoder_all/width_16k_l0_small"
        or transcoder.get("layer_count") != 18
        or transcoder.get("total_bytes") != 1_512_362_420
    ):
        fail("asset model identity differs")
    environment = artifacts["environment_manifest.json"]
    accelerator = environment.get("accelerator")
    if (
        environment.get("execution_commit") != run.get("execution_commit")
        or environment.get("status") != "passed"
        or environment.get("platform")
        != {"system": "Darwin", "machine": "arm64", "python": "3.11.13"}
        or environment.get("packages")
        != {
            "circuit-tracer": "0.5.2",
            "torch": "2.6.0",
            "nnsight": "0.6.1",
            "transformers": "4.57.3",
        }
        or environment.get("privacy")
        != {
            "network_accessed": False,
            "credential_values_read": False,
            "secret_values_recorded": False,
            "private_paths_recorded": False,
        }
        or type(accelerator) is not dict
        or accelerator.get("device") != "mps:0"
        or accelerator.get("dtype") != "torch.bfloat16"
        or accelerator.get("fallback_variable_present") is not False
        or accelerator.get("outer_autocast_enabled") is not False
        or accelerator.get("scientific_tensor_device") != "mps"
    ):
        fail("environment runtime identity differs")
    memory = artifacts["memory_timing_summary.json"]
    if set(memory) != {
        "artifact_type",
        "experiment_class",
        "intervention",
        "intervention_supervisor",
        "prediction",
        "prediction_supervisor",
        "schema_version",
        "status",
        "worker",
    }:
        fail("memory/timing artifact schema differs")
    prediction_telemetry = _validate_telemetry(memory.get("prediction"), "prediction")
    _validate_supervisor(memory.get("prediction_supervisor"), "prediction")
    _validate_supervisor(memory.get("intervention_supervisor"), "intervention")
    del prediction_telemetry
    if primary_count:
        intervention_telemetry = _validate_telemetry(
            memory.get("intervention"), "intervention"
        )
        if memory.get("worker") != intervention_telemetry:
            fail("worker/intervention telemetry differs")
        stages = intervention_telemetry["stage_peaks"]
        for item in sweep_rows:
            for point in item["points"]:
                if point["memory_stage_identity"] not in stages:
                    fail("point memory-stage identity is absent from telemetry")
    elif memory.get("intervention") is not None or memory.get("worker") is not None:
        fail("no-eligible result contains intervention telemetry")
    return {
        "status": "passed",
        "artifact_count": len(ALLOWLIST),
        "verdict": run["verdict"],
        "point_count": point_count,
        "api_call_count": calls,
    }


def validate_records(
    artifacts: dict[str, dict[str, Any]], execution_commit: str | None = None
) -> dict[str, Any]:
    """Validate detached in-memory records before publishing any file."""

    if set(artifacts) != JSON_NAMES:
        fail("in-memory artifact records differ from the exact JSON allowlist")
    canonical = {
        name: strict_json(_canonical_json_bytes(artifacts[name]), name)
        for name in JSON_NAMES
    }
    return _validate_artifacts(canonical, execution_commit)


def validate_bundle(
    directory: Path, execution_commit: str | None = None
) -> dict[str, Any]:
    """Validate a fully serialized directory bundle and its checksums."""

    return _validate_artifacts(load_bundle(directory), execution_commit)


def validate_zip(path: Path) -> None:
    path = _assert_no_symlink_ancestors(path)
    try:
        info = path.lstat()
    except OSError as error:
        raise ValidationError("ZIP is unreadable") from error
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
    ):
        fail("ZIP must be a single-link regular file")
    if info.st_size > MAX_BUNDLE:
        fail("ZIP exceeds compressed size cap")
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.comment:
                fail("ZIP archive comment is forbidden")
            infos = archive.infolist()
            if len(infos) != len(ALLOWLIST):
                fail("ZIP member count differs from exact allowlist")
            names: list[str] = []
            member_bytes: dict[str, bytes] = {}
            expanded = 0
            for member in infos:
                name = member.filename
                normalized = name.replace("\\", "/")
                pure = PurePosixPath(normalized)
                if (
                    normalized != name
                    or not name
                    or pure.is_absolute()
                    or ".." in pure.parts
                    or "." in pure.parts
                    or name.endswith("/")
                    or name not in ALLOWLIST
                    or name in names
                ):
                    fail("ZIP path is unsafe or differs from exact allowlist")
                if member.extra or member.comment:
                    fail("ZIP extra field is forbidden")
                if member.flag_bits & 0x1:
                    fail("encrypted ZIP member is forbidden")
                mode = (member.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(mode)
                if file_type not in (0, stat.S_IFREG):
                    fail("ZIP contains a link or special file")
                if (
                    member.file_size < 0
                    or member.compress_size < 0
                    or member.file_size > MAX_FILE
                ):
                    fail("ZIP member exceeds file cap")
                expanded += member.file_size
                if expanded > MAX_BUNDLE:
                    fail("ZIP expanded size exceeds bundle cap")
                try:
                    raw = archive.read(member)
                except (
                    OSError,
                    RuntimeError,
                    zipfile.BadZipFile,
                    zipfile.LargeZipFile,
                ) as error:
                    raise ValidationError("ZIP member cannot be read safely") from error
                if len(raw) != member.file_size or b"\0" in raw:
                    fail("ZIP member size or binary content differs")
                member_bytes[name] = raw
                names.append(name)
            if set(names) != set(ALLOWLIST):
                fail("ZIP members differ from exact allowlist")
            checksum_text = member_bytes["checksums.sha256"].decode("utf-8")
            expected_lines = "".join(
                f"{hashlib.sha256(member_bytes[name]).hexdigest()}  {name}\n"
                for name in sorted(JSON_NAMES)
            )
            if checksum_text != expected_lines:
                fail("ZIP checksum manifest is not exact")
            for name in sorted(JSON_NAMES):
                strict_json(member_bytes[name], name)
    except ValidationError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise ValidationError("invalid ZIP bundle") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bundle", type=Path)
    mode.add_argument("--zip", dest="zip_path", type=Path)
    mode.add_argument("--prediction", type=Path)
    parser.add_argument("--execution-commit")
    args = parser.parse_args(argv)
    try:
        if args.zip_path is not None:
            validate_zip(args.zip_path)
            print(
                json.dumps(
                    {"status": "passed", "zip": str(args.zip_path)}, sort_keys=True
                )
            )
            return 0
        if args.prediction is not None:
            raw = _open_regular(args.prediction, maximum=MAX_FILE)
            prediction = strict_json(raw, args.prediction.name)
            selected = scan_prediction(prediction)
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "prediction": str(args.prediction),
                        "selected_pair_count": len(selected),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.bundle is None:  # pragma: no cover - argparse enforces a mode
            parser.error("a validation mode is required")
        print(
            json.dumps(
                validate_bundle(args.bundle, args.execution_commit),
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    except ValidationError as error:
        print(f"validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
