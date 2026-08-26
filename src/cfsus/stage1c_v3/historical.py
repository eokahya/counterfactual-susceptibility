"""Authenticated historical exclusion and preregistered prompt primitives."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from cfsus.exceptions import ScientificInputError

EndpointKey = tuple[int, int, int]
ExactPairKey = tuple[EndpointKey, EndpointKey]

HISTORICAL_MANIFEST_PATH = Path(
    "results/stage1c_first_prospective_prediction/prediction_manifest.json"
)
HISTORICAL_MANIFEST_SHA256 = (
    "43cf17f3f87ff97f9fa2aa6b827c84416add5dced2824b69c057d99a5f2b882a"
)
HISTORICAL_MANIFEST_GIT_BLOB_SHA1 = "847e9c3389097b529c0ac2861b3d519afe18d050"
HISTORICAL_MANIFEST_FREEZE_COMMIT = "6ec950d93fe1215fdcfee68c87e1f58a23a78ae8"
HISTORICAL_MANIFEST_FREEZE_PARENT = "efbf70a7e462e640a0e1819a93f3b92727bbd193"
HISTORICAL_BRANCH = "stage-1c-first-prospective-prediction"
HISTORICAL_EXPERIMENT_CLASS = "stage1c_first_prospective_prediction"
HISTORICAL_ARTIFACT_TYPE = "stage1c_prediction_manifest"
HISTORICAL_PAIR_COUNT = 28

DENYLIST_PATH = Path("configs/stage1c_v3_v1_exact_pair_denylist.json")
DENYLIST_SHA256 = "ee31e29e2eb2be5aa5cbf72b95d75ea275098592eb54106f20aa7b3ba87405ad"
DENYLIST_ARTIFACT_TYPE = "stage1c_v3_historical_exact_pair_denylist"

PROMPT_POOL = (
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
)
PROMPT_SELECTION_BASE_COMMIT = "ee9cc944fbdabaa6437b7be3c997725fce5de0a6"
PROMPT_SELECTION_SALT = "stage1c-v3-prompt-v1"
PROMPT_SELECTION_SHA256 = (
    "66e7d4281197efefdbc83bf369d9d317faa7641990c27fa1c3842de99c358e41"
)
PROMPT_SELECTION_INDEX = 7
PROMPT_ID = "capital_norway_preregistered_v3"

_FORBIDDEN_HISTORICAL_OUTCOME_KEYS = frozenset(
    {
        "actual_bf16_value_passed",
        "intervention_sweeps",
        "observed_alpha",
        "observed_crossing",
        "observed_outcome",
        "realized_suppression",
        "scientific_outcome",
        "target_activation_after_intervention",
    }
)


class EndpointOverlapCategory(StrEnum):
    """Audit-only endpoint recurrence category for one surviving v3 pair."""

    NEITHER = "neither_endpoint_seen_in_v1"
    SOURCE_ONLY = "source_endpoint_seen_in_v1_only"
    TARGET_ONLY = "target_endpoint_seen_in_v1_only"
    BOTH_SEPARATE = "both_endpoints_seen_but_not_as_exact_v1_pair"


@dataclass(frozen=True, slots=True)
class PromptDerivation:
    message: str
    sha256_hex: str
    index: int
    prompt: str
    prompt_id: str


@dataclass(frozen=True, slots=True)
class HistoricalPairMetadata:
    exact_pairs: tuple[ExactPairKey, ...]
    historical_endpoints: frozenset[EndpointKey]
    source_manifest_sha256: str
    source_manifest_git_blob_sha1: str
    denylist_sha256: str


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _reject_outcome_fields(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ScientificInputError(f"non-string key at {path}")
            if _normalized_key(key) in _FORBIDDEN_HISTORICAL_OUTCOME_KEYS:
                raise ScientificInputError(
                    f"historical intervention outcome field is forbidden: {path}.{key}"
                )
            _reject_outcome_fields(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_outcome_fields(item, f"{path}[{index}]")


def _strict_json_bytes(data: bytes, label: str) -> Any:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ScientificInputError(f"{label} is not strict UTF-8") from error

    def reject_constant(value: str) -> Any:
        raise ScientificInputError(f"{label} contains non-finite {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ScientificInputError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, RecursionError) as error:
        raise ScientificInputError(f"{label} is invalid JSON") from error


def _regular_bytes(path: Path, label: str, *, maximum_bytes: int) -> bytes:
    try:
        info = path.lstat()
    except OSError as error:
        raise ScientificInputError(f"{label} is unreadable") from error
    if (
        path.is_symlink()
        or not path.is_file()
        or info.st_nlink != 1
        or info.st_size < 1
        or info.st_size > maximum_bytes
    ):
        raise ScientificInputError(f"{label} is not a bounded single-link file")
    try:
        return path.read_bytes()
    except OSError as error:
        raise ScientificInputError(f"{label} is unreadable") from error


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ScientificInputError(f"{label} must be an object")
    return dict(value)


def _feature_key(value: Any, label: str) -> EndpointKey:
    record = _mapping(value, label)
    if set(record) != {"layer", "position", "feature_id"}:
        raise ScientificInputError(f"{label} has an invalid FeatureRef schema")
    raw = (record["layer"], record["position"], record["feature_id"])
    if any(isinstance(item, bool) or not isinstance(item, int) for item in raw):
        raise ScientificInputError(f"{label} has a non-integer FeatureRef")
    layer, position, feature_id = raw
    if not 0 <= layer < 18 or position < 0 or not 0 <= feature_id < 16_384:
        raise ScientificInputError(f"{label} FeatureRef is out of range")
    return int(layer), int(position), int(feature_id)


def feature_key(value: Any) -> EndpointKey:
    """Convert a FeatureRef-like object or strict record to a key."""

    if isinstance(value, Mapping):
        return _feature_key(value, "feature")
    try:
        raw = (value.layer, value.position, value.feature_id)
    except AttributeError as error:
        raise ScientificInputError("feature value is not FeatureRef-like") from error
    return _feature_key(
        {"layer": raw[0], "position": raw[1], "feature_id": raw[2]},
        "feature",
    )


def exact_pair_key(source: Any, target: Any) -> ExactPairKey:
    return feature_key(source), feature_key(target)


def exact_pair_record(pair: ExactPairKey) -> dict[str, dict[str, int]]:
    source, target = pair

    def record(endpoint: EndpointKey) -> dict[str, int]:
        return {
            "layer": endpoint[0],
            "position": endpoint[1],
            "feature_id": endpoint[2],
        }

    return {"source": record(source), "target": record(target)}


def extract_historical_exact_pairs(value: Any) -> tuple[ExactPairKey, ...]:
    """Extract only selected FeatureRefs from the frozen baseline-only manifest."""

    manifest = _mapping(value, "historical prediction manifest")
    _reject_outcome_fields(manifest)
    expected_identity = {
        "schema_version": 1,
        "artifact_type": HISTORICAL_ARTIFACT_TYPE,
        "experiment_class": HISTORICAL_EXPERIMENT_CLASS,
        "branch": HISTORICAL_BRANCH,
        "base_commit": HISTORICAL_MANIFEST_FREEZE_PARENT,
    }
    if any(
        manifest.get(key) != expected for key, expected in expected_identity.items()
    ):
        raise ScientificInputError("historical prediction identity differs")
    guards = _mapping(
        manifest.get("prediction_only_guards"), "historical prediction guards"
    )
    if (
        guards.get("source_suppression_api_calls") != 0
        or guards.get("prior_inactive_target_outcome_read") is not False
    ):
        raise ScientificInputError("historical manifest is not baseline-only")
    groups = _mapping(manifest.get("selected_groups"), "historical selected groups")
    expected_counts = {"primary": 12, "near_boundary": 8, "directional": 8}
    if set(groups) != set(expected_counts):
        raise ScientificInputError("historical selected-group schema differs")
    pairs: list[ExactPairKey] = []
    for group in ("primary", "near_boundary", "directional"):
        rows = groups[group]
        if not isinstance(rows, list) or len(rows) != expected_counts[group]:
            raise ScientificInputError(f"historical {group} count differs")
        for index, raw in enumerate(rows):
            row = _mapping(raw, f"historical {group}[{index}]")
            pairs.append(
                (
                    _feature_key(row.get("source"), f"{group}[{index}].source"),
                    _feature_key(row.get("target"), f"{group}[{index}].target"),
                )
            )
    ordered = tuple(sorted(pairs))
    if len(ordered) != HISTORICAL_PAIR_COUNT or len(set(ordered)) != len(ordered):
        raise ScientificInputError("historical exact-pair set is not 28 unique pairs")
    return ordered


def historical_endpoints(pairs: Sequence[ExactPairKey]) -> frozenset[EndpointKey]:
    return frozenset(endpoint for pair in pairs for endpoint in pair)


def sanitized_denylist_record(
    pairs: Sequence[ExactPairKey],
) -> dict[str, Any]:
    ordered = tuple(sorted(pairs))
    if len(ordered) != HISTORICAL_PAIR_COUNT or len(set(ordered)) != len(ordered):
        raise ScientificInputError("sanitized denylist requires 28 unique pairs")
    return {
        "artifact_type": DENYLIST_ARTIFACT_TYPE,
        "pair_count": HISTORICAL_PAIR_COUNT,
        "pairs": [exact_pair_record(pair) for pair in ordered],
        "schema_version": 1,
        "source_manifest": {
            "artifact_type": HISTORICAL_ARTIFACT_TYPE,
            "branch": HISTORICAL_BRANCH,
            "experiment_class": HISTORICAL_EXPERIMENT_CLASS,
            "freeze_commit": HISTORICAL_MANIFEST_FREEZE_COMMIT,
            "git_blob_sha1": HISTORICAL_MANIFEST_GIT_BLOB_SHA1,
            "path": HISTORICAL_MANIFEST_PATH.as_posix(),
            "schema_version": 1,
            "sha256": HISTORICAL_MANIFEST_SHA256,
        },
    }


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def load_authenticated_historical_metadata(
    repository_root: Path,
) -> HistoricalPairMetadata:
    source_path = repository_root / HISTORICAL_MANIFEST_PATH
    source_bytes = _regular_bytes(
        source_path, "historical prediction manifest", maximum_bytes=2 * 1024 * 1024
    )
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    source_blob = git_blob_sha1(source_bytes)
    if (
        source_sha256 != HISTORICAL_MANIFEST_SHA256
        or source_blob != HISTORICAL_MANIFEST_GIT_BLOB_SHA1
    ):
        raise ScientificInputError(
            "historical prediction manifest authentication failed"
        )
    manifest = _strict_json_bytes(source_bytes, "historical prediction manifest")
    pairs = extract_historical_exact_pairs(manifest)

    denylist_path = repository_root / DENYLIST_PATH
    denylist_bytes = _regular_bytes(
        denylist_path, "historical exact-pair denylist", maximum_bytes=128 * 1024
    )
    denylist_sha256 = hashlib.sha256(denylist_bytes).hexdigest()
    if denylist_sha256 != DENYLIST_SHA256:
        raise ScientificInputError("historical denylist SHA-256 differs")
    denylist = _strict_json_bytes(denylist_bytes, "historical exact-pair denylist")
    _reject_outcome_fields(denylist)
    if denylist != sanitized_denylist_record(pairs):
        raise ScientificInputError("sanitized denylist differs from source extraction")
    return HistoricalPairMetadata(
        exact_pairs=pairs,
        historical_endpoints=historical_endpoints(pairs),
        source_manifest_sha256=source_sha256,
        source_manifest_git_blob_sha1=source_blob,
        denylist_sha256=denylist_sha256,
    )


def mask_historical_exact_pairs(
    pairs: Sequence[Any], denylist: frozenset[ExactPairKey]
) -> tuple[tuple[Any, ...], int]:
    """Remove exact historical pairs while preserving pre-ranking input order."""

    kept = tuple(
        pair
        for pair in pairs
        if exact_pair_key(pair.source, pair.target) not in denylist
    )
    excluded = len(pairs) - len(kept)
    if any(exact_pair_key(pair.source, pair.target) in denylist for pair in kept):
        raise ScientificInputError(
            "historical exact pair survived the pre-ranking mask"
        )
    return kept, excluded


def assert_no_historical_exact_pairs(
    pairs: Sequence[Any], denylist: frozenset[ExactPairKey]
) -> None:
    if any(exact_pair_key(pair.source, pair.target) in denylist for pair in pairs):
        raise ScientificInputError("historical exact pair entered a v3 manifest")


def endpoint_overlap_category(
    source: Any,
    target: Any,
    *,
    denylist: frozenset[ExactPairKey],
    endpoints: frozenset[EndpointKey],
) -> EndpointOverlapCategory:
    """Classify role-independent endpoint recurrence after exact-pair masking."""

    pair = exact_pair_key(source, target)
    if pair in denylist:
        raise ScientificInputError("cannot categorize an excluded historical pair")
    source_seen = pair[0] in endpoints
    target_seen = pair[1] in endpoints
    if source_seen and target_seen:
        return EndpointOverlapCategory.BOTH_SEPARATE
    if source_seen:
        return EndpointOverlapCategory.SOURCE_ONLY
    if target_seen:
        return EndpointOverlapCategory.TARGET_ONLY
    return EndpointOverlapCategory.NEITHER


def derive_preregistered_prompt(
    *,
    base_commit: str = PROMPT_SELECTION_BASE_COMMIT,
    salt: str = PROMPT_SELECTION_SALT,
) -> PromptDerivation:
    if re.fullmatch(r"[0-9a-f]{40}", base_commit) is None or not salt:
        raise ScientificInputError("prompt derivation identity is invalid")
    message = f"{base_commit}|{salt}"
    digest = hashlib.sha256(message.encode("utf-8")).hexdigest()
    index = int(digest[:16], 16) % len(PROMPT_POOL)
    prompt = PROMPT_POOL[index]
    prompt_id = PROMPT_ID if index == PROMPT_SELECTION_INDEX else "invalid"
    return PromptDerivation(message, digest, index, prompt, prompt_id)


def assert_expected_prompt_derivation() -> PromptDerivation:
    result = derive_preregistered_prompt()
    if (
        result.sha256_hex != PROMPT_SELECTION_SHA256
        or result.index != PROMPT_SELECTION_INDEX
        or result.prompt != PROMPT_POOL[PROMPT_SELECTION_INDEX]
        or result.prompt_id != PROMPT_ID
    ):
        raise ScientificInputError("preregistered prompt derivation differs")
    return result


__all__ = [
    "DENYLIST_PATH",
    "DENYLIST_SHA256",
    "HISTORICAL_MANIFEST_FREEZE_COMMIT",
    "HISTORICAL_MANIFEST_GIT_BLOB_SHA1",
    "HISTORICAL_MANIFEST_PATH",
    "HISTORICAL_MANIFEST_SHA256",
    "HISTORICAL_PAIR_COUNT",
    "PROMPT_ID",
    "PROMPT_POOL",
    "PROMPT_SELECTION_BASE_COMMIT",
    "PROMPT_SELECTION_INDEX",
    "PROMPT_SELECTION_SALT",
    "PROMPT_SELECTION_SHA256",
    "EndpointKey",
    "EndpointOverlapCategory",
    "ExactPairKey",
    "HistoricalPairMetadata",
    "PromptDerivation",
    "assert_expected_prompt_derivation",
    "assert_no_historical_exact_pairs",
    "derive_preregistered_prompt",
    "endpoint_overlap_category",
    "exact_pair_key",
    "exact_pair_record",
    "extract_historical_exact_pairs",
    "feature_key",
    "git_blob_sha1",
    "historical_endpoints",
    "load_authenticated_historical_metadata",
    "mask_historical_exact_pairs",
    "sanitized_denylist_record",
]
