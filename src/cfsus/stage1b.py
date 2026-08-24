"""Frozen configuration contract for Stage 1B measurement primitives."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cfsus.exceptions import ScientificInputError

CONFIG_PATH = Path("configs/stage1b_measurement_primitives_gemma3_270m_mps_bf16.yaml")
BRANCH = "stage-1b-measurement-primitives"
BASE_COMMIT = "fb2fc158b45c842743804040e4e273776e666a48"
STAGE1A_EXECUTION_COMMIT = "6a5c21027fbb6b83e34c39db75987b0ce5b72d17"
COMPLETED_STATUS = "completed_stage1b_measurement_primitives"
UPSTREAM_REVISION = "8f1e2438df612464e229e44c4a00ff637bf9379b"
MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
TRANSCODER_REVISION = "fada11860ac1d337c1e41e9da308798405b94c8e"
TRANSCODER_SUBFOLDER = "transcoder_all/width_16k_l0_small"
SHA40_RE = re.compile(r"\A[0-9a-f]{40}\Z")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ScientificInputError(f"{label} must be an object")
    return dict(value)


def _require(mapping: Mapping[str, Any], key: str, expected: Any, label: str) -> None:
    observed = mapping.get(key)
    if observed != expected:
        raise ScientificInputError(
            f"{label}.{key} must be {expected!r}, got {observed!r}"
        )


def load_stage1b_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load YAML with duplicate object keys rejected."""

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as error:
        raise ScientificInputError(
            "PyYAML is required to load Stage 1B config"
        ) from error

    class UniqueKeyLoader(yaml.SafeLoader):  # type: ignore[misc]
        pass

    def construct_mapping(loader: Any, node: Any, deep: bool = False) -> Any:
        pairs = loader.construct_pairs(node, deep=deep)
        result: dict[Any, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ScientificInputError(f"duplicate YAML key: {key!r}")
            result[key] = value
        return result

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    try:
        value = yaml.load(
            Path(path).read_text(encoding="utf-8"), Loader=UniqueKeyLoader
        )
    except OSError as error:
        raise ScientificInputError("Stage 1B config is unreadable") from error
    return validate_stage1b_config(value)


def validate_stage1b_config(value: Any) -> dict[str, Any]:
    """Validate immutable identity and calibration/canonical phase gates."""

    config = _mapping(value, "config")
    for key, expected in (
        ("schema_version", 1),
        ("experiment_class", "stage1b_measurement_primitives"),
        ("completed_status", COMPLETED_STATUS),
        ("branch", BRANCH),
        ("base_commit", BASE_COMMIT),
        ("accepted_stage1a_execution_commit", STAGE1A_EXECUTION_COMMIT),
    ):
        _require(config, key, expected, "config")
    phase = config.get("phase")
    if phase not in {"calibration", "canonical_frozen"}:
        raise ScientificInputError("config.phase is invalid")

    upstream = _mapping(config.get("upstream"), "upstream")
    model = _mapping(config.get("model"), "model")
    transcoder = _mapping(config.get("transcoder"), "transcoder")
    runtime = _mapping(config.get("runtime"), "runtime")
    for mapping, checks, label in (
        (
            upstream,
            (("version", "0.5.2"), ("revision", UPSTREAM_REVISION)),
            "upstream",
        ),
        (
            model,
            (
                ("identifier", "google/gemma-3-270m"),
                ("revision", MODEL_REVISION),
                ("layer_count", 18),
                ("hidden_size", 640),
            ),
            "model",
        ),
        (
            transcoder,
            (
                ("identifier", "mwhanna/gemma-scope-2-270m-pt"),
                ("revision", TRANSCODER_REVISION),
                ("subfolder", TRANSCODER_SUBFOLDER),
                ("layer_count", 18),
                ("feature_width", 16_384),
                ("execution_dtype", "bfloat16"),
            ),
            "transcoder",
        ),
        (
            runtime,
            (
                ("backend", "nnsight"),
                ("device", "mps"),
                ("dtype", "bfloat16"),
                ("python", "3.11.13"),
                ("torch", "2.6.0"),
                ("nnsight", "0.6.1"),
                ("fallback_allowed", False),
                ("outer_autocast_allowed", False),
            ),
            "runtime",
        ),
    ):
        for key, expected in checks:
            _require(mapping, key, expected, label)

    prompt = _mapping(config.get("prompt"), "prompt")
    _require(prompt, "id", "pilot", "prompt")
    _require(prompt, "text", "The capital of France is", "prompt")
    _require(prompt, "expected_token_ids", [2, 818, 5279, 529, 7001, 563], "prompt")
    _require(prompt, "zero_positions", [0], "prompt")

    scanner = _mapping(config.get("scanner"), "scanner")
    _require(scanner, "selected_layers", list(range(18)), "scanner")
    _require(scanner, "selected_positions", [1, 2, 3, 4, 5], "scanner")
    _require(scanner, "feature_width", 16_384, "scanner")
    _require(scanner, "chunk_sizes", [257, 1024, 4096], "scanner")
    _require(scanner, "canonical_chunk_size", 1024, "scanner")
    _require(scanner, "top_k_per_group", 8, "scanner")
    _require(scanner, "global_top_k", 128, "scanner")
    _require(
        scanner,
        "tie_break",
        ["margin", "layer", "position", "feature_id"],
        "scanner",
    )

    responses = _mapping(config.get("responses"), "responses")
    response_checks: tuple[tuple[str, object], ...] = (
        ("method", "target_encoder_reverse_vjp_source_decoder_contraction"),
        ("convention", "attribution_matched_target_preactivation_pre_gate"),
        ("graph_edge_orientation", "target_row_source_column"),
        (
            "graph_edge_formula",
            "raw_edge_equals_source_activation_times_targeted_response",
        ),
        ("graph_edge_input_to_targeted_path", "forbidden"),
        ("calibration_pair_count", 16),
        ("canonical_pair_count", 64),
        ("minimum_target_layers", 6),
        ("minimum_target_positions", 3),
        ("require_both_edge_signs", True),
        ("proposed_edge_floor", 0.015625),
    )
    for response_key, response_expected in response_checks:
        _require(responses, response_key, response_expected, "responses")
    calibration_ids = responses.get("calibration_pair_ids")
    canonical_ids = responses.get("canonical_pair_ids")
    canonical_endpoint_digest = responses.get("canonical_endpoint_manifest_sha256")
    edge_floor = responses.get("edge_floor")
    if phase == "calibration":
        if (
            calibration_ids != []
            or canonical_ids != []
            or canonical_endpoint_digest is not None
            or edge_floor is not None
        ):
            raise ScientificInputError("calibration config contains frozen outcomes")
    else:
        if not isinstance(calibration_ids, list) or len(calibration_ids) != 16:
            raise ScientificInputError("frozen calibration pair IDs are invalid")
        if not isinstance(canonical_ids, list) or len(canonical_ids) != 64:
            raise ScientificInputError("frozen canonical pair IDs are invalid")
        combined = calibration_ids + canonical_ids
        if len(combined) != len(set(combined)) or any(
            not isinstance(item, str) for item in combined
        ):
            raise ScientificInputError("frozen pair IDs are malformed or duplicated")
        if any(
            len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
            for item in combined
        ):
            raise ScientificInputError("frozen pair IDs must be lowercase SHA-256")
        if (
            not isinstance(canonical_endpoint_digest, str)
            or len(canonical_endpoint_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in canonical_endpoint_digest
            )
        ):
            raise ScientificInputError(
                "frozen canonical endpoint digest must be lowercase SHA-256"
            )
        if not isinstance(edge_floor, (float, int)) or isinstance(edge_floor, bool):
            raise ScientificInputError("frozen edge floor must be numeric")
        if edge_floor != responses["proposed_edge_floor"]:
            raise ScientificInputError("frozen edge floor differs from calibration")

    tolerances = _mapping(config.get("tolerances"), "tolerances")
    for tolerance_key, expected_tolerance in (
        ("spearman_minimum", 0.98),
        ("sign_agreement_minimum", 0.95),
        ("median_symmetric_normalized_error_maximum", 0.05),
        ("p95_symmetric_normalized_error_maximum", 0.20),
    ):
        _require(
            tolerances,
            tolerance_key,
            expected_tolerance,
            "tolerances",
        )
    return config


__all__ = [
    "BASE_COMMIT",
    "BRANCH",
    "COMPLETED_STATUS",
    "CONFIG_PATH",
    "MODEL_REVISION",
    "STAGE1A_EXECUTION_COMMIT",
    "TRANSCODER_REVISION",
    "TRANSCODER_SUBFOLDER",
    "UPSTREAM_REVISION",
    "load_stage1b_config",
    "validate_stage1b_config",
]
