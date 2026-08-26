"""Immutable configuration contract for the Stage 1C-v3 preregistered run."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cfsus.exceptions import ScientificInputError
from cfsus.stage1c_v3.historical import (
    DENYLIST_PATH,
    DENYLIST_SHA256,
    HISTORICAL_MANIFEST_FREEZE_COMMIT,
    HISTORICAL_MANIFEST_GIT_BLOB_SHA1,
    HISTORICAL_MANIFEST_PATH,
    HISTORICAL_MANIFEST_SHA256,
    HISTORICAL_PAIR_COUNT,
    PROMPT_POOL,
    PROMPT_SELECTION_BASE_COMMIT,
    PROMPT_SELECTION_INDEX,
    PROMPT_SELECTION_SALT,
    PROMPT_SELECTION_SHA256,
    assert_expected_prompt_derivation,
)

CONFIG_PATH = Path("configs/stage1c_v3_preregistered_prospective_prediction.yaml")
SCHEMA_PATH = Path(
    "configs/stage1c_v3_preregistered_prospective_prediction_artifact_schema.json"
)
BRANCH = "stage-1c-v3-preregistered-prospective-prediction"
BASE_COMMIT = "ee9cc944fbdabaa6437b7be3c997725fce5de0a6"
EXPERIMENT_CLASS = "stage1c_v3_preregistered_prospective_prediction"
COMPLETED_STATUS = "completed_stage1c_v3_preregistered_prospective_prediction"
PROMPT_ID = "capital_norway_preregistered_v3"
PROMPT_TEXT = "The capital of Norway is"
EXPECTED_TOKEN_IDS = (2, 818, 5279, 529, 32649, 563)
SELECTED_POSITIONS = (1, 2, 3, 4, 5)
PAIR_SEED = "stage1c-v3-preregistered-prospective-prediction"
UPSTREAM_REVISION = "8f1e2438df612464e229e44c4a00ff637bf9379b"
MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
TRANSCODER_REVISION = "fada11860ac1d337c1e41e9da308798405b94c8e"
TRANSCODER_SUBFOLDER = "transcoder_all/width_16k_l0_small"
SHA40_RE = re.compile(r"\A[0-9a-f]{40}\Z")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ScientificInputError(f"{label} must be an object")
    return dict(value)


def _require(value: Mapping[str, Any], key: str, expected: Any, label: str) -> None:
    if value.get(key) != expected:
        raise ScientificInputError(f"{label}.{key} differs from the frozen value")


def _unique_yaml(path: str | Path) -> Any:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as error:
        raise ScientificInputError("PyYAML is required for Stage 1C-v3") from error

    class UniqueKeyLoader(yaml.SafeLoader):  # type: ignore[misc]
        pass

    def construct_mapping(loader: Any, node: Any, deep: bool = False) -> Any:
        pairs = loader.construct_pairs(node, deep=deep)
        result: dict[Any, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ScientificInputError(f"duplicate YAML key: {key!r}")
            result[key] = item
        return result

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    try:
        return yaml.load(Path(path).read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    except OSError as error:
        raise ScientificInputError("Stage 1C-v3 config is unreadable") from error


def _validate_token_ids(
    value: Any, *, label: str = "prompt.expected_token_ids"
) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ScientificInputError(f"{label} must be a non-empty token list")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in value
    ):
        raise ScientificInputError(f"{label} must contain non-negative integers")
    return list(value)


def selected_positions_for_token_ids(token_ids: Sequence[int]) -> list[int]:
    """Return the only allowed scanner positions: every non-BOS position."""

    ids = _validate_token_ids(list(token_ids), label="token_ids")
    return list(range(1, len(ids)))


def validate_prompt_token_ids(
    config: Mapping[str, Any], token_ids: Sequence[int]
) -> None:
    """Require an exact preflight token list before empirical execution."""

    prompt = _mapping(config.get("prompt"), "prompt")
    observed = _validate_token_ids(list(token_ids), label="token_ids")
    expected = prompt.get("expected_token_ids")
    if expected is None:
        raise ScientificInputError(
            "exact tokenizer IDs are not frozen; empirical execution is forbidden"
        )
    if _validate_token_ids(expected) != observed:
        raise ScientificInputError("preregistered prompt token identity changed")
    positions = _mapping(config.get("scanner"), "scanner").get("selected_positions")
    if positions is not None and positions != selected_positions_for_token_ids(
        observed
    ):
        raise ScientificInputError("scanner positions are not all non-BOS positions")


def load_stage1c_v3_config(
    path: str | Path = CONFIG_PATH, *, require_token_ids: bool = False
) -> dict[str, Any]:
    """Load v3 configuration, optionally requiring frozen tokenizer IDs."""

    return validate_stage1c_v3_config(
        _unique_yaml(path), require_token_ids=require_token_ids
    )


def validate_stage1c_v3_config(
    value: Any, *, require_token_ids: bool = False
) -> dict[str, Any]:
    config = _mapping(value, "config")
    expected: Any
    for key, expected in (
        ("schema_version", 3),
        ("experiment_class", EXPERIMENT_CLASS),
        ("completed_status", COMPLETED_STATUS),
        ("branch", BRANCH),
        ("base_commit", BASE_COMMIT),
        ("phase", "prediction_only_open"),
    ):
        _require(config, key, expected, "config")

    upstream = _mapping(config.get("upstream"), "upstream")
    model = _mapping(config.get("model"), "model")
    transcoder = _mapping(config.get("transcoder"), "transcoder")
    runtime = _mapping(config.get("runtime"), "runtime")
    for current, checks, label in (
        (
            upstream,
            (("version", "0.5.2"), ("revision", UPSTREAM_REVISION)),
            "upstream",
        ),
        (
            model,
            (("identifier", "google/gemma-3-270m"), ("revision", MODEL_REVISION)),
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
                ("transformers", "4.57.3"),
                ("fallback_allowed", False),
                ("outer_autocast_allowed", False),
                ("scientific_tensor_device", "mps"),
                ("metadata_device", "cpu"),
            ),
            "runtime",
        ),
    ):
        for key, expected in checks:
            _require(current, key, expected, label)

    derivation = _mapping(config.get("prompt_derivation"), "prompt_derivation")
    derived = assert_expected_prompt_derivation()
    for key, expected in (
        ("algorithm", "sha256_prefix16_mod_pool_length"),
        ("base_commit", PROMPT_SELECTION_BASE_COMMIT),
        ("salt", PROMPT_SELECTION_SALT),
        ("message", derived.message),
        ("sha256_hex", PROMPT_SELECTION_SHA256),
        ("index", PROMPT_SELECTION_INDEX),
        ("pool", list(PROMPT_POOL)),
    ):
        _require(derivation, key, expected, "prompt_derivation")

    historical = _mapping(config.get("historical_exclusion"), "historical_exclusion")
    for key, expected in (
        ("unit", "context_independent_exact_pair_key"),
        ("source_manifest", HISTORICAL_MANIFEST_PATH.as_posix()),
        ("source_manifest_sha256", HISTORICAL_MANIFEST_SHA256),
        ("source_manifest_git_blob_sha1", HISTORICAL_MANIFEST_GIT_BLOB_SHA1),
        ("source_manifest_freeze_commit", HISTORICAL_MANIFEST_FREEZE_COMMIT),
        ("denylist", DENYLIST_PATH.as_posix()),
        ("denylist_sha256", DENYLIST_SHA256),
        ("exact_pair_count", HISTORICAL_PAIR_COUNT),
        ("mask_stage", "before_ranking_and_quota_selection"),
        ("single_endpoint_overlap", "audit_only"),
        ("historical_intervention_outcome_reads", "forbidden"),
        ("v2_temporary_baseline_reads", "forbidden"),
    ):
        _require(historical, key, expected, "historical_exclusion")

    prompt = _mapping(config.get("prompt"), "prompt")
    _require(prompt, "id", PROMPT_ID, "prompt")
    _require(prompt, "text", PROMPT_TEXT, "prompt")
    _require(prompt, "expected_token_ids", list(EXPECTED_TOKEN_IDS), "prompt")
    _validate_token_ids(prompt["expected_token_ids"])

    scanner = _mapping(config.get("scanner"), "scanner")
    _require(scanner, "selected_layers", list(range(18)), "scanner")
    _require(scanner, "selected_positions", list(SELECTED_POSITIONS), "scanner")
    for key, expected in (
        ("feature_width", 16_384),
        ("dense_oracle_chunk_size", 16_384),
        ("canonical_chunk_size", 1_024),
        ("top_k_per_group", 8),
        ("global_top_k", 128),
        ("tie_break", ["margin", "layer", "position", "feature_id"]),
        ("dense_oracle_lifecycle", "one_group_ephemeral"),
    ):
        _require(scanner, key, expected, "scanner")

    sources = _mapping(config.get("source_pool"), "source_pool")
    for key, expected in (
        ("selection", "all_exact_loaded_active_features_with_causal_target"),
        ("ordering", ["layer", "position", "feature_id"]),
        ("require_positive_activation", True),
        ("require_strictly_earlier_layer", True),
        ("require_causal_position", True),
        ("raw_graph_input", "forbidden"),
        ("maximum_active_sources", 10_000),
    ):
        _require(sources, key, expected, "source_pool")

    responses = _mapping(config.get("responses"), "responses")
    for key, expected in (
        ("method", "target_encoder_reverse_vjp_many_source_contraction"),
        ("convention", "attribution_matched_target_preactivation_pre_gate"),
        ("graph_edge_input", "forbidden"),
        ("target_batch_size", 8),
        ("maximum_eligible_pairs", 500_000),
    ):
        _require(responses, key, expected, "responses")

    calibration = _mapping(
        config.get("engineering_calibration"), "engineering_calibration"
    )
    for key, expected in (
        (
            "endpoint_class",
            "baseline_active_source_active_target_disjoint_from_inactive_pool",
        ),
        ("pair_count", 4),
        ("reference_method", "stage1b_independent_pairwise_targeted_vjp"),
        ("comparison", "exact_bf16_identity"),
        ("inactive_target_intervention_calls", 0),
    ):
        _require(calibration, key, expected, "engineering_calibration")

    scoring = _mapping(config.get("scoring"), "scoring")
    _require(scoring, "epsilon", 1.0e-12, "scoring")
    _require(scoring, "crossing_tolerance", 1.0e-9, "scoring")
    _require(scoring, "pair_seed", PAIR_SEED, "scoring")

    selection = _mapping(config.get("selection"), "selection")
    for key, expected in (
        ("primary_maximum", 12),
        ("near_boundary_maximum", 8),
        ("directional_maximum", 8),
        ("maximum_per_target", 1),
        ("maximum_primary_per_source", 2),
        ("primary_order", ["susceptibility_desc", "alpha_hat_asc", "target", "source"]),
        ("near_order", ["distance_above_one_asc", "target", "source"]),
        ("directional_order", ["movement_over_margin_desc", "target", "source"]),
        ("prefer_unused_control_targets", True),
        ("control_overlap_fallback", "deterministic_after_unique_exhausted"),
    ):
        _require(selection, key, expected, "selection")

    schedule = _mapping(config.get("schedule"), "schedule")
    for key, expected in (
        ("coarse_alphas", [0.0, 0.25, 0.5, 0.75, 1.0]),
        ("alpha_hat_offset", 0.015625),
        ("maximum_bisection_steps", 8),
        ("deduplicate_applied_bf16", True),
    ):
        _require(schedule, key, expected, "schedule")

    intervention = _mapping(config.get("intervention"), "intervention")
    for key, expected in (
        ("source_count", 1),
        ("mapping", "desired=(1-alpha)*baseline"),
        ("freeze_attention", True),
        ("constrained_layers", None),
        ("target_clamp_allowed", False),
        ("canonical_attempts", 1),
        ("scientific_retries", 0),
    ):
        _require(intervention, key, expected, "intervention")

    analysis = _mapping(config.get("analysis"), "analysis")
    for key, expected in (
        ("minimum_nonzero_points", 3),
        ("movement_sign_agreement_minimum", 0.80),
        ("median_movement_sne_maximum", 0.50),
        ("p95_movement_sne_maximum", 1.00),
        ("critical_bracket_distance_maximum", 0.125),
        ("undefined_metric_policy", "null_with_reason"),
    ):
        _require(analysis, key, expected, "analysis")

    artifacts = _mapping(config.get("artifacts"), "artifacts")
    _require(
        artifacts,
        "canonical_attempt_lock",
        (
            "results/generated/stage1c_v3_preregistered_prospective_prediction/"
            "canonical_attempt_v1.lock"
        ),
        "artifacts",
    )
    expected_files = [
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
    ]
    if artifacts.get("required_files") != expected_files:
        raise ScientificInputError("v3 artifact allowlist differs from the frozen set")
    return config


__all__ = [
    "BASE_COMMIT",
    "BRANCH",
    "COMPLETED_STATUS",
    "CONFIG_PATH",
    "EXPECTED_TOKEN_IDS",
    "EXPERIMENT_CLASS",
    "MODEL_REVISION",
    "PAIR_SEED",
    "PROMPT_ID",
    "PROMPT_TEXT",
    "SCHEMA_PATH",
    "SELECTED_POSITIONS",
    "TRANSCODER_REVISION",
    "TRANSCODER_SUBFOLDER",
    "UPSTREAM_REVISION",
    "load_stage1c_v3_config",
    "selected_positions_for_token_ids",
    "validate_prompt_token_ids",
    "validate_stage1c_v3_config",
]
