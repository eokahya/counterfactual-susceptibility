"""Immutable configuration contract for the Stage 1C-v2 held-out run.

The prompt's tokenizer output is intentionally not checked into this initial
configuration.  A preflight must supply the exact non-empty token list before
the prediction or intervention workers are allowed to execute.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cfsus.exceptions import ScientificInputError

CONFIG_PATH = Path("configs/stage1c_v2_heldout_prospective_prediction.yaml")
SCHEMA_PATH = Path(
    "configs/stage1c_v2_heldout_prospective_prediction_artifact_schema.json"
)
BRANCH = "stage-1c-v2-heldout-prospective-prediction"
BASE_COMMIT = "cc47cb604fc2422deb50aacbc7fde77499b532c5"
EXPERIMENT_CLASS = "stage1c_v2_heldout_prospective_prediction"
COMPLETED_STATUS = "completed_stage1c_v2_heldout_prospective_prediction"
PROMPT_ID = "capital_germany_heldout_v2"
PROMPT_TEXT = "The capital of Germany is"
PAIR_SEED = "stage1c-v2-heldout-prospective-prediction"
UPSTREAM_REVISION = "8f1e2438df612464e229e44c4a00ff637bf9379b"
MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
TRANSCODER_REVISION = "fada11860ac1d337c1e41e9da308798405b94c8e"
TRANSCODER_SUBFOLDER = "transcoder_all/width_16k_l0_small"
SHA40_RE = re.compile(r"\A[0-9a-f]{40}\Z")

# Metadata-only endpoint denylist.  It exists solely to fail closed if the
# deterministic v2 selection accidentally repeats a historical selected pair;
# it is never used to rank, filter, or reselect candidates.
V1_SELECTED_ENDPOINTS: frozenset[tuple[tuple[int, int, int], tuple[int, int, int]]] = (
    frozenset(
        {
            ((14, 1, 234), (15, 1, 771)),
            ((1, 3, 111), (10, 3, 5755)),
            ((2, 3, 582), (3, 3, 15884)),
            ((9, 1, 761), (10, 1, 2072)),
            ((11, 1, 326), (12, 1, 11208)),
            ((9, 5, 2031), (10, 5, 7271)),
            ((9, 1, 761), (10, 1, 2082)),
            ((4, 3, 324), (8, 3, 10215)),
            ((2, 5, 375), (3, 5, 166)),
            ((0, 1, 1386), (1, 1, 4617)),
            ((10, 1, 206), (11, 1, 774)),
            ((7, 1, 503), (8, 1, 30)),
            ((3, 3, 7468), (5, 4, 3745)),
            ((1, 4, 12501), (13, 4, 14586)),
            ((5, 1, 316), (8, 4, 1464)),
            ((0, 2, 724), (9, 5, 153)),
            ((6, 4, 1472), (9, 4, 3017)),
            ((10, 3, 324), (11, 4, 9817)),
            ((2, 3, 419), (15, 4, 30)),
            ((3, 3, 1566), (9, 4, 161)),
            ((3, 3, 380), (8, 3, 1485)),
            ((3, 1, 537), (10, 1, 1008)),
            ((1, 1, 266), (2, 2, 3293)),
            ((1, 5, 453), (2, 5, 278)),
            ((3, 1, 537), (6, 1, 4612)),
            ((5, 1, 7717), (6, 1, 451)),
            ((3, 1, 537), (10, 1, 4484)),
            ((1, 4, 546), (2, 4, 542)),
        }
    )
)


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
        raise ScientificInputError("PyYAML is required for Stage 1C-v2") from error

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
        raise ScientificInputError("Stage 1C-v2 config is unreadable") from error


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
        raise ScientificInputError("held-out prompt token identity changed")
    positions = _mapping(config.get("scanner"), "scanner").get("selected_positions")
    if positions is not None and positions != selected_positions_for_token_ids(
        observed
    ):
        raise ScientificInputError("scanner positions are not all non-BOS positions")


def load_stage1c_v2_config(
    path: str | Path = CONFIG_PATH, *, require_token_ids: bool = False
) -> dict[str, Any]:
    """Load v2 configuration, optionally requiring frozen tokenizer IDs."""

    return validate_stage1c_v2_config(
        _unique_yaml(path), require_token_ids=require_token_ids
    )


def validate_stage1c_v2_config(
    value: Any, *, require_token_ids: bool = False
) -> dict[str, Any]:
    config = _mapping(value, "config")
    expected: Any
    for key, expected in (
        ("schema_version", 2),
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

    prompt = _mapping(config.get("prompt"), "prompt")
    _require(prompt, "id", PROMPT_ID, "prompt")
    _require(prompt, "text", PROMPT_TEXT, "prompt")
    if "expected_token_ids" in prompt:
        _validate_token_ids(prompt["expected_token_ids"])
    elif require_token_ids:
        raise ScientificInputError("exact held-out tokenizer IDs are not frozen")

    scanner = _mapping(config.get("scanner"), "scanner")
    _require(scanner, "selected_layers", list(range(18)), "scanner")
    positions = scanner.get("selected_positions")
    if positions is not None and (
        not isinstance(positions, list)
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in positions
        )
        or positions != list(range(1, len(positions) + 1))
    ):
        raise ScientificInputError(
            "scanner.selected_positions must be omitted/null until tokenization "
            "is frozen"
        )
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
        raise ScientificInputError("v2 artifact allowlist differs from the frozen set")
    return config


def assert_v2_selection_disjoint_from_v1(pairs: Sequence[Any]) -> None:
    """Fail closed after selection if an endpoint repeats a v1 selected pair."""

    overlaps: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    for pair in pairs:
        source = pair.source
        target = pair.target
        endpoint = (
            (int(source.layer), int(source.position), int(source.feature_id)),
            (int(target.layer), int(target.position), int(target.feature_id)),
        )
        if endpoint in V1_SELECTED_ENDPOINTS:
            overlaps.append(endpoint)
    if overlaps:
        raise ScientificInputError(
            "v2 deterministic selection overlaps a historical selected endpoint"
        )


__all__ = [
    "BASE_COMMIT",
    "BRANCH",
    "COMPLETED_STATUS",
    "CONFIG_PATH",
    "EXPERIMENT_CLASS",
    "MODEL_REVISION",
    "PAIR_SEED",
    "PROMPT_ID",
    "PROMPT_TEXT",
    "SCHEMA_PATH",
    "TRANSCODER_REVISION",
    "TRANSCODER_SUBFOLDER",
    "UPSTREAM_REVISION",
    "V1_SELECTED_ENDPOINTS",
    "assert_v2_selection_disjoint_from_v1",
    "load_stage1c_v2_config",
    "selected_positions_for_token_ids",
    "validate_prompt_token_ids",
    "validate_stage1c_v2_config",
]
