"""Frozen configuration contract for the Stage 1C prospective pilot."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cfsus.exceptions import ScientificInputError

CONFIG_PATH = Path("configs/stage1c_first_prospective_prediction.yaml")
BRANCH = "stage-1c-first-prospective-prediction"
BASE_COMMIT = "efbf70a7e462e640a0e1819a93f3b92727bbd193"
COMPLETED_STATUS = "completed_stage1c_first_prospective_prediction"
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
        raise ScientificInputError("PyYAML is required for Stage 1C") from error

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
        raise ScientificInputError("Stage 1C config is unreadable") from error


def load_stage1c_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the prospective protocol without outcome-dependent fields."""

    return validate_stage1c_config(_unique_yaml(path))


def validate_stage1c_config(value: Any) -> dict[str, Any]:
    config = _mapping(value, "config")
    expected: Any
    for key, expected in (
        ("schema_version", 1),
        ("experiment_class", "stage1c_first_prospective_prediction"),
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
        (upstream, (("version", "0.5.2"), ("revision", UPSTREAM_REVISION)), "upstream"),
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
                ("fallback_allowed", False),
                ("outer_autocast_allowed", False),
            ),
            "runtime",
        ),
    ):
        for key, expected in checks:
            _require(current, key, expected, label)

    prompt = _mapping(config.get("prompt"), "prompt")
    _require(prompt, "id", "pilot", "prompt")
    _require(prompt, "text", "The capital of France is", "prompt")
    _require(prompt, "expected_token_ids", [2, 818, 5279, 529, 7001, 563], "prompt")

    scanner = _mapping(config.get("scanner"), "scanner")
    for key, expected in (
        ("selected_layers", list(range(18))),
        ("selected_positions", [1, 2, 3, 4, 5]),
        ("feature_width", 16_384),
        ("dense_oracle_chunk_size", 16_384),
        ("canonical_chunk_size", 1_024),
        ("top_k_per_group", 8),
        ("global_top_k", 128),
        ("tie_break", ["margin", "layer", "position", "feature_id"]),
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

    response = _mapping(config.get("responses"), "responses")
    for key, expected in (
        ("method", "target_encoder_reverse_vjp_many_source_contraction"),
        ("convention", "attribution_matched_target_preactivation_pre_gate"),
        ("graph_edge_input", "forbidden"),
        ("target_batch_size", 8),
        ("maximum_eligible_pairs", 500_000),
    ):
        _require(response, key, expected, "responses")

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
    _require(scoring, "pair_seed", "stage1c-first-prospective-v1", "scoring")

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
    ):
        _require(selection, key, expected, "selection")

    schedule = _mapping(config.get("schedule"), "schedule")
    _require(schedule, "coarse_alphas", [0.0, 0.25, 0.5, 0.75, 1.0], "schedule")
    _require(schedule, "alpha_hat_offset", 0.015625, "schedule")
    _require(schedule, "maximum_bisection_steps", 8, "schedule")
    _require(schedule, "deduplicate_applied_bf16", True, "schedule")

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
    required = artifacts.get("required_files")
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
    if required != expected_files:
        raise ScientificInputError("artifact allowlist differs from the frozen set")
    return config


__all__ = [
    "BASE_COMMIT",
    "BRANCH",
    "COMPLETED_STATUS",
    "CONFIG_PATH",
    "MODEL_REVISION",
    "TRANSCODER_REVISION",
    "TRANSCODER_SUBFOLDER",
    "UPSTREAM_REVISION",
    "load_stage1c_config",
    "validate_stage1c_config",
]
