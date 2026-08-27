"""Strict immutable configuration contract for Stage 1D."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cfsus.exceptions import ScientificInputError

CONFIG_PATH = Path("configs/stage1d_multiprompt_gate_benchmark.yaml")
SCHEMA_PATH = Path("configs/stage1d_multiprompt_gate_benchmark_artifact_schema.json")
BRANCH = "stage-1d-multiprompt-gate-benchmark"
BASE_COMMIT = "d4fdcc2c2f0040654af17e21f396f1d26072aa0e"
EXPERIMENT_CLASS = "stage1d_multiprompt_gate_benchmark"
COMPLETED_STATUS = "completed_stage1d_multiprompt_gate_benchmark"
SHA40 = re.compile(r"\A[0-9a-f]{40}\Z")
PROMPTS = (
    ("P01", "The capital of Sweden is", (2, 818, 5279, 529, 27556, 563)),
    ("P02", "The capital of Canada is", (2, 818, 5279, 529, 7203, 563)),
    ("P03", "The capital of Japan is", (2, 818, 5279, 529, 6056, 563)),
    ("P04", "The currency of Japan is", (2, 818, 15130, 529, 6056, 563)),
    (
        "P05",
        "The chemical symbol for oxygen is",
        (2, 818, 7395, 5404, 573, 12123, 563),
    ),
    (
        "P06",
        "The largest planet in the Solar System is",
        (2, 818, 7488, 13401, 528, 506, 29277, 1804, 563),
    ),
    ("P07", "The author of Hamlet was", (2, 818, 3260, 529, 124600, 691)),
    (
        "P08",
        "Water freezes at a temperature of",
        (2, 17390, 126521, 657, 496, 4022, 529),
    ),
)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ScientificInputError(f"{label} must be an object")
    return dict(value)


def _expect(value: Mapping[str, Any], key: str, expected: Any, label: str) -> None:
    if value.get(key) != expected:
        raise ScientificInputError(f"{label}.{key} differs from the frozen value")


def _unique_yaml(path: str | Path) -> Any:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as error:
        raise ScientificInputError("PyYAML is required for Stage 1D") from error

    class UniqueLoader(yaml.SafeLoader):  # type: ignore[misc]
        pass

    def construct(loader: Any, node: Any, deep: bool = False) -> Any:
        result: dict[Any, Any] = {}
        for key, item in loader.construct_pairs(node, deep=deep):
            if key in result:
                raise ScientificInputError(f"duplicate YAML key: {key!r}")
            result[key] = item
        return result

    UniqueLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct
    )
    return yaml.load(Path(path).read_text(encoding="utf-8"), Loader=UniqueLoader)


def validate_stage1d_config(value: Any) -> dict[str, Any]:
    config = _object(value, "config")
    for key, expected in (
        ("schema_version", 1),
        ("experiment_class", EXPERIMENT_CLASS),
        ("completed_status", COMPLETED_STATUS),
        ("branch", BRANCH),
        ("base_commit", BASE_COMMIT),
        ("phase", "protocol_frozen_before_baseline"),
    ):
        _expect(config, key, expected, "config")
    runtime = _object(config.get("runtime"), "runtime")
    runtime_expected = {
        "model_identifier": "google/gemma-3-270m",
        "model_revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
        "transcoder_identifier": "mwhanna/gemma-scope-2-270m-pt",
        "transcoder_revision": "fada11860ac1d337c1e41e9da308798405b94c8e",
        "transcoder_subfolder": "transcoder_all/width_16k_l0_small",
        "layer_count": 18,
        "feature_width": 16_384,
        "backend": "nnsight",
        "device": "mps:0",
        "dtype": "torch.bfloat16",
        "python": "3.11.13",
        "torch": "2.6.0",
        "transformers": "4.57.3",
        "nnsight": "0.6.1",
        "circuit_tracer": "0.5.2",
        "upstream_revision": "8f1e2438df612464e229e44c4a00ff637bf9379b",
        "host_class": "Apple M2 Max, 32 GiB unified memory",
        "fallback_allowed": False,
        "offline_required": True,
    }
    if runtime != runtime_expected:
        raise ScientificInputError("Stage 1D runtime identity differs")
    prompts = config.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != len(PROMPTS):
        raise ScientificInputError("Stage 1D prompt list differs")
    for observed, (prompt_id, text, token_ids) in zip(prompts, PROMPTS, strict=True):
        row = _object(observed, "prompt")
        prompt_expected: dict[str, Any] = {
            "id": prompt_id,
            "text": text,
            "token_ids": list(token_ids),
            "selected_positions": list(range(1, len(token_ids))),
        }
        if row != prompt_expected:
            raise ScientificInputError(f"frozen prompt differs: {prompt_id}")
    scanner = _object(config.get("scanner"), "scanner")
    if scanner != {
        "selected_layers": list(range(18)),
        "dense_oracle_chunk_size": 16_384,
        "canonical_chunk_size": 1_024,
        "top_k_per_group": 8,
        "global_top_k": 128,
        "tie_break": ["margin", "layer", "position", "feature_id"],
    }:
        raise ScientificInputError("scanner protocol differs")
    expected_sections = {
        "quantization_resolvability",
        "full_ablation_panel",
        "detailed_panel",
        "schedules",
        "metrics",
        "decision_rule",
        "intervention",
        "engineering_rehearsal",
        "safety_limits",
        "artifacts",
        "claim_boundary",
        "source_pool",
        "responses",
        "scoring",
    }
    if not expected_sections.issubset(config):
        raise ScientificInputError("Stage 1D protocol section is missing")
    panel = _object(config["full_ablation_panel"], "full_ablation_panel")
    if panel.get("per_method_k") != 4 or panel.get("directional_k") != 2:
        raise ScientificInputError("panel quotas differ")
    quant = _object(config["quantization_resolvability"], "quantization")
    if (
        quant.get("predicted_alpha_minimum") != 0.02
        or quant.get("predicted_alpha_maximum") != 0.95
        or quant.get("minimum_distinct_nonzero_at_or_below_twice_alpha") != 3
    ):
        raise ScientificInputError("quantization rule differs")
    schedule = _object(config["schedules"], "schedules")
    if schedule.get("detailed_coarse") != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise ScientificInputError("detailed schedule differs")
    if schedule.get("maximum_bisection_steps") != 6:
        raise ScientificInputError("bisection limit differs")
    decision = _object(config["decision_rule"], "decision_rule")
    if decision != {
        "precedence": [
            "continue_all_criteria",
            "redesign_if_neither_baseline_or_directional_failure",
            "retain_otherwise",
        ],
        "s_minimum_absolute_advantage": 0.10,
        "critical_spearman_minimum": 0.50,
        "critical_pair_count_minimum": 20,
        "directional_violation_fraction_maximum": 0.10,
        "detailed_nonmonotonic_fraction_maximum": 0.20,
    }:
        raise ScientificInputError("decision thresholds differ")
    return config


def load_stage1d_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    return validate_stage1d_config(_unique_yaml(path))


__all__ = [
    "BASE_COMMIT",
    "BRANCH",
    "COMPLETED_STATUS",
    "CONFIG_PATH",
    "EXPERIMENT_CLASS",
    "PROMPTS",
    "SCHEMA_PATH",
    "load_stage1d_config",
    "validate_stage1d_config",
]
