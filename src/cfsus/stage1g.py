"""Prospective Stage 1G behavioral-importance and mediation pilot.

This module is intentionally a thin scientific layer over the accepted
Stage 1C-v4/Stage 1D/Stage 1F runtime, journal, scanner, and targeted-VJP
primitives.  It owns the frozen protocol, deterministic panel selection,
serialized-point analysis, and fail-closed artifact validation.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any, cast

from cfsus.exceptions import ScientificInputError
from cfsus.scanning.near_threshold import compare_scanner_results
from cfsus.stage1c_v3.execution_journal import CanonicalExecutionJournal
from cfsus.stage1c_v3.prediction import (
    canonical_v3_pair_id,
    causally_eligible,
    filter_source_pool,
    filter_target_pool,
    source_pool_digest,
    target_pool_digest,
)
from cfsus.stage1c_v3.quantization_audit import bf16_round
from cfsus.stage1c_v3.serialization import detach_json, read_json_strict, write_json_new
from cfsus.stage1d.metrics import spearman
from cfsus.stage1d.protocol import sha256_file
from cfsus.stage1g_runtime import Stage1GInterventionBackend, Stage1GPredictionBackend
from cfsus.types import FeatureActivity, FeatureRef

CONFIG_PATH = Path("configs/stage1g_behavioral_mediation_pilot.json")
SCHEMA_PATH = Path("configs/stage1g_behavioral_mediation_artifact_schema.json")
EXPERIMENT_CLASS = "stage1g_behavioral_mediation_pilot"
BRANCH = "stage-1g-behavioral-mediation-pilot"
BASE_COMMIT = "c1bb6a3bbab3de945767eded4503b17343ba88e6"
SOURCE_BRANCH = "stage-1f-prospective-one-probe-confirmation"
RUNTIME_FINGERPRINT = (
    "gemma3-270m@9b0cfec892e2/plt@fada11860ac1/"
    "circuit-tracer@8f1e2438df61/nnsight/mps/bf16/stage1g"
)
JSON_ARTIFACTS = (
    "protocol_manifest.json",
    "output_sensitivity_validation.json",
    "prediction_manifest.json",
    "point_records.json",
    "analysis_summary.json",
    "run_manifest.json",
    "environment_manifest.json",
)
PROTOCOL_FILES = (
    str(CONFIG_PATH),
    str(SCHEMA_PATH),
    "src/cfsus/stage1b_runtime.py",
    "src/cfsus/stage1c_v3/execution_journal.py",
    "src/cfsus/stage1c_v3/intervention_runtime.py",
    "src/cfsus/stage1c_v3/prediction.py",
    "src/cfsus/stage1c_v3/runtime.py",
    "src/cfsus/stage1c_v3/serialization.py",
    "src/cfsus/stage1d/metrics.py",
    "src/cfsus/stage1g_runtime.py",
    "src/cfsus/stage1g.py",
    "scripts/stage1g.py",
    "scripts/validate_stage1g_artifacts.py",
)
PROTECTED_ORIGIN_REFS = {
    "main": "7aacf30d888f96a29a1cfc82d035fca489ed0c17",
    "stage-1c-v4-protocol-preserving-execution": (
        "d4fdcc2c2f0040654af17e21f396f1d26072aa0e"
    ),
    "stage-1d-multiprompt-gate-benchmark": ("b71df55fdeb2fb66601af56207b6fbe5238e57d8"),
    "stage-1e-finite-probe-calibration": ("f7aae1f3ce3b1b8d98e850093a3cb5ca480277ea"),
    SOURCE_BRANCH: "6434e72964d8fc9d14e2a6b4cdd9109d7c29e273",
}
SHA40 = re.compile(r"\A[0-9a-f]{40}\Z")
SHA64 = re.compile(r"\A[0-9a-f]{64}\Z")
PANEL_ORDER = ("B", "Q", "G", "D")


def _strict_json(path: str | Path) -> Any:
    def reject_constant(value: str) -> Any:
        raise ScientificInputError(f"non-finite JSON constant: {value}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ScientificInputError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique,
    )


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    value = _strict_json(path)
    if type(value) is not dict:
        raise ScientificInputError("Stage 1G config must be an object")
    config = cast(dict[str, Any], value)
    expected = {
        "schema_version": 1,
        "experiment_class": EXPERIMENT_CLASS,
        "branch": BRANCH,
        "base_commit": BASE_COMMIT,
        "source_branch": SOURCE_BRANCH,
        "phase": "protocol_frozen_before_any_stage1g_baseline_model_call",
    }
    if any(config.get(key) != item for key, item in expected.items()):
        raise ScientificInputError("Stage 1G config identity differs")
    prompts = config.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != 20:
        raise ScientificInputError("Stage 1G requires exactly twenty prompts")
    if [row.get("id") for row in prompts] != [f"G{i:02d}" for i in range(1, 21)]:
        raise ScientificInputError("Stage 1G prompt IDs differ")
    for row in prompts:
        if (
            set(row) != {"id", "text", "answer", "contrast"}
            or not all(isinstance(row[key], str) for key in row)
            or not row["text"].strip()
            or not row["answer"].startswith(" ")
            or not row["contrast"].startswith(" ")
        ):
            raise ScientificInputError("Stage 1G prompt triple differs")
    if config["prompt_ordering"] != {
        "domain": "stage1g-behavior-v1",
        "formula": "SHA256(base_commit|domain|prompt_id)",
        "ascending_hex": True,
        "take_first_eligible": 8,
        "minimum_eligible": 6,
        "backfill_outside_pool_allowed": False,
    }:
        raise ScientificInputError("Stage 1G prompt ordering differs")
    if config["panels"] != {
        "B": {"quota": 4, "rank": "descending_abs_predicted_signed_mediation"},
        "Q": {"quota": 4, "rank": "descending_positive_q"},
        "G": {"quota": 4, "rank": "descending_abs_g_among_predicted_crossings"},
        "D": {"quota": 2, "rank": "descending_abs_g_among_q_nonpositive_causal_pairs"},
        "unique_full_identity_within_panel": True,
        "target_preference": "two_pass_unique_target_then_ranked_fill",
        "execution_deduplication": "prompt_source_target_identity",
        "tie_break": [
            "source_layer",
            "source_position",
            "source_feature_id",
            "target_layer",
            "target_position",
            "target_feature_id",
            "pair_id",
        ],
    }:
        raise ScientificInputError("Stage 1G panel contract differs")
    decision = config["decision_rule"]
    frozen_decision = {
        "minimum_eligible_prompts": 6,
        "minimum_B_crossing_memberships": 24,
        "minimum_B_crossing_prompts": 6,
        "B_sign_accuracy_minimum": 0.7,
        "B_sign_bootstrap_lower_strictly_greater_than": 0.5,
        "B_minus_Q_mean_abs_mediation_bootstrap_lower_strictly_greater_than": 0.0,
        "B_above_floor_fraction_minimum": 0.5,
        "B_injection_sign_agreement_minimum": 0.65,
        "directional_movement_toward_gate_violation_fraction_maximum": 0.1,
        "mixed_minimum_supported_criteria": 4,
        "supported_criteria_count": 7,
        "precedence": [
            "inconclusive_runtime",
            "underpowered",
            "supported",
            "mixed",
            "not_supported",
        ],
    }
    if decision != frozen_decision:
        raise ScientificInputError("Stage 1G terminal classifier differs")
    validation = config["output_sensitivity_validation"]
    if (
        validation["minimum_active_features"] < 8
        or validation["sign_agreement_minimum"] != 0.9
        or validation["spearman_minimum"] != 0.9
        or validation["median_symmetric_normalized_error_maximum"] != 0.05
    ):
        raise ScientificInputError("Stage 1G output-sensitivity tolerances differ")
    expected_files = [*JSON_ARTIFACTS, "checksums.sha256"]
    if config["artifacts"]["required_files"] != expected_files:
        raise ScientificInputError("Stage 1G artifact allowlist differs")
    return config


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env={
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        },
    ).stdout.rstrip("\n")


def verify_git(
    repository: Path, *, expected_head: str | None = None, require_pushed: bool = True
) -> dict[str, Any]:
    head = _git(repository, "rev-parse", "HEAD")
    branch = _git(repository, "branch", "--show-current")
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    if branch != BRANCH or status:
        raise RuntimeError("Stage 1G requires its clean isolated worktree")
    if expected_head is not None and head != expected_head:
        raise RuntimeError("Stage 1G HEAD differs from the frozen commit")
    _git(repository, "merge-base", "--is-ancestor", BASE_COMMIT, head)
    _git(
        repository,
        "merge-base",
        "--is-ancestor",
        BASE_COMMIT,
        f"origin/{SOURCE_BRANCH}",
    )
    protected = {
        name: _git(repository, "rev-parse", f"refs/remotes/origin/{name}")
        for name in PROTECTED_ORIGIN_REFS
    }
    if protected != PROTECTED_ORIGIN_REFS:
        raise RuntimeError("a protected origin ref differs from Stage 1G preflight")
    origin_head: str | None = None
    upstream: str | None = None
    if require_pushed:
        origin_head = _git(repository, "rev-parse", f"refs/remotes/origin/{BRANCH}")
        upstream = _git(
            repository, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"
        )
        if origin_head != head or upstream != f"origin/{BRANCH}":
            raise RuntimeError("Stage 1G local/origin byte identity differs")
    return {
        "branch": branch,
        "head": head,
        "origin_branch_head": origin_head,
        "upstream": upstream,
        "working_tree_clean": True,
        "base_commit": BASE_COMMIT,
        "base_ancestry_verified": True,
        "source_remote_contains_base": True,
        "protected_origin_refs": protected,
    }


def prompt_digest(prompt_id: str, config: Mapping[str, Any]) -> str:
    domain = str(config["prompt_ordering"]["domain"])
    return hashlib.sha256(f"{BASE_COMMIT}|{domain}|{prompt_id}".encode()).hexdigest()


def ordered_prompts(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    prompts = [dict(row) for row in config["prompts"]]
    prompts.sort(key=lambda row: (prompt_digest(str(row["id"]), config), row["id"]))
    return prompts


def _historical_prompt_evidence(
    repository: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    observed: dict[str, list[str]] = {}
    digests: dict[str, str] = {}
    all_texts: set[str] = set()
    for relative in config["eligibility"]["historical_exact_prompt_sources"]:
        path = repository / str(relative)
        value = read_json_strict(path)
        if not isinstance(value, dict):
            raise RuntimeError("historical prediction manifest is malformed")
        prompt_records = value.get("prompts")
        if prompt_records is None and isinstance(value.get("prompt"), dict):
            prompt_records = [value["prompt"]]
        if not isinstance(prompt_records, list):
            raise RuntimeError("historical prediction manifest is malformed")
        texts: list[str] = []
        for prompt in prompt_records:
            if not isinstance(prompt, dict) or not isinstance(prompt.get("text"), str):
                raise RuntimeError("historical prompt record is malformed")
            texts.append(str(prompt["text"]))
        observed[str(relative)] = sorted(set(texts))
        all_texts.update(texts)
        digests[str(relative)] = sha256_file(path)
    fresh = {str(row["text"]) for row in config["prompts"]}
    overlap = sorted(fresh & all_texts)
    if overlap:
        raise RuntimeError(
            "Stage 1G fresh prompt pool overlaps historical interventions"
        )
    return {
        "manifest_sha256": digests,
        "historical_prompt_texts": observed,
        "fresh_exact_prompt_overlap": overlap,
        "intervention_outcomes_read": False,
    }


def protocol_hashes(repository: Path) -> dict[str, str]:
    return {relative: sha256_file(repository / relative) for relative in PROTOCOL_FILES}


def _map_digest(value: Mapping[str, str]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_protocol_manifest(
    repository: Path, *, protocol_commit: str
) -> dict[str, Any]:
    if SHA40.fullmatch(protocol_commit) is None:
        raise ValueError("Stage 1G protocol commit is malformed")
    config = load_config(repository / CONFIG_PATH)
    hashes = protocol_hashes(repository)
    order = ordered_prompts(config)
    return {
        "schema_version": 1,
        "artifact_type": "stage1g_protocol_manifest",
        "status": "protocol_frozen_before_any_stage1g_baseline_model_call",
        "experiment_class": EXPERIMENT_CLASS,
        "branch": BRANCH,
        "base_commit": BASE_COMMIT,
        "source_branch": SOURCE_BRANCH,
        "protocol_commit": protocol_commit,
        "protocol_file_sha256": hashes,
        "protocol_map_sha256": _map_digest(hashes),
        "prompt_order": [row["id"] for row in order],
        "prompt_order_sha256": {
            str(row["id"]): prompt_digest(str(row["id"]), config) for row in order
        },
        "historical_prompt_evidence": _historical_prompt_evidence(repository, config),
        "stage1g_baseline_model_calls_before_freeze": 0,
        "stage1g_scientific_intervention_calls_before_freeze": 0,
        "scientific_attempt_started": False,
        "historical_stage1f_terminal_class_preserved": (
            "completed_stage1f_e1_not_supported"
        ),
        "simple_critical_alpha_calibration": "retired",
    }


def validate_protocol(
    repository: Path, protocol: dict[str, Any], config: dict[str, Any]
) -> None:
    if (
        protocol.get("schema_version") != 1
        or protocol.get("artifact_type") != "stage1g_protocol_manifest"
        or protocol.get("status")
        != "protocol_frozen_before_any_stage1g_baseline_model_call"
        or protocol.get("experiment_class") != EXPERIMENT_CLASS
        or protocol.get("branch") != BRANCH
        or protocol.get("base_commit") != BASE_COMMIT
        or protocol.get("source_branch") != SOURCE_BRANCH
        or protocol.get("stage1g_baseline_model_calls_before_freeze") != 0
        or protocol.get("stage1g_scientific_intervention_calls_before_freeze") != 0
        or protocol.get("scientific_attempt_started") is not False
        or protocol.get("historical_stage1f_terminal_class_preserved")
        != "completed_stage1f_e1_not_supported"
        or protocol.get("simple_critical_alpha_calibration") != "retired"
    ):
        raise ValueError("Stage 1G protocol identity or freeze boundary differs")
    if SHA40.fullmatch(str(protocol.get("protocol_commit"))) is None:
        raise ValueError("Stage 1G protocol commit is malformed")
    hashes = protocol.get("protocol_file_sha256")
    if type(hashes) is not dict or set(hashes) != set(PROTOCOL_FILES):
        raise ValueError("Stage 1G protocol file allowlist differs")
    for relative, digest in hashes.items():
        if (
            SHA64.fullmatch(str(digest)) is None
            or sha256_file(repository / relative) != digest
        ):
            raise ValueError(f"Stage 1G protocol file digest differs: {relative}")
    if protocol.get("protocol_map_sha256") != _map_digest(hashes):
        raise ValueError("Stage 1G protocol map digest differs")
    order = ordered_prompts(config)
    if protocol.get("prompt_order") != [row["id"] for row in order]:
        raise ValueError("Stage 1G prompt order differs")
    if protocol.get("prompt_order_sha256") != {
        row["id"]: prompt_digest(row["id"], config) for row in order
    }:
        raise ValueError("Stage 1G prompt order digests differ")
    if protocol.get("historical_prompt_evidence") != _historical_prompt_evidence(
        repository, config
    ):
        raise ValueError("Stage 1G historical prompt evidence differs")


def _coordinate_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["source"]["layer"]),
        int(row["source"]["position"]),
        int(row["source"]["feature_id"]),
        int(row["target"]["layer"]),
        int(row["target"]["position"]),
        int(row["target"]["feature_id"]),
        str(row["pair_id"]),
    )


def _target_key(row: Mapping[str, Any]) -> tuple[int, int, int]:
    target = row["target"]
    return (int(target["layer"]), int(target["position"]), int(target["feature_id"]))


def _prefer_unique_targets(
    ordered: Sequence[dict[str, Any]], *, quota: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    targets: set[tuple[int, int, int]] = set()
    for row in ordered:
        target = _target_key(row)
        if target in targets:
            continue
        selected.append(row)
        targets.add(target)
        if len(selected) == quota:
            return selected
    for row in ordered:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) == quota:
            break
    return selected


def select_panels(
    pairs: Sequence[dict[str, Any]], *, prompt_id: str, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select frozen equal-budget B/Q/G panels and directional controls."""

    positive = [
        row
        for row in pairs
        if _finite(row["q"], "q") > 0.0
        and bool(row["predicted_full_ablation_crossing"])
    ]
    directional = [row for row in pairs if _finite(row["q"], "q") <= 0.0]
    rankings = {
        "B": sorted(
            positive,
            key=lambda row: (
                -abs(_finite(row["predicted_signed_mediation"], "Mhat")),
                *_coordinate_key(row),
            ),
        ),
        "Q": sorted(
            positive,
            key=lambda row: (-_finite(row["q"], "q"), *_coordinate_key(row)),
        ),
        "G": sorted(
            positive,
            key=lambda row: (-abs(_finite(row["g_i"], "g")), *_coordinate_key(row)),
        ),
        "D": sorted(
            directional,
            key=lambda row: (-abs(_finite(row["g_i"], "g")), *_coordinate_key(row)),
        ),
    }
    selected: dict[str, list[dict[str, Any]]] = {}
    for panel in PANEL_ORDER:
        quota = int(config["panels"][panel]["quota"])
        selected[panel] = _prefer_unique_targets(rankings[panel], quota=quota)
    execution: dict[str, dict[str, Any]] = {}
    memberships: list[dict[str, Any]] = []
    for panel in PANEL_ORDER:
        for rank, row in enumerate(selected[panel], start=1):
            pair_id = str(row["pair_id"])
            current = execution.setdefault(pair_id, dict(row))
            labels = current.setdefault("method_memberships", [])
            ranks = current.setdefault("panel_ranks", {})
            if panel in labels or panel in ranks:
                raise RuntimeError("Stage 1G panel membership duplicated")
            labels.append(panel)
            ranks[panel] = rank
            memberships.append(
                {
                    "prompt_id": prompt_id,
                    "panel": panel,
                    "rank": rank,
                    "pair_id": pair_id,
                }
            )
    rows = sorted(execution.values(), key=_coordinate_key)
    for row in rows:
        row["prompt_id"] = prompt_id
        row["method_memberships"] = [
            panel for panel in PANEL_ORDER if panel in row["method_memberships"]
        ]
        row["panel_ranks"] = {
            panel: row["panel_ranks"][panel]
            for panel in PANEL_ORDER
            if panel in row["panel_ranks"]
        }
    return rows, {
        "candidate_count": len(pairs),
        "predicted_crossing_candidate_count": len(positive),
        "directional_candidate_count": len(directional),
        "selected_membership_count_by_panel": {
            panel: len(selected[panel]) for panel in PANEL_ORDER
        },
        "shortfall_by_panel": {
            panel: int(config["panels"][panel]["quota"]) - len(selected[panel])
            for panel in PANEL_ORDER
        },
        "execution_pair_count": len(rows),
        "overlap_membership_count": len(memberships) - len(rows),
        "memberships": memberships,
        "target_preference": "two_pass_unique_target_then_ranked_fill",
    }


def symmetric_normalized_error(predicted: float, observed: float) -> float:
    denominator = abs(predicted) + abs(observed)
    return 0.0 if denominator == 0.0 else 2.0 * abs(predicted - observed) / denominator


def _sign(value: float) -> int:
    return 1 if value > 0.0 else -1 if value < 0.0 else 0


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("percentile input differs")
    ordered = sorted(values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return float(ordered[min(rank - 1, len(ordered) - 1)])


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(
        math.comb(n, index) * probability**index * (1.0 - probability) ** (n - index)
        for index in range(k + 1)
    )


def clopper_pearson(
    successes: int, total: int, confidence: float = 0.95
) -> list[float]:
    if total == 0 and successes == 0 and 0.0 < confidence < 1.0:
        return [0.0, 1.0]
    if total < 1 or not 0 <= successes <= total or not 0.0 < confidence < 1.0:
        raise ValueError("exact-binomial interval input differs")
    alpha = 1.0 - confidence
    if successes == 0:
        lower = 0.0
    else:
        lo, hi = 0.0, successes / total
        target = 1.0 - alpha / 2.0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if _binomial_cdf(successes - 1, total, mid) > target:
                lo = mid
            else:
                hi = mid
        lower = (lo + hi) / 2.0
    if successes == total:
        upper = 1.0
    else:
        lo, hi = successes / total, 1.0
        target = alpha / 2.0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if _binomial_cdf(successes, total, mid) > target:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2.0
    return [lower, upper]


def _continuation_token(
    tokenizer: Any, prompt: str, continuation: str
) -> tuple[int | None, str | None]:
    """Resolve an exact one-token suffix under concatenated tokenization."""

    prompt_ids = list(tokenizer(prompt, add_special_tokens=False).input_ids)
    combined = list(
        tokenizer(prompt + continuation, add_special_tokens=False).input_ids
    )
    if len(combined) != len(prompt_ids) + 1 or combined[:-1] != prompt_ids:
        return None, "continuation_is_not_exactly_one_contextual_suffix_token"
    token = combined[-1]
    if isinstance(token, bool) or not isinstance(token, int) or token < 0:
        return None, "continuation_token_id_is_invalid"
    return token, None


def _active_calibration_pairs(
    sources: Sequence[Any], *, pair_count: int
) -> tuple[tuple[Any, Any], ...]:
    selected: list[tuple[Any, Any]] = []
    used_sources: set[FeatureRef] = set()
    used_targets: set[FeatureRef] = set()
    for target_state in sources:
        for source_state in sources:
            source = source_state.feature
            target = target_state.feature
            if (
                source in used_sources
                or target in used_targets
                or source.layer >= target.layer
                or source.position > target.position
            ):
                continue
            selected.append((source_state, target_state))
            used_sources.add(source)
            used_targets.add(target)
            break
        if len(selected) == pair_count:
            break
    if len(selected) != pair_count:
        raise RuntimeError("insufficient active-only Stage 1G J calibration pairs")
    return tuple(selected)


def _feature_record(feature: FeatureRef) -> dict[str, int]:
    return {
        "layer": feature.layer,
        "position": feature.position,
        "feature_id": feature.feature_id,
    }


def _pair_pool_digest(rows: Sequence[dict[str, Any]]) -> str:
    hasher = hashlib.sha256()
    ordered = sorted(rows, key=lambda row: (_target_key(row), _coordinate_key(row)))
    for row in ordered:
        hasher.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
        hasher.update(b"\n")
    return hasher.hexdigest()


def _validation_feature_order(
    candidates: Sequence[tuple[Any, float, float, float]], *, count: int
) -> list[tuple[Any, float, float, float]]:
    ordered = sorted(
        candidates,
        key=lambda item: (-abs(item[1]), item[0].feature),
    )
    selected: list[tuple[Any, float, float, float]] = []
    layers: set[int] = set()
    for item in ordered:
        if item[0].feature.layer in layers:
            continue
        selected.append(item)
        layers.add(item[0].feature.layer)
        if len(selected) == count:
            return selected
    for item in ordered:
        if item in selected:
            continue
        selected.append(item)
        if len(selected) == count:
            break
    return selected


def run_output_sensitivity_validation(
    model: Any,
    torch: Any,
    sampler: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate independent output VJPs against disjoint BF16 finite edits."""

    settings = config["output_sensitivity_validation"]
    prompt = str(settings["prompt"])
    prompt_id = str(settings["prompt_id"])
    answer_id, answer_error = _continuation_token(
        model.tokenizer, prompt, str(settings["answer"])
    )
    contrast_id, contrast_error = _continuation_token(
        model.tokenizer, prompt, str(settings["contrast"])
    )
    if (
        answer_error is not None
        or contrast_error is not None
        or answer_id is None
        or contrast_id is None
        or answer_id == contrast_id
    ):
        raise RuntimeError("output-sensitivity rehearsal continuation tokens differ")
    tokens = model.ensure_tokenized(prompt)
    token_ids = [int(item) for item in tokens.detach().cpu().tolist()]
    position = len(token_ids) - 1
    backend = Stage1GPredictionBackend(
        model, prompt=prompt, prompt_id=prompt_id, torch=torch
    )
    with sampler.stage("stage1g_output_sensitivity_baseline"):
        baseline = backend.baseline_behavior(
            answer_token_id=answer_id, contrast_token_id=contrast_id
        )
    groups = tuple((layer, position) for layer in range(18))
    with sampler.stage("stage1g_output_sensitivity_active_pool"):
        active = backend.collect_active_sources(
            groups=groups, chunk_size=int(config["scanner"]["canonical_chunk_size"])
        )
    active = tuple(item for item in active if item.feature.layer > 0)
    target_batch = int(config["responses"]["output_target_batch_size"])
    g_rows: list[tuple[Any, float]] = []
    for start in range(0, len(active), target_batch):
        batch = tuple(item.feature for item in active[start : start + target_batch])
        with sampler.stage(f"stage1g_output_sensitivity_vjp_{start // target_batch}"):
            values = backend.output_sensitivities(
                batch,
                answer_token_id=answer_id,
                contrast_token_id=contrast_id,
                maximum_targets=target_batch,
            )
        g_rows.extend(zip(active[start : start + target_batch], values, strict=True))
    half_width = float(settings["central_relative_half_width"])
    candidates: list[tuple[Any, float, float, float]] = []
    for state, g_value in sorted(
        g_rows, key=lambda item: (-abs(item[1]), item[0].feature)
    )[: int(settings["candidate_limit"])]:
        low = float(
            torch.tensor(
                state.activation * (1.0 - half_width),
                device="mps",
                dtype=torch.bfloat16,
            ).item()
        )
        high = float(
            torch.tensor(
                state.activation * (1.0 + half_width),
                device="mps",
                dtype=torch.bfloat16,
            ).item()
        )
        if low < high and all(math.isfinite(item) for item in (g_value, low, high)):
            candidates.append((state, float(g_value), low, high))
    count = int(settings["minimum_active_features"])
    selected = _validation_feature_order(candidates, count=count)
    if len(selected) < count:
        raise RuntimeError("insufficient BF16-resolvable output-sensitivity features")
    validation_rows: list[dict[str, Any]] = []
    call_count = 0
    for index, (state, g_value, low, high) in enumerate(selected):
        source_state = next(
            item
            for item in active
            if item.feature.layer < state.feature.layer
            and item.feature.position <= state.feature.position
        )
        pair_id = canonical_v3_pair_id(
            source=source_state.feature,
            target=state.feature,
            runtime_fingerprint=RUNTIME_FINGERPRINT,
            prompt_id=prompt_id,
            seed="stage1g-output-sensitivity-validation-v1",
            experiment_class="stage1g_output_sensitivity_validation",
        )
        pair = {
            "pair_id": pair_id,
            "source": _feature_record(source_state.feature),
            "target": _feature_record(state.feature),
        }
        intervention = Stage1GInterventionBackend(
            model,
            prompt=prompt,
            prompt_id=prompt_id,
            torch=torch,
            answer_token_id=answer_id,
            contrast_token_id=contrast_id,
            token_count=len(token_ids),
            call_index_offset=call_count,
        )
        with sampler.stage(f"stage1g_output_sensitivity_fd_{index}_low"):
            low_point = intervention.measure_condition(
                pair,
                condition="target_only_injection",
                desired_source_activation=None,
                desired_target_activation=low,
                stage=f"validation_{index}_low",
            )
        with sampler.stage(f"stage1g_output_sensitivity_fd_{index}_high"):
            high_point = intervention.measure_condition(
                pair,
                condition="target_only_injection",
                desired_source_activation=None,
                desired_target_activation=high,
                stage=f"validation_{index}_high",
            )
        call_count = intervention.source_suppression_api_calls
        applied_low = float(low_point["actual_bf16_target_activation"])
        applied_high = float(high_point["actual_bf16_target_activation"])
        finite_response = (
            float(high_point["behavior_T"]) - float(low_point["behavior_T"])
        ) / (applied_high - applied_low)
        validation_rows.append(
            {
                "feature": _feature_record(state.feature),
                "baseline_activation": state.activation,
                "autograd_g_i": g_value,
                "requested_low": low,
                "requested_high": high,
                "applied_bf16_low": applied_low,
                "applied_bf16_high": applied_high,
                "behavior_low": float(low_point["behavior_T"]),
                "behavior_high": float(high_point["behavior_T"]),
                "finite_secant": finite_response,
                "sign_agreement": _sign(g_value) == _sign(finite_response),
                "symmetric_normalized_error": symmetric_normalized_error(
                    g_value, finite_response
                ),
            }
        )
        del intervention, low_point, high_point
        gc.collect()
        torch.mps.empty_cache()
    autograd = [float(row["autograd_g_i"]) for row in validation_rows]
    finite = [float(row["finite_secant"]) for row in validation_rows]
    sign_accuracy = sum(bool(row["sign_agreement"]) for row in validation_rows) / len(
        validation_rows
    )
    rank_result = spearman(autograd, finite)
    if rank_result is None:
        raise RuntimeError("Stage 1G output-sensitivity rank correlation is undefined")
    rank_correlation = float(rank_result)
    median_error = float(
        median(float(row["symmetric_normalized_error"]) for row in validation_rows)
    )
    passed = (
        all(math.isfinite(item) for item in (*autograd, *finite))
        and sign_accuracy >= float(settings["sign_agreement_minimum"])
        and rank_correlation >= float(settings["spearman_minimum"])
        and median_error <= float(settings["median_symmetric_normalized_error_maximum"])
    )
    record = {
        "schema_version": 1,
        "artifact_type": "stage1g_output_sensitivity_validation",
        "status": "passed" if passed else "failed",
        "prompt_id": prompt_id,
        "prompt": prompt,
        "token_ids": token_ids,
        "answer_token_id": answer_id,
        "contrast_token_id": contrast_id,
        "baseline_behavior": baseline,
        "selected_feature_count": len(validation_rows),
        "selection_rule": settings["selection"],
        "central_relative_half_width": half_width,
        "rows": validation_rows,
        "metrics": {
            "all_finite": True,
            "sign_agreement": sign_accuracy,
            "spearman": rank_correlation,
            "median_symmetric_normalized_error": median_error,
        },
        "frozen_tolerances": {
            "sign_agreement_minimum": settings["sign_agreement_minimum"],
            "spearman_minimum": settings["spearman_minimum"],
            "median_symmetric_normalized_error_maximum": settings[
                "median_symmetric_normalized_error_maximum"
            ],
        },
        "instrumented_target_injection_calls": call_count,
        "scientific_attempt_consumed": False,
        "scientific_pair_overlap": False,
        "gradient_tensor_persisted": False,
        "full_logits_persisted": False,
    }
    if not passed:
        raise RuntimeError("Stage 1G output-sensitivity validation failed")
    return record


def validate_output_sensitivity(value: dict[str, Any], config: dict[str, Any]) -> None:
    settings = config["output_sensitivity_validation"]
    rows = value.get("rows")
    metrics = value.get("metrics")
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != "stage1g_output_sensitivity_validation"
        or value.get("status") != "passed"
        or value.get("prompt_id") != settings["prompt_id"]
        or value.get("prompt") != settings["prompt"]
        or value.get("selected_feature_count") < settings["minimum_active_features"]
        or value.get("scientific_attempt_consumed") is not False
        or value.get("scientific_pair_overlap") is not False
        or value.get("gradient_tensor_persisted") is not False
        or value.get("full_logits_persisted") is not False
        or not isinstance(rows, list)
        or not isinstance(metrics, dict)
    ):
        raise ValueError("Stage 1G output-sensitivity identity differs")
    autograd: list[float] = []
    finite: list[float] = []
    errors: list[float] = []
    signs = 0
    features: set[tuple[int, int, int]] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("feature"), dict):
            raise ValueError("Stage 1G output-sensitivity row is malformed")
        feature = FeatureRef(**row["feature"])
        key = (feature.layer, feature.position, feature.feature_id)
        if key in features or feature.layer <= 0:
            raise ValueError("Stage 1G validation feature identity differs")
        features.add(key)
        g_value = _finite(row.get("autograd_g_i"), "validation g")
        secant = _finite(row.get("finite_secant"), "validation secant")
        low = _finite(row.get("applied_bf16_low"), "validation low")
        high = _finite(row.get("applied_bf16_high"), "validation high")
        if low >= high:
            raise ValueError("Stage 1G validation BF16 perturbation collapsed")
        expected_sign = _sign(g_value) == _sign(secant)
        expected_error = symmetric_normalized_error(g_value, secant)
        if (
            row.get("sign_agreement") is not expected_sign
            or _finite(row.get("symmetric_normalized_error"), "validation error")
            != expected_error
        ):
            raise ValueError("Stage 1G validation derived value differs")
        autograd.append(g_value)
        finite.append(secant)
        errors.append(expected_error)
        signs += int(expected_sign)
    rank_result = spearman(autograd, finite)
    if rank_result is None:
        raise ValueError("Stage 1G output-sensitivity rank correlation is undefined")
    expected_metrics = {
        "all_finite": True,
        "sign_agreement": signs / len(rows),
        "spearman": float(rank_result),
        "median_symmetric_normalized_error": float(median(errors)),
    }
    if metrics != expected_metrics:
        raise ValueError("Stage 1G output-sensitivity aggregate differs")
    if (
        expected_metrics["sign_agreement"] < settings["sign_agreement_minimum"]
        or expected_metrics["spearman"] < settings["spearman_minimum"]
        or expected_metrics["median_symmetric_normalized_error"]
        > settings["median_symmetric_normalized_error_maximum"]
    ):
        raise ValueError("Stage 1G output-sensitivity gate failed")


def _prompt_prediction(
    model: Any,
    torch: Any,
    sampler: Any,
    prompt: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    prompt_id = str(prompt["id"])
    prompt_text = str(prompt["text"])
    answer_id = int(prompt["answer_token_id"])
    contrast_id = int(prompt["contrast_token_id"])
    token_ids = [int(item) for item in prompt["token_ids"]]
    positions = list(range(1, len(token_ids)))
    backend = Stage1GPredictionBackend(
        model, prompt=prompt_text, prompt_id=prompt_id, torch=torch
    )
    scanner_config = config["scanner"]
    groups = tuple(
        (layer, position)
        for layer in scanner_config["selected_layers"]
        for position in positions
    )
    with sampler.stage(f"{prompt_id}_scanner_dense_oracle"):
        oracle = backend.scan(
            groups=groups,
            chunk_size=int(scanner_config["dense_oracle_chunk_size"]),
            top_k_per_group=int(scanner_config["top_k_per_group"]),
            global_top_k=int(scanner_config["global_top_k"]),
        )
    with sampler.stage(f"{prompt_id}_scanner_chunked"):
        scanner = backend.scan(
            groups=groups,
            chunk_size=int(scanner_config["canonical_chunk_size"]),
            top_k_per_group=int(scanner_config["top_k_per_group"]),
            global_top_k=int(scanner_config["global_top_k"]),
        )
        compare_scanner_results(oracle, scanner)
    del oracle
    with sampler.stage(f"{prompt_id}_active_source_pool"):
        raw_sources = backend.collect_active_sources(
            groups=groups, chunk_size=int(scanner_config["canonical_chunk_size"])
        )
        targets = filter_target_pool(scanner.global_candidates, raw_sources)
        sources = filter_source_pool(
            raw_sources,
            targets,
            maximum_sources=int(config["source_pool"]["maximum_active_sources"]),
        )
        targets = filter_target_pool(targets, sources)
        eligible_pair_count = sum(
            causally_eligible(source.feature, target.feature)
            for target in targets
            for source in sources
        )
        if eligible_pair_count > int(config["responses"]["maximum_eligible_pairs"]):
            raise RuntimeError(f"{prompt_id} eligible pair cap exceeded")
    calibration_count = int(config["responses"]["active_j_health_pair_count"])
    calibration_pairs = _active_calibration_pairs(
        raw_sources, pair_count=calibration_count
    )
    with sampler.stage(f"{prompt_id}_active_j_health"):
        pairwise = backend.targeted_local_responses(
            tuple(
                (source.feature, target.feature) for source, target in calibration_pairs
            ),
            maximum_pairs=calibration_count,
        )
        calibration_tile = backend.response_tile(
            targets=tuple(target.feature for _, target in calibration_pairs),
            sources=tuple(source for source, _ in calibration_pairs),
            maximum_targets=calibration_count,
        )
    if tuple(
        float(calibration_tile[index][index]) for index in range(calibration_count)
    ) != tuple(float(item.response) for item in pairwise):
        raise RuntimeError(f"{prompt_id} targeted J health calibration differs")
    del calibration_pairs, pairwise, calibration_tile
    gc.collect()
    torch.mps.empty_cache()
    target_by_feature = {item.feature: item for item in targets}
    target_features = tuple(sorted(target_by_feature))
    output_batch = int(config["responses"]["output_target_batch_size"])
    g_by_target: dict[FeatureRef, float] = {}
    for start in range(0, len(target_features), output_batch):
        batch = target_features[start : start + output_batch]
        with sampler.stage(f"{prompt_id}_output_g_batch_{start // output_batch}"):
            values = backend.output_sensitivities(
                batch,
                answer_token_id=answer_id,
                contrast_token_id=contrast_id,
                maximum_targets=output_batch,
            )
        g_by_target.update(zip(batch, values, strict=True))
    response_batch = int(config["responses"]["target_batch_size"])
    pair_rows: list[dict[str, Any]] = []
    q_counts: Counter[str] = Counter()
    for start in range(0, len(target_features), response_batch):
        selected_targets = target_features[start : start + response_batch]
        with sampler.stage(f"{prompt_id}_target_j_batch_{start // response_batch}"):
            response_tile = backend.response_tile(
                targets=selected_targets,
                sources=sources,
                maximum_targets=response_batch,
            )
        for target, responses in zip(selected_targets, response_tile, strict=True):
            target_state = target_by_feature[target]
            g_value = float(g_by_target[target])
            for source_state, response in zip(sources, responses, strict=True):
                if not causally_eligible(source_state.feature, target):
                    continue
                response_value = float(response)
                q_value = -source_state.activation * response_value
                margin = target_state.threshold - target_state.preactivation
                predicted_crossing = (
                    q_value > 0.0
                    and target_state.preactivation + q_value > target_state.threshold
                )
                predicted_activation = (
                    target_state.preactivation + q_value if predicted_crossing else 0.0
                )
                row = {
                    "pair_id": canonical_v3_pair_id(
                        source=source_state.feature,
                        target=target,
                        runtime_fingerprint=RUNTIME_FINGERPRINT,
                        prompt_id=prompt_id,
                        seed=str(config["scoring"]["pair_seed"]),
                        experiment_class=EXPERIMENT_CLASS,
                    ),
                    "source": _feature_record(source_state.feature),
                    "target": _feature_record(target),
                    "source_activation": source_state.activation,
                    "target_preactivation": target_state.preactivation,
                    "target_threshold": target_state.threshold,
                    "margin": margin,
                    "targeted_response": response_value,
                    "q": q_value,
                    "g_i": g_value,
                    "predicted_full_ablation_crossing": predicted_crossing,
                    "predicted_target_activation": predicted_activation,
                    "predicted_signed_mediation": g_value * predicted_activation,
                    "q_over_margin_computed_or_used": False,
                    "intervention_outcome_used": False,
                }
                if not all(
                    math.isfinite(_finite(row[key], key))
                    for key in (
                        "source_activation",
                        "target_preactivation",
                        "target_threshold",
                        "margin",
                        "targeted_response",
                        "q",
                        "g_i",
                        "predicted_target_activation",
                        "predicted_signed_mediation",
                    )
                ):
                    raise RuntimeError(
                        f"{prompt_id} baseline pair scalar is non-finite"
                    )
                pair_rows.append(row)
                q_counts[
                    "positive"
                    if q_value > 0
                    else "zero"
                    if q_value == 0
                    else "negative"
                ] += 1
        del response_tile
        gc.collect()
        torch.mps.empty_cache()
    if len(pair_rows) != eligible_pair_count:
        raise RuntimeError(f"{prompt_id} pair enumeration is incomplete")
    execution_pairs, panel_audit = select_panels(
        pair_rows, prompt_id=prompt_id, config=config
    )
    for row in execution_pairs:
        row["answer_token_id"] = answer_id
        row["contrast_token_id"] = contrast_id
    result = {
        "id": prompt_id,
        "text": prompt_text,
        "answer": prompt["answer"],
        "contrast": prompt["contrast"],
        "token_ids": token_ids,
        "selected_positions": positions,
        "answer_token_id": answer_id,
        "contrast_token_id": contrast_id,
        "baseline_behavior": prompt["baseline_behavior"],
        "baseline_pools": {
            "scanner_candidate_count": len(scanner.global_candidates),
            "eligible_target_count": len(targets),
            "target_pool_sha256": target_pool_digest(targets),
            "raw_active_source_count": len(raw_sources),
            "eligible_source_count": len(sources),
            "source_pool_sha256": source_pool_digest(sources),
            "eligible_pair_count": len(pair_rows),
            "pair_score_sha256": _pair_pool_digest(pair_rows),
            "q_sign_counts": dict(sorted(q_counts.items())),
            "dense_scanner_arrays_persisted": False,
            "complete_derivative_matrix_persisted": False,
            "gradient_tensor_persisted": False,
            "scanner_dense_oracle_validation": {
                "group_count": len(groups),
                "exact_identity_and_order": True,
            },
            "active_j_health_calibration": {
                "pair_count": calibration_count,
                "pairwise_vs_many_source_exact_bf16_identity": True,
                "graph_edge_input_used": False,
                "intervention_calls": 0,
            },
        },
        "panel_audit": panel_audit,
        "execution_pairs": execution_pairs,
    }
    pair_rows.clear()
    del raw_sources, sources, targets, scanner, backend
    gc.collect()
    torch.mps.empty_cache()
    return result


def build_prediction_manifest(
    model: Any,
    torch: Any,
    sampler: Any,
    config: dict[str, Any],
    *,
    protocol: dict[str, Any],
    sensitivity_validation: dict[str, Any],
    git: dict[str, Any],
) -> dict[str, Any]:
    validate_output_sensitivity(sensitivity_validation, config)
    historical = {
        text
        for texts in protocol["historical_prompt_evidence"][
            "historical_prompt_texts"
        ].values()
        for text in texts
    }
    eligible: list[dict[str, Any]] = []
    eligibility_rows: list[dict[str, Any]] = []
    quota = int(config["prompt_ordering"]["take_first_eligible"])
    for frozen in ordered_prompts(config):
        row = dict(frozen)
        digest = prompt_digest(str(row["id"]), config)
        if len(eligible) >= quota:
            eligibility_rows.append(
                {
                    "id": row["id"],
                    "order_sha256": digest,
                    "status": "not_evaluated_after_quota",
                    "eligible": None,
                    "model_calls": 0,
                }
            )
            continue
        reasons: list[str] = []
        answer_id, answer_error = _continuation_token(
            model.tokenizer, str(row["text"]), str(row["answer"])
        )
        contrast_id, contrast_error = _continuation_token(
            model.tokenizer, str(row["text"]), str(row["contrast"])
        )
        if answer_error:
            reasons.append(f"answer:{answer_error}")
        if contrast_error:
            reasons.append(f"contrast:{contrast_error}")
        if answer_id is not None and answer_id == contrast_id:
            reasons.append("answer_and_contrast_token_ids_equal")
        token_ids = [
            int(item)
            for item in model.ensure_tokenized(str(row["text"])).detach().cpu().tolist()
        ]
        baseline: dict[str, Any] | None = None
        if not reasons and answer_id is not None and contrast_id is not None:
            backend = Stage1GPredictionBackend(
                model,
                prompt=str(row["text"]),
                prompt_id=str(row["id"]),
                torch=torch,
            )
            with sampler.stage(f"{row['id']}_eligibility_baseline"):
                baseline = backend.baseline_behavior(
                    answer_token_id=answer_id, contrast_token_id=contrast_id
                )
            if float(baseline["behavior_T"]) <= 0.0:
                reasons.append("baseline_answer_minus_contrast_not_positive")
            if baseline["answer_in_top64"] is not True:
                reasons.append("answer_not_in_top64")
            del backend
        if str(row["text"]) in historical:
            reasons.append("historical_exact_prompt_identity")
        is_eligible = not reasons
        eligibility_rows.append(
            {
                "id": row["id"],
                "order_sha256": digest,
                "status": "eligible" if is_eligible else "ineligible",
                "eligible": is_eligible,
                "reasons": reasons,
                "token_ids": token_ids,
                "answer_token_id": answer_id,
                "contrast_token_id": contrast_id,
                "baseline_behavior": baseline,
                "model_calls": 0 if baseline is None else 1,
            }
        )
        if is_eligible:
            row.update(
                {
                    "token_ids": token_ids,
                    "answer_token_id": answer_id,
                    "contrast_token_id": contrast_id,
                    "baseline_behavior": baseline,
                }
            )
            eligible.append(row)
    prompts = [
        _prompt_prediction(model, torch, sampler, row, config) for row in eligible
    ]
    minimum = int(config["prompt_ordering"]["minimum_eligible"])
    total_pairs = sum(len(prompt["execution_pairs"]) for prompt in prompts)
    totals = {
        "eligible_prompt_count": len(prompts),
        "execution_pair_count": total_pairs,
        "membership_count_by_panel": {
            panel: sum(
                int(prompt["panel_audit"]["selected_membership_count_by_panel"][panel])
                for prompt in prompts
            )
            for panel in PANEL_ORDER
        },
        "shortfall_by_panel": {
            panel: sum(
                int(prompt["panel_audit"]["shortfall_by_panel"][panel])
                for prompt in prompts
            )
            for panel in PANEL_ORDER
        },
    }
    return {
        "schema_version": 1,
        "artifact_type": "stage1g_prediction_manifest",
        "status": (
            "prediction_frozen_ready_for_commit"
            if len(prompts) >= minimum
            else "underpowered_before_intervention"
        ),
        "experiment_class": EXPERIMENT_CLASS,
        "branch": BRANCH,
        "base_commit": BASE_COMMIT,
        "protocol_commit": protocol["protocol_commit"],
        "protocol_map_sha256": protocol["protocol_map_sha256"],
        "prediction_execution_commit": git["head"],
        "runtime_identity": config["runtime"],
        "output_sensitivity_validation": sensitivity_validation,
        "ordered_prompt_eligibility": eligibility_rows,
        "eligible_prompt_order": [prompt["id"] for prompt in prompts],
        "prompts": prompts,
        "selection_totals": totals,
        "prediction_only_guards": {
            "fresh_scientific_intervention_api_calls": 0,
            "historical_intervention_outcomes_used": False,
            "graph_edge_input_used": False,
            "q_over_margin_discovery_used": False,
            "E1_or_E2_computed": False,
            "network_accessed": False,
        },
        "claim_boundary": config["claim_boundary"],
    }


def validate_prediction(
    prediction: dict[str, Any], config: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Validate the frozen baseline-only panel without reading interventions."""

    minimum = int(config["prompt_ordering"]["minimum_eligible"])
    if (
        prediction.get("schema_version") != 1
        or prediction.get("artifact_type") != "stage1g_prediction_manifest"
        or prediction.get("status")
        not in {
            "prediction_frozen_ready_for_commit",
            "underpowered_before_intervention",
        }
        or prediction.get("experiment_class") != EXPERIMENT_CLASS
        or prediction.get("branch") != BRANCH
        or prediction.get("base_commit") != BASE_COMMIT
        or prediction.get("protocol_commit") != protocol["protocol_commit"]
        or prediction.get("protocol_map_sha256") != protocol["protocol_map_sha256"]
        or prediction.get("runtime_identity") != config["runtime"]
        or prediction.get("prediction_only_guards")
        != {
            "fresh_scientific_intervention_api_calls": 0,
            "historical_intervention_outcomes_used": False,
            "graph_edge_input_used": False,
            "q_over_margin_discovery_used": False,
            "E1_or_E2_computed": False,
            "network_accessed": False,
        }
        or prediction.get("claim_boundary") != config["claim_boundary"]
    ):
        raise ValueError("Stage 1G prediction identity or no-outcome guard differs")
    sensitivity = prediction.get("output_sensitivity_validation")
    if not isinstance(sensitivity, dict):
        raise ValueError("Stage 1G output-sensitivity evidence is missing")
    validate_output_sensitivity(sensitivity, config)
    eligibility = prediction.get("ordered_prompt_eligibility")
    prompts = prediction.get("prompts")
    if not isinstance(eligibility, list) or len(eligibility) != 20:
        raise ValueError("Stage 1G prompt eligibility audit differs")
    if not isinstance(prompts, list) or len(prompts) > 8:
        raise ValueError("Stage 1G eligible prompt count differs")
    ordered = ordered_prompts(config)
    for observed, frozen in zip(eligibility, ordered, strict=True):
        if (
            not isinstance(observed, dict)
            or observed.get("id") != frozen["id"]
            or observed.get("order_sha256") != prompt_digest(str(frozen["id"]), config)
            or observed.get("status")
            not in {"eligible", "ineligible", "not_evaluated_after_quota"}
        ):
            raise ValueError("Stage 1G prompt eligibility row differs")
    eligible_ids = [row["id"] for row in eligibility if row.get("eligible") is True]
    if prediction.get("eligible_prompt_order") != eligible_ids:
        raise ValueError("Stage 1G eligible prompt order differs")
    if [prompt.get("id") for prompt in prompts] != eligible_ids:
        raise ValueError("Stage 1G prompt records differ from eligibility")
    expected_status = (
        "prediction_frozen_ready_for_commit"
        if len(prompts) >= minimum
        else "underpowered_before_intervention"
    )
    if prediction.get("status") != expected_status:
        raise ValueError("Stage 1G pre-intervention power classification differs")
    all_pairs: dict[str, dict[str, Any]] = {}
    membership_totals = {panel: 0 for panel in PANEL_ORDER}
    shortfall_totals = {panel: 0 for panel in PANEL_ORDER}
    for prompt in prompts:
        if not isinstance(prompt, dict):
            raise ValueError("Stage 1G prediction prompt is malformed")
        eligibility_row = next(row for row in eligibility if row["id"] == prompt["id"])
        for key in (
            "token_ids",
            "answer_token_id",
            "contrast_token_id",
            "baseline_behavior",
        ):
            if prompt.get(key) != eligibility_row.get(key):
                raise ValueError(f"Stage 1G prompt {prompt['id']} {key} differs")
        rows = prompt.get("execution_pairs")
        audit = prompt.get("panel_audit")
        if not isinstance(rows, list) or len(rows) > 14 or not isinstance(audit, dict):
            raise ValueError("Stage 1G prompt execution panel differs")
        prompt_pairs: dict[str, dict[str, Any]] = {}
        by_panel: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Stage 1G pair row is malformed")
            pair_id = row.get("pair_id")
            source_raw = row.get("source")
            target_raw = row.get("target")
            if (
                not isinstance(pair_id, str)
                or SHA64.fullmatch(pair_id) is None
                or pair_id in all_pairs
                or pair_id in prompt_pairs
                or not isinstance(source_raw, dict)
                or not isinstance(target_raw, dict)
            ):
                raise ValueError("Stage 1G pair identity is malformed or duplicated")
            source = FeatureRef(**source_raw)
            target = FeatureRef(**target_raw)
            token_count = len(prompt["token_ids"])
            if not (
                0 <= source.layer < target.layer < 18
                and 1 <= source.position <= target.position < token_count
                and 0 <= source.feature_id < 16384
                and 0 <= target.feature_id < 16384
            ):
                raise ValueError("Stage 1G pair violates causal domain")
            expected_id = canonical_v3_pair_id(
                source=source,
                target=target,
                runtime_fingerprint=RUNTIME_FINGERPRINT,
                prompt_id=str(prompt["id"]),
                seed=str(config["scoring"]["pair_seed"]),
                experiment_class=EXPERIMENT_CLASS,
            )
            activation = _finite(row.get("source_activation"), "source activation")
            z_value = _finite(row.get("target_preactivation"), "target z")
            threshold = _finite(row.get("target_threshold"), "target threshold")
            margin = _finite(row.get("margin"), "target margin")
            response = _finite(row.get("targeted_response"), "targeted J")
            q_value = _finite(row.get("q"), "q")
            g_value = _finite(row.get("g_i"), "g")
            predicted_crossing = q_value > 0.0 and z_value + q_value > threshold
            predicted_activation = z_value + q_value if predicted_crossing else 0.0
            predicted_mediation = g_value * predicted_activation
            memberships = row.get("method_memberships")
            ranks = row.get("panel_ranks")
            if (
                pair_id != expected_id
                or row.get("prompt_id") != prompt["id"]
                or row.get("answer_token_id") != prompt["answer_token_id"]
                or row.get("contrast_token_id") != prompt["contrast_token_id"]
                or activation <= 0.0
                or z_value > threshold
                or margin != threshold - z_value
                or q_value != -activation * response
                or row.get("predicted_full_ablation_crossing") is not predicted_crossing
                or _finite(row.get("predicted_target_activation"), "predicted a")
                != predicted_activation
                or _finite(row.get("predicted_signed_mediation"), "predicted M")
                != predicted_mediation
                or row.get("q_over_margin_computed_or_used") is not False
                or row.get("intervention_outcome_used") is not False
                or not isinstance(memberships, list)
                or not memberships
                or memberships
                != [panel for panel in PANEL_ORDER if panel in memberships]
                or not isinstance(ranks, dict)
                or set(ranks) != set(memberships)
            ):
                raise ValueError("Stage 1G pair scalar or membership differs")
            if any(panel not in PANEL_ORDER for panel in memberships):
                raise ValueError("Stage 1G pair has an unknown panel")
            if any(panel != "D" for panel in memberships) and not predicted_crossing:
                raise ValueError("Stage 1G B/Q/G membership lacks predicted crossing")
            if "D" in memberships and q_value > 0.0:
                raise ValueError("Stage 1G directional pair has positive q")
            for panel in memberships:
                by_panel[panel].append(row)
            prompt_pairs[pair_id] = row
            all_pairs[pair_id] = row
        memberships_expected: list[dict[str, Any]] = []
        for panel in PANEL_ORDER:
            panel_rows = sorted(
                by_panel[panel], key=lambda row: int(row["panel_ranks"][panel])
            )
            quota = int(config["panels"][panel]["quota"])
            if len(panel_rows) > quota or [
                row["panel_ranks"][panel] for row in panel_rows
            ] != list(range(1, len(panel_rows) + 1)):
                raise ValueError("Stage 1G panel quota or rank differs")

            def ranking_key(
                row: dict[str, Any], panel_name: str = panel
            ) -> tuple[Any, ...]:
                if panel_name == "B":
                    return (
                        -abs(float(row["predicted_signed_mediation"])),
                        *_coordinate_key(row),
                    )
                if panel_name == "Q":
                    return (-float(row["q"]), *_coordinate_key(row))
                return (-abs(float(row["g_i"])), *_coordinate_key(row))

            reconstructed = _prefer_unique_targets(
                sorted(panel_rows, key=ranking_key), quota=len(panel_rows)
            )
            if panel_rows != reconstructed:
                raise ValueError("Stage 1G selected panel ordering differs")
            membership_totals[panel] += len(panel_rows)
            shortfall_totals[panel] += quota - len(panel_rows)
            memberships_expected.extend(
                {
                    "prompt_id": prompt["id"],
                    "panel": panel,
                    "rank": rank,
                    "pair_id": row["pair_id"],
                }
                for rank, row in enumerate(panel_rows, start=1)
            )
        if audit.get("memberships") != memberships_expected:
            raise ValueError("Stage 1G panel membership table differs")
        if audit.get("selected_membership_count_by_panel") != {
            panel: len(by_panel[panel]) for panel in PANEL_ORDER
        }:
            raise ValueError("Stage 1G panel membership count differs")
        if audit.get("shortfall_by_panel") != {
            panel: int(config["panels"][panel]["quota"]) - len(by_panel[panel])
            for panel in PANEL_ORDER
        }:
            raise ValueError("Stage 1G panel shortfall differs")
        if audit.get("execution_pair_count") != len(rows):
            raise ValueError("Stage 1G execution deduplication count differs")
    totals = prediction.get("selection_totals")
    if not isinstance(totals, dict) or totals != {
        "eligible_prompt_count": len(prompts),
        "execution_pair_count": len(all_pairs),
        "membership_count_by_panel": membership_totals,
        "shortfall_by_panel": shortfall_totals,
    }:
        raise ValueError("Stage 1G prediction totals differ")
    return all_pairs


def _matching_baseline(
    pair: dict[str, Any], states: dict[FeatureRef, Any], *, allow_active_target: bool
) -> tuple[Any, Any]:
    source = FeatureRef(**pair["source"])
    target = FeatureRef(**pair["target"])
    source_state = states[source]
    target_state = states[target]
    if (
        source_state.activity is not FeatureActivity.ACTIVE
        or source_state.activation <= 0.0
    ):
        raise RuntimeError("Stage 1G baseline source is not active")
    if (
        not allow_active_target
        and target_state.activity is not FeatureActivity.INACTIVE
    ):
        raise RuntimeError("Stage 1G scientific target is not baseline inactive")
    if not allow_active_target and target_state.preactivation > target_state.threshold:
        raise RuntimeError("Stage 1G scientific target violates strict inactive gate")
    if "source_activation" in pair and source_state.activation != float(
        pair["source_activation"]
    ):
        raise RuntimeError("Stage 1G source baseline differs from prediction")
    if "target_preactivation" in pair and target_state.preactivation != float(
        pair["target_preactivation"]
    ):
        raise RuntimeError("Stage 1G target baseline z differs from prediction")
    if "target_threshold" in pair and target_state.threshold != float(
        pair["target_threshold"]
    ):
        raise RuntimeError("Stage 1G target baseline threshold differs from prediction")
    return source_state, target_state


def _run_condition(
    backend: Any,
    journal: CanonicalExecutionJournal,
    sampler: Any,
    pair: dict[str, Any],
    *,
    source_state: Any,
    target_state: Any,
    condition: str,
    desired_source_activation: float | None,
    desired_target_activation: float | None,
    stage: str,
) -> dict[str, Any]:
    with sampler.stage(stage):
        point = backend.measure_condition(
            pair,
            condition=condition,
            desired_source_activation=desired_source_activation,
            desired_target_activation=desired_target_activation,
            stage=stage,
        )
    point.update(
        {
            "pair_id": pair["pair_id"],
            "prompt_id": pair["prompt_id"],
            "method_memberships": list(pair.get("method_memberships", [])),
            "panel_ranks": dict(pair.get("panel_ranks", {})),
            "baseline_source_activation": source_state.activation,
            "baseline_target_preactivation": target_state.preactivation,
            "baseline_target_threshold": target_state.threshold,
            "baseline_target_activation": target_state.activation,
            "baseline_target_active": target_state.activity is FeatureActivity.ACTIVE,
            "q": pair.get("q"),
            "g_i": pair.get("g_i"),
            "predicted_target_activation": pair.get("predicted_target_activation"),
            "predicted_signed_mediation": pair.get("predicted_signed_mediation"),
        }
    )
    detached = detach_json(point)
    if not isinstance(detached, dict):
        raise RuntimeError("Stage 1G condition did not detach")
    journal.append_completed_point(detached)
    return detached


def execute_frozen_pairs(
    *,
    model: Any,
    torch: Any,
    sampler: Any,
    prompts: list[dict[str, Any]],
    journal: CanonicalExecutionJournal,
    backend_factory: Callable[..., Any] = Stage1GInterventionBackend,
    allow_active_target: bool = False,
    force_all_conditions: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Execute baseline/repeat/ablation and conditional clamp/injection once."""

    sweeps: list[dict[str, Any]] = []
    global_calls = 0
    for prompt in prompts:
        backend = backend_factory(
            model,
            prompt=str(prompt["text"]),
            prompt_id=str(prompt["id"]),
            torch=torch,
            answer_token_id=int(prompt["answer_token_id"]),
            contrast_token_id=int(prompt["contrast_token_id"]),
            token_count=len(prompt["token_ids"]),
            attempt_recorder=journal.before_source_suppression,
            call_index_offset=global_calls,
        )
        for pair in prompt["execution_pairs"]:
            source = FeatureRef(**pair["source"])
            target = FeatureRef(**pair["target"])
            short = str(pair["pair_id"])[:12]
            with sampler.stage(f"{prompt['id']}_{short}_baseline_remeasurement"):
                states = backend.measure_states((source, target))
            source_state, target_state = _matching_baseline(
                pair, states, allow_active_target=allow_active_target
            )
            points: list[dict[str, Any]] = []
            for condition in ("baseline_noop", "baseline_repeat"):
                points.append(
                    _run_condition(
                        backend,
                        journal,
                        sampler,
                        pair,
                        source_state=source_state,
                        target_state=target_state,
                        condition=condition,
                        desired_source_activation=source_state.activation,
                        desired_target_activation=None,
                        stage=f"{prompt['id']}_{short}_{condition}",
                    )
                )
            full = _run_condition(
                backend,
                journal,
                sampler,
                pair,
                source_state=source_state,
                target_state=target_state,
                condition="source_full_ablation",
                desired_source_activation=0.0,
                desired_target_activation=None,
                stage=f"{prompt['id']}_{short}_source_full_ablation",
            )
            points.append(full)
            scientific_panels = any(
                panel in pair.get("method_memberships", []) for panel in ("B", "Q", "G")
            )
            if (
                bool(full["target_active"]) and scientific_panels
            ) or force_all_conditions:
                observed_amplitude = float(full["target_natural_activation"])
                if not math.isfinite(observed_amplitude) or observed_amplitude <= 0.0:
                    raise RuntimeError("Stage 1G crossing target amplitude is invalid")
                points.append(
                    _run_condition(
                        backend,
                        journal,
                        sampler,
                        pair,
                        source_state=source_state,
                        target_state=target_state,
                        condition="source_ablation_target_clamp",
                        desired_source_activation=0.0,
                        desired_target_activation=0.0,
                        stage=f"{prompt['id']}_{short}_source_ablation_target_clamp",
                    )
                )
                points.append(
                    _run_condition(
                        backend,
                        journal,
                        sampler,
                        pair,
                        source_state=source_state,
                        target_state=target_state,
                        condition="target_only_injection",
                        desired_source_activation=None,
                        desired_target_activation=observed_amplitude,
                        stage=f"{prompt['id']}_{short}_target_only_injection",
                    )
                )
            if {point["condition"] for point in points} not in (
                {"baseline_noop", "baseline_repeat", "source_full_ablation"},
                {
                    "baseline_noop",
                    "baseline_repeat",
                    "source_full_ablation",
                    "source_ablation_target_clamp",
                    "target_only_injection",
                },
            ):
                raise RuntimeError("Stage 1G condition set differs")
            sweeps.append(
                {
                    "prompt_id": prompt["id"],
                    "pair_id": pair["pair_id"],
                    "method_memberships": list(pair.get("method_memberships", [])),
                    "point_count": len(points),
                    "points": points,
                }
            )
            global_calls = backend.source_suppression_api_calls
            del states, points
            gc.collect()
            torch.mps.empty_cache()
    journal.verify_complete(expected_point_count=global_calls)
    return sweeps, global_calls


def read_completed_journal(path: Path) -> list[dict[str, Any]]:
    """Read only complete start/point pairs from the durable canonical journal."""

    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > 32 * 1024 * 1024
    ):
        raise ValueError("Stage 1G journal is missing, unsafe, or oversized")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or len(lines) % 2:
        raise ValueError("Stage 1G journal lacks complete record pairs")
    points: list[dict[str, Any]] = []
    for expected, (left, right) in enumerate(
        zip(lines[::2], lines[1::2], strict=True), start=1
    ):
        started = json.loads(left)
        completed = json.loads(right)
        if (
            type(started) is not dict
            or type(completed) is not dict
            or set(started) != {"record_type", "call_index", "pair_id"}
            or started.get("record_type") != "source_suppression_call_started"
            or started.get("call_index") != expected
            or set(completed) != {"record_type", "call_index", "pair_id", "point"}
            or completed.get("record_type") != "point_completed"
            or completed.get("call_index") != expected
            or completed.get("pair_id") != started.get("pair_id")
            or type(completed.get("point")) is not dict
            or completed["point"].get("pair_id") != started.get("pair_id")
            or completed["point"].get("source_suppression_api_call_index") != expected
        ):
            raise ValueError("Stage 1G journal ordering or identity differs")
        points.append(cast(dict[str, Any], completed["point"]))
    return points


def group_points(
    prediction: dict[str, Any], points: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    pairs = {
        pair["pair_id"]: pair
        for prompt in prediction["prompts"]
        for pair in prompt["execution_pairs"]
    }
    grouped: dict[str, list[dict[str, Any]]] = {pair_id: [] for pair_id in pairs}
    for expected, point in enumerate(points, start=1):
        pair_id = point.get("pair_id")
        if (
            pair_id not in grouped
            or point.get("source_suppression_api_call_index") != expected
        ):
            raise ValueError("Stage 1G journal contains a non-frozen pair or call gap")
        grouped[str(pair_id)].append(point)
    if any(not rows for rows in grouped.values()):
        raise ValueError("a Stage 1G frozen pair lacks completed points")
    return pairs, grouped


def _condition_map(points: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for point in points:
        condition = point.get("condition")
        if not isinstance(condition, str) or condition in result:
            raise ValueError("Stage 1G condition is missing or duplicated")
        result[condition] = point
    return result


def validate_serialized_point(
    point: dict[str, Any], pair: dict[str, Any], *, expected_call_index: int
) -> None:
    """Validate exact activation mappings and strict loaded-gate semantics."""

    condition = point.get("condition")
    if condition not in {
        "baseline_noop",
        "baseline_repeat",
        "source_full_ablation",
        "source_ablation_target_clamp",
        "target_only_injection",
    }:
        raise ValueError("Stage 1G point condition differs")
    memberships = pair["method_memberships"]
    if (
        point.get("source_suppression_api_call_index") != expected_call_index
        or point.get("pair_id") != pair["pair_id"]
        or point.get("prompt_id") != pair["prompt_id"]
        or point.get("method_memberships") != memberships
        or point.get("panel_ranks") != pair["panel_ranks"]
        or point.get("answer_token_id") != pair["answer_token_id"]
        or point.get("contrast_token_id") != pair["contrast_token_id"]
        or point.get("loaded_gate") != "a=z*1[z>tau]"
        or point.get("threshold_equality_activity") != "inactive"
        or point.get("freeze_attention") is not True
        or point.get("constrained_layers") is not None
        or point.get("logits_finite") is not True
        or point.get("preactivation_cache_persisted") is not False
        or point.get("full_logits_persisted") is not False
    ):
        raise ValueError("Stage 1G point identity or safety evidence differs")
    baseline_source = _finite(pair["source_activation"], "pair source activation")
    baseline_z = _finite(pair["target_preactivation"], "pair target z")
    threshold = _finite(point.get("target_threshold"), "point threshold")
    z_value = _finite(point.get("target_preactivation"), "point target z")
    natural = _finite(point.get("target_natural_activation"), "point target activation")
    effective = _finite(
        point.get("target_effective_activation"), "point effective activation"
    )
    answer = _finite(point.get("answer_logit"), "answer logit")
    contrast = _finite(point.get("contrast_logit"), "contrast logit")
    behavior = _finite(point.get("behavior_T"), "behavior T")
    elapsed = _finite(point.get("point_elapsed_seconds"), "point elapsed")
    active = z_value > threshold
    expected_natural = z_value if active else 0.0
    if (
        elapsed < 0.0
        or threshold != _finite(pair["target_threshold"], "pair threshold")
        or point.get("target_active") is not active
        or point.get("strict_crossing") is not active
        or natural != expected_natural
        or behavior != answer - contrast
        or _finite(point.get("baseline_source_activation"), "baseline source")
        != baseline_source
        or _finite(point.get("baseline_target_preactivation"), "baseline z")
        != baseline_z
        or _finite(point.get("baseline_target_threshold"), "baseline threshold")
        != threshold
        or point.get("baseline_target_active") is not False
    ):
        raise ValueError("Stage 1G point finite, baseline, or gate evidence differs")
    desired_source = point.get("desired_source_activation")
    actual_source = point.get("actual_bf16_source_activation")
    desired_target = point.get("desired_target_activation")
    actual_target = point.get("actual_bf16_target_activation")
    expected_edits = {
        "baseline_noop": (baseline_source, None),
        "baseline_repeat": (baseline_source, None),
        "source_full_ablation": (0.0, None),
        "source_ablation_target_clamp": (0.0, 0.0),
        "target_only_injection": (None, "observed"),
    }[str(condition)]
    if desired_source != expected_edits[0]:
        raise ValueError("Stage 1G desired source mapping differs")
    if desired_source is None:
        if actual_source is not None or point.get("source_value_device") is not None:
            raise ValueError("Stage 1G absent source edit has tensor evidence")
    elif (
        _finite(actual_source, "actual source") != bf16_round(float(desired_source))
        or point.get("source_value_device") != "mps:0"
        or point.get("source_value_dtype") != "torch.bfloat16"
    ):
        raise ValueError("Stage 1G BF16 source mapping differs")
    if expected_edits[1] is None:
        if desired_target is not None or actual_target is not None:
            raise ValueError("Stage 1G unexpected target edit")
    else:
        target_desired = _finite(desired_target, "desired target")
        if expected_edits[1] == 0.0 and target_desired != 0.0:
            raise ValueError("Stage 1G target clamp differs")
        if (
            _finite(actual_target, "actual target") != bf16_round(target_desired)
            or point.get("target_value_device") != "mps:0"
            or point.get("target_value_dtype") != "torch.bfloat16"
        ):
            raise ValueError("Stage 1G BF16 target mapping differs")
    expected_effective = natural if actual_target is None else float(actual_target)
    if effective != expected_effective:
        raise ValueError("Stage 1G effective target activation differs")
    for key in (
        "q",
        "g_i",
        "predicted_target_activation",
        "predicted_signed_mediation",
    ):
        if point.get(key) != pair.get(key):
            raise ValueError(f"Stage 1G point {key} differs from freeze")


def _validate_schedules(
    pairs: dict[str, dict[str, Any]], grouped: dict[str, list[dict[str, Any]]]
) -> None:
    for pair_id, pair in pairs.items():
        conditions = _condition_map(grouped[pair_id])
        required = {"baseline_noop", "baseline_repeat", "source_full_ablation"}
        scientific = any(
            panel in pair["method_memberships"] for panel in ("B", "Q", "G")
        )
        crossing = bool(conditions["source_full_ablation"]["target_active"])
        if scientific and crossing:
            required |= {"source_ablation_target_clamp", "target_only_injection"}
        if set(conditions) != required:
            raise ValueError("Stage 1G frozen conditional schedule differs")
        baseline = conditions["baseline_noop"]
        repeat = conditions["baseline_repeat"]
        full = conditions["source_full_ablation"]
        if (
            float(baseline["target_preactivation"])
            != float(pair["target_preactivation"])
            or float(repeat["target_preactivation"])
            != float(pair["target_preactivation"])
            or float(full["actual_bf16_source_activation"]) != 0.0
        ):
            raise ValueError("Stage 1G baseline repeat or full ablation differs")
        if scientific and crossing:
            clamp = conditions["source_ablation_target_clamp"]
            injection = conditions["target_only_injection"]
            amplitude = float(full["target_natural_activation"])
            if (
                amplitude <= 0.0
                or float(clamp["actual_bf16_source_activation"]) != 0.0
                or float(clamp["actual_bf16_target_activation"]) != 0.0
                or float(injection["actual_bf16_target_activation"]) != amplitude
                or float(clamp["target_preactivation"])
                != float(full["target_preactivation"])
                or float(injection["target_preactivation"])
                != float(pair["target_preactivation"])
            ):
                raise ValueError("Stage 1G clamp/injection schedule differs")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pair_effect(
    pair: dict[str, Any], points: Sequence[dict[str, Any]], floor: float
) -> dict[str, Any]:
    conditions = _condition_map(points)
    baseline = conditions["baseline_noop"]
    repeat = conditions["baseline_repeat"]
    full = conditions["source_full_ablation"]
    crossing = bool(full["target_active"])
    scientific = any(panel in pair["method_memberships"] for panel in ("B", "Q", "G"))
    source_effect = float(full["behavior_T"]) - float(baseline["behavior_T"])
    mediation: float | None = None
    injection: float | None = None
    oracle: float | None = None
    fraction: float | None = None
    if crossing and scientific:
        clamp = conditions["source_ablation_target_clamp"]
        injected = conditions["target_only_injection"]
        mediation = float(full["behavior_T"]) - float(clamp["behavior_T"])
        injection = float(injected["behavior_T"]) - float(baseline["behavior_T"])
        oracle = float(pair["g_i"]) * float(full["target_natural_activation"])
        if abs(source_effect) > 10.0 * floor:
            fraction = mediation / source_effect
    predicted = float(pair["predicted_signed_mediation"])
    return {
        "prompt_id": pair["prompt_id"],
        "pair_id": pair["pair_id"],
        "method_memberships": list(pair["method_memberships"]),
        "panel_ranks": dict(pair["panel_ranks"]),
        "observed_crossing": crossing,
        "baseline_behavior_T": float(baseline["behavior_T"]),
        "baseline_repeat_behavior_T": float(repeat["behavior_T"]),
        "no_op_absolute_error": abs(
            float(repeat["behavior_T"]) - float(baseline["behavior_T"])
        ),
        "source_ablation_behavior_T": float(full["behavior_T"]),
        "source_effect": source_effect,
        "mediation_effect_M": mediation,
        "target_injection_effect_I": injection,
        "predicted_signed_mediation": predicted,
        "oracle_amplitude_prediction": oracle,
        "above_no_effect_floor": (mediation is not None and abs(mediation) > floor),
        "prospective_sign_correct": (
            mediation is not None
            and abs(mediation) > floor
            and _sign(predicted) == _sign(mediation)
        ),
        "injection_sign_agrees_with_M": (
            mediation is not None
            and injection is not None
            and _sign(mediation) == _sign(injection)
        ),
        "clamp_reduces_absolute_source_effect": (
            mediation is not None
            and abs(
                float(conditions["source_ablation_target_clamp"]["behavior_T"])
                - float(baseline["behavior_T"])
            )
            < abs(source_effect)
        ),
        "mediation_fraction": fraction,
        "target_amplitude_under_source_ablation": (
            float(full["target_natural_activation"]) if crossing else 0.0
        ),
        "directional_movement_toward_gate_violation": (
            "D" in pair["method_memberships"]
            and float(full["target_preactivation"])
            > float(pair["target_preactivation"]) + 1e-9
        ),
        "point_count": len(points),
    }


def _panel_prompt_stat(
    effects: Sequence[dict[str, Any]], prompt_id: str, panel: str
) -> dict[str, float]:
    rows = [
        row
        for row in effects
        if row["prompt_id"] == prompt_id
        and panel in row["method_memberships"]
        and row["observed_crossing"]
        and row["mediation_effect_M"] is not None
    ]
    absolute = [abs(float(row["mediation_effect_M"])) for row in rows]
    return {
        "mean_abs_M": _mean(absolute),
        "median_abs_M": float(median(absolute)) if absolute else 0.0,
        "above_floor_fraction": _mean(
            [float(bool(row["above_no_effect_floor"])) for row in rows]
        ),
        "clamp_reduction_fraction": _mean(
            [float(bool(row["clamp_reduces_absolute_source_effect"])) for row in rows]
        ),
        "crossing_membership_count": float(len(rows)),
    }


def _bootstrap_metrics(
    prompt_ids: Sequence[str],
    prompt_stats: Mapping[str, Mapping[str, Mapping[str, float]]],
    effects: Sequence[dict[str, Any]],
    *,
    count: int,
    seed: str,
) -> dict[str, Any]:
    generator = random.Random(int(hashlib.sha256(seed.encode()).hexdigest(), 16))
    differences: dict[str, list[float]] = {
        f"B_minus_{other}_{metric}": []
        for other in ("Q", "G")
        for metric in (
            "mean_abs_M",
            "median_abs_M",
            "above_floor_fraction",
            "clamp_reduction_fraction",
        )
    }
    sign_accuracy: list[float] = []
    for _ in range(count):
        sampled = [prompt_ids[generator.randrange(len(prompt_ids))] for _ in prompt_ids]
        for other in ("Q", "G"):
            for metric in (
                "mean_abs_M",
                "median_abs_M",
                "above_floor_fraction",
                "clamp_reduction_fraction",
            ):
                differences[f"B_minus_{other}_{metric}"].append(
                    _mean(
                        [
                            prompt_stats[prompt_id]["B"][metric]
                            - prompt_stats[prompt_id][other][metric]
                            for prompt_id in sampled
                        ]
                    )
                )
        per_prompt_accuracy: list[float] = []
        for prompt_id in sampled:
            rows = [
                row
                for row in effects
                if row["prompt_id"] == prompt_id
                and "B" in row["method_memberships"]
                and row["observed_crossing"]
                and row["above_no_effect_floor"]
            ]
            per_prompt_accuracy.append(
                _mean([float(bool(row["prospective_sign_correct"])) for row in rows])
            )
        sign_accuracy.append(_mean(per_prompt_accuracy))

    def interval(values: Sequence[float]) -> dict[str, Any]:
        return {
            "lower": _percentile(values, 0.025),
            "upper": _percentile(values, 0.975),
            "requested_resamples": count,
            "defined_resamples": len(values),
        }

    return {
        "panel_differences": {
            key: interval(values) for key, values in sorted(differences.items())
        },
        "B_prospective_sign_accuracy": interval(sign_accuracy),
    }


def classify_terminal(
    *,
    eligible_prompt_count: int,
    B_crossing_count: int,
    B_crossing_prompt_count: int,
    B_sign_accuracy: float,
    B_sign_bootstrap_lower: float,
    B_minus_Q_mean_abs_M: float,
    B_minus_Q_bootstrap_lower: float,
    B_above_floor_fraction: float,
    B_injection_sign_agreement: float,
    directional_violation_fraction: float,
    rules: Mapping[str, Any],
) -> tuple[str, str, dict[str, bool]]:
    checks = {
        "minimum_eligible_prompts": eligible_prompt_count
        >= int(rules["minimum_eligible_prompts"]),
        "minimum_B_crossing_memberships_and_prompts": (
            B_crossing_count >= int(rules["minimum_B_crossing_memberships"])
            and B_crossing_prompt_count >= int(rules["minimum_B_crossing_prompts"])
        ),
        "B_prospective_sign_accuracy": (
            B_sign_accuracy >= float(rules["B_sign_accuracy_minimum"])
            and B_sign_bootstrap_lower
            > float(rules["B_sign_bootstrap_lower_strictly_greater_than"])
        ),
        "B_minus_Q_prompt_mean_abs_M": (
            B_minus_Q_mean_abs_M > 0.0
            and B_minus_Q_bootstrap_lower
            > float(
                rules[
                    "B_minus_Q_mean_abs_mediation_bootstrap_lower_strictly_greater_than"
                ]
            )
        ),
        "B_above_no_effect_floor_fraction": B_above_floor_fraction
        >= float(rules["B_above_floor_fraction_minimum"]),
        "B_target_injection_sign_agreement": B_injection_sign_agreement
        >= float(rules["B_injection_sign_agreement_minimum"]),
        "directional_movement_toward_gate_violation": directional_violation_fraction
        <= float(rules["directional_movement_toward_gate_violation_fraction_maximum"]),
    }
    powered = (
        checks["minimum_eligible_prompts"]
        and checks["minimum_B_crossing_memberships_and_prompts"]
    )
    passed = sum(checks.values())
    if not powered:
        return (
            "underpowered_behavioral_mediation_pilot",
            "report_why_do_not_expand_frozen_prompt_pool",
            checks,
        )
    if passed == int(rules["supported_criteria_count"]):
        return (
            "supported_behavioral_mediation_pilot",
            "proceed_to_reference_model_or_clt_replication_before_paper_claim",
            checks,
        )
    if passed >= int(rules["mixed_minimum_supported_criteria"]):
        return (
            "mixed_behavioral_mediation_pilot",
            "inspect_feature_group_or_split_mediation_before_scaling",
            checks,
        )
    return (
        "not_supported_behavioral_mediation_pilot",
        "stop_behavioral_mediation_claim_and_reconsider_paper_contribution",
        checks,
    )


def compute_analysis(
    prediction: dict[str, Any],
    grouped: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    prompt_ids = [str(prompt["id"]) for prompt in prediction["prompts"]]
    pair_map = {
        pair["pair_id"]: pair
        for prompt in prediction["prompts"]
        for pair in prompt["execution_pairs"]
    }
    floors: dict[str, float] = {}
    for prompt_id in prompt_ids:
        errors = [
            abs(
                float(_condition_map(grouped[pair_id])["baseline_repeat"]["behavior_T"])
                - float(_condition_map(grouped[pair_id])["baseline_noop"]["behavior_T"])
            )
            for pair_id, pair in pair_map.items()
            if pair["prompt_id"] == prompt_id
        ]
        floors[prompt_id] = max(1e-4, 5.0 * max(errors, default=0.0))
    effects = [
        _pair_effect(pair, grouped[pair_id], floors[str(pair["prompt_id"])])
        for pair_id, pair in pair_map.items()
    ]
    prompt_stats: dict[str, dict[str, dict[str, float]]] = {
        prompt_id: {
            panel: _panel_prompt_stat(effects, prompt_id, panel)
            for panel in ("B", "Q", "G")
        }
        for prompt_id in prompt_ids
    }
    panel_metrics: dict[str, Any] = {}
    for panel in PANEL_ORDER:
        memberships = [row for row in effects if panel in row["method_memberships"]]
        crossings = [row for row in memberships if row["observed_crossing"]]
        analyzable = [row for row in crossings if row["mediation_effect_M"] is not None]
        panel_metrics[panel] = {
            "membership_count": len(memberships),
            "crossing_count": len(crossings),
            "crossing_prompt_count": len({row["prompt_id"] for row in crossings}),
            "crossing_rate": len(crossings) / len(memberships) if memberships else 0.0,
            "analyzable_mediation_count": len(analyzable),
            "mean_abs_M": _mean(
                [abs(float(row["mediation_effect_M"])) for row in analyzable]
            ),
            "median_abs_M": (
                float(
                    median(abs(float(row["mediation_effect_M"])) for row in analyzable)
                )
                if analyzable
                else 0.0
            ),
            "above_floor_fraction": _mean(
                [float(bool(row["above_no_effect_floor"])) for row in analyzable]
            ),
            "clamp_reduction_fraction": _mean(
                [
                    float(bool(row["clamp_reduces_absolute_source_effect"]))
                    for row in analyzable
                ]
            ),
        }
    bootstrap = _bootstrap_metrics(
        prompt_ids,
        prompt_stats,
        effects,
        count=int(config["metrics"]["bootstrap_resamples"]),
        seed=str(config["metrics"]["bootstrap_seed"]),
    )
    B_crossings = [
        row
        for row in effects
        if "B" in row["method_memberships"] and row["observed_crossing"]
    ]
    B_sign_rows = [row for row in B_crossings if row["above_no_effect_floor"]]
    sign_success = sum(bool(row["prospective_sign_correct"]) for row in B_sign_rows)
    sign_accuracy = sign_success / len(B_sign_rows) if B_sign_rows else 0.0
    injection_rows = [
        row for row in B_crossings if row["target_injection_effect_I"] is not None
    ]
    injection_success = sum(
        bool(row["injection_sign_agrees_with_M"]) for row in injection_rows
    )
    injection_accuracy = (
        injection_success / len(injection_rows) if injection_rows else 0.0
    )
    directional = [row for row in effects if "D" in row["method_memberships"]]
    directional_violations = sum(
        bool(row["directional_movement_toward_gate_violation"]) for row in directional
    )
    directional_fraction = (
        directional_violations / len(directional) if directional else 0.0
    )
    prompt_differences: dict[str, dict[str, float]] = {}
    for other in ("Q", "G"):
        prompt_differences[f"B_minus_{other}"] = {
            metric: _mean(
                [
                    prompt_stats[prompt_id]["B"][metric]
                    - prompt_stats[prompt_id][other][metric]
                    for prompt_id in prompt_ids
                ]
            )
            for metric in (
                "mean_abs_M",
                "median_abs_M",
                "above_floor_fraction",
                "clamp_reduction_fraction",
            )
        }
    mediation_rows = [row for row in effects if row["mediation_effect_M"] is not None]
    observed_M = [float(row["mediation_effect_M"]) for row in mediation_rows]
    injection_M = [float(row["target_injection_effect_I"]) for row in mediation_rows]
    predicted_M = [float(row["predicted_signed_mediation"]) for row in mediation_rows]
    oracle_M = [float(row["oracle_amplitude_prediction"]) for row in mediation_rows]
    lower = float(bootstrap["B_prospective_sign_accuracy"]["lower"])
    BQ = prompt_differences["B_minus_Q"]["mean_abs_M"]
    BQ_lower = float(bootstrap["panel_differences"]["B_minus_Q_mean_abs_M"]["lower"])
    terminal, decision, checks = classify_terminal(
        eligible_prompt_count=len(prompt_ids),
        B_crossing_count=len(B_crossings),
        B_crossing_prompt_count=len({row["prompt_id"] for row in B_crossings}),
        B_sign_accuracy=sign_accuracy,
        B_sign_bootstrap_lower=lower,
        B_minus_Q_mean_abs_M=BQ,
        B_minus_Q_bootstrap_lower=BQ_lower,
        B_above_floor_fraction=panel_metrics["B"]["above_floor_fraction"],
        B_injection_sign_agreement=injection_accuracy,
        directional_violation_fraction=directional_fraction,
        rules=config["decision_rule"],
    )
    return {
        "schema_version": 1,
        "artifact_type": "stage1g_analysis_summary",
        "status": "passed",
        "experiment_class": EXPERIMENT_CLASS,
        "terminal_class": terminal,
        "project_decision": decision,
        "eligible_prompt_count": len(prompt_ids),
        "unique_execution_pair_count": len(effects),
        "no_effect_floor_by_prompt": floors,
        "panel_metrics": panel_metrics,
        "prompt_panel_metrics": prompt_stats,
        "prompt_level_panel_differences": prompt_differences,
        "bootstrap": bootstrap,
        "prospective_sign": {
            "eligible_above_floor_count": len(B_sign_rows),
            "success_count": sign_success,
            "accuracy": sign_accuracy,
            "exact_binomial_95_interval": clopper_pearson(
                sign_success, len(B_sign_rows)
            ),
        },
        "necessity_and_sufficiency": {
            "B_injection_denominator": len(injection_rows),
            "B_injection_sign_successes": injection_success,
            "B_injection_sign_agreement": injection_accuracy,
            "B_injection_exact_binomial_95_interval": clopper_pearson(
                injection_success, len(injection_rows)
            ),
            "M_vs_I_spearman": spearman(observed_M, injection_M),
            "predicted_M_vs_observed_M_spearman": spearman(predicted_M, observed_M),
            "predicted_M_median_symmetric_normalized_error": (
                float(
                    median(
                        symmetric_normalized_error(a, b)
                        for a, b in zip(predicted_M, observed_M, strict=True)
                    )
                )
                if observed_M
                else None
            ),
            "oracle_amplitude_diagnostic_M_vs_observed_M_spearman": spearman(
                oracle_M, observed_M
            ),
            "oracle_amplitude_diagnostic_median_symmetric_normalized_error": (
                float(
                    median(
                        symmetric_normalized_error(a, b)
                        for a, b in zip(oracle_M, observed_M, strict=True)
                    )
                )
                if observed_M
                else None
            ),
        },
        "directional_controls": {
            "membership_count": len(directional),
            "movement_toward_gate_violation_count": directional_violations,
            "movement_toward_gate_violation_fraction": directional_fraction,
        },
        "coverage": {
            "panel_shortfalls": prediction["selection_totals"]["shortfall_by_panel"],
            "noncrossing_membership_count_by_panel": {
                panel: panel_metrics[panel]["membership_count"]
                - panel_metrics[panel]["crossing_count"]
                for panel in PANEL_ORDER
            },
            "missing_condition_count": 0,
            "quantization_collapse_count": 0,
        },
        "supported_criteria": checks,
        "supported_criteria_passed": sum(checks.values()),
        "pair_effects": effects,
        "claim_boundary": config["claim_boundary"],
    }


def build_records(
    *,
    protocol: dict[str, Any],
    sensitivity: dict[str, Any],
    prediction: dict[str, Any],
    worker: dict[str, Any],
    points: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    pairs, grouped = group_points(prediction, points)
    for index, point in enumerate(points, start=1):
        validate_serialized_point(
            point, pairs[str(point["pair_id"])], expected_call_index=index
        )
    _validate_schedules(pairs, grouped)
    analysis = compute_analysis(prediction, grouped, config)
    point_record = {
        "schema_version": 1,
        "artifact_type": "stage1g_point_records",
        "status": "passed",
        "experiment_class": EXPERIMENT_CLASS,
        "point_count": len(points),
        "sweep_count": len(grouped),
        "sweeps": [
            {
                "prompt_id": pair["prompt_id"],
                "pair_id": pair_id,
                "method_memberships": list(pair["method_memberships"]),
                "panel_ranks": dict(pair["panel_ranks"]),
                "point_count": len(grouped[pair_id]),
                "points": grouped[pair_id],
            }
            for pair_id, pair in pairs.items()
        ],
    }
    telemetry = worker["telemetry"]
    environment = {
        "schema_version": 1,
        "artifact_type": "stage1g_environment_manifest",
        "status": "passed",
        "experiment_class": EXPERIMENT_CLASS,
        "runtime": worker["environment"],
        "runtime_evidence": worker["runtime_evidence"],
        "telemetry": {
            "started_at_unix": telemetry["started_at_unix"],
            "finished_at_unix": telemetry["finished_at_unix"],
            "sample_count": telemetry["sample_count"],
            "sampling_interval_seconds": telemetry["sampling_interval_seconds"],
            "attempt_peaks": telemetry["attempt_peaks"],
            "thermal_states": telemetry["thermal_states"],
            "violations": telemetry["violations"],
            "telemetry_failures": telemetry["telemetry_failures"],
        },
        "privacy": {
            "network_accessed": False,
            "credential_values_read": False,
            "secret_values_recorded": False,
            "private_paths_recorded": False,
            "gradient_tensors_recorded": False,
            "full_logits_recorded": False,
        },
    }
    run = {
        "schema_version": 1,
        "artifact_type": "stage1g_run_manifest",
        "status": analysis["terminal_class"],
        "project_decision": analysis["project_decision"],
        "experiment_class": EXPERIMENT_CLASS,
        "branch": BRANCH,
        "base_commit": BASE_COMMIT,
        "protocol_freeze_commit": protocol["protocol_commit"],
        "prediction_freeze_commit": worker["prediction_freeze_commit"],
        "execution_commit": worker["pre_run_commit"],
        "canonical_attempt_count": 1,
        "scientific_retry_count": 0,
        "instrumented_intervention_api_calls": len(points),
        "journal_completed_point_count": len(points),
        "serialized_point_count": len(points),
        "final_artifacts_rebuilt_from_journal_in_fresh_process": True,
        "standalone_recomputation_required": True,
        "historical_stage1f_terminal_class": "completed_stage1f_e1_not_supported",
        "simple_critical_alpha_calibration": "retired",
        "claim_boundary": config["claim_boundary"],
    }
    return {
        "protocol_manifest.json": protocol,
        "output_sensitivity_validation.json": sensitivity,
        "prediction_manifest.json": prediction,
        "point_records.json": point_record,
        "analysis_summary.json": analysis,
        "run_manifest.json": run,
        "environment_manifest.json": environment,
    }


def publish_records(output: Path, records: dict[str, dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise ValueError("Stage 1G artifact output is a symlink")
    for name in JSON_ARTIFACTS:
        path = output / name
        value = records[name]
        if path.exists():
            if read_json_strict(path) != value:
                raise ValueError(f"existing frozen Stage 1G artifact differs: {name}")
        else:
            write_json_new(path, value)
    checksum = output / "checksums.sha256"
    if checksum.exists():
        raise ValueError("Stage 1G checksum sidecar already exists")
    encoded = (
        "\n".join(
            f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}"
            for name in JSON_ARTIFACTS
        )
        + "\n"
    ).encode("ascii")
    descriptor = os.open(
        checksum,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        if os.write(descriptor, encoded) != len(encoded):
            raise ValueError("short Stage 1G checksum write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_bundle(repository: Path, output: Path) -> dict[str, Any]:
    """Hostile-input validation and fresh serialized-point recomputation."""

    config = load_config(repository / CONFIG_PATH)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("Stage 1G bundle directory is unsafe")
    names = {path.name for path in output.iterdir()}
    if names != {*JSON_ARTIFACTS, "checksums.sha256"}:
        raise ValueError("Stage 1G bundle file allowlist differs")
    maximum_file = int(config["safety_limits"]["maximum_artifact_file_bytes"])
    maximum_bundle = int(config["safety_limits"]["maximum_artifact_bundle_bytes"])
    total_size = 0
    for path in output.iterdir():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > maximum_file
        ):
            raise ValueError("Stage 1G artifact file is unsafe or oversized")
        total_size += path.stat().st_size
    if total_size > maximum_bundle:
        raise ValueError("Stage 1G artifact bundle is oversized")
    expected_checksums = [
        f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}"
        for name in JSON_ARTIFACTS
    ]
    if (output / "checksums.sha256").read_text(
        encoding="ascii"
    ).splitlines() != expected_checksums:
        raise ValueError("Stage 1G bundle checksum sidecar differs")
    records = {name: _strict_json(output / name) for name in JSON_ARTIFACTS}
    if any(type(value) is not dict for value in records.values()):
        raise ValueError("Stage 1G artifact is not an object")
    protocol = cast(dict[str, Any], records["protocol_manifest.json"])
    sensitivity = cast(dict[str, Any], records["output_sensitivity_validation.json"])
    prediction = cast(dict[str, Any], records["prediction_manifest.json"])
    validate_protocol(repository, protocol, config)
    validate_output_sensitivity(sensitivity, config)
    pairs = validate_prediction(prediction, config, protocol)
    if prediction.get("output_sensitivity_validation") != sensitivity:
        raise ValueError("Stage 1G sensitivity artifacts differ")
    point_artifact = cast(dict[str, Any], records["point_records.json"])
    sweeps = point_artifact.get("sweeps")
    if (
        point_artifact.get("schema_version") != 1
        or point_artifact.get("artifact_type") != "stage1g_point_records"
        or point_artifact.get("status") != "passed"
        or point_artifact.get("experiment_class") != EXPERIMENT_CLASS
        or not isinstance(sweeps, list)
    ):
        raise ValueError("Stage 1G point artifact identity differs")
    grouped: dict[str, list[dict[str, Any]]] = {}
    points: list[dict[str, Any]] = []
    for raw in sweeps:
        if type(raw) is not dict or not isinstance(raw.get("points"), list):
            raise ValueError("Stage 1G serialized sweep is malformed")
        sweep = cast(dict[str, Any], raw)
        pair_id = str(sweep.get("pair_id"))
        rows = cast(list[dict[str, Any]], sweep["points"])
        pair = pairs.get(pair_id)
        if (
            pair is None
            or pair_id in grouped
            or sweep.get("prompt_id") != pair["prompt_id"]
            or sweep.get("method_memberships") != pair["method_memberships"]
            or sweep.get("panel_ranks") != pair["panel_ranks"]
            or sweep.get("point_count") != len(rows)
        ):
            raise ValueError("Stage 1G serialized sweep identity differs")
        grouped[pair_id] = rows
        points.extend(rows)
    if (
        set(grouped) != set(pairs)
        or point_artifact.get("point_count") != len(points)
        or point_artifact.get("sweep_count") != len(grouped)
    ):
        raise ValueError("Stage 1G serialized pair or point count differs")
    points.sort(key=lambda row: int(row["source_suppression_api_call_index"]))
    if [row["source_suppression_api_call_index"] for row in points] != list(
        range(1, len(points) + 1)
    ):
        raise ValueError("Stage 1G serialized call indices differ")
    for index, point in enumerate(points, start=1):
        validate_serialized_point(
            point, pairs[str(point["pair_id"])], expected_call_index=index
        )
    _validate_schedules(pairs, grouped)
    recomputed = compute_analysis(prediction, grouped, config)
    if records["analysis_summary.json"] != recomputed:
        raise ValueError("Stage 1G analysis differs from standalone recomputation")
    run = cast(dict[str, Any], records["run_manifest.json"])
    if (
        run.get("status") != recomputed["terminal_class"]
        or run.get("project_decision") != recomputed["project_decision"]
        or run.get("canonical_attempt_count") != 1
        or run.get("scientific_retry_count") != 0
        or run.get("instrumented_intervention_api_calls") != len(points)
        or run.get("journal_completed_point_count") != len(points)
        or run.get("serialized_point_count") != len(points)
        or run.get("final_artifacts_rebuilt_from_journal_in_fresh_process") is not True
        or run.get("standalone_recomputation_required") is not True
        or run.get("claim_boundary") != config["claim_boundary"]
    ):
        raise ValueError("Stage 1G run manifest differs")
    environment = cast(dict[str, Any], records["environment_manifest.json"])
    telemetry = environment.get("telemetry")
    if (
        environment.get("artifact_type") != "stage1g_environment_manifest"
        or environment.get("status") != "passed"
        or not isinstance(telemetry, dict)
        or telemetry.get("violations") != []
        or telemetry.get("telemetry_failures") != 0
        or environment.get("privacy")
        != {
            "network_accessed": False,
            "credential_values_read": False,
            "secret_values_recorded": False,
            "private_paths_recorded": False,
            "gradient_tensors_recorded": False,
            "full_logits_recorded": False,
        }
    ):
        raise ValueError("Stage 1G environment or telemetry evidence differs")
    forbidden = re.compile(
        r"(?i)(/Users/|/home/|hf_[A-Za-z0-9]{20,}|BEGIN [A-Z ]*PRIVATE KEY|api[_-]?key)"
    )
    for name in JSON_ARTIFACTS:
        if forbidden.search((output / name).read_text(encoding="utf-8")):
            raise ValueError(
                "Stage 1G artifact contains a private path or secret pattern"
            )
    return {
        "status": "passed",
        "terminal_class": recomputed["terminal_class"],
        "eligible_prompt_count": recomputed["eligible_prompt_count"],
        "unique_pair_count": len(pairs),
        "instrumented_intervention_api_calls": len(points),
        "journal_completed_points": len(points),
        "serialized_points": len(points),
        "bundle_bytes": total_size,
        "standalone_recomputation": True,
    }
