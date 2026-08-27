"""Thin prospective E1 layer over the accepted Stage 1C-v4/Stage 1D core."""

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
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any, cast

from cfsus.exceptions import ScientificInputError
from cfsus.scanning.near_threshold import compare_scanner_results
from cfsus.stage1c_v3.execution_journal import CanonicalExecutionJournal
from cfsus.stage1c_v3.intervention import (
    bisection_requested_alpha,
    plan_applied_values,
)
from cfsus.stage1c_v3.prediction import (
    build_prospective_pair,
    canonical_v3_pair_id,
    causally_eligible,
    filter_source_pool,
    filter_target_pool,
    pair_score_digest,
    prospective_pair_record,
    source_pool_digest,
    target_pool_digest,
)
from cfsus.stage1c_v3.quantization_audit import bf16_round
from cfsus.stage1c_v3.runtime import Stage1CPredictionBackend
from cfsus.stage1c_v3.serialization import read_json_strict, write_json_new
from cfsus.stage1d.execution import (
    _crossing_bracket,
    _feature,
    _matching_baseline,
    _run_plan,
)
from cfsus.stage1d.metrics import spearman
from cfsus.stage1d.protocol import sha256_file
from cfsus.stage1e.offline import estimate_e0, estimate_e1
from cfsus.types import FeatureRef

CONFIG_PATH = Path("configs/stage1f_prospective_one_probe_confirmation.json")
SCHEMA_PATH = Path("configs/stage1f_prospective_one_probe_artifact_schema.json")
EXPERIMENT_CLASS = "stage1f_prospective_one_probe_confirmation"
BRANCH = "stage-1f-prospective-one-probe-confirmation"
BASE_COMMIT = "f7aae1f3ce3b1b8d98e850093a3cb5ca480277ea"
REQUIRED_ANCESTOR = "f1cbaa29ba4d7ee0133a4b6c5011709f723e8980"
RUNTIME_FINGERPRINT = (
    "gemma3-270m@9b0cfec892e2/plt@fada11860ac1/"
    "circuit-tracer@8f1e2438df61/nnsight/mps/bf16/stage1f"
)
PROTOCOL_FILES = (
    str(CONFIG_PATH),
    str(SCHEMA_PATH),
    "src/cfsus/stage1b_runtime.py",
    "src/cfsus/stage1c_v3/execution_journal.py",
    "src/cfsus/stage1c_v3/intervention.py",
    "src/cfsus/stage1c_v3/intervention_runtime.py",
    "src/cfsus/stage1c_v3/prediction.py",
    "src/cfsus/stage1c_v3/quantization_audit.py",
    "src/cfsus/stage1c_v3/runtime.py",
    "src/cfsus/stage1c_v3/serialization.py",
    "src/cfsus/stage1d/execution.py",
    "src/cfsus/stage1d/preflight.py",
    "src/cfsus/stage1e/offline.py",
    "src/cfsus/stage1f.py",
    "scripts/stage1f.py",
)
JSON_ARTIFACTS = (
    "protocol_manifest.json",
    "prediction_manifest.json",
    "point_sweeps.json",
    "analysis_summary.json",
    "run_manifest.json",
    "environment_manifest.json",
)
SHA40 = re.compile(r"\A[0-9a-f]{40}\Z")
SHA64 = re.compile(r"\A[0-9a-f]{64}\Z")
PROTECTED_ORIGIN_REFS = {
    "main": "7aacf30d888f96a29a1cfc82d035fca489ed0c17",
    "stage-1c-v4-protocol-preserving-execution": (
        "d4fdcc2c2f0040654af17e21f396f1d26072aa0e"
    ),
    "stage-1d-multiprompt-gate-benchmark": ("b71df55fdeb2fb66601af56207b6fbe5238e57d8"),
    "stage-1e-finite-probe-calibration": BASE_COMMIT,
}


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


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    value = _strict_json(path)
    if type(value) is not dict:
        raise ScientificInputError("Stage 1F config must be an object")
    config = cast(dict[str, Any], value)
    expected = {
        "schema_version": 1,
        "experiment_class": EXPERIMENT_CLASS,
        "branch": BRANCH,
        "base_commit": BASE_COMMIT,
        "required_stage1e_ancestor": REQUIRED_ANCESTOR,
        "phase": "protocol_frozen_before_fresh_baseline",
    }
    if any(config.get(key) != item for key, item in expected.items()):
        raise ScientificInputError("Stage 1F config identity differs")
    prompts = config.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != 10:
        raise ScientificInputError("Stage 1F requires exactly ten prompts")
    if [item.get("id") for item in prompts] != [f"F{i:02d}" for i in range(1, 11)]:
        raise ScientificInputError("Stage 1F prompt IDs differ")
    for prompt in prompts:
        tokens = prompt.get("token_ids")
        positions = prompt.get("selected_positions")
        if (
            not isinstance(prompt.get("text"), str)
            or not isinstance(tokens, list)
            or tokens[:1] != [2]
            or positions != list(range(1, len(tokens)))
        ):
            raise ScientificInputError(
                "Stage 1F prompt/token position contract differs"
            )
    if config["estimators"]["e1_nominal_probe_grid"] != [0.125, 0.1875, 0.25]:
        raise ScientificInputError("Stage 1F E1 probe grid differs")
    if config["scoring"]["discovery_ranker"] != "inhibitory_influence_q_descending":
        raise ScientificInputError("Stage 1F discovery ranker differs")
    if config["artifacts"]["required_files"] != [*JSON_ARTIFACTS, "checksums.sha256"]:
        raise ScientificInputError("Stage 1F artifact allowlist differs")
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
    """Require the isolated branch, exact ancestry, protected refs, and clean state."""

    head = _git(repository, "rev-parse", "HEAD")
    branch = _git(repository, "branch", "--show-current")
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    if branch != BRANCH or status:
        raise RuntimeError("Stage 1F requires its clean isolated worktree")
    if expected_head is not None and head != expected_head:
        raise RuntimeError("Stage 1F HEAD differs from the frozen commit")
    _git(repository, "merge-base", "--is-ancestor", BASE_COMMIT, head)
    _git(repository, "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, head)
    protected = {
        name: _git(repository, "rev-parse", f"refs/remotes/origin/{name}")
        for name in PROTECTED_ORIGIN_REFS
    }
    if protected != PROTECTED_ORIGIN_REFS:
        raise RuntimeError("a protected origin ref differs from its preflight SHA")
    origin_head: str | None = None
    upstream: str | None = None
    if require_pushed:
        origin_head = _git(repository, "rev-parse", f"refs/remotes/origin/{BRANCH}")
        upstream = _git(
            repository, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"
        )
        if origin_head != head or upstream != f"origin/{BRANCH}":
            raise RuntimeError("Stage 1F local/origin branch identity differs")
    return {
        "branch": branch,
        "head": head,
        "origin_branch_head": origin_head,
        "upstream": upstream,
        "working_tree_clean": True,
        "base_commit": BASE_COMMIT,
        "base_ancestry_verified": True,
        "required_stage1e_ancestor_verified": True,
        "protected_origin_refs": protected,
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
        raise ValueError("protocol commit is malformed")
    hashes = protocol_hashes(repository)
    return {
        "schema_version": 1,
        "artifact_type": "stage1f_protocol_manifest",
        "status": "protocol_frozen_before_fresh_baseline",
        "experiment_class": EXPERIMENT_CLASS,
        "branch": BRANCH,
        "base_commit": BASE_COMMIT,
        "required_stage1e_ancestor": REQUIRED_ANCESTOR,
        "protocol_commit": protocol_commit,
        "protocol_file_sha256": hashes,
        "protocol_map_sha256": _map_digest(hashes),
        "fresh_baseline_model_calls_before_freeze": 0,
        "fresh_source_suppression_calls_before_freeze": 0,
        "scientific_attempt_started": False,
        "historical_stage1e_status_preserved": "completed_stage1e_offline_negative",
    }


def validate_protocol(
    repository: Path, protocol: dict[str, Any], config: dict[str, Any]
) -> None:
    if (
        protocol.get("schema_version") != 1
        or protocol.get("artifact_type") != "stage1f_protocol_manifest"
        or protocol.get("status") != "protocol_frozen_before_fresh_baseline"
        or protocol.get("experiment_class") != EXPERIMENT_CLASS
        or protocol.get("branch") != BRANCH
        or protocol.get("base_commit") != BASE_COMMIT
        or protocol.get("required_stage1e_ancestor") != REQUIRED_ANCESTOR
        or protocol.get("fresh_baseline_model_calls_before_freeze") != 0
        or protocol.get("fresh_source_suppression_calls_before_freeze") != 0
        or protocol.get("scientific_attempt_started") is not False
        or protocol.get("historical_stage1e_status_preserved")
        != "completed_stage1e_offline_negative"
    ):
        raise ValueError("Stage 1F protocol identity or freeze boundary differs")
    if SHA40.fullmatch(str(protocol.get("protocol_commit"))) is None:
        raise ValueError("Stage 1F protocol commit is malformed")
    hashes = protocol.get("protocol_file_sha256")
    if type(hashes) is not dict or set(hashes) != set(PROTOCOL_FILES):
        raise ValueError("Stage 1F protocol file allowlist differs")
    for relative, digest in hashes.items():
        if (
            SHA64.fullmatch(str(digest)) is None
            or sha256_file(repository / relative) != digest
        ):
            raise ValueError(f"Stage 1F protocol file digest differs: {relative}")
    if protocol.get("protocol_map_sha256") != _map_digest(hashes):
        raise ValueError("Stage 1F protocol map digest differs")
    load_config(repository / CONFIG_PATH)


def first_probe_mapping(
    source_activation: float, grid: Sequence[float]
) -> dict[str, float] | None:
    """Return Stage 1E's first nonzero BF16-realized probe mapping."""

    if not math.isfinite(source_activation) or source_activation <= 0.0:
        return None
    for nominal in grid:
        desired = (1.0 - float(nominal)) * source_activation
        applied = bf16_round(desired)
        realized = 1.0 - applied / source_activation
        if realized > 0.0 and all(
            math.isfinite(x) for x in (desired, applied, realized)
        ):
            return {
                "nominal_requested_alpha": float(nominal),
                "desired_source_activation": desired,
                "applied_bf16_source_activation": applied,
                "expected_realized_probe_alpha": realized,
            }
    return None


def _coordinate_key(pair: Any) -> tuple[Any, ...]:
    return (
        pair.source.layer,
        pair.source.position,
        pair.source.feature_id,
        pair.target.layer,
        pair.target.position,
        pair.target.feature_id,
        pair.pair_id,
    )


def select_q_panel(
    pairs: Sequence[Any], *, prompt_id: str, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select up to four highest-q pairs independently in each frozen E0 bin."""

    grid = config["estimators"]["e1_nominal_probe_grid"]
    eligible: list[tuple[Any, dict[str, float]]] = []
    reason_counts: Counter[str] = Counter()
    for pair in pairs:
        if pair.q <= 0.0 or not math.isfinite(pair.q):
            reason_counts["q_not_finite_positive"] += 1
            continue
        alpha = pair.margin / pair.q
        if not 0.02 <= alpha <= 0.95:
            reason_counts["e0_alpha_outside_0.02_0.95"] += 1
            continue
        probe = first_probe_mapping(pair.source_activation, grid)
        if probe is None:
            reason_counts["no_nonzero_bf16_e1_probe"] += 1
            continue
        eligible.append((pair, probe))
    selected: list[dict[str, Any]] = []
    shortfalls: dict[str, int] = {}
    counts: dict[str, int] = {}
    for name, bounds in config["panel"]["bins"].items():
        low = float(bounds["low"])
        high = float(bounds["high"])
        inclusive = bool(bounds["high_inclusive"])
        candidates = [
            (pair, probe)
            for pair, probe in eligible
            if low <= pair.margin / pair.q
            and (
                pair.margin / pair.q <= high
                if inclusive
                else pair.margin / pair.q < high
            )
        ]
        candidates.sort(key=lambda item: (-item[0].q, *_coordinate_key(item[0])))
        quota = int(config["panel"]["per_prompt_per_bin_maximum"])
        chosen = candidates[:quota]
        counts[name] = len(chosen)
        shortfalls[name] = quota - len(chosen)
        for pair, probe in chosen:
            row = prospective_pair_record(pair)
            row.update(
                {
                    "prompt_id": prompt_id,
                    "stratum": name,
                    "selection_rank_in_stratum": len(
                        [x for x in selected if x["stratum"] == name]
                    )
                    + 1,
                    "discovery_ranker": "q",
                    "e0_predicted_alpha": pair.margin / pair.q,
                    "e1_probe_mapping": probe,
                    "requested_alphas": [0.0, probe["nominal_requested_alpha"], 1.0],
                    "method_memberships": ["influence_only"],
                    "detailed_role": name,
                }
            )
            selected.append(row)
    return selected, {
        "eligible_candidate_count": len(eligible),
        "selected_count_by_stratum": counts,
        "shortfall_by_stratum": shortfalls,
        "ineligibility_reason_counts": dict(sorted(reason_counts.items())),
        "cross_bin_backfill_used": False,
    }


def _active_calibration_pairs(
    sources: tuple[Any, ...], *, pair_count: int
) -> tuple[tuple[Any, Any], ...]:
    selected: list[tuple[Any, Any]] = []
    used_sources: set[Any] = set()
    used_targets: set[Any] = set()
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
        raise RuntimeError("insufficient active-only VJP health-calibration endpoints")
    return tuple(selected)


def _prompt_prediction(
    model: Any,
    torch: Any,
    sampler: Any,
    prompt: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    prompt_id = str(prompt["id"])
    prompt_text = str(prompt["text"])
    backend = Stage1CPredictionBackend(
        model, prompt=prompt_text, prompt_id=prompt_id, torch=torch
    )
    with sampler.stage(f"{prompt_id}_tokenization"):
        tokens = model.ensure_tokenized(prompt_text)
        token_ids = [int(item) for item in tokens.detach().cpu().tolist()]
    if token_ids != prompt["token_ids"]:
        raise RuntimeError(f"{prompt_id} token IDs differ from protocol freeze")
    positions = list(range(1, len(token_ids)))
    if positions != prompt["selected_positions"]:
        raise RuntimeError(f"{prompt_id} selected positions differ")
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
    calibration_count = int(config["responses"]["active_calibration_pair_count"])
    calibration_pairs = _active_calibration_pairs(
        raw_sources, pair_count=calibration_count
    )
    with sampler.stage(f"{prompt_id}_active_vjp_health_calibration"):
        pairwise_responses = backend.targeted_local_responses(
            tuple(
                (source.feature, target.feature) for source, target in calibration_pairs
            ),
            maximum_pairs=calibration_count,
        )
        tile = backend.response_tile(
            targets=tuple(target.feature for _, target in calibration_pairs),
            sources=tuple(source for source, _ in calibration_pairs),
            maximum_targets=calibration_count,
        )
    if tuple(float(tile[i][i]) for i in range(calibration_count)) != tuple(
        float(item.response) for item in pairwise_responses
    ):
        raise RuntimeError(f"{prompt_id} many-source VJP health calibration differs")
    del calibration_pairs, pairwise_responses, tile
    gc.collect()
    torch.mps.empty_cache()
    target_by_feature = {item.feature: item for item in targets}
    target_features = tuple(sorted(target_by_feature))
    batch_size = int(config["responses"]["target_batch_size"])
    pairs: list[Any] = []
    for start in range(0, len(target_features), batch_size):
        selected_targets = target_features[start : start + batch_size]
        with sampler.stage(f"{prompt_id}_target_vjp_batch_{start // batch_size}"):
            response_tile = backend.response_tile(
                targets=selected_targets, sources=sources, maximum_targets=batch_size
            )
        for target, responses in zip(selected_targets, response_tile, strict=True):
            candidate = target_by_feature[target]
            for source, response in zip(sources, responses, strict=True):
                if not causally_eligible(source.feature, target):
                    continue
                pairs.append(
                    build_prospective_pair(
                        source=source,
                        target=candidate,
                        targeted_response=response,
                        seed=str(config["scoring"]["pair_seed"]),
                        prompt_id=prompt_id,
                        runtime_fingerprint=RUNTIME_FINGERPRINT,
                        epsilon=float(config["scoring"]["epsilon"]),
                        tolerance=float(config["scoring"]["crossing_tolerance"]),
                        experiment_class=EXPERIMENT_CLASS,
                    )
                )
        del response_tile
        gc.collect()
        torch.mps.empty_cache()
    if len(pairs) != eligible_pair_count:
        raise RuntimeError(f"{prompt_id} eligible pair enumeration is incomplete")
    selected, panel_audit = select_q_panel(pairs, prompt_id=prompt_id, config=config)
    for row in selected:
        row["prompt_id"] = prompt_id
    q_counts = Counter(
        "positive" if pair.q > 0 else "zero" if pair.q == 0 else "negative"
        for pair in pairs
    )
    result = {
        "id": prompt_id,
        "text": prompt_text,
        "token_ids": token_ids,
        "selected_positions": positions,
        "baseline_pools": {
            "scanner_candidate_count": len(scanner.global_candidates),
            "eligible_target_count": len(targets),
            "target_pool_sha256": target_pool_digest(targets),
            "raw_active_source_count": len(raw_sources),
            "eligible_source_count": len(sources),
            "source_pool_sha256": source_pool_digest(sources),
            "eligible_pair_count": len(pairs),
            "pair_score_sha256": pair_score_digest(pairs),
            "q_sign_counts": dict(q_counts),
            "dense_scanner_arrays_persisted": False,
            "complete_derivative_matrix_persisted": False,
            "scanner_dense_oracle_validation": {
                "group_count": len(groups),
                "exact_identity_and_order": True,
            },
            "active_vjp_health_calibration": {
                "pair_count": calibration_count,
                "pairwise_vs_many_source_exact_bf16_identity": True,
                "graph_edge_input_used": False,
                "intervention_calls": 0,
            },
        },
        "panel_audit": panel_audit,
        "execution_pairs": selected,
    }
    pairs.clear()
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
    git: dict[str, Any],
) -> dict[str, Any]:
    prompts = [
        _prompt_prediction(model, torch, sampler, row, config)
        for row in config["prompts"]
    ]
    total = sum(len(prompt["execution_pairs"]) for prompt in prompts)
    if total > int(config["panel"]["maximum_total_pairs"]):
        raise RuntimeError("Stage 1F panel exceeds 120 pairs")
    return {
        "schema_version": 1,
        "artifact_type": "stage1f_prediction_manifest",
        "status": "prediction_frozen_ready_for_commit",
        "experiment_class": EXPERIMENT_CLASS,
        "branch": BRANCH,
        "base_commit": BASE_COMMIT,
        "protocol_commit": protocol["protocol_commit"],
        "protocol_map_sha256": protocol["protocol_map_sha256"],
        "prediction_execution_commit": git["head"],
        "runtime_identity": config["runtime"],
        "prompt_provenance": config["prompt_provenance"],
        "prompt_order": [item["id"] for item in config["prompts"]],
        "prompts": prompts,
        "selection_totals": {
            "prompt_count": 10,
            "selected_pair_count": total,
            "selected_count_by_stratum": {
                name: sum(
                    int(prompt["panel_audit"]["selected_count_by_stratum"][name])
                    for prompt in prompts
                )
                for name in ("B1", "B2", "B3")
            },
        },
        "prediction_only_guards": {
            "fresh_source_suppression_api_calls": 0,
            "fresh_target_responses_inspected": False,
            "historical_intervention_outcomes_used": False,
            "graph_edge_input_used_for_inactive_predictions": False,
            "network_accessed": False,
            "discovery_ranker": "q",
        },
        "claim_boundary": config["claim_boundary"],
    }


def validate_prediction(
    prediction: dict[str, Any], config: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Validate the baseline-only frozen panel without reading outcomes."""

    if (
        prediction.get("schema_version") != 1
        or prediction.get("artifact_type") != "stage1f_prediction_manifest"
        or prediction.get("status") != "prediction_frozen_ready_for_commit"
        or prediction.get("experiment_class") != EXPERIMENT_CLASS
        or prediction.get("branch") != BRANCH
        or prediction.get("base_commit") != BASE_COMMIT
        or prediction.get("protocol_commit") != protocol["protocol_commit"]
        or prediction.get("protocol_map_sha256") != protocol["protocol_map_sha256"]
        or prediction.get("runtime_identity") != config["runtime"]
        or prediction.get("prompt_provenance") != config["prompt_provenance"]
        or prediction.get("prompt_order") != [row["id"] for row in config["prompts"]]
        or prediction.get("claim_boundary") != config["claim_boundary"]
        or prediction.get("prediction_only_guards")
        != {
            "fresh_source_suppression_api_calls": 0,
            "fresh_target_responses_inspected": False,
            "historical_intervention_outcomes_used": False,
            "graph_edge_input_used_for_inactive_predictions": False,
            "network_accessed": False,
            "discovery_ranker": "q",
        }
    ):
        raise ValueError("Stage 1F prediction identity or no-outcome guards differ")
    prompts = prediction.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != 10:
        raise ValueError("Stage 1F prediction prompt count differs")
    all_pairs: dict[str, dict[str, Any]] = {}
    stratum_totals = {"B1": 0, "B2": 0, "B3": 0}
    for observed, frozen in zip(prompts, config["prompts"], strict=True):
        if type(observed) is not dict:
            raise ValueError("Stage 1F prediction prompt is malformed")
        prompt = cast(dict[str, Any], observed)
        for key in ("id", "text", "token_ids", "selected_positions"):
            if prompt.get(key) != frozen[key]:
                raise ValueError(f"Stage 1F prompt {frozen['id']} {key} differs")
        rows = prompt.get("execution_pairs")
        if not isinstance(rows, list) or len(rows) > 12:
            raise ValueError("Stage 1F prompt pair quota differs")
        by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
        prompt_ids: set[str] = set()
        for raw in rows:
            if type(raw) is not dict:
                raise ValueError("Stage 1F prediction pair is malformed")
            row = cast(dict[str, Any], raw)
            pair_id = row.get("pair_id")
            if (
                not isinstance(pair_id, str)
                or SHA64.fullmatch(pair_id) is None
                or pair_id in all_pairs
                or pair_id in prompt_ids
            ):
                raise ValueError("Stage 1F pair ID is malformed or duplicated")
            source_raw = row.get("source")
            target_raw = row.get("target")
            if type(source_raw) is not dict or type(target_raw) is not dict:
                raise ValueError("Stage 1F feature coordinate is malformed")
            source = FeatureRef(**source_raw)
            target = FeatureRef(**target_raw)
            if not (
                0 <= source.layer < target.layer < 18
                and 1 <= source.position <= target.position < len(frozen["token_ids"])
                and 0 <= source.feature_id < 16384
                and 0 <= target.feature_id < 16384
            ):
                raise ValueError("Stage 1F pair violates the causal feature domain")
            expected_id = canonical_v3_pair_id(
                source=source,
                target=target,
                runtime_fingerprint=RUNTIME_FINGERPRINT,
                prompt_id=frozen["id"],
                seed=config["scoring"]["pair_seed"],
                experiment_class=EXPERIMENT_CLASS,
            )
            activation = _finite(row.get("source_activation"), "source activation")
            z = _finite(row.get("target_preactivation"), "target preactivation")
            threshold = _finite(row.get("target_threshold"), "target threshold")
            margin = _finite(row.get("margin"), "margin")
            response = _finite(row.get("targeted_response"), "targeted response")
            q = _finite(row.get("q"), "q")
            alpha = _finite(row.get("e0_predicted_alpha"), "E0 alpha")
            stratum = row.get("stratum")
            if (
                pair_id != expected_id
                or row.get("prompt_id") != frozen["id"]
                or activation <= 0.0
                or z > threshold
                or margin != threshold - z
                or q != -activation * response
                or q <= 0.0
                or alpha != margin / q
                or not 0.02 <= alpha <= 0.95
                or stratum not in {"B1", "B2", "B3"}
                or row.get("discovery_ranker") != "q"
                or row.get("method_memberships") != ["influence_only"]
                or row.get("detailed_role") != stratum
            ):
                raise ValueError("Stage 1F pair scalar or selection definition differs")
            bounds = config["panel"]["bins"][stratum]
            if not (
                float(bounds["low"]) <= alpha
                and (
                    alpha <= float(bounds["high"])
                    if bounds["high_inclusive"]
                    else alpha < float(bounds["high"])
                )
            ):
                raise ValueError("Stage 1F pair is in the wrong E0 stratum")
            expected_probe = first_probe_mapping(
                activation, config["estimators"]["e1_nominal_probe_grid"]
            )
            if (
                expected_probe is None
                or row.get("e1_probe_mapping") != expected_probe
                or row.get("requested_alphas")
                != [0.0, expected_probe["nominal_requested_alpha"], 1.0]
            ):
                raise ValueError("Stage 1F frozen E1 probe mapping differs")
            prompt_ids.add(pair_id)
            all_pairs[pair_id] = row
            by_stratum[stratum].append(row)
        audit = prompt.get("panel_audit")
        if type(audit) is not dict:
            raise ValueError("Stage 1F panel audit is malformed")
        for name in ("B1", "B2", "B3"):
            selected = by_stratum[name]
            if len(selected) > 4 or selected != sorted(
                selected,
                key=lambda row: (
                    -float(row["q"]),
                    int(row["source"]["layer"]),
                    int(row["source"]["position"]),
                    int(row["source"]["feature_id"]),
                    int(row["target"]["layer"]),
                    int(row["target"]["position"]),
                    int(row["target"]["feature_id"]),
                    str(row["pair_id"]),
                ),
            ):
                raise ValueError("Stage 1F q ranking or tie-break differs")
            if [row["selection_rank_in_stratum"] for row in selected] != list(
                range(1, len(selected) + 1)
            ):
                raise ValueError("Stage 1F stratum ranks differ")
            if audit.get("selected_count_by_stratum", {}).get(name) != len(selected):
                raise ValueError("Stage 1F panel selected counts differ")
            if audit.get("shortfall_by_stratum", {}).get(name) != 4 - len(selected):
                raise ValueError("Stage 1F panel shortfall differs")
            stratum_totals[name] += len(selected)
    totals = prediction.get("selection_totals")
    if (
        type(totals) is not dict
        or totals.get("prompt_count") != 10
        or totals.get("selected_pair_count") != len(all_pairs)
        or totals.get("selected_count_by_stratum") != stratum_totals
        or len(all_pairs) > 120
    ):
        raise ValueError("Stage 1F selection totals differ")
    return all_pairs


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _requested(point: dict[str, Any]) -> set[float]:
    mappings = point.get("requested_mappings")
    if not isinstance(mappings, list) or not mappings:
        raise ValueError("point requested mappings are missing")
    return {float(item["requested_alpha"]) for item in mappings}


def execute_frozen_pairs(
    *,
    model: Any,
    torch: Any,
    sampler: Any,
    prompts: list[dict[str, Any]],
    journal: CanonicalExecutionJournal,
    backend_factory: Callable[..., Any],
    maximum_bisection_steps: int,
    stop_bracket_width: float,
) -> tuple[list[dict[str, Any]], int]:
    """Execute the exact baseline/probe/full policy and conditional refinement."""

    sweeps: list[dict[str, Any]] = []
    global_calls = 0
    for prompt in prompts:
        backend = backend_factory(
            model,
            prompt=str(prompt["text"]),
            prompt_id=str(prompt["id"]),
            torch=torch,
            token_count=len(prompt["token_ids"]),
            attempt_recorder=journal.before_source_suppression,
            call_index_offset=global_calls,
        )
        for pair in prompt["execution_pairs"]:
            source = _feature(pair["source"])
            target = _feature(pair["target"])
            short = pair["pair_id"][:12]
            with sampler.stage(f"{prompt['id']}_{short}_baseline_remeasurement"):
                states = backend.measure_states((source, target))
            source_state, target_state = _matching_baseline(pair, states)
            source_tensor = torch.tensor(
                source_state.activation, device="mps", dtype=torch.bfloat16
            ).reshape(())
            requested = tuple(float(item) for item in pair["requested_alphas"])
            plans = plan_applied_values(source_tensor, requested, torch)
            if len(plans) != 3:
                raise RuntimeError("frozen baseline/probe/full schedule BF16-collapsed")
            points: list[dict[str, Any]] = []
            applied_values: set[float] = set()
            for index, plan in enumerate(plans):
                applied_values.add(plan.applied_bf16)
                points.append(
                    _run_plan(
                        backend,
                        journal,
                        sampler,
                        pair,
                        plan,
                        source_state=source_state,
                        target_state=target_state,
                        stage=f"{prompt['id']}_{short}_required_{index}",
                    )
                )
            baseline_point = next(point for point in points if 0.0 in _requested(point))
            probe_nominal = float(pair["e1_probe_mapping"]["nominal_requested_alpha"])
            probe_point = next(
                point for point in points if probe_nominal in _requested(point)
            )
            full_point = next(point for point in points if 1.0 in _requested(point))
            e1 = estimate_e1(
                margin=float(pair["margin"]),
                baseline_z=float(baseline_point["target_preactivation"]),
                probe={
                    "nominal_requested_alpha": probe_nominal,
                    "desired_source_activation": float(
                        pair["e1_probe_mapping"]["desired_source_activation"]
                    ),
                    "applied_bf16_source_activation": float(
                        probe_point["actual_bf16_value_passed"]
                    ),
                    "realized_alpha": float(probe_point["realized_suppression"]),
                    "target_preactivation": float(probe_point["target_preactivation"]),
                },
            )
            if bool(full_point["target_active"]) and e1["status"] == "accepted":
                for step in range(maximum_bisection_steps):
                    bracket = _crossing_bracket(points)
                    if bracket is None:
                        break
                    lower = float(bracket[0]["realized_suppression"])
                    upper = float(bracket[1]["realized_suppression"])
                    if upper - lower <= stop_bracket_width:
                        break
                    requested_alpha = bisection_requested_alpha(lower, upper)
                    plan = plan_applied_values(
                        source_tensor, (requested_alpha,), torch
                    )[0]
                    if (
                        plan.applied_bf16 in applied_values
                        or not lower < plan.realized_suppression < upper
                    ):
                        break
                    applied_values.add(plan.applied_bf16)
                    points.append(
                        _run_plan(
                            backend,
                            journal,
                            sampler,
                            pair,
                            plan,
                            source_state=source_state,
                            target_state=target_state,
                            stage=f"{prompt['id']}_{short}_bisect_{step}",
                        )
                    )
            points.sort(key=lambda item: float(item["realized_suppression"]))
            sweeps.append(
                {
                    "prompt_id": prompt["id"],
                    "pair_id": pair["pair_id"],
                    "stratum": pair["stratum"],
                    "point_count": len(points),
                    "points": points,
                }
            )
            global_calls = backend.source_suppression_api_calls
            del states, source_tensor, plans, points
            gc.collect()
            torch.mps.empty_cache()
    journal.verify_complete(expected_point_count=global_calls)
    return sweeps, global_calls


def read_completed_journal(path: Path) -> list[dict[str, Any]]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > 32 * 1024 * 1024
    ):
        raise ValueError("Stage 1F journal is missing, unsafe, or oversized")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or len(lines) % 2:
        raise ValueError("Stage 1F journal lacks complete record pairs")
    points: list[dict[str, Any]] = []
    for expected, (left, right) in enumerate(
        zip(lines[::2], lines[1::2], strict=True), start=1
    ):
        started = json.loads(left)
        completed = json.loads(right)
        if (
            type(started) is not dict
            or type(completed) is not dict
            or started.get("record_type") != "source_suppression_call_started"
            or started.get("call_index") != expected
            or set(started) != {"record_type", "call_index", "pair_id"}
            or completed.get("record_type") != "point_completed"
            or completed.get("call_index") != expected
            or completed.get("pair_id") != started.get("pair_id")
            or set(completed) != {"record_type", "call_index", "pair_id", "point"}
            or type(completed.get("point")) is not dict
            or completed["point"].get("pair_id") != started.get("pair_id")
            or completed["point"].get("source_suppression_api_call_index") != expected
        ):
            raise ValueError("Stage 1F journal ordering or identity differs")
        points.append(completed["point"])
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
    for point in points:
        pair_id = point.get("pair_id")
        if pair_id not in grouped:
            raise ValueError("Stage 1F journal contains a non-frozen pair")
        grouped[pair_id].append(point)
    if any(not rows for rows in grouped.values()):
        raise ValueError("a Stage 1F frozen pair lacks completed points")
    for pair_id, rows in grouped.items():
        rows.sort(key=lambda item: float(item["realized_suppression"]))
        if len({float(row["actual_bf16_value_passed"]) for row in rows}) != len(rows):
            raise ValueError(f"Stage 1F pair repeats an applied BF16 value: {pair_id}")
    return pairs, grouped


def _nearest_rank(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    return sorted(values)[max(1, math.ceil(probability * len(values))) - 1]


def _first_bracket(
    points: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    for left, right in pairwise(
        sorted(points, key=lambda row: float(row["realized_suppression"]))
    ):
        if not bool(left["target_active"]) and bool(right["target_active"]):
            return left, right
    return None


def _nonmonotonic(points: Sequence[dict[str, Any]]) -> bool:
    active_seen = False
    for point in sorted(points, key=lambda row: float(row["realized_suppression"])):
        if bool(point["target_active"]):
            active_seen = True
        elif active_seen:
            return True
    return False


def _pair_analysis(
    pair: dict[str, Any], points: list[dict[str, Any]]
) -> dict[str, Any]:
    baseline = [point for point in points if 0.0 in _requested(point)]
    probe_nominal = float(pair["e1_probe_mapping"]["nominal_requested_alpha"])
    probe = [point for point in points if probe_nominal in _requested(point)]
    full = [point for point in points if 1.0 in _requested(point)]
    if len(baseline) != 1 or len(probe) != 1 or len(full) != 1:
        raise ValueError("Stage 1F required baseline/probe/full point identity differs")
    baseline_z = float(baseline[0]["target_preactivation"])
    if baseline_z != float(pair["target_preactivation"]):
        raise ValueError("Stage 1F serialized no-op differs from frozen baseline")
    e0 = estimate_e0(margin=float(pair["margin"]), q=float(pair["q"]))
    e1 = estimate_e1(
        margin=float(pair["margin"]),
        baseline_z=baseline_z,
        probe={
            "nominal_requested_alpha": probe_nominal,
            "desired_source_activation": float(
                pair["e1_probe_mapping"]["desired_source_activation"]
            ),
            "applied_bf16_source_activation": float(
                probe[0]["actual_bf16_value_passed"]
            ),
            "realized_alpha": float(probe[0]["realized_suppression"]),
            "target_preactivation": float(probe[0]["target_preactivation"]),
        },
    )
    nonmonotonic = _nonmonotonic(points)
    bracket = _first_bracket(points)
    bracket_record = None
    midpoint = None
    if bracket is not None:
        lower = float(bracket[0]["realized_suppression"])
        upper = float(bracket[1]["realized_suppression"])
        bracket_record = {
            "lower_realized_alpha": lower,
            "upper_realized_alpha": upper,
            "width": upper - lower,
        }
        midpoint = (lower + upper) / 2.0
    observed_crossing = bool(full[0]["target_active"])
    e1_accepted = e1["status"] == "accepted"
    reference_denominator = (
        observed_crossing and not nonmonotonic and bracket is not None
    )
    paired_reference = reference_denominator and e1_accepted
    e0_alpha = float(e0["predicted_alpha"])
    e1_alpha = None if not e1_accepted else float(e1["predicted_alpha"])
    return {
        "prompt_id": pair["prompt_id"],
        "pair_id": pair["pair_id"],
        "stratum": pair["stratum"],
        "q": float(pair["q"]),
        "margin": float(pair["margin"]),
        "e0_predicted_alpha": e0_alpha,
        "e1_status": e1["status"],
        "e1_abstention_reason": e1["abstention_reason"],
        "e1_estimated_secant_drive": e1.get("estimated_drive"),
        "e1_predicted_alpha": e1_alpha,
        "e1_probe_nominal_alpha": probe_nominal,
        "e1_probe_realized_alpha": float(probe[0]["realized_suppression"]),
        "observed_full_ablation_crossing": observed_crossing,
        "observed_critical_bracket": bracket_record,
        "observed_critical_midpoint": midpoint,
        "nonmonotonic": nonmonotonic,
        "reference_denominator": reference_denominator,
        "paired_reference": paired_reference,
        "e0_absolute_error": (
            abs(e0_alpha - midpoint)
            if paired_reference and midpoint is not None
            else None
        ),
        "e1_absolute_error": (
            abs(cast(float, e1_alpha) - midpoint)
            if paired_reference and midpoint is not None
            else None
        ),
        "e0_bracket_distance": (
            0.0
            if paired_reference
            and bracket_record is not None
            and bracket_record["lower_realized_alpha"]
            <= e0_alpha
            <= bracket_record["upper_realized_alpha"]
            else min(
                abs(e0_alpha - bracket_record["lower_realized_alpha"]),
                abs(e0_alpha - bracket_record["upper_realized_alpha"]),
            )
            if paired_reference and bracket_record is not None
            else None
        ),
        "e1_bracket_distance": (
            0.0
            if paired_reference
            and bracket_record is not None
            and bracket_record["lower_realized_alpha"]
            <= cast(float, e1_alpha)
            <= bracket_record["upper_realized_alpha"]
            else min(
                abs(cast(float, e1_alpha) - bracket_record["lower_realized_alpha"]),
                abs(cast(float, e1_alpha) - bracket_record["upper_realized_alpha"]),
            )
            if paired_reference and bracket_record is not None
            else None
        ),
        "e0_predicted_crossing": 0.0 < e0_alpha <= 1.0,
        "e1_predicted_crossing": e1_alpha is not None and 0.0 < e1_alpha <= 1.0,
        "point_count": len(points),
    }


def _cluster_bootstrap(
    paired: list[dict[str, Any]], prompt_ids: list[str], *, count: int, seed: str
) -> dict[str, Any]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        by_prompt[str(row["prompt_id"])].append(row)
    generator = random.Random(int(hashlib.sha256(seed.encode()).hexdigest(), 16))
    improvement: list[float] = []
    e0_medians: list[float] = []
    e1_medians: list[float] = []
    e0_spearman: list[float] = []
    e1_spearman: list[float] = []
    for _ in range(count):
        sampled = [prompt_ids[generator.randrange(len(prompt_ids))] for _ in prompt_ids]
        rows = [row for prompt_id in sampled for row in by_prompt[prompt_id]]
        if not rows:
            continue
        e0_errors = [float(row["e0_absolute_error"]) for row in rows]
        e1_errors = [float(row["e1_absolute_error"]) for row in rows]
        e0_median = median(e0_errors)
        e1_median = median(e1_errors)
        e0_medians.append(e0_median)
        e1_medians.append(e1_median)
        improvement.append(e0_median - e1_median)
        observed = [float(row["observed_critical_midpoint"]) for row in rows]
        e0_rho = spearman([float(row["e0_predicted_alpha"]) for row in rows], observed)
        e1_rho = spearman([float(row["e1_predicted_alpha"]) for row in rows], observed)
        if e0_rho is not None:
            e0_spearman.append(e0_rho)
        if e1_rho is not None:
            e1_spearman.append(e1_rho)

    def interval(values: list[float]) -> dict[str, Any]:
        return {
            "lower": _nearest_rank(values, 0.025),
            "upper": _nearest_rank(values, 0.975),
            "defined_resamples": len(values),
            "requested_resamples": count,
        }

    return {
        "paired_median_absolute_error_improvement": interval(improvement),
        "E0_median_absolute_error": interval(e0_medians),
        "E1_median_absolute_error": interval(e1_medians),
        "E0_spearman": interval(e0_spearman),
        "E1_spearman": interval(e1_spearman),
    }


def _classification(rows: Sequence[dict[str, Any]], estimator: str) -> dict[str, Any]:
    predicted_key = f"{estimator.lower()}_predicted_crossing"
    confusion = {
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": 0,
        "false_negative": 0,
    }
    for row in rows:
        predicted = bool(row[predicted_key])
        observed = bool(row["observed_full_ablation_crossing"])
        key = (
            "true_positive"
            if predicted and observed
            else "false_positive"
            if predicted
            else "false_negative"
            if observed
            else "true_negative"
        )
        confusion[key] += 1
    total = len(rows)
    return {
        **confusion,
        "accuracy": (
            (confusion["true_positive"] + confusion["true_negative"]) / total
            if total
            else None
        ),
        "evaluated_pair_count": total,
        "abstention_treated_as_no_crossing": estimator == "E1",
    }


def _subset_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paired = [row for row in rows if row["paired_reference"]]
    return {
        "panel_pair_count": len(rows),
        "observed_crossing_count": sum(
            bool(row["observed_full_ablation_crossing"]) for row in rows
        ),
        "reference_denominator_count": sum(
            bool(row["reference_denominator"]) for row in rows
        ),
        "paired_reference_count": len(paired),
        "e1_accepted_count": sum(row["e1_status"] == "accepted" for row in rows),
        "e1_median_absolute_error": (
            median(float(row["e1_absolute_error"]) for row in paired)
            if paired
            else None
        ),
        "e0_median_absolute_error": (
            median(float(row["e0_absolute_error"]) for row in paired)
            if paired
            else None
        ),
    }


def classify_terminal(
    *,
    paired_count: int,
    coverage: float,
    e0_median: float | None,
    e1_median: float | None,
    improvement_lower: float | None,
    e0_spearman: float | None,
    e1_spearman: float | None,
    e0_accuracy: float | None,
    e1_accuracy: float | None,
    rules: dict[str, Any],
) -> tuple[str, str, dict[str, bool]]:
    if paired_count < int(rules["minimum_paired_reference_pairs"]):
        return (
            "completed_stage1f_underpowered",
            "insufficient_power_reassess_panel_size",
            {"minimum_paired_reference_pairs": False},
        )
    if None in (
        e0_median,
        e1_median,
        e0_spearman,
        e1_spearman,
        e0_accuracy,
        e1_accuracy,
    ):
        raise ValueError(
            "Stage 1F adequately powered result has undefined primary metrics"
        )
    assert e0_median is not None and e1_median is not None
    assert e0_spearman is not None and e1_spearman is not None
    assert e0_accuracy is not None and e1_accuracy is not None
    ratio = (
        e1_median / e0_median
        if e0_median > 0.0
        else (0.0 if e1_median == 0.0 else None)
    )
    checks = {
        "minimum_paired_reference_pairs": True,
        "coverage_at_least_0.60": coverage
        >= float(rules["confirmed_coverage_minimum"]),
        "e1_median_at_most_0.75_e0": ratio is not None
        and ratio <= float(rules["confirmed_e1_to_e0_median_error_ratio_maximum"]),
        "paired_improvement_bootstrap_lower_positive": improvement_lower is not None
        and improvement_lower
        > float(
            rules["confirmed_paired_improvement_bootstrap_lower_strictly_greater_than"]
        ),
        "e1_spearman_at_least_0.70": e1_spearman
        >= float(rules["confirmed_e1_spearman_minimum"]),
        "e1_spearman_drop_no_more_than_0.05": e1_spearman
        >= e0_spearman - float(rules["confirmed_e1_spearman_drop_from_e0_maximum"]),
        "e1_classification_drop_no_more_than_0.02": e1_accuracy
        >= e0_accuracy
        - float(rules["confirmed_e1_classification_drop_from_e0_maximum"]),
    }
    if all(checks.values()):
        return (
            "completed_stage1f_e1_confirmed",
            "proceed_to_behavioral_mediation_with_q_plus_e1",
            checks,
        )
    improvement_fraction = (
        (e0_median - e1_median) / e0_median
        if e0_median > 0.0
        else (1.0 if e1_median == 0.0 else -1.0)
    )
    if improvement_fraction >= float(rules["mixed_median_error_improvement_minimum"]):
        return (
            "completed_stage1f_e1_mixed",
            "proceed_to_directional_curvature_hvp",
            checks,
        )
    materially_worse = (
        ratio is None
        or ratio
        > float(
            rules["not_supported_e1_to_e0_median_error_ratio_strictly_greater_than"]
        )
        or e1_accuracy < e0_accuracy - float(rules["material_classification_drop"])
    )
    checks["material_not_supported_condition"] = materially_worse
    return (
        "completed_stage1f_e1_not_supported",
        "retire_simple_critical_alpha_calibration",
        checks,
    )


def compute_analysis(
    prediction: dict[str, Any],
    grouped: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    pairs = {
        pair["pair_id"]: pair
        for prompt in prediction["prompts"]
        for pair in prompt["execution_pairs"]
    }
    rows = [_pair_analysis(pairs[pair_id], grouped[pair_id]) for pair_id in pairs]
    paired = [row for row in rows if row["paired_reference"]]
    denominator = sum(bool(row["reference_denominator"]) for row in rows)
    coverage = len(paired) / denominator if denominator else 0.0
    e0_errors = [float(row["e0_absolute_error"]) for row in paired]
    e1_errors = [float(row["e1_absolute_error"]) for row in paired]
    observed = [float(row["observed_critical_midpoint"]) for row in paired]
    e0_predicted = [float(row["e0_predicted_alpha"]) for row in paired]
    e1_predicted = [float(row["e1_predicted_alpha"]) for row in paired]
    e0_classification = _classification(rows, "E0")
    e1_classification = _classification(rows, "E1")
    bootstrap = _cluster_bootstrap(
        paired,
        [str(prompt["id"]) for prompt in prediction["prompts"]],
        count=int(config["metrics"]["bootstrap_resamples"]),
        seed=str(config["metrics"]["bootstrap_seed"]),
    )
    e0_median = median(e0_errors) if e0_errors else None
    e1_median = median(e1_errors) if e1_errors else None
    terminal, project, checks = classify_terminal(
        paired_count=len(paired),
        coverage=coverage,
        e0_median=e0_median,
        e1_median=e1_median,
        improvement_lower=bootstrap["paired_median_absolute_error_improvement"][
            "lower"
        ],
        e0_spearman=spearman(e0_predicted, observed),
        e1_spearman=spearman(e1_predicted, observed),
        e0_accuracy=e0_classification["accuracy"],
        e1_accuracy=e1_classification["accuracy"],
        rules=config["decision_rule"],
    )
    abstentions = Counter(
        str(row["e1_abstention_reason"])
        for row in rows
        if row["e1_status"] != "accepted"
    )
    accepted_count = sum(row["e1_status"] == "accepted" for row in rows)
    total_calls = sum(len(points) for points in grouped.values())
    return {
        "schema_version": 1,
        "artifact_type": "stage1f_analysis_summary",
        "status": "passed",
        "experiment_class": EXPERIMENT_CLASS,
        "terminal_class": terminal,
        "project_decision": project,
        "panel_pair_count": len(rows),
        "paired_reference_pair_count": len(paired),
        "reference_denominator_pair_count": denominator,
        "selective_risk_and_coverage": {
            "e1_reference_coverage": coverage,
            "e1_accepted_count_full_panel": accepted_count,
            "e1_acceptance_coverage_full_panel": accepted_count / len(rows)
            if rows
            else 0.0,
            "e1_abstention_count": len(rows) - accepted_count,
            "e1_abstention_rate": (len(rows) - accepted_count) / len(rows)
            if rows
            else 0.0,
            "e1_abstention_reason_counts": dict(sorted(abstentions.items())),
        },
        "critical_alpha": {
            "E0": {
                "median_absolute_error": e0_median,
                "p95_absolute_error": _nearest_rank(e0_errors, 0.95),
                "median_bracket_distance": median(
                    float(row["e0_bracket_distance"]) for row in paired
                )
                if paired
                else None,
                "spearman": spearman(e0_predicted, observed),
            },
            "E1": {
                "median_absolute_error": e1_median,
                "p95_absolute_error": _nearest_rank(e1_errors, 0.95),
                "median_bracket_distance": median(
                    float(row["e1_bracket_distance"]) for row in paired
                )
                if paired
                else None,
                "spearman": spearman(e1_predicted, observed),
            },
            "paired_median_absolute_error_reduction": (
                e0_median - e1_median
                if e0_median is not None and e1_median is not None
                else None
            ),
            "paired_median_error_improvement_fraction": (
                (e0_median - e1_median) / e0_median
                if e0_median is not None and e1_median is not None and e0_median > 0.0
                else None
            ),
            "prompt_cluster_bootstrap_95": bootstrap,
        },
        "full_ablation_classification": {
            "E0": e0_classification,
            "E1": e1_classification,
        },
        "nonmonotonicity": {
            "pair_count": sum(bool(row["nonmonotonic"]) for row in rows),
            "fraction": sum(bool(row["nonmonotonic"]) for row in rows) / len(rows)
            if rows
            else 0.0,
        },
        "intervention_cost": {
            "total_instrumented_calls": total_calls,
            "required_calls_per_frozen_pair": 3,
            "refinement_calls": total_calls - 3 * len(rows),
            "total_calls_per_accepted_e1_prediction": total_calls / accepted_count
            if accepted_count
            else None,
            "one_probe_calls_per_accepted_e1_prediction": len(rows) / accepted_count
            if accepted_count
            else None,
        },
        "by_prompt": {
            prompt["id"]: _subset_metrics(
                [row for row in rows if row["prompt_id"] == prompt["id"]]
            )
            for prompt in prediction["prompts"]
        },
        "by_stratum": {
            name: _subset_metrics([row for row in rows if row["stratum"] == name])
            for name in ("B1", "B2", "B3")
        },
        "decision_checks": checks,
        "pair_results": rows,
        "claim_boundary": config["claim_boundary"],
    }


def validate_serialized_point(
    point: dict[str, Any], pair: dict[str, Any], *, expected_call_index: int
) -> None:
    z = _finite(point.get("target_preactivation"), "target preactivation")
    threshold = _finite(point.get("target_threshold"), "target threshold")
    baseline = _finite(pair.get("source_activation"), "frozen source activation")
    baseline_z = _finite(
        pair.get("target_preactivation"), "frozen target preactivation"
    )
    applied = _finite(point.get("actual_bf16_value_passed"), "applied BF16 value")
    realized = 1.0 - applied / baseline
    mappings = point.get("requested_mappings")
    if not isinstance(mappings, list) or not mappings:
        raise ValueError("Stage 1F point requested mapping provenance is missing")
    if (
        point.get("pair_id") != pair["pair_id"]
        or point.get("prompt_id") != pair["prompt_id"]
        or point.get("source_suppression_api_call_index") != expected_call_index
        or _finite(point.get("baseline_source_activation"), "point baseline source")
        != baseline
        or _finite(point.get("baseline_target_preactivation"), "point baseline z")
        != baseline_z
        or _finite(point.get("baseline_target_threshold"), "point baseline threshold")
        != float(pair["target_threshold"])
        or point.get("baseline_target_active") is not False
        or threshold != float(pair["target_threshold"])
        or point.get("target_active") is not (z > threshold)
        or point.get("strict_crossing") is not (z > threshold)
        or _finite(point.get("target_activation"), "target activation")
        != (z if z > threshold else 0.0)
        or _finite(point.get("realized_suppression"), "realized suppression")
        != realized
        or point.get("source_value_device") != "mps:0"
        or point.get("source_value_dtype") != "torch.bfloat16"
    ):
        raise ValueError(
            "Stage 1F point baseline, gate, dtype, or device evidence differs"
        )
    for raw in mappings:
        if type(raw) is not dict:
            raise ValueError("Stage 1F requested mapping is malformed")
        mapping = cast(dict[str, Any], raw)
        requested = _finite(mapping.get("requested_alpha"), "requested alpha")
        desired = (1.0 - requested) * baseline
        if (
            not 0.0 <= requested <= 1.0
            or _finite(mapping.get("desired_high_precision"), "desired source")
            != desired
            or _finite(mapping.get("actual_bf16_value_passed"), "mapping applied")
            != applied
            or applied != bf16_round(desired)
            or _finite(mapping.get("realized_suppression"), "mapping realized")
            != realized
        ):
            raise ValueError("Stage 1F desired/applied/realized mapping differs")


def _validate_schedules(
    pairs: dict[str, dict[str, Any]],
    grouped: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> None:
    maximum = int(config["refinement"]["maximum_bisection_steps"])
    stop = float(config["refinement"]["stop_bracket_width"])
    for pair_id, pair in pairs.items():
        rows = grouped[pair_id]
        if not 3 <= len(rows) <= 3 + maximum:
            raise ValueError("Stage 1F point count is outside the frozen schedule")
        required = {float(alpha) for alpha in pair["requested_alphas"]}
        observed = {alpha for point in rows for alpha in _requested(point)}
        if not required.issubset(observed):
            raise ValueError("Stage 1F required point schedule is incomplete")
        analysis = _pair_analysis(pair, rows)
        if (
            not analysis["observed_full_ablation_crossing"]
            or analysis["e1_status"] != "accepted"
        ):
            if len(rows) != 3:
                raise ValueError("Stage 1F refinement ran for an ineligible pair")
            continue
        bracket = analysis["observed_critical_bracket"]
        if bracket is None:
            if not analysis["nonmonotonic"]:
                raise ValueError("Stage 1F crossing pair lacks a final bracket")
            continue
        width = float(bracket["width"])
        refinement_calls = len(rows) - 3
        if width > stop and refinement_calls < maximum:
            lower = float(bracket["lower_realized_alpha"])
            upper = float(bracket["upper_realized_alpha"])
            midpoint_request = (lower + upper) / 2.0
            baseline = float(pair["source_activation"])
            midpoint_applied = bf16_round((1.0 - midpoint_request) * baseline)
            existing = {float(row["actual_bf16_value_passed"]) for row in rows}
            if midpoint_applied not in existing:
                raise ValueError(
                    "Stage 1F refinement stopped before a frozen stop condition"
                )


def build_records(
    *,
    protocol: dict[str, Any],
    prediction: dict[str, Any],
    worker: dict[str, Any],
    points: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    pairs, grouped = group_points(prediction, points)
    _validate_schedules(pairs, grouped, config)
    analysis = compute_analysis(prediction, grouped, config)
    sweeps = {
        "schema_version": 1,
        "artifact_type": "stage1f_point_sweeps",
        "status": "passed",
        "experiment_class": EXPERIMENT_CLASS,
        "sweeps": [
            {
                "prompt_id": pair["prompt_id"],
                "pair_id": pair_id,
                "stratum": pair["stratum"],
                "point_count": len(grouped[pair_id]),
                "points": grouped[pair_id],
            }
            for pair_id, pair in pairs.items()
        ],
    }
    telemetry = worker["telemetry"]
    environment = {
        "schema_version": 1,
        "artifact_type": "stage1f_environment_manifest",
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
        },
    }
    run = {
        "schema_version": 1,
        "artifact_type": "stage1f_run_manifest",
        "status": analysis["terminal_class"],
        "project_decision": analysis["project_decision"],
        "experiment_class": EXPERIMENT_CLASS,
        "branch": BRANCH,
        "base_commit": BASE_COMMIT,
        "protocol_freeze_commit": protocol["protocol_commit"],
        "prediction_freeze_commit": worker["prediction_freeze_commit"],
        "pre_run_commit": worker["pre_run_commit"],
        "canonical_attempt_count": 1,
        "scientific_retry_count": 0,
        "instrumented_source_suppression_api_calls": len(points),
        "journal_completed_point_count": len(points),
        "serialized_unique_point_count": len(points),
        "final_artifacts_rebuilt_from_journal_in_fresh_process": True,
        "standalone_recomputation_required": True,
        "historical_stage1e_status": "completed_stage1e_offline_negative",
        "claim_boundary": config["claim_boundary"],
    }
    return {
        "protocol_manifest.json": protocol,
        "prediction_manifest.json": prediction,
        "point_sweeps.json": sweeps,
        "analysis_summary.json": analysis,
        "run_manifest.json": run,
        "environment_manifest.json": environment,
    }


def publish_records(output: Path, records: dict[str, dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise ValueError("Stage 1F artifact output is a symlink")
    for name in JSON_ARTIFACTS:
        path = output / name
        value = records[name]
        if path.exists():
            if read_json_strict(path) != value:
                raise ValueError(f"existing frozen Stage 1F artifact differs: {name}")
        else:
            write_json_new(path, value)
    checksum = output / "checksums.sha256"
    if checksum.exists():
        raise ValueError("Stage 1F checksum sidecar already exists")
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
            raise ValueError("short Stage 1F checksum write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_bundle(repository: Path, output: Path) -> dict[str, Any]:
    config = load_config(repository / CONFIG_PATH)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("Stage 1F bundle directory is unsafe")
    names = {path.name for path in output.iterdir()}
    if names != {*JSON_ARTIFACTS, "checksums.sha256"}:
        raise ValueError("Stage 1F bundle file allowlist differs")
    checksum_lines = (
        (output / "checksums.sha256").read_text(encoding="ascii").splitlines()
    )
    expected_lines = [
        f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}"
        for name in JSON_ARTIFACTS
    ]
    if checksum_lines != expected_lines:
        raise ValueError("Stage 1F bundle checksum sidecar differs")
    records = {name: read_json_strict(output / name) for name in JSON_ARTIFACTS}
    if any(type(value) is not dict for value in records.values()):
        raise ValueError("Stage 1F artifact is not an object")
    protocol = cast(dict[str, Any], records["protocol_manifest.json"])
    prediction = cast(dict[str, Any], records["prediction_manifest.json"])
    validate_protocol(repository, protocol, config)
    pairs = validate_prediction(prediction, config, protocol)
    point_artifact = cast(dict[str, Any], records["point_sweeps.json"])
    sweeps = point_artifact.get("sweeps")
    if (
        point_artifact.get("artifact_type") != "stage1f_point_sweeps"
        or point_artifact.get("status") != "passed"
        or not isinstance(sweeps, list)
    ):
        raise ValueError("Stage 1F point artifact identity differs")
    grouped: dict[str, list[dict[str, Any]]] = {}
    points: list[dict[str, Any]] = []
    for raw in sweeps:
        if type(raw) is not dict or not isinstance(raw.get("points"), list):
            raise ValueError("Stage 1F serialized sweep is malformed")
        sweep = cast(dict[str, Any], raw)
        pair_id = str(sweep.get("pair_id"))
        rows = cast(list[dict[str, Any]], sweep["points"])
        if (
            pair_id in grouped
            or pair_id not in pairs
            or sweep.get("prompt_id") != pairs[pair_id]["prompt_id"]
            or sweep.get("stratum") != pairs[pair_id]["stratum"]
            or sweep.get("point_count") != len(rows)
        ):
            raise ValueError("Stage 1F serialized sweep identity differs")
        grouped[pair_id] = rows
        points.extend(rows)
    if set(grouped) != set(pairs):
        raise ValueError("Stage 1F serialized pair set differs from prediction freeze")
    points.sort(key=lambda row: int(row["source_suppression_api_call_index"]))
    if [row["source_suppression_api_call_index"] for row in points] != list(
        range(1, len(points) + 1)
    ):
        raise ValueError("Stage 1F serialized call indices differ")
    for index, point in enumerate(points, start=1):
        validate_serialized_point(
            point, pairs[str(point["pair_id"])], expected_call_index=index
        )
    _validate_schedules(pairs, grouped, config)
    recomputed = compute_analysis(prediction, grouped, config)
    if records["analysis_summary.json"] != recomputed:
        raise ValueError(
            "Stage 1F analysis differs from standalone serialized recomputation"
        )
    run = cast(dict[str, Any], records["run_manifest.json"])
    if (
        run.get("status") != recomputed["terminal_class"]
        or run.get("project_decision") != recomputed["project_decision"]
        or run.get("canonical_attempt_count") != 1
        or run.get("scientific_retry_count") != 0
        or run.get("instrumented_source_suppression_api_calls") != len(points)
        or run.get("journal_completed_point_count") != len(points)
        or run.get("serialized_unique_point_count") != len(points)
        or run.get("historical_stage1e_status") != "completed_stage1e_offline_negative"
        or run.get("claim_boundary") != config["claim_boundary"]
    ):
        raise ValueError(
            "Stage 1F run counts, terminal result, or claim boundary differs"
        )
    environment = cast(dict[str, Any], records["environment_manifest.json"])
    telemetry = environment.get("telemetry")
    if (
        type(telemetry) is not dict
        or telemetry.get("violations") != []
        or telemetry.get("telemetry_failures") != 0
        or environment.get("privacy")
        != {
            "network_accessed": False,
            "credential_values_read": False,
            "secret_values_recorded": False,
            "private_paths_recorded": False,
        }
    ):
        raise ValueError("Stage 1F telemetry or privacy evidence differs")
    total_size = sum(path.stat().st_size for path in output.iterdir())
    if total_size > int(config["safety_limits"]["maximum_artifact_bundle_bytes"]):
        raise ValueError("Stage 1F artifact bundle exceeds the frozen size cap")
    return {
        "status": "passed",
        "terminal_class": recomputed["terminal_class"],
        "project_decision": recomputed["project_decision"],
        "prompt_count": 10,
        "pair_count": len(pairs),
        "paired_reference_pair_count": recomputed["paired_reference_pair_count"],
        "instrumented_api_call_count": len(points),
        "journal_completed_point_count": len(points),
        "serialized_unique_point_count": len(points),
        "checksums_verified": len(JSON_ARTIFACTS),
    }


__all__ = [
    "BASE_COMMIT",
    "BRANCH",
    "CONFIG_PATH",
    "EXPERIMENT_CLASS",
    "JSON_ARTIFACTS",
    "PROTOCOL_FILES",
    "REQUIRED_ANCESTOR",
    "RUNTIME_FINGERPRINT",
    "SCHEMA_PATH",
    "build_prediction_manifest",
    "build_protocol_manifest",
    "build_records",
    "classify_terminal",
    "compute_analysis",
    "execute_frozen_pairs",
    "first_probe_mapping",
    "group_points",
    "load_config",
    "protocol_hashes",
    "publish_records",
    "read_completed_journal",
    "select_q_panel",
    "validate_bundle",
    "validate_prediction",
    "validate_protocol",
    "verify_git",
]
