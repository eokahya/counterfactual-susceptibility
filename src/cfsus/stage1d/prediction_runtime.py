"""Baseline-only eight-prompt prediction runtime for Stage 1D."""

from __future__ import annotations

import gc
from collections import Counter
from typing import Any

from cfsus.scanning.near_threshold import compare_scanner_results
from cfsus.stage1c_v3.prediction import (
    build_prospective_pair,
    causally_eligible,
    filter_source_pool,
    filter_target_pool,
    pair_score_digest,
    source_pool_digest,
    target_pool_digest,
)
from cfsus.stage1c_v3.runtime import Stage1CPredictionBackend
from cfsus.stage1d.benchmark import select_prompt_panels
from cfsus.stage1d.config import BRANCH, EXPERIMENT_CLASS

RUNTIME_FINGERPRINT = (
    "gemma3-270m@9b0cfec892e2/plt@fada11860ac1/"
    "circuit-tracer@8f1e2438df61/nnsight/mps/bf16/stage1d"
)


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
        raise RuntimeError(f"{prompt_id} token IDs differ from the protocol freeze")
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
        pairwise = backend.targeted_local_responses(
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
    if tuple(float(tile[index][index]) for index in range(calibration_count)) != tuple(
        float(item.response) for item in pairwise
    ):
        raise RuntimeError(f"{prompt_id} many-source VJP health calibration differs")
    del calibration_pairs, pairwise, tile
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
                targets=selected_targets,
                sources=sources,
                maximum_targets=batch_size,
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
    selection = select_prompt_panels(pairs, prompt_id=prompt_id, config=config)
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
            "q_sign_counts": {
                "positive": q_counts["positive"],
                "zero": q_counts["zero"],
                "negative": q_counts["negative"],
            },
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
        "method_pair_ids": selection.method_pair_ids,
        "directional_pair_ids": selection.directional_pair_ids,
        "detailed_pair_ids": selection.detailed_pair_ids,
        "execution_pairs": selection.execution_pairs,
        "quantization_audit": selection.quantization_audit,
        "missing_strata": selection.missing_strata,
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
    protocol_manifest: dict[str, Any],
    git_identity: dict[str, Any],
) -> dict[str, Any]:
    """Run all eight baseline-only predictions with one loaded model."""

    prompt_rows = [
        _prompt_prediction(model, torch, sampler, prompt, config)
        for prompt in config["prompts"]
    ]
    total_pairs = sum(len(prompt["execution_pairs"]) for prompt in prompt_rows)
    return {
        "schema_version": 1,
        "artifact_type": "stage1d_prediction_manifest",
        "status": "prediction_frozen_ready_for_commit",
        "experiment_class": EXPERIMENT_CLASS,
        "branch": BRANCH,
        "base_commit": config["base_commit"],
        "protocol_commit": protocol_manifest["protocol_commit"],
        "protocol_map_sha256": protocol_manifest["protocol_map_sha256"],
        "prediction_execution_commit": git_identity["head"],
        "runtime_identity": dict(config["runtime"]),
        "protocol": {
            key: config[key]
            for key in (
                "scanner",
                "source_pool",
                "responses",
                "scoring",
                "quantization_resolvability",
                "full_ablation_panel",
                "detailed_panel",
                "schedules",
                "metrics",
                "decision_rule",
                "intervention",
            )
        },
        "prompt_order": [prompt["id"] for prompt in config["prompts"]],
        "prompts": prompt_rows,
        "selection_totals": {
            "prompt_count": len(prompt_rows),
            "execution_pair_membership_row_count": total_pairs,
            "unique_experiment_pair_count": len(
                {
                    pair["pair_id"]
                    for prompt in prompt_rows
                    for pair in prompt["execution_pairs"]
                }
            ),
        },
        "prediction_only_guards": {
            "evaluation_source_suppression_api_calls": 0,
            "historical_intervention_outcomes_read": False,
            "norway_development_outcome_used": False,
            "graph_edge_input_used_for_inactive_predictions": False,
            "network_accessed": False,
        },
        "claim_boundary": dict(config["claim_boundary"]),
    }


__all__ = ["RUNTIME_FINGERPRINT", "build_prediction_manifest"]
