#!/usr/bin/env python3
"""Run the baseline-only Stage 1C-v3 prediction phase.

This worker has no intervention-worker import. It reads only the authenticated
frozen v1 baseline prediction manifest and its sanitized exact-pair denylist;
historical intervention results and v2 temporary outputs are never read.
Runtime/model code is imported only after the offline identity gates pass.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cfsus.mps_telemetry import MPSTelemetrySampler  # noqa: E402
from cfsus.reproduction.small_model_mps_bf16 import (  # noqa: E402
    assert_fallback_disabled,
)
from cfsus.scanning.near_threshold import compare_scanner_results  # noqa: E402
from cfsus.stage1b_runtime import (  # noqa: E402
    build_mps_bf16_replacement,
    resolve_offline_snapshots,
)
from cfsus.stage1c_v3.config import (  # noqa: E402
    BASE_COMMIT,
    BRANCH,
    CONFIG_PATH,
    EXPERIMENT_CLASS,
    SCHEMA_PATH,
    load_stage1c_v3_config,
    selected_positions_for_token_ids,
    validate_prompt_token_ids,
)
from cfsus.stage1c_v3.historical import (  # noqa: E402
    DENYLIST_PATH,
    HISTORICAL_MANIFEST_FREEZE_COMMIT,
    HISTORICAL_MANIFEST_PATH,
    assert_expected_prompt_derivation,
    assert_no_historical_exact_pairs,
    load_authenticated_historical_metadata,
    mask_historical_exact_pairs,
)
from cfsus.stage1c_v3.prediction import (  # noqa: E402
    build_prospective_pair,
    causally_eligible,
    filter_source_pool,
    filter_target_pool,
    pair_score_digest,
    select_pair_groups,
    selected_group_records,
    source_pool_digest,
    target_pool_digest,
)
from cfsus.stage1c_v3.runtime import Stage1CPredictionBackend  # noqa: E402
from cfsus.stage1c_v3.serialization import (  # noqa: E402
    SerializationError,
    detach_json,
    write_json_new,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / CONFIG_PATH)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emergency-output", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise RuntimeError(f"protocol file is not a single-link regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol_hashes() -> dict[str, str]:
    paths = (
        CONFIG_PATH.as_posix(),
        SCHEMA_PATH.as_posix(),
        DENYLIST_PATH.as_posix(),
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
    return {path: _sha256(REPOSITORY_ROOT / path) for path in paths}


def _git_identity() -> dict[str, Any]:
    from preflight_stage1c_v3 import verify_git

    return verify_git("prediction")


def _verify_assets(cache: Path) -> dict[str, Any]:
    """Verify already-downloaded Stage 1A snapshots without network access."""

    model, transcoder = resolve_offline_snapshots(cache, REPOSITORY_ROOT)
    result = subprocess.run(
        [
            sys.executable,
            str(
                REPOSITORY_ROOT
                / "scripts/stage1a/verify_small_model_mps_bf16_assets.py"
            ),
            "--hf-cache",
            str(cache),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
        env={
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "HF_TOKEN",
                "HUGGING_FACE_HUB_TOKEN",
                "GITHUB_TOKEN",
                "GH_TOKEN",
                "PYTORCH_ENABLE_MPS_FALLBACK",
            }
        }
        | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    record = json.loads(result.stdout)
    if record.get("status") != "verified":
        raise RuntimeError("immutable v3 asset verification failed")
    model_record = record.get("model")
    transcoder_record = record.get("transcoder")
    if not isinstance(model_record, dict) or not isinstance(transcoder_record, dict):
        raise RuntimeError("immutable v3 asset byte evidence is missing")
    return {
        "status": "verified",
        "download_performed": False,
        "network_accessed": False,
        "authentication_used": False,
        "authentication_value_recorded": False,
        "actual_total_bytes": record.get("actual_total_bytes"),
        "model_total_bytes": model_record.get("total_bytes"),
        "transcoder_total_bytes": transcoder_record.get("total_bytes"),
        "model_revision": model.name,
        "transcoder_revision": transcoder.name,
        "exact_allowlist_hashes_verified": True,
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
        raise RuntimeError("insufficient active-only VJP calibration endpoints")
    return tuple(selected)


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    config = load_stage1c_v3_config(args.config, require_token_ids=True)
    prompt_derivation = assert_expected_prompt_derivation()
    historical = load_authenticated_historical_metadata(REPOSITORY_ROOT)
    denylist = frozenset(historical.exact_pairs)
    assert_fallback_disabled()
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
    ):
        raise RuntimeError("v3 prediction worker requires offline mode")
    git_start = _git_identity()
    if git_start["branch"] != BRANCH or git_start.get("working_tree_clean") is not True:
        raise RuntimeError("v3 prediction worker requires the clean protocol commit")

    # These imports are deliberately below all fail-closed identity gates.
    import nnsight  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    import transformers  # type: ignore[import-not-found]

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("native MPS is unavailable")
    if torch.is_autocast_enabled():
        raise RuntimeError("outer autocast must be disabled")
    asset_manifest = _verify_assets(args.hf_cache)
    model_snapshot, transcoder_snapshot = resolve_offline_snapshots(
        args.hf_cache, REPOSITORY_ROOT
    )
    sampler = MPSTelemetrySampler(torch, config["safety_limits"], args.emergency_output)
    sampler_finished = False
    model: Any = None
    pairs: list[Any] = []
    try:
        with sampler.stage("replacement_runtime_loading"):
            model, module_guard = build_mps_bf16_replacement(
                model_snapshot, transcoder_snapshot, torch
            )
            prompt = str(config["prompt"]["text"])
            backend = Stage1CPredictionBackend(
                model,
                prompt=prompt,
                prompt_id=str(config["prompt"]["id"]),
                torch=torch,
            )
            tokens = model.ensure_tokenized(prompt)
            token_ids = [int(item) for item in tokens.detach().cpu().tolist()]
            validate_prompt_token_ids(config, token_ids)
        positions = selected_positions_for_token_ids(token_ids)
        scanner_config = dict(config["scanner"])
        frozen_positions = scanner_config.get("selected_positions")
        if frozen_positions is not None and frozen_positions != positions:
            raise RuntimeError("scanner positions do not match the frozen token length")
        scanner_config["selected_positions"] = positions
        groups = tuple(
            (layer, position)
            for layer in scanner_config["selected_layers"]
            for position in positions
        )
        with sampler.stage("prediction_dense_oracle"):
            oracle = backend.scan(
                groups=groups,
                chunk_size=int(scanner_config["dense_oracle_chunk_size"]),
                top_k_per_group=int(scanner_config["top_k_per_group"]),
                global_top_k=int(scanner_config["global_top_k"]),
            )
        with sampler.stage("prediction_scanner_chunk_1024"):
            scanner = backend.scan(
                groups=groups,
                chunk_size=int(scanner_config["canonical_chunk_size"]),
                top_k_per_group=int(scanner_config["top_k_per_group"]),
                global_top_k=int(scanner_config["global_top_k"]),
            )
            compare_scanner_results(oracle, scanner)
        with sampler.stage("prediction_active_source_pool"):
            raw_sources = backend.collect_active_sources(
                groups=groups, chunk_size=int(scanner_config["canonical_chunk_size"])
            )
            eligible_targets = filter_target_pool(
                scanner.global_candidates, raw_sources
            )
            sources = filter_source_pool(
                raw_sources,
                eligible_targets,
                maximum_sources=int(config["source_pool"]["maximum_active_sources"]),
            )
            eligible_targets = filter_target_pool(eligible_targets, sources)
            eligible_pair_count = sum(
                causally_eligible(source.feature, target.feature)
                for target in eligible_targets
                for source in sources
            )
            if eligible_pair_count > int(config["responses"]["maximum_eligible_pairs"]):
                raise RuntimeError("v3 eligible pair pool exceeds the frozen cap")
        calibration_count = int(config["engineering_calibration"]["pair_count"])
        calibration_pairs = _active_calibration_pairs(
            raw_sources, pair_count=calibration_count
        )
        with sampler.stage("prediction_many_source_vjp_active_calibration"):
            reference = backend.targeted_local_responses(
                tuple(
                    (source.feature, target.feature)
                    for source, target in calibration_pairs
                ),
                maximum_pairs=calibration_count,
            )
            calibration_tile = backend.response_tile(
                targets=tuple(target.feature for _, target in calibration_pairs),
                sources=tuple(source for source, _ in calibration_pairs),
                maximum_targets=calibration_count,
            )
        if tuple(
            float(calibration_tile[i][i]) for i in range(calibration_count)
        ) != tuple(float(item.response) for item in reference):
            raise RuntimeError("many-source VJP differs from pairwise calibration")
        del calibration_tile, reference
        gc.collect()
        torch.mps.empty_cache()

        target_by_feature = {item.feature: item for item in eligible_targets}
        target_features = tuple(sorted(target_by_feature))
        batch_size = int(config["responses"]["target_batch_size"])
        runtime_fingerprint = (
            "gemma3-270m@9b0cfec892e2/plt@fada11860ac1/"
            "circuit-tracer@8f1e2438df61/nnsight/mps/bf16/stage1c-v3"
        )
        for start in range(0, len(target_features), batch_size):
            selected_targets = target_features[start : start + batch_size]
            with sampler.stage(f"prediction_target_vjp_batch_{start // batch_size}"):
                tile = backend.response_tile(
                    targets=selected_targets,
                    sources=sources,
                    maximum_targets=batch_size,
                )
            for target, row in zip(selected_targets, tile, strict=True):
                candidate = target_by_feature[target]
                for source, response in zip(sources, row, strict=True):
                    if (
                        source.feature.layer >= target.layer
                        or source.feature.position > target.position
                    ):
                        continue
                    pairs.append(
                        build_prospective_pair(
                            source=source,
                            target=candidate,
                            targeted_response=response,
                            seed=str(config["scoring"]["pair_seed"]),
                            prompt_id=str(config["prompt"]["id"]),
                            runtime_fingerprint=runtime_fingerprint,
                            epsilon=float(config["scoring"]["epsilon"]),
                            tolerance=float(config["scoring"]["crossing_tolerance"]),
                        )
                    )
            del tile
            gc.collect()
            torch.mps.empty_cache()
        if len(pairs) != eligible_pair_count:
            raise RuntimeError("v3 eligible pair enumeration is incomplete")
        pre_mask_score_digest = pair_score_digest(pairs)
        masked_pairs, exact_pair_exclusion_count = mask_historical_exact_pairs(
            pairs, denylist
        )
        selected = select_pair_groups(
            masked_pairs,
            selection=config["selection"],
            tolerance=float(config["scoring"]["crossing_tolerance"]),
        )
        selected_pairs = (
            *selected.primary,
            *selected.near_boundary,
            *selected.directional,
        )
        assert_no_historical_exact_pairs(selected_pairs, denylist)
        selected_records = selected_group_records(
            selected,
            schedule=config["schedule"],
            denylist=denylist,
            historical_endpoints=historical.historical_endpoints,
        )
        selected_rows = [row for rows in selected_records.values() for row in rows]
        overlap_counts = Counter(
            str(row["endpoint_overlap_category"]) for row in selected_rows
        )
        raw_status_counts = Counter(item.status.value for item in masked_pairs)
        status_counts = {
            key: raw_status_counts.get(key, 0)
            for key in (
                "boundary_ambiguous",
                "definitely_crossing",
                "not_crossing",
            )
        }
        q_counts = {
            "positive": sum(item.q > 0.0 for item in masked_pairs),
            "zero": sum(item.q == 0.0 for item in masked_pairs),
            "negative": sum(item.q < 0.0 for item in masked_pairs),
        }
        telemetry = sampler.finish()
        sampler_finished = True
        if telemetry["violations"] or telemetry["telemetry_failures"]:
            raise RuntimeError("v3 prediction telemetry contains a safety failure")
        git_end = _git_identity()
        if git_end != git_start:
            raise RuntimeError("v3 prediction worktree identity changed")
        manifest = {
            "schema_version": 3,
            "artifact_type": "stage1c_v3_prediction_manifest",
            "status": "prediction_frozen_ready_for_commit",
            "experiment_class": EXPERIMENT_CLASS,
            "base_commit": BASE_COMMIT,
            "branch": BRANCH,
            "pair_id_domain": f"{EXPERIMENT_CLASS}:capital_norway_preregistered_v3",
            "runtime_identity": {
                "backend": "nnsight",
                "device": "mps:0",
                "dtype": "torch.bfloat16",
                "model_identifier": config["model"]["identifier"],
                "model_revision": config["model"]["revision"],
                "transcoder_identifier": config["transcoder"]["identifier"],
                "transcoder_revision": config["transcoder"]["revision"],
                "transcoder_subfolder": config["transcoder"]["subfolder"],
                "layer_count": config["transcoder"]["layer_count"],
                "feature_width": config["transcoder"]["feature_width"],
                "upstream_revision": config["upstream"]["revision"],
            },
            "prompt": {
                "id": config["prompt"]["id"],
                "text": config["prompt"]["text"],
                "token_ids": token_ids,
            },
            "prompt_derivation": {
                "algorithm": config["prompt_derivation"]["algorithm"],
                "base_commit": config["prompt_derivation"]["base_commit"],
                "salt": config["prompt_derivation"]["salt"],
                "message": prompt_derivation.message,
                "sha256_hex": prompt_derivation.sha256_hex,
                "index": prompt_derivation.index,
                "prompt": prompt_derivation.prompt,
                "prompt_id": prompt_derivation.prompt_id,
                "pool": list(config["prompt_derivation"]["pool"]),
            },
            "historical_independence": {
                "source_manifest_path": HISTORICAL_MANIFEST_PATH.as_posix(),
                "source_manifest_sha256": historical.source_manifest_sha256,
                "source_manifest_git_blob_sha1": (
                    historical.source_manifest_git_blob_sha1
                ),
                "source_manifest_freeze_commit": (HISTORICAL_MANIFEST_FREEZE_COMMIT),
                "denylist_path": DENYLIST_PATH.as_posix(),
                "denylist_sha256": historical.denylist_sha256,
                "exact_pair_count": len(historical.exact_pairs),
                "historical_endpoint_count": len(historical.historical_endpoints),
                "mask_applied_before_ranking": True,
                "endpoint_overlap_policy": "audit_only",
                "historical_intervention_outcome_read": False,
                "v2_temporary_baseline_artifact_read": False,
            },
            "protocol": {
                "scanner": {**scanner_config, "selected_positions": positions},
                "source_pool": dict(config["source_pool"]),
                "responses": dict(config["responses"]),
                "engineering_calibration": dict(config["engineering_calibration"]),
                "scoring": dict(config["scoring"]),
                "selection": dict(config["selection"]),
                "schedule": dict(config["schedule"]),
                "intervention_regime": dict(config["intervention"]),
                "analysis": dict(config["analysis"]),
            },
            "baseline_pools": {
                "scanner_candidate_count": len(scanner.global_candidates),
                "eligible_target_count": len(eligible_targets),
                "excluded_no_causal_source_target_count": len(scanner.global_candidates)
                - len(eligible_targets),
                "target_pool_sha256": target_pool_digest(eligible_targets),
                "raw_active_source_count": len(raw_sources),
                "eligible_source_count": len(sources),
                "source_pool_sha256": source_pool_digest(sources),
                "eligible_pair_count_before_historical_mask": eligible_pair_count,
                "excluded_exact_historical_pair_count": (exact_pair_exclusion_count),
                "eligible_pair_count_after_historical_mask": len(masked_pairs),
                "pair_score_sha256_before_historical_mask": (pre_mask_score_digest),
                "pair_score_sha256_after_historical_mask": (
                    pair_score_digest(masked_pairs)
                ),
                "predicted_status_counts": status_counts,
                "q_sign_counts": q_counts,
                "complete_derivative_matrix_persisted": False,
                "dense_scanner_arrays_persisted": False,
                "many_source_vjp_engineering_calibration": {
                    "endpoint_class": config["engineering_calibration"][
                        "endpoint_class"
                    ],
                    "pair_count": calibration_count,
                    "reference_method": config["engineering_calibration"][
                        "reference_method"
                    ],
                    "comparison": "exact_bf16_identity",
                    "passed": True,
                    "inactive_target_intervention_calls": 0,
                },
                "scanner_dense_oracle_validation": {
                    "group_count": len(groups),
                    "exact_dense_oracle_identity_and_order": True,
                    "bounded_oracle_recall": 1.0,
                    "dense_oracle_persisted": False,
                },
            },
            "selected_groups": selected_records,
            "selection_audit": {
                "primary_count": len(selected.primary),
                "near_boundary_count": len(selected.near_boundary),
                "directional_count": len(selected.directional),
                "near_overlap_fallback_count": selected.near_overlap_fallback_count,
                "directional_overlap_fallback_count": (
                    selected.directional_overlap_fallback_count
                ),
                "groups_disjoint": True,
                "primary_target_unique": True,
                "primary_source_cap": 2,
                "exact_pair_mask_applied_before_ranking": True,
                "selected_exact_pair_overlap_count": 0,
                "endpoint_overlap_category_counts": {
                    key: overlap_counts.get(key, 0)
                    for key in (
                        "neither_endpoint_seen_in_v1",
                        "source_endpoint_seen_in_v1_only",
                        "target_endpoint_seen_in_v1_only",
                        "both_endpoints_seen_but_not_as_exact_v1_pair",
                    )
                },
                "endpoint_overlap_used_for_ranking_or_quota": False,
            },
            "prediction_only_guards": {
                "source_suppression_api_calls": 0,
                "prior_inactive_target_outcome_read": False,
                "historical_intervention_outcome_read": False,
                "v2_temporary_baseline_artifact_read": False,
                "intervention_worker_imported": False,
                "raw_graph_read": False,
                "raw_adjacency_read": False,
            },
            "protocol_file_sha256": _protocol_hashes(),
            "config_sha256": _sha256(args.config),
            "artifact_schema_sha256": _sha256(REPOSITORY_ROOT / SCHEMA_PATH),
            "claim_boundary": {
                "behavioral_importance_result": "none",
                "mediation_result": "none",
                "official_bf16_reproduction": "pending",
                "reference_clt_reproduction": "pending",
                "paper_results_readiness": False,
            },
        }
        detached = detach_json(manifest)
        if not isinstance(detached, dict):  # pragma: no cover
            raise SerializationError("v3 prediction manifest is not an object")
        return {
            "schema_version": 3,
            "artifact_type": "stage1c_v3_prediction_worker",
            "status": "passed",
            "git": git_end,
            "prediction_manifest": detached,
            "asset_manifest": asset_manifest,
            "environment": {
                "system": platform.system(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "circuit-tracer": importlib.metadata.version("circuit-tracer"),
                "torch": torch.__version__,
                "nnsight": nnsight.__version__,
                "transformers": transformers.__version__,
                "mps_built": torch.backends.mps.is_built(),
                "mps_available": torch.backends.mps.is_available(),
                "fallback_variable_present": "PYTORCH_ENABLE_MPS_FALLBACK"
                in os.environ,
                "outer_autocast_enabled": torch.is_autocast_enabled(),
            },
            "module_guard": module_guard,
            "scanner_validation": {
                "group_count": len(groups),
                "exact_dense_oracle_identity_and_order": True,
                "bounded_oracle_recall": 1.0,
                "dense_oracle_persisted": False,
            },
            "telemetry": telemetry,
        }
    finally:
        if not sampler_finished:
            with contextlib.suppress(Exception):
                sampler.finish()
        pairs.clear()
        if model is not None:
            del model
        gc.collect()
        if "torch" in locals():
            with contextlib.suppress(Exception):
                torch.mps.empty_cache()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists() or args.emergency_output.exists():
        raise RuntimeError("v3 prediction output paths must be new")
    if not args.output.parent.is_dir() or not args.emergency_output.parent.is_dir():
        raise RuntimeError("v3 prediction output parents must exist")
    result = _execute(args)
    write_json_new(args.output, result)
    print(
        json.dumps({"status": result["status"], "phase": "prediction"}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
