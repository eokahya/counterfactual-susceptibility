#!/usr/bin/env python3
"""Fresh-process Stage 1B calibration/canonical measurement worker."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cfsus.backends.nnsight_plt import (  # noqa: E402
    NNSightPLTMeasurementBackend,
)
from cfsus.mps_telemetry import MPSTelemetrySampler  # noqa: E402
from cfsus.reproduction.artifacts import write_json_atomic  # noqa: E402
from cfsus.reproduction.small_model_mps_bf16 import (  # noqa: E402
    assert_fallback_disabled,
)
from cfsus.responses.validation import (  # noqa: E402
    compute_local_response_metrics,
    extract_active_pair_references,
    select_disjoint_pair_references,
    symmetric_normalized_error,
    validate_pair_distribution,
)
from cfsus.scanning.near_threshold import compare_scanner_results  # noqa: E402
from cfsus.stage1b import CONFIG_PATH, load_stage1b_config  # noqa: E402
from cfsus.stage1b_runtime import (  # noqa: E402
    FEATURE_WIDTH,
    build_mps_bf16_replacement,
    build_raw_reference_graph,
    resolve_offline_snapshots,
)
from cfsus.types import ActivePairReference, NearThresholdCandidate  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / CONFIG_PATH)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--mode", choices=("calibration", "canonical"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emergency-output", type=Path, required=True)
    return parser


def _git_identity() -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            env={
                key: value
                for key, value in os.environ.items()
                if not key.startswith("GIT_")
            },
        )
        return result.stdout.strip()

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "working_tree_clean": run("status", "--porcelain") == "",
    }


def _verify_assets(cache: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/stage1a/verify_small_model_mps_bf16_assets.py"),
        "--hf-cache",
        str(cache),
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
        env={
            **{
                key: value
                for key, value in os.environ.items()
                if key not in {"HF_TOKEN", "PYTORCH_ENABLE_MPS_FALLBACK"}
            },
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )
    value = json.loads(result.stdout)
    if value.get("status") != "verified":
        raise RuntimeError("immutable asset verifier did not pass")
    return dict(value)


def _feature_dict(feature: Any) -> dict[str, int]:
    return {
        "layer": int(feature.layer),
        "position": int(feature.position),
        "feature_id": int(feature.feature_id),
    }


def _candidate_dict(candidate: NearThresholdCandidate) -> dict[str, Any]:
    return {
        "feature": _feature_dict(candidate.feature),
        "preactivation": candidate.preactivation,
        "activation": candidate.activation,
        "threshold": candidate.threshold,
        "margin": candidate.margin,
        "activity": "inactive",
        "device": candidate.device,
        "dtype": candidate.dtype,
    }


def _reference_dict(reference: ActivePairReference) -> dict[str, Any]:
    return {
        "pair_id": reference.pair_id,
        "source": _feature_dict(reference.source),
        "target": _feature_dict(reference.target),
        "source_activation": reference.source_activation,
        "raw_edge": reference.raw_edge,
    }


def _capability_dict(backend: NNSightPLTMeasurementBackend) -> dict[str, Any]:
    report = backend.capability_report()
    return {
        "backend_name": report.backend_name,
        "dependency_available": report.dependency_available,
        "dependency_version": report.dependency_version,
        "evidence": [asdict(item) for item in report.evidence],
    }


def _endpoint_records(
    references: tuple[ActivePairReference, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "pair_id": item.pair_id,
            "source": _feature_dict(item.source),
            "target": _feature_dict(item.target),
        }
        for item in references
    ]


def _endpoint_digest(references: tuple[ActivePairReference, ...]) -> str:
    encoded = json.dumps(
        _endpoint_records(references),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pair_rows(
    references: tuple[ActivePairReference, ...], estimates: tuple[Any, ...]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reference, estimate in zip(references, estimates, strict=True):
        rebuilt = estimate.source_activation * estimate.response
        rows.append(
            {
                "pair_id": reference.pair_id,
                "source": _feature_dict(reference.source),
                "target": _feature_dict(reference.target),
                "source_activation": reference.source_activation,
                "target_preactivation": estimate.target_preactivation,
                "raw_edge": reference.raw_edge,
                "targeted_response": estimate.response,
                "reconstructed_edge": rebuilt,
                "symmetric_normalized_error": symmetric_normalized_error(
                    reference.raw_edge, rebuilt
                ),
                "device": estimate.device,
                "dtype": estimate.dtype,
                "method": estimate.method,
                "convention": estimate.convention,
                "graph_edge_used": estimate.graph_edge_used,
            }
        )
    return rows


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    config = load_stage1b_config(args.config)
    expected_phase = "calibration" if args.mode == "calibration" else "canonical_frozen"
    if config["phase"] != expected_phase:
        raise RuntimeError("worker mode and frozen config phase differ")
    assert_fallback_disabled()
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
    ):
        raise RuntimeError("measurement worker requires offline asset mode")

    import nnsight  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    import transformers  # type: ignore[import-not-found]

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("native MPS is unavailable")
    if torch.is_autocast_enabled():
        raise RuntimeError("outer autocast must be disabled")
    git_start = _git_identity()
    if args.mode == "canonical" and (
        git_start["branch"] != config["branch"]
        or not git_start["working_tree_clean"]
        or git_start["head"] == config["base_commit"]
    ):
        raise RuntimeError("canonical worker requires the clean pre-run commit")
    asset_manifest = _verify_assets(args.hf_cache)
    model_snapshot, transcoder_snapshot = resolve_offline_snapshots(
        args.hf_cache, REPOSITORY_ROOT
    )
    sampler = MPSTelemetrySampler(torch, config["safety_limits"], args.emergency_output)
    sampler_finished = False
    model: Any = None
    graph: Any = None
    try:
        with sampler.stage("replacement_runtime_loading"):
            model, module_guard = build_mps_bf16_replacement(
                model_snapshot, transcoder_snapshot, torch
            )
            backend = NNSightPLTMeasurementBackend(
                model,
                prompt=str(config["prompt"]["text"]),
                prompt_id=str(config["prompt"]["id"]),
                torch=torch,
            )
            tokens = model.ensure_tokenized(str(config["prompt"]["text"]))
            token_ids = [int(item) for item in tokens.detach().cpu().tolist()]
            if token_ids != config["prompt"]["expected_token_ids"]:
                raise RuntimeError("accepted prompt token identity changed")

        scanner_config = config["scanner"]
        groups = tuple(
            (layer, position)
            for layer in scanner_config["selected_layers"]
            for position in scanner_config["selected_positions"]
        )
        with sampler.stage("scanner_dense_oracle"):
            oracle = backend.scan(
                groups=groups,
                chunk_size=FEATURE_WIDTH,
                top_k_per_group=int(scanner_config["top_k_per_group"]),
                global_top_k=int(scanner_config["global_top_k"]),
            )
        scanner_results: dict[int, Any] = {}
        for chunk_size in scanner_config["chunk_sizes"]:
            with sampler.stage(f"scanner_chunk_{chunk_size}"):
                result = backend.scan(
                    groups=groups,
                    chunk_size=int(chunk_size),
                    top_k_per_group=int(scanner_config["top_k_per_group"]),
                    global_top_k=int(scanner_config["global_top_k"]),
                )
                compare_scanner_results(oracle, result)
                scanner_results[int(chunk_size)] = result

        response_config = config["responses"]
        with sampler.stage("ephemeral_graph_reference"):
            graph, graph_usage = build_raw_reference_graph(
                model,
                prompt=str(config["prompt"]["text"]),
                graph_config=dict(response_config["graph"]),
                maximum_graph_buffer_bytes=int(
                    config["safety_limits"]["maximum_graph_buffer_bytes"]
                ),
                torch=torch,
            )
            pool = extract_active_pair_references(
                graph,
                seed=str(response_config["pair_seed"]),
                prompt_id=str(config["prompt"]["id"]),
                runtime_fingerprint=(
                    "gemma3-270m@9b0cfec892e2/plt@fada11860ac1/"
                    "circuit-tracer@8f1e2438df61/nnsight/mps/bf16"
                ),
                per_target_per_sign=int(response_config["per_target_per_sign"]),
            )
            calibration_refs, canonical_refs = select_disjoint_pair_references(
                pool,
                calibration_count=int(response_config["calibration_pair_count"]),
                canonical_count=int(response_config["canonical_pair_count"]),
                minimum_target_layers=int(response_config["minimum_target_layers"]),
                minimum_target_positions=int(
                    response_config["minimum_target_positions"]
                ),
            )
            validate_pair_distribution(
                canonical_refs,
                minimum_pairs=int(response_config["canonical_pair_count"]),
                minimum_target_layers=int(response_config["minimum_target_layers"]),
                minimum_target_positions=int(
                    response_config["minimum_target_positions"]
                ),
                require_both_signs=bool(response_config["require_both_edge_signs"]),
            )
            if args.mode == "canonical":
                if [item.pair_id for item in calibration_refs] != response_config[
                    "calibration_pair_ids"
                ]:
                    raise RuntimeError("frozen calibration pair IDs changed")
                if [item.pair_id for item in canonical_refs] != response_config[
                    "canonical_pair_ids"
                ]:
                    raise RuntimeError("frozen canonical pair IDs changed")
                if (
                    _endpoint_digest(canonical_refs)
                    != response_config["canonical_endpoint_manifest_sha256"]
                ):
                    raise RuntimeError("frozen canonical pair endpoints changed")
        del graph
        graph = None
        gc.collect()
        torch.mps.empty_cache()

        targeted_references = (
            calibration_refs if args.mode == "calibration" else canonical_refs
        )
        with sampler.stage(f"targeted_vjp_{args.mode}"):
            endpoint_pairs = tuple(
                (item.source, item.target) for item in targeted_references
            )
            estimates = backend.targeted_local_responses(
                endpoint_pairs,
                maximum_pairs=int(response_config["canonical_pair_count"]),
            )
            edge_floor = float(
                response_config[
                    "proposed_edge_floor"
                    if args.mode == "calibration"
                    else "edge_floor"
                ]
            )
            metrics = compute_local_response_metrics(
                targeted_references,
                estimates,
                edge_floor=edge_floor,
            )
            tolerance = config["tolerances"]
            if metrics.spearman < float(tolerance["spearman_minimum"]):
                raise RuntimeError("targeted-response Spearman is below the hard floor")
            if metrics.sign_agreement < float(tolerance["sign_agreement_minimum"]):
                raise RuntimeError(
                    "targeted-response sign agreement is below the hard floor"
                )
            if metrics.median_symmetric_normalized_error > float(
                tolerance["median_symmetric_normalized_error_maximum"]
            ):
                raise RuntimeError("targeted-response median error exceeds the ceiling")
            if metrics.p95_symmetric_normalized_error > float(
                tolerance["p95_symmetric_normalized_error_maximum"]
            ):
                raise RuntimeError("targeted-response p95 error exceeds the ceiling")
            backend.mark_measurement_primitives_validated()

        canonical_scanner = scanner_results[int(scanner_config["canonical_chunk_size"])]
        telemetry = sampler.finish()
        sampler_finished = True
        if telemetry["violations"] or telemetry["telemetry_failures"]:
            raise RuntimeError("measurement telemetry contains a safety failure")
        git_end = _git_identity()
        if args.mode == "canonical" and git_end != git_start:
            raise RuntimeError("canonical worktree identity changed during execution")
        environment = {
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "nnsight": nnsight.__version__,
            "transformers": transformers.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "fallback_variable_present": ("PYTORCH_ENABLE_MPS_FALLBACK" in os.environ),
            "outer_autocast_enabled": torch.is_autocast_enabled(),
        }
        scanner_record = {
            "group_count": len(groups),
            "chunk_sizes": list(scanner_config["chunk_sizes"]),
            "dense_oracle_chunk_size": FEATURE_WIDTH,
            "canonical_chunk_size": int(scanner_config["canonical_chunk_size"]),
            "top_k_per_group": int(scanner_config["top_k_per_group"]),
            "global_top_k": int(scanner_config["global_top_k"]),
            "exact_candidate_identity_and_order": True,
            "bounded_oracle_recall": 1.0,
            "candidate_count": len(canonical_scanner.global_candidates),
            "candidates": [
                _candidate_dict(item) for item in canonical_scanner.global_candidates
            ],
            "maximum_retained_candidates": max(
                item.maximum_retained_candidates
                for result in scanner_results.values()
                for item in result.groups
            ),
            "persisted_dense_arrays": False,
            "loaded_gate": "a=z*1[z>tau]",
            "threshold_equality_activity": "inactive",
            "device": "mps:0",
            "dtype": "torch.bfloat16",
        }
        graph_record = {
            "orientation": "target_row_source_column",
            "formula": "raw_edge=source_activation*targeted_response",
            "persisted": False,
            "eligible_pool_count": len(pool),
            "usage": graph_usage,
        }
        if args.mode == "canonical":
            return {
                "schema_version": 1,
                "artifact_type": "stage1b_measurement_primitives_canonical_worker",
                "status": "passed",
                "fresh_canonical_run": True,
                "scientific_retry_count": 0,
                "calibration_artifact_read": False,
                "git": git_end,
                "environment": environment,
                "asset_manifest": asset_manifest,
                "prompt_id": config["prompt"]["id"],
                "token_ids": token_ids,
                "module_guard": module_guard,
                "capabilities": _capability_dict(backend),
                "scanner": scanner_record,
                "graph_reference": graph_record,
                "pair_selection": {
                    "calibration_pair_ids": [item.pair_id for item in calibration_refs],
                    "canonical_pair_ids": [item.pair_id for item in canonical_refs],
                    "canonical_endpoint_manifest_sha256": _endpoint_digest(
                        canonical_refs
                    ),
                    "disjoint": not bool(
                        {item.pair_id for item in calibration_refs}
                        & {item.pair_id for item in canonical_refs}
                    ),
                    "edge_floor": edge_floor,
                },
                "local_response_validation": {
                    "metrics": asdict(metrics),
                    "pairs": _pair_rows(canonical_refs, estimates),
                    "graph_edge_used_by_targeted_path": False,
                },
                "telemetry": telemetry,
            }
        return {
            "schema_version": 1,
            "artifact_type": "stage1b_measurement_primitives_calibration",
            "status": "passed",
            "git": git_end,
            "environment": environment,
            "asset_manifest": asset_manifest,
            "prompt_id": config["prompt"]["id"],
            "token_ids": token_ids,
            "module_guard": module_guard,
            "capabilities": _capability_dict(backend),
            "scanner": scanner_record,
            "graph_reference": graph_record,
            "pair_freeze": {
                "calibration_pair_ids": [item.pair_id for item in calibration_refs],
                "canonical_pair_ids": [item.pair_id for item in canonical_refs],
                "canonical_pairs": _endpoint_records(canonical_refs),
                "disjoint": not bool(
                    {item.pair_id for item in calibration_refs}
                    & {item.pair_id for item in canonical_refs}
                ),
                "edge_floor": edge_floor,
            },
            "calibration_validation": {
                "references": [_reference_dict(item) for item in calibration_refs],
                "metrics": asdict(metrics),
                "graph_edge_used_by_targeted_path": False,
            },
            "telemetry": telemetry,
        }
    finally:
        if not sampler_finished:
            with contextlib.suppress(Exception):
                sampler.finish()
        if graph is not None:
            del graph
        if model is not None:
            del model
        gc.collect()
        torch.mps.empty_cache()


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists() or args.emergency_output.exists():
        raise RuntimeError("worker output paths must not already exist")
    if not args.output.parent.is_dir() or not args.emergency_output.parent.is_dir():
        raise RuntimeError("worker output parents must already exist")
    result = _execute(args)
    write_json_atomic(args.output, result)
    print(json.dumps({"status": result["status"], "mode": args.mode}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
