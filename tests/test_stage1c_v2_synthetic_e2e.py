"""Nonempty synthetic worker -> assembler -> standalone-validator coverage."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

from cfsus.stage1c_v2.config import load_stage1c_v2_config
from cfsus.stage1c_v2.worker_result import build_detached_worker_result

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts/stage1c_v2/validate_stage1c_v2_artifacts.py"
ASSEMBLER = ROOT / "scripts/stage1c_v2/assemble_stage1c_artifacts.py"
RUNTIME_FINGERPRINT = (
    "gemma3-270m@9b0cfec892e2/plt@fada11860ac1/"
    "circuit-tracer@8f1e2438df61/nnsight/mps/bf16/stage1c-v2"
)


def _load_module(name: str, path: Path) -> Any:
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pair_id(source: tuple[int, int, int], target: tuple[int, int, int]) -> str:
    payload = {
        "experiment_class": "stage1c_v2_heldout_prospective_prediction",
        "prompt_id": "capital_germany_heldout_v2",
        "runtime_fingerprint": RUNTIME_FINGERPRINT,
        "seed": "stage1c-v2-heldout-prospective-prediction",
        "source": list(source),
        "target": list(target),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _synthetic_prediction(config: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for source_key, target_key, threshold, q, alpha in (
        ((0, 1, 1), (1, 1, 2), 0.5, 1.0, 0.5),
        ((0, 1, 3), (1, 1, 4), 0.6, 0.8, 0.6 / 0.8),
    ):
        source = dict(zip(("layer", "position", "feature_id"), source_key, strict=True))
        target = dict(zip(("layer", "position", "feature_id"), target_key, strict=True))
        requested = sorted(
            {0.0, 0.25, 0.5, 0.75, 1.0, alpha - 0.015625, alpha, alpha + 0.015625}
        )
        rows.append(
            {
                "pair_id": _pair_id(source_key, target_key),
                "group": "primary",
                "source": source,
                "target": target,
                "source_activation": 2.0,
                "target_preactivation": 0.0,
                "target_threshold": threshold,
                "margin": threshold,
                "targeted_response": -q / 2.0,
                "q": q,
                "susceptibility": q / (threshold + 1.0e-12),
                "predicted_alpha_star": alpha,
                "predicted_status": "definitely_crossing",
                "requested_alphas": requested,
            }
        )
    return {
        "schema_version": 2,
        "artifact_type": "stage1c_v2_prediction_manifest",
        "experiment_class": "stage1c_v2_heldout_prospective_prediction",
        "status": "prediction_frozen_ready_for_commit",
        "base_commit": "cc47cb604fc2422deb50aacbc7fde77499b532c5",
        "branch": "stage-1c-v2-heldout-prospective-prediction",
        "config_sha256": "a" * 64,
        "artifact_schema_sha256": "b" * 64,
        "pair_id_domain": (
            "stage1c_v2_heldout_prospective_prediction:capital_germany_heldout_v2"
        ),
        "prompt": {
            "id": "capital_germany_heldout_v2",
            "text": "The capital of Germany is",
            "token_ids": [2, 818, 5279, 529, 9405, 563],
        },
        "runtime_identity": {
            "backend": "nnsight",
            "device": "mps:0",
            "dtype": "torch.bfloat16",
            "model_identifier": "google/gemma-3-270m",
            "model_revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
            "transcoder_identifier": "mwhanna/gemma-scope-2-270m-pt",
            "transcoder_revision": "fada11860ac1d337c1e41e9da308798405b94c8e",
            "transcoder_subfolder": "transcoder_all/width_16k_l0_small",
            "layer_count": 18,
            "feature_width": 16384,
            "upstream_revision": "8f1e2438df612464e229e44c4a00ff637bf9379b",
        },
        "protocol": {
            "scanner": dict(config["scanner"]),
            "source_pool": dict(config["source_pool"]),
            "responses": dict(config["responses"]),
            "scoring": dict(config["scoring"]),
            "selection": dict(config["selection"]),
            "schedule": dict(config["schedule"]),
            "intervention_regime": dict(config["intervention"]),
            "analysis": dict(config["analysis"]),
        },
        "selected_groups": {"primary": rows, "near_boundary": [], "directional": []},
        "selection_audit": {
            "primary_count": 2,
            "near_boundary_count": 0,
            "directional_count": 0,
            "groups_disjoint": True,
            "primary_target_unique": True,
            "primary_source_cap": 2,
        },
        "prediction_only_guards": {
            "source_suppression_api_calls": 0,
            "prior_inactive_target_outcome_read": False,
            "intervention_worker_imported": False,
            "raw_graph_read": False,
            "raw_adjacency_read": False,
        },
        "protocol_file_sha256": {"synthetic_protocol.py": "c" * 64},
    }


def _bf16_round(value: float) -> float:
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    lower, upper = bits & 0xFFFF, bits >> 16
    if lower > 0x8000 or (lower == 0x8000 and upper & 1):
        upper += 1
    return float(struct.unpack(">f", struct.pack(">I", upper << 16))[0])


def _points(pair: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[float, list[float]] = {}
    for requested in pair["requested_alphas"]:
        desired = (1.0 - requested) * pair["source_activation"]
        grouped.setdefault(_bf16_round(desired), []).append(requested)
    return [
        _point(pair, requests)
        for requests in sorted(
            grouped.values(),
            key=lambda values: (
                1.0
                - _bf16_round((1.0 - min(values)) * pair["source_activation"])
                / pair["source_activation"]
            ),
        )
    ]


def _point(pair: dict[str, Any], requested_values: list[float]) -> dict[str, Any]:
    requested = min(requested_values)
    desired = (1.0 - requested) * pair["source_activation"]
    applied = _bf16_round(desired)
    realized = 1.0 - applied / pair["source_activation"]
    z = realized * pair["q"]
    active = z > pair["target_threshold"]
    mapping = [
        {
            "requested_alpha": alpha,
            "desired_high_precision": (1.0 - alpha) * pair["source_activation"],
            "actual_bf16_value_passed": applied,
            "realized_suppression": realized,
        }
        for alpha in requested_values
    ]
    return {
        "pair_id": pair["pair_id"],
        "group": pair["group"],
        "source": pair["source"],
        "target": pair["target"],
        "requested_alpha": requested,
        "desired_high_precision": desired,
        "actual_bf16_value_passed": applied,
        "realized_suppression": realized,
        "requested_mappings": mapping,
        "representative_requested_alpha": requested,
        "collapsed_request_count": len(mapping),
        "source_value_device": "mps:0",
        "source_value_dtype": "torch.bfloat16",
        "target_preactivation": z,
        "target_threshold": pair["target_threshold"],
        "target_activation": z if active else 0.0,
        "target_active": active,
        "loaded_gate": "a=z*1[z>tau]",
        "threshold_equality_activity": "inactive",
        "target_clamped": False,
        "freeze_attention": True,
        "constrained_layers": None,
        "predicted_target_preactivation": z,
        "predicted_target_activation": z if active else 0.0,
        "predicted_target_active": active,
        "target_preactivation_absolute_error": 0.0,
        "target_preactivation_symmetric_normalized_error": 0.0,
        "stage": "synthetic",
        "bisection_step": None,
    }


def test_nonempty_synthetic_worker_assembler_standalone_validator(
    tmp_path: Path,
) -> None:
    validator = _load_module("stage1c_v2_standalone_validator", VALIDATOR)
    assembler = _load_module("stage1c_v2_assembler", ASSEMBLER)
    prediction = _synthetic_prediction(load_stage1c_v2_config())
    working_sweeps: list[dict[str, Any]] = []
    for pair in prediction["selected_groups"]["primary"]:
        points = _points(pair)
        working_sweeps.append(
            {
                "pair_id": pair["pair_id"],
                "group": "primary",
                "source": pair["source"],
                "target": pair["target"],
                "target_preactivation": pair["target_preactivation"],
                "target_threshold": pair["target_threshold"],
                "q": pair["q"],
                "predicted_alpha_star": pair["predicted_alpha_star"],
                "predicted_status": pair["predicted_status"],
                "baseline_source_activation": pair["source_activation"],
                "point_count": len(points),
                "bisection_step_count": 0,
                "points": points,
            }
        )
    result = build_detached_worker_result(
        working_sweeps,
        intervention_artifacts={"intervention_sweeps": {"pairs": working_sweeps}},
        canonical_source_suppression_api_calls=sum(
            item["point_count"] for item in working_sweeps
        ),
        status="passed",
    )
    assert len(result["sweeps"]) == 2
    assert (
        result["sweeps"]
        == result["intervention_artifacts"]["intervention_sweeps"]["pairs"]
    )
    worker = {
        **result,
        "schema_version": 2,
        "artifact_type": "stage1c_v2_intervention_worker",
        "prediction_manifest_sha256": hashlib.sha256(
            assembler._canonical_bytes(prediction)
        ).hexdigest(),
        "scientific_outcome": "supported",
        "attempt_count": 1,
        "asset_manifest": {"status": "verified"},
        "environment": {},
    }
    prediction_worker = {
        "schema_version": 2,
        "artifact_type": "stage1c_v2_prediction_worker",
        "status": "passed",
        "prediction_manifest": prediction,
        "asset_manifest": {"status": "verified"},
        "environment": {},
    }
    analyses = [
        validator._analysis(pair, sweep["points"], prediction["protocol"]["analysis"])
        for pair, sweep in zip(
            prediction["selected_groups"]["primary"], working_sweeps, strict=True
        )
    ]
    worker["intervention_artifacts"]["crossing_summary"] = {
        "scientific_outcome": "supported",
        "aggregate_metrics": validator._aggregate(analyses),
    }
    worker["intervention_artifacts"]["attempts"] = {
        "attempt_count": 1,
        "scientific_retry_count": 0,
        "intervention_required": True,
    }
    artifacts = assembler.records(
        prediction, prediction_worker, worker, execution="d" * 40
    )
    assembler.write_bundle(artifacts, tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--bundle",
            str(tmp_path),
            "--execution-commit",
            "d" * 40,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert '"status": "passed"' in completed.stdout
