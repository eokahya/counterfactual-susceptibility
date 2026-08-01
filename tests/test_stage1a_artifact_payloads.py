from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

STAGE1A_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "stage1a"
if str(STAGE1A_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STAGE1A_SCRIPTS))

from validate_artifacts import (  # noqa: E402
    validate_artifact_payload,
    validate_present_artifacts,
)

from cfsus.reproduction.artifacts import (  # noqa: E402
    ArtifactValidationError,
    make_artifact_envelope,
    write_json_atomic,
)

UPSTREAM = "8f1e2438df612464e229e44c4a00ff637bf9379b"
MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
TRANSCODER_REVISION = "bd5773156dea09893636c801df1237d0410307d2"


def _runtime_provenance() -> dict[str, object]:
    return {
        "upstream_repository": "https://github.com/decoderesearch/circuit-tracer",
        "upstream_revision": UPSTREAM,
        "model_identifier": "google/gemma-2-2b",
        "model_revision": MODEL_REVISION,
        "transcoder_identifier": "mwhanna/gemma-scope-transcoders",
        "transcoder_revision": TRANSCODER_REVISION,
        "backend": "transformerlens",
        "device": "cuda",
        "dtype": "bfloat16",
    }


def _timing() -> dict[str, object]:
    return {
        "wall_seconds": 1.0,
        "process_peak_rss_bytes": 1024,
        "cuda_peak_allocated_bytes": 512,
    }


def _envelope(artifact_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    return make_artifact_envelope(
        artifact_type=artifact_type,
        run_id=f"test-{artifact_type}",
        status="completed",
        provenance=_runtime_provenance(),
        payload=payload,
    )


def _attribution() -> dict[str, Any]:
    return _envelope(
        "attribution_summary",
        {
            "source_notebook": {"path": "demos/attribute_demo.ipynb"},
            "prompt": "The capital of state containing Dallas is",
            "token_ids": [1],
            "tokens": ["<bos>"],
            "parameters": {
                "max_n_logits": 10,
                "desired_logit_prob": 0.95,
                "max_feature_nodes": 8192,
                "batch_size": 256,
                "offload": "disk",
            },
            "graph": {
                "adjacency_shape": [2, 2],
                "active_feature_count": 1,
                "selected_feature_count": 1,
                "error_node_count": 1,
                "input_node_count": 1,
                "logit_node_count": 1,
                "nonzero_edge_count": 1,
                "finite": True,
            },
            "raw_validation": {"passed": True},
            "logit_targets": [
                {"token_id": 1, "token": " Texas", "probability_weight": 0.95}
            ],
            "raw_artifact": {
                "path": "results/generated/stage1a/graph.pt",
                "sha256": "a" * 64,
                "size_bytes": 10,
            },
            "timing": _timing(),
            "seed": 0,
            "classification": "exact",
            "claim_boundary": "Upstream reproduction only.",
        },
    )


def _condition_row() -> dict[str, object]:
    return {
        "token_id": 1,
        "token": " fútbol",
        "logit": 1.0,
        "probability": 0.5,
        "signed_logit_change_from_baseline": 0.0,
        "signed_probability_change_from_baseline": 0.0,
    }


def _intervention() -> dict[str, Any]:
    return _envelope(
        "intervention_summary",
        {
            "source_notebook": {"path": "demos/intervention_demo.ipynb"},
            "prompt": "Hecho: Michael Jordan juega al",
            "token_ids": [1],
            "feature": {
                "layer": 20,
                "requested_position": -1,
                "resolved_position": 6,
                "feature_id": 341,
            },
            "baseline_activation": 2.0,
            "desired_values": [
                {"alpha": 0.0, "desired_post_gate_activation": 2.0},
                {"alpha": 0.5, "desired_post_gate_activation": 1.0},
                {"alpha": 1.0, "desired_post_gate_activation": 0.0},
            ],
            "fixed_top_k_union_token_ids": [1],
            "conditions": {
                name: [_condition_row()]
                for name in ("baseline", "noop", "half_suppression", "full_ablation")
            },
            "baseline_noop_comparison": {"within_tolerance": True},
            "determinism": {"within_tolerance": True},
            "regime": {"freeze_attention": True, "constrained_layers": None},
            "timing": _timing(),
            "seed": 0,
            "claim_boundary": "API reproduction only.",
        },
    )


def _sample(*, active: bool) -> dict[str, object]:
    return {
        "layer": 20,
        "position": 6,
        "feature_id": 1,
        "preactivation": 2.0 if active else 0.0,
        "threshold": 1.0,
        "post_gate_activation": 2.0 if active else 0.0,
        "active": active,
        "signed_margin": 1.0 if active else -1.0,
    }


def _semantics() -> dict[str, Any]:
    return _envelope(
        "semantics_summary",
        {
            "prompt": "Hecho: Michael Jordan juega al",
            "token_ids": [1],
            "cache_shape": [26, 7, 16384],
            "cache_index_order": ["layer", "token_position", "feature_id"],
            "cache_flags": {
                "preactivation_apply_activation_function": False,
                "post_gate_apply_activation_function": True,
                "sparse": False,
            },
            "parameters": {
                "layer_count": 26,
                "d_model": 2304,
                "d_transcoder": 16384,
                "activation_function": "JumpReLU",
            },
            "preactivation_equation": {
                "b_enc_included": True,
                "b_dec_included": False,
            },
            "gate_check": {
                "strict_greater_than": True,
                "equality_inactive": True,
                "samples": {
                    "active": _sample(active=True),
                    "inactive": _sample(active=False),
                    "closest_margin": _sample(active=False),
                    "official_intervention_source": _sample(active=True),
                },
            },
            "intervention_value_check": {
                "upstream_argument": "absolute_desired_post_gate_activation",
                "project_mapping": "desired = (1 - alpha) * baseline_activation",
                "official_feature_baseline_activation": 2.0,
                "alpha": 0.0,
                "desired_noop_activation": 2.0,
                "delta_logic_still_uses_post_gate_activation": True,
                "baseline_noop_maximum_absolute_logit_error": 0.0,
                "noop_repeat_maximum_absolute_logit_error": 0.0,
                "absolute_tolerance": 0.02,
                "relative_tolerance": 0.002,
            },
            "timing": _timing(),
            "seed": 0,
            "claim_boundary": "Runtime semantics only.",
        },
    )


@pytest.mark.parametrize("factory", [_attribution, _intervention, _semantics])
def test_complete_empirical_payloads_pass_strict_validation(factory: Any) -> None:
    validate_artifact_payload(factory())


@pytest.mark.parametrize(
    "artifact_type",
    ["attribution_summary", "intervention_summary", "semantics_summary"],
)
def test_empty_empirical_payload_is_rejected(artifact_type: str) -> None:
    with pytest.raises(ArtifactValidationError, match="missing required keys"):
        validate_artifact_payload(_envelope(artifact_type, {}))


def test_attribution_requires_raw_checksum_and_nonempty_graph() -> None:
    missing_checksum = _attribution()
    del missing_checksum["payload"]["raw_artifact"]["sha256"]
    with pytest.raises(ArtifactValidationError, match="sha256"):
        validate_artifact_payload(missing_checksum)

    empty_graph = _attribution()
    empty_graph["payload"]["graph"]["selected_feature_count"] = 0
    with pytest.raises(ArtifactValidationError, match="selected_feature_count"):
        validate_artifact_payload(empty_graph)


def test_intervention_requires_exact_alpha_mapping_and_condition_alignment() -> None:
    wrong_desired = _intervention()
    wrong_desired["payload"]["desired_values"][1]["desired_post_gate_activation"] = 0.0
    with pytest.raises(ArtifactValidationError, match="alpha mapping"):
        validate_artifact_payload(wrong_desired)

    wrong_token = _intervention()
    wrong_token["payload"]["conditions"]["noop"][0]["token_id"] = 2
    with pytest.raises(ArtifactValidationError, match="fixed union"):
        validate_artifact_payload(wrong_token)


def test_semantics_requires_inactive_visibility_and_absolute_noop() -> None:
    active_inactive_sample = _semantics()
    active_inactive_sample["payload"]["gate_check"]["samples"]["inactive"]["active"] = (
        True
    )
    with pytest.raises(ArtifactValidationError, match=r"inactive sample\.active"):
        validate_artifact_payload(active_inactive_sample)

    wrong_noop = _semantics()
    wrong_noop["payload"]["intervention_value_check"]["desired_noop_activation"] = 0.0
    with pytest.raises(ArtifactValidationError, match="equal baseline"):
        validate_artifact_payload(wrong_noop)


def test_current_blocked_metadata_artifacts_pass_strict_validation() -> None:
    artifact_directory = Path(__file__).resolve().parents[1] / "results" / "stage1a"
    for name in (
        "asset_manifest.json",
        "colab_handoff_manifest.json",
        "environment_manifest.json",
    ):
        value = json.loads((artifact_directory / name).read_text(encoding="utf-8"))
        validate_artifact_payload(value)


def test_unknown_json_filename_is_rejected(tmp_path: Path) -> None:
    artifact_directory = tmp_path / "results" / "stage1a"
    artifact_directory.mkdir(parents=True)
    write_json_atomic(
        artifact_directory / "attributon_summary.json",
        _attribution(),
    )

    with pytest.raises(ArtifactValidationError, match="unsupported Stage 1A artifact"):
        validate_present_artifacts(artifact_directory)


def test_directory_validation_enforces_strict_payloads_when_requested(
    tmp_path: Path,
) -> None:
    artifact_directory = tmp_path / "results" / "stage1a"
    artifact_directory.mkdir(parents=True)
    write_json_atomic(
        artifact_directory / "attribution_summary.json",
        _envelope("attribution_summary", {}),
    )

    with pytest.raises(ArtifactValidationError, match="missing required keys"):
        validate_present_artifacts(artifact_directory, strict_payloads=True)


def test_runtime_provenance_is_required_and_immutable() -> None:
    missing = _attribution()
    missing["provenance"] = {}
    with pytest.raises(ArtifactValidationError, match=r"provenance.*missing required"):
        validate_artifact_payload(missing)

    mutable = copy.deepcopy(_intervention())
    mutable["provenance"]["model_revision"] = "0" * 40
    with pytest.raises(ArtifactValidationError, match="required revision"):
        validate_artifact_payload(mutable)
