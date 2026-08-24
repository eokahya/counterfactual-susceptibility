"""Offline policy tests for the isolated Stage 1A-S runtime."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from cfsus.reproduction.artifacts import ArtifactValidationError
from cfsus.reproduction.small_model_mps_fp16 import (
    ARTIFACT_ALLOWLIST,
    COMPLETED_STATUS,
    EXPERIMENT_CLASS,
    MODEL_REVISION,
    TRANSCODER_REVISION,
    UPSTREAM_REVISION,
    assert_fallback_disabled,
    conservative_memory_feasibility,
    intervention_values,
    load_small_model_config,
    projected_graph_bytes,
    select_feature_from_graph,
    validate_projected_manifest,
    validate_small_artifact_directory,
    validate_small_model_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage1a_small_model_mps_fp16_pilot.yaml"
PROJECTED = ROOT / "configs/stage1a_small_model_projected_download.json"


def _config() -> dict[str, object]:
    return {
        "experiment_name": EXPERIMENT_CLASS,
        "experiment_class": EXPERIMENT_CLASS,
        "completed_status": COMPLETED_STATUS,
        "claim_class": "local_small_model_runtime_validation",
        "upstream": {
            "repository": "https://github.com/decoderesearch/circuit-tracer.git",
            "version": "0.5.2",
            "revision": UPSTREAM_REVISION,
        },
        "model": {
            "identifier": "google/gemma-3-270m",
            "revision": MODEL_REVISION,
            "architecture": "Gemma3ForCausalLM",
            "pretrained_variant": "base",
        },
        "transcoder": {
            "identifier": "mwhanna/gemma-scope-2-270m-pt",
            "revision": TRANSCODER_REVISION,
            "subfolder": "transcoder_all/width_16k_l0_small",
            "model_kind": "transcoder_set",
            "feature_input_hook": "mlp.hook_in",
            "feature_output_hook": "hook_mlp_out",
            "layer_count": 18,
            "feature_width": 16384,
        },
        "runtime": {
            "backend": "nnsight",
            "device": "mps",
            "dtype": "float16",
            "python": "3.11",
            "platform": "macos-arm64",
            "fallback_environment_variable": "PYTORCH_ENABLE_MPS_FALLBACK",
            "fallback_allowed": False,
            "lazy_encoder": True,
            "lazy_decoder": True,
            "graph_metadata_device": "cpu",
            "scientific_tensor_device": "mps",
        },
        "accepted": {
            "max_n_logits": 10,
            "desired_logit_probability": 0.95,
            "max_feature_nodes": 4096,
            "attribution_batch_sizes": [64, 32, 16],
            "intervention_alphas": [0.0, 0.5, 1.0],
            "baseline_repeat": True,
        },
        "feature_selection": {
            "manual_selection_allowed": False,
            "primary_rule": (
                "highest_absolute_direct_contribution_to_baseline_top_logit_"
                "at_final_token"
            ),
            "fallback_rule": (
                "highest_absolute_active_baseline_activation_at_final_token"
            ),
        },
        "safety_limits": {
            "maximum_mps_driver_bytes": 24 * 1024**3,
            "maximum_process_rss_bytes": 24 * 1024**3,
            "maximum_swap_growth_bytes": 4 * 1024**3,
            "minimum_available_memory_bytes": 4 * 1024**3,
            "maximum_graph_buffer_bytes": 6 * 1024**3,
            "maximum_transcoder_download_bytes": 6 * 1024**3,
            "accepted_thermal_states": ["nominal", "fair"],
        },
        "artifacts": {
            "result_directory": "results/stage1a_small_model_mps_fp16",
            "generated_directory": "results/generated/stage1a_small_model_mps_fp16",
            "environment_lock": (
                "environments/stage1a_small_model_mps/requirements-lock.txt"
            ),
            "projected_download_manifest": (
                "configs/stage1a_small_model_projected_download.json"
            ),
        },
        "prompt": "The capital of France is",
    }


def test_small_model_config_is_exact_and_immutable() -> None:
    config = validate_small_model_config(_config())
    assert config["experiment_class"] == EXPERIMENT_CLASS
    assert config["completed_status"] == COMPLETED_STATUS
    assert config["upstream"]["revision"] == UPSTREAM_REVISION
    assert config["model"]["revision"] == MODEL_REVISION
    assert config["transcoder"]["revision"] == TRANSCODER_REVISION


def test_yaml_config_matches_the_validated_mapping_when_pyyaml_is_available() -> None:
    pytest.importorskip("yaml")
    config = load_small_model_config(CONFIG)
    assert config["experiment_class"] == EXPERIMENT_CLASS


@pytest.mark.parametrize(
    ("section", "key", "bad_value"),
    [
        ("runtime", "backend", "transformerlens"),
        ("runtime", "device", "cuda"),
        ("runtime", "dtype", "bfloat16"),
        ("model", "identifier", "google/gemma-2-2b"),
        ("model", "revision", "main"),
        ("transcoder", "revision", "main"),
        ("transcoder", "subfolder", "transcoder_all/width_65k_l0_small"),
        ("feature_selection", "manual_selection_allowed", True),
        ("accepted", "attribution_batch_sizes", [32, 16]),
    ],
)
def test_small_model_config_rejects_cross_experiment_or_mutable_values(
    section: str, key: str, bad_value: object
) -> None:
    config = copy.deepcopy(_config())
    config[section][key] = bad_value  # type: ignore[index]
    with pytest.raises(ArtifactValidationError):
        validate_small_model_config(config)


def test_projected_download_manifest_is_exact_and_under_limit() -> None:
    value = json.loads(PROJECTED.read_text(encoding="utf-8"))
    manifest = validate_projected_manifest(value)
    assert manifest["projected_total_bytes"] == 2_087_816_677
    assert manifest["transcoder"]["projected_bytes"] < 6 * 1024**3


def test_projected_download_rejects_extra_file() -> None:
    value = json.loads(PROJECTED.read_text(encoding="utf-8"))
    value["transcoder"]["files"].append(
        {
            "path": "transcoder_all/width_16k_l0_small/features/extra.json",
            "reported_bytes": 1,
        }
    )
    with pytest.raises(ArtifactValidationError, match="allowlist"):
        validate_projected_manifest(value)


def test_memory_projection_fits_and_eager_transcoder_is_informational() -> None:
    estimate = conservative_memory_feasibility()
    assert estimate.feasible
    assert estimate.total_conservative_bytes < estimate.maximum_process_rss_bytes
    assert estimate.full_eager_transcoder_fp16_bytes > 700 * 1024**2
    assert estimate.one_lazy_matrix_fp16_bytes == 20_971_520


def test_graph_projection_rejects_invalid_dimensions() -> None:
    assert (
        projected_graph_bytes(
            active_features=100, selected_features=10, token_count=6, logits=3
        )
        > 0
    )
    with pytest.raises(Exception, match="positive"):
        projected_graph_bytes(
            active_features=0, selected_features=10, token_count=6, logits=3
        )


def test_fallback_detector_is_fail_closed() -> None:
    assert_fallback_disabled({})
    assert_fallback_disabled({"PYTORCH_ENABLE_MPS_FALLBACK": "false"})
    with pytest.raises(ArtifactValidationError, match="must be absent or false"):
        assert_fallback_disabled({"PYTORCH_ENABLE_MPS_FALLBACK": "1"})


def test_suppression_mapping_is_absolute() -> None:
    assert intervention_values(2.0, [0.0, 0.5, 1.0]) == [
        {
            "alpha": 0.0,
            "baseline_activation": 2.0,
            "desired_absolute_activation": 2.0,
        },
        {
            "alpha": 0.5,
            "baseline_activation": 2.0,
            "desired_absolute_activation": 1.0,
        },
        {
            "alpha": 1.0,
            "baseline_activation": 2.0,
            "desired_absolute_activation": 0.0,
        },
    ]


def test_graph_feature_selection_uses_direct_effect_then_stable_tie_break() -> None:
    torch = pytest.importorskip("torch")
    graph = SimpleNamespace(
        active_features=torch.tensor([[2, 5, 9], [1, 5, 8], [0, 4, 7]]),
        activation_values=torch.tensor([1.0, 2.0, 3.0]),
        selected_features=torch.tensor([0, 1]),
        adjacency_matrix=torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.5, -0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
        logit_targets=[1, 2],
    )
    selection = select_feature_from_graph(graph, final_position=5)
    assert (selection.layer, selection.position, selection.feature) == (1, 5, 8)
    assert selection.rule.startswith("highest_absolute_direct_contribution")


def test_small_artifact_directory_rejects_extra_and_hardlink(tmp_path: Path) -> None:
    directory = tmp_path / "artifacts"
    directory.mkdir()
    for name in ARTIFACT_ALLOWLIST:
        (directory / name).write_text("{}\n", encoding="utf-8")
    validate_small_artifact_directory(directory)
    extra = directory / "model.safetensors"
    extra.write_bytes(b"weight")
    with pytest.raises(ArtifactValidationError):
        validate_small_artifact_directory(directory)
    extra.unlink()
    hardlink = directory / "attempts.json"
    alias = tmp_path / "alias"
    os.link(hardlink, alias)
    try:
        with pytest.raises(ArtifactValidationError, match="hardlinked"):
            validate_small_artifact_directory(directory)
    finally:
        alias.unlink()
