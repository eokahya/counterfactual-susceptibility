from __future__ import annotations

import importlib
from copy import deepcopy
from pathlib import Path
from typing import TextIO

import pytest

from cfsus.reproduction.config import (
    ConfigDependencyError,
    DeviceName,
    Stage1AConfig,
    Stage1AConfigError,
    load_stage1a_config,
)


def _valid_mapping() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_name": "stage1a_gemma2_2b_official_reproduction",
        "upstream": {
            "repository": "https://github.com/decoderesearch/circuit-tracer",
            "revision": "8f1e2438df612464e229e44c4a00ff637bf9379b",
        },
        "model": {
            "identifier": "google/gemma-2-2b",
            "revision": "c5ebcd40d208330abc697524c919956e692655cf",
            "snapshot_path": "results/generated/stage1a/assets/google-gemma-2-2b",
        },
        "transcoder": {
            "identifier": "mwhanna/gemma-scope-transcoders",
            "revision": "bd5773156dea09893636c801df1237d0410307d2",
            "snapshot_path": (
                "results/generated/stage1a/assets/mwhanna-gemma-scope-transcoders"
            ),
        },
        "runtime": {
            "backend": "transformerlens",
            "device": "mps",
            "dtype": "bfloat16",
        },
        "seeds": {"python": 0, "numpy": 0, "torch": 0},
        "asset_policy": {
            "allow_download": True,
            "require_offline_execution": True,
        },
        "attribution": {
            "prompt": "The capital of state containing Dallas is",
            "max_n_logits": 10,
            "desired_logit_probability": 0.95,
            "max_feature_nodes": 8192,
            "batch_size": 256,
            "offload": "cpu",
        },
        "intervention": {
            "prompt": "Hecho: Michael Jordan juega al",
            "feature": {"layer": 20, "position": -1, "feature_id": 341},
            "alphas": [0.0, 0.5, 1.0],
            "freeze_attention": True,
            "constrained_layers": None,
        },
        "artifacts": {
            "raw_graph": "results/generated/stage1a/official_graph.pt",
            "environment_manifest": "results/stage1a/environment_manifest.json",
            "asset_manifest": "results/stage1a/asset_manifest.json",
            "attribution_summary": "results/stage1a/attribution_summary.json",
            "intervention_summary": "results/stage1a/intervention_summary.json",
            "semantics_summary": "results/stage1a/semantics_summary.json",
            "checksums": "results/stage1a/checksums.sha256",
        },
    }


def _section(config: dict[str, object], key: str) -> dict[str, object]:
    value = config[key]
    assert isinstance(value, dict)
    return value


def test_resolved_official_configuration_is_accepted() -> None:
    config = Stage1AConfig.from_mapping(_valid_mapping())

    assert config.model.revision == "c5ebcd40d208330abc697524c919956e692655cf"
    assert config.transcoder.revision == "bd5773156dea09893636c801df1237d0410307d2"
    assert config.runtime.device is DeviceName.MPS
    assert config.intervention.feature.position == -1
    assert config.intervention.alphas == (0.0, 0.5, 1.0)


@pytest.mark.parametrize(
    "revision",
    [
        "main",
        "latest",
        "TO_BE_PINNED",
        "a" * 39,
        "A" * 40,
        "0" * 40,
    ],
)
def test_mutable_placeholder_and_malformed_revisions_are_rejected(
    revision: str,
) -> None:
    raw = _valid_mapping()
    _section(raw, "model")["revision"] = revision

    with pytest.raises(Stage1AConfigError, match=r"SHA|placeholder"):
        Stage1AConfig.from_mapping(raw)


@pytest.mark.parametrize("identifier", ["gemma", "google/gemma-2-2b@main", "x"])
def test_aliases_and_revision_bearing_identifiers_are_rejected(
    identifier: str,
) -> None:
    raw = _valid_mapping()
    _section(raw, "transcoder")["identifier"] = identifier

    with pytest.raises(Stage1AConfigError, match="identifier"):
        Stage1AConfig.from_mapping(raw)


def test_other_well_formed_asset_revision_is_rejected() -> None:
    raw = _valid_mapping()
    _section(raw, "model")["revision"] = "1" * 40

    with pytest.raises(Stage1AConfigError, match="resolved immutable revision"):
        Stage1AConfig.from_mapping(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend", "nnsight"),
        ("device", "auto"),
        ("dtype", "auto"),
        ("dtype", "float16"),
        ("dtype", "float32"),
    ],
)
def test_runtime_values_are_explicit_and_strict(field: str, value: str) -> None:
    raw = _valid_mapping()
    _section(raw, "runtime")[field] = value

    with pytest.raises(Stage1AConfigError, match=f"runtime.{field}"):
        Stage1AConfig.from_mapping(raw)


def test_official_prompts_and_scientific_attribution_parameters_are_fixed() -> None:
    changed_prompt = _valid_mapping()
    _section(changed_prompt, "attribution")["prompt"] = "The capital of France is"
    with pytest.raises(Stage1AConfigError, match="official demo"):
        Stage1AConfig.from_mapping(changed_prompt)

    changed_target = _valid_mapping()
    _section(changed_target, "attribution")["max_feature_nodes"] = 1024
    with pytest.raises(Stage1AConfigError, match="scientific target"):
        Stage1AConfig.from_mapping(changed_target)


def test_official_intervention_coordinates_and_regime_are_fixed() -> None:
    changed_feature = _valid_mapping()
    feature = _section(_section(changed_feature, "intervention"), "feature")
    feature["position"] = 0
    with pytest.raises(Stage1AConfigError, match="official coordinates"):
        Stage1AConfig.from_mapping(changed_feature)

    changed_regime = _valid_mapping()
    _section(changed_regime, "intervention")["freeze_attention"] = False
    with pytest.raises(Stage1AConfigError, match="frozen attention"):
        Stage1AConfig.from_mapping(changed_regime)


@pytest.mark.parametrize("bad_seed", [-1, True, 2**32])
def test_seeds_are_explicit_bounded_integers(bad_seed: object) -> None:
    raw = _valid_mapping()
    _section(raw, "seeds")["torch"] = bad_seed

    with pytest.raises(Stage1AConfigError, match="seed"):
        Stage1AConfig.from_mapping(raw)


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/model",
        "~/model",
        "results//generated/stage1a/assets/model",
        "results/generated/stage1a/../model",
        "models/google-gemma-2-2b",
    ],
)
def test_snapshot_paths_are_safe_and_ignored_by_design(path: str) -> None:
    raw = _valid_mapping()
    _section(raw, "model")["snapshot_path"] = path

    with pytest.raises(Stage1AConfigError, match=r"path|under"):
        Stage1AConfig.from_mapping(raw)


def test_summary_and_raw_artifact_paths_have_separate_roots() -> None:
    raw_summary = _valid_mapping()
    _section(raw_summary, "artifacts")["attribution_summary"] = (
        "results/generated/stage1a/attribution_summary.json"
    )
    with pytest.raises(Stage1AConfigError, match="results/stage1a"):
        Stage1AConfig.from_mapping(raw_summary)

    committed_raw = _valid_mapping()
    _section(committed_raw, "artifacts")["raw_graph"] = (
        "results/stage1a/official_graph.pt"
    )
    with pytest.raises(Stage1AConfigError, match="results/generated/stage1a"):
        Stage1AConfig.from_mapping(committed_raw)


def test_resolved_execution_must_be_offline_after_snapshot_resolution() -> None:
    raw = _valid_mapping()
    _section(raw, "asset_policy")["require_offline_execution"] = False

    with pytest.raises(Stage1AConfigError, match="must be true"):
        Stage1AConfig.from_mapping(raw)


def test_unknown_and_missing_keys_are_rejected() -> None:
    unknown = _valid_mapping()
    unknown["token"] = "must-never-be-accepted"
    with pytest.raises(Stage1AConfigError, match="unknown"):
        Stage1AConfig.from_mapping(unknown)

    missing = _valid_mapping()
    del missing["seeds"]
    with pytest.raises(Stage1AConfigError, match="missing"):
        Stage1AConfig.from_mapping(missing)


def test_yaml_dependency_is_loaded_only_on_demand(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("ignored by fake parser\n", encoding="utf-8")
    original_import = importlib.import_module

    def missing_yaml(name: str, package: str | None = None) -> object:
        if name == "yaml":
            raise ModuleNotFoundError("No module named 'yaml'")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", missing_yaml)
    with pytest.raises(ConfigDependencyError, match="dedicated Stage 1A"):
        load_stage1a_config(config_path)


def test_yaml_loader_validates_the_loaded_mapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeYaml:
        @staticmethod
        def safe_load(stream: TextIO) -> object:
            assert stream.read()
            return deepcopy(_valid_mapping())

    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    monkeypatch.setattr(importlib, "import_module", lambda name: FakeYaml())

    config = load_stage1a_config(config_path)

    assert config.attribution.prompt.endswith("Dallas is")


def test_tracked_yaml_contains_exact_pins_and_no_placeholders() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "stage1a_gemma2_2b_official_reproduction.yaml"
    )
    text = config_path.read_text(encoding="utf-8")

    assert "c5ebcd40d208330abc697524c919956e692655cf" in text
    assert "bd5773156dea09893636c801df1237d0410307d2" in text
    assert "TO_BE_" not in text
    assert "device: auto" not in text
