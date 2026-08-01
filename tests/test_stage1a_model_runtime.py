"""Explicitly opted-in checks against the pinned loaded Stage 1A runtime.

These tests never download assets. They are excluded from the default suite and
require explicit environment variables pointing at immutable local snapshots.
Only the official feature's layer and resolved token position are inspected;
this is a semantics check, not Stage 1B candidate scanning.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from cfsus.reproduction.config import load_stage1a_config
from cfsus.reproduction.runtime_helpers import (
    apply_strict_jumprelu,
    compare_numeric_sequences,
    desired_activation,
    resolve_position_selector,
    select_gate_samples,
    strict_jumprelu,
)

pytestmark = pytest.mark.model

_OPT_IN_VARIABLE = "CFSUS_RUN_STAGE1A_MODEL_TESTS"
_CONFIG_VARIABLE = "CFSUS_STAGE1A_CONFIG"
_MODEL_SNAPSHOT_VARIABLE = "CFSUS_STAGE1A_MODEL_SNAPSHOT"
_TRANSCODER_SNAPSHOT_VARIABLE = "CFSUS_STAGE1A_TRANSCODER_SNAPSHOT"


def _explicit_path(variable: str) -> Path:
    raw_value = os.environ.get(variable)
    if raw_value is None:
        pytest.fail(f"{variable} is required for opted-in Stage 1A model tests")
    path = Path(raw_value).expanduser().resolve()
    if not path.exists():
        pytest.fail(f"{variable} does not exist")
    return path


def _require_snapshot_revision(snapshot: Path, revision: str, role: str) -> None:
    """Require an HF commit-named snapshot or a local immutable marker."""

    marker = snapshot / ".cfsus-revision"
    marker_revision = (
        marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
    )
    if snapshot.name != revision and marker_revision != revision:
        pytest.fail(
            f"{role} snapshot does not prove revision {revision}; use an HF "
            "snapshot directory named by the commit or add .cfsus-revision"
        )


def _runtime_inputs_or_skip() -> tuple[Any, Path, Path]:
    if os.environ.get(_OPT_IN_VARIABLE) != "1":
        pytest.skip(f"set {_OPT_IN_VARIABLE}=1 to enable Stage 1A model tests")

    config_path = _explicit_path(_CONFIG_VARIABLE)
    model_snapshot = _explicit_path(_MODEL_SNAPSHOT_VARIABLE)
    transcoder_snapshot = _explicit_path(_TRANSCODER_SNAPSHOT_VARIABLE)
    if not config_path.is_file():
        pytest.fail(f"{_CONFIG_VARIABLE} must identify a configuration file")
    if not model_snapshot.is_dir() or not transcoder_snapshot.is_dir():
        pytest.fail("Stage 1A model and transcoder snapshots must be directories")

    config = load_stage1a_config(config_path)
    if not config.asset_policy.require_offline_execution:
        pytest.fail("Stage 1A model tests require offline execution in configuration")
    _require_snapshot_revision(model_snapshot, config.model.revision, "model")
    _require_snapshot_revision(
        transcoder_snapshot, config.transcoder.revision, "transcoder"
    )

    required_model_files = (model_snapshot / "config.json",)
    required_transcoder_files = (transcoder_snapshot / "config.yaml",)
    if not all(path.is_file() for path in required_model_files):
        pytest.fail("model snapshot is missing config.json")
    if not any(model_snapshot.glob("*.safetensors")):
        pytest.fail("model snapshot contains no safetensors weights")
    if not all(path.is_file() for path in required_transcoder_files):
        pytest.fail("transcoder snapshot is missing config.yaml")
    if not any(transcoder_snapshot.glob("layer_*.safetensors")):
        pytest.fail("transcoder snapshot contains no layer safetensors")
    return config, model_snapshot, transcoder_snapshot


def _optional_runtime_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pytest.fail(f"the opted-in Stage 1A runtime is missing dependency {name!r}")


def _layer_paths(snapshot: Path) -> dict[int, str]:
    indexed_paths: list[tuple[int, Path]] = []
    for path in snapshot.glob("layer_*.safetensors"):
        suffix = path.stem.removeprefix("layer_")
        if suffix.isdigit():
            indexed_paths.append((int(suffix), path))
    indexed_paths.sort()
    indices = [index for index, _ in indexed_paths]
    if indices != list(range(len(indices))):
        pytest.fail("transcoder layer snapshots must be contiguous from layer zero")
    return {index: str(path) for index, path in indexed_paths}


def _load_offline_model(
    config: Any, model_snapshot: Path, transcoder_snapshot: Path
) -> Any:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    torch = _optional_runtime_module("torch")
    transformers = _optional_runtime_module("transformers")
    yaml = _optional_runtime_module("yaml")
    replacement_module = _optional_runtime_module(
        "circuit_tracer.replacement_model.replacement_model"
    )
    transcoder_module = _optional_runtime_module(
        "circuit_tracer.transcoder.single_layer_transcoder"
    )

    device_name = config.runtime.device.value
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.fail("resolved Stage 1A configuration requires unavailable MPS")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.fail("resolved Stage 1A configuration requires unavailable CUDA")
    device = torch.device(device_name)
    dtype = getattr(torch, config.runtime.dtype.value)

    transcoder_config = yaml.safe_load(
        (transcoder_snapshot / "config.yaml").read_text(encoding="utf-8")
    )
    if not isinstance(transcoder_config, dict):
        pytest.fail("transcoder config.yaml must contain a mapping")
    if transcoder_config.get("model_kind") != "transcoder_set":
        pytest.fail("official Stage 1A reproduction requires a PLT TranscoderSet")

    transcoders = transcoder_module.load_transcoder_set(
        _layer_paths(transcoder_snapshot),
        scan_name=f"{config.transcoder.identifier}@{config.transcoder.revision}",
        feature_input_hook=transcoder_config["feature_input_hook"],
        feature_output_hook=transcoder_config["feature_output_hook"],
        activation=transcoder_config.get("activation"),
        k=transcoder_config.get("k"),
        device=device,
        dtype=dtype,
        lazy_encoder=False,
        lazy_decoder=False,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_snapshot), local_files_only=True
    )
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_snapshot),
        local_files_only=True,
        torch_dtype=dtype,
    )
    model = replacement_module.ReplacementModel.from_pretrained_and_transcoders(
        model_name=config.model.identifier,
        transcoders=transcoders,
        backend=config.runtime.backend.value,
        device=device,
        dtype=dtype,
        hf_model=hf_model,
        tokenizer=tokenizer,
    )
    model.eval()
    return model


def test_loaded_official_feature_gate_and_cache_contract() -> None:
    """Verify the pinned public cache and strict gate at one layer/position."""

    config, model_snapshot, transcoder_snapshot = _runtime_inputs_or_skip()
    model = _load_offline_model(config, model_snapshot, transcoder_snapshot)
    torch = importlib.import_module("torch")

    prompt = config.intervention.prompt
    tokens = model.ensure_tokenized(prompt)
    token_count = int(tokens.numel())
    feature = config.intervention.feature
    position = resolve_position_selector(feature.position, token_count)

    _, preactivation_cache = model.get_activations(
        tokens,
        sparse=False,
        apply_activation_function=False,
    )
    _, activation_cache = model.get_activations(
        tokens,
        sparse=False,
        apply_activation_function=True,
    )

    assert preactivation_cache.ndim == 3
    assert activation_cache.shape == preactivation_cache.shape
    assert preactivation_cache.shape[1] == token_count
    assert preactivation_cache.shape[0] > feature.layer
    assert preactivation_cache.shape[2] > feature.feature_id

    layer_transcoder = model.transcoders[feature.layer]
    threshold = layer_transcoder.activation_function.threshold.detach()
    assert threshold.ndim == 1
    assert threshold.shape[0] == preactivation_cache.shape[2]

    # Bounded semantics slice only: one official layer and one resolved position.
    z_vector = (
        preactivation_cache[feature.layer, position].detach().float().cpu().tolist()
    )
    a_vector = activation_cache[feature.layer, position].detach().float().cpu().tolist()
    tau_vector = threshold.float().cpu().tolist()
    expected = apply_strict_jumprelu(z_vector, tau_vector)
    comparison = compare_numeric_sequences(
        expected,
        a_vector,
        absolute_tolerance=5e-3,
        relative_tolerance=1e-5,
    )
    assert comparison.within_tolerance, comparison
    samples = select_gate_samples(z_vector, tau_vector)
    assert a_vector[samples.inactive_feature_id] == 0.0

    equality_output = layer_transcoder.activation_function(threshold.clone())
    assert int(torch.count_nonzero(equality_output).item()) == 0

    official_z = float(preactivation_cache[feature.layer, position, feature.feature_id])
    official_a = float(activation_cache[feature.layer, position, feature.feature_id])
    official_tau = float(threshold[feature.feature_id])
    assert official_a == pytest.approx(strict_jumprelu(official_z, official_tau))
    assert official_z > official_tau, (
        "official feature is inactive; investigate asset or tokenization mismatch"
    )

    def feature_input_only(name: str) -> bool:
        return name.endswith(model.feature_input_hook)

    with torch.inference_mode():
        _, input_cache = model.run_with_cache(prompt, names_filter=feature_input_only)
    input_name = f"blocks.{feature.layer}.{model.feature_input_hook}"
    feature_input = input_cache[input_name]
    manual_preactivation = torch.nn.functional.linear(
        feature_input.to(layer_transcoder.W_enc.dtype),
        layer_transcoder.W_enc,
        layer_transcoder.b_enc,
    ).squeeze(0)
    manual_preactivation[model.zero_positions] = 0
    assert torch.allclose(
        manual_preactivation[position],
        preactivation_cache[feature.layer, position],
        atol=5e-3,
        rtol=1e-5,
    ), "public preactivation must equal W_enc projection plus b_enc, without b_dec"

    desired_noop = desired_activation(official_a, 0.0)
    assert desired_activation(official_a, 0.5) == pytest.approx(0.5 * official_a)
    assert desired_activation(official_a, 1.0) == 0.0
    with torch.inference_mode():
        baseline_logits, _ = model.feature_intervention(
            prompt, [], return_activations=False
        )
        noop_logits, _ = model.feature_intervention(
            prompt,
            [(feature.layer, feature.position, feature.feature_id, desired_noop)],
            freeze_attention=True,
            constrained_layers=None,
            return_activations=False,
        )
        noop_repeat_logits, _ = model.feature_intervention(
            prompt,
            [(feature.layer, feature.position, feature.feature_id, desired_noop)],
            freeze_attention=True,
            constrained_layers=None,
            return_activations=False,
        )
    assert torch.allclose(baseline_logits, noop_logits, atol=2e-2, rtol=2e-3)
    assert torch.allclose(noop_logits, noop_repeat_logits, atol=2e-2, rtol=2e-3)

    # Make the snapshot identities visible in failure output without serializing paths.
    assert json.loads((model_snapshot / "config.json").read_text())["model_type"]
