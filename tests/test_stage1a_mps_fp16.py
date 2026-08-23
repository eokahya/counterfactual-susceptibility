from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cfsus.reproduction.artifacts import ArtifactValidationError
from cfsus.reproduction.config import Stage1AConfigError
from cfsus.reproduction.mps_fp16 import (
    EXECUTION_CLASS,
    MODEL_METADATA_FILES,
    MODEL_REQUIRED_FILES,
    MODEL_WEIGHT_FILES,
    MPS_CLAIM_BOUNDARY,
    OOM_BATCH_SEQUENCE,
    PROJECT_BASE_COMMIT,
    REPRODUCTION_CLASS,
    TRANSCODER_METADATA_FILES,
    TRANSCODER_REQUIRED_FILES,
    TRANSCODER_WEIGHT_FILES,
    MPSRunStatus,
    MPSTelemetrySample,
    _build_asset_manifest,
    _cache_root_for_override,
    _capture_feature_intervention_write,
    _dense_to_cpu_sparse_metadata,
    _download_snapshot_phase,
    _mps_sparse_attribution_adapter,
    _summarize_graph,
    _validate_required_snapshot,
    aggregate_mps_telemetry,
    aggregate_stage_attempt_telemetry,
    batch_deviation,
    classify_mps_failure,
    is_mps_out_of_memory,
    make_sparse_coo_metadata,
    mps_runtime_identity,
    should_retry_attempt,
    sparse_coo_metadata_to_dense_cpu,
    sparse_coo_to_dense_on_device,
    validate_attempt_peak_invariants,
    validate_mps_fp16_mapping,
    validate_mps_runtime_guards,
)


def _config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_name": "stage1a_gemma2_2b_mps_fp16_hardware_adaptation",
        "reproduction_class": REPRODUCTION_CLASS,
        "project_base_commit": PROJECT_BASE_COMMIT,
        "reference_dtype": "bfloat16",
        "execution_dtype": "float16",
        "reference_status": "pending",
        "claim_boundary": MPS_CLAIM_BOUNDARY,
        "environment": {
            "python": "3.11.13",
            "pytorch": "2.6.0",
            "platform": "macos-arm64",
            "hardware_family": "Apple M2 Max",
            "physical_memory_gib": 32,
        },
        "upstream": {
            "repository": "https://github.com/decoderesearch/circuit-tracer",
            "revision": "8f1e2438df612464e229e44c4a00ff637bf9379b",
            "version": "0.5.2",
        },
        "model": {
            "identifier": "google/gemma-2-2b",
            "revision": "c5ebcd40d208330abc697524c919956e692655cf",
            "snapshot_path": (
                "results/generated/stage1a_mps_fp16/assets/google-gemma-2-2b"
            ),
        },
        "transcoder": {
            "identifier": "mwhanna/gemma-scope-transcoders",
            "revision": "bd5773156dea09893636c801df1237d0410307d2",
            "snapshot_path": (
                "results/generated/stage1a_mps_fp16/assets/"
                "mwhanna-gemma-scope-transcoders"
            ),
        },
        "runtime": {
            "backend": "transformerlens",
            "device": "mps",
            "dtype": "float16",
            "execution_class": EXECUTION_CLASS,
            "hardware_family": "Apple M2 Max",
            "official_bf16_reproduction": False,
            "t4_fp16_reproduction": False,
            "fallback_enabled": False,
            "offload": "disk",
        },
        "seeds": {"python": 0, "numpy": 0, "torch": 0},
        "asset_policy": {
            "allow_download": True,
            "require_offline_execution": True,
            "cache_location": "project_external_huggingface_cache",
            "immutable_revisions_only": True,
        },
        "attribution": {
            "prompt": "The capital of state containing Dallas is",
            "max_n_logits": 10,
            "desired_logit_probability": 0.95,
            "max_feature_nodes": 8192,
            "batch_size": 256,
            "offload": "disk",
        },
        "intervention": {
            "prompt": "Hecho: Michael Jordan juega al",
            "feature": {"layer": 20, "position": -1, "feature_id": 341},
            "alphas": [0.0, 0.5, 1.0],
            "freeze_attention": True,
            "constrained_layers": None,
        },
        "numerics": {
            "gate_absolute_tolerance": 0.005,
            "projection_absolute_tolerance": 0.005,
            "noop_absolute_tolerance": 0.02,
            "noop_relative_tolerance": 0.002,
            "determinism_absolute_tolerance": 0.02,
            "determinism_relative_tolerance": 0.002,
            "model_parameter_samples_per_tensor": 16,
            "preflight_absolute_tolerance": 0.005,
            "preflight_relative_tolerance": 0.002,
        },
        "oom_retry": {
            "batch_sizes": [256, 128, 64],
            "trigger": "mps_out_of_memory_only",
            "fresh_process": True,
            "clear_mps_cache_between_attempts": True,
            "retry_on_unknown_failure": False,
        },
        "memory_budget": {
            "physical_memory_gib": 32,
            "safety_reserve_gib": 6,
            "telemetry": "sampled_mps_and_process_rss",
            "do_not_disable_guardrails": True,
            "high_watermark_ratio_override": None,
        },
        "artifacts": {
            "preflight_summary": (
                "results/stage1a_mps_fp16/preflight/preflight_summary.json"
            ),
            "environment_manifest": (
                "results/stage1a_mps_fp16/environment_manifest.json"
            ),
            "asset_manifest": "results/stage1a_mps_fp16/asset_manifest.json",
            "attribution_summary": "results/stage1a_mps_fp16/attribution_summary.json",
            "intervention_summary": (
                "results/stage1a_mps_fp16/intervention_summary.json"
            ),
            "semantics_summary": "results/stage1a_mps_fp16/semantics_summary.json",
            "memory_summary": "results/stage1a_mps_fp16/memory_summary.json",
            "checksums": "results/stage1a_mps_fp16/checksums.sha256",
            "run_manifest": "results/stage1a_mps_fp16/stage1a_mps_run_manifest.json",
        },
    }


def test_mps_config_is_strictly_pinned_and_separate() -> None:
    config = validate_mps_fp16_mapping(_config())
    assert config.runtime.device == "mps"
    assert config.runtime.dtype == "float16"
    assert config.oom_retry.batch_sizes == OOM_BATCH_SEQUENCE

    for section, field, replacement in (
        ("runtime", "device", "cuda"),
        ("runtime", "dtype", "bfloat16"),
        ("runtime", "official_bf16_reproduction", True),
        ("environment", "platform", "linux-x86_64"),
        ("model", "revision", "main"),
        ("attribution", "batch_size", 128),
    ):
        value = _config()
        value[section][field] = replacement
        with pytest.raises(Stage1AConfigError):
            validate_mps_fp16_mapping(value)


def test_mps_identity_and_guards_reject_mislabels_and_fallback() -> None:
    identity = mps_runtime_identity()
    assert identity["backend"] == "mps"
    assert identity["precision"] == "float16"
    assert identity["official_bf16_reproduction"] is False
    with pytest.raises(ArtifactValidationError):
        validate_mps_runtime_guards(
            machine="x86_64", mps_built=True, mps_available=True
        )
    with pytest.raises(ArtifactValidationError):
        validate_mps_runtime_guards(
            machine="arm64", mps_built=True, mps_available=True, fallback_enabled=True
        )
    with pytest.raises(ArtifactValidationError):
        validate_mps_runtime_guards(
            machine="arm64",
            mps_built=True,
            mps_available=True,
            environ={"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0"},
        )


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", " FALSE "])
def test_disabled_fallback_spellings_are_accepted(value: str) -> None:
    validate_mps_runtime_guards(
        machine="arm64",
        mps_built=True,
        mps_available=True,
        environ={"PYTORCH_ENABLE_MPS_FALLBACK": value},
    )


def test_ambiguous_fallback_value_is_rejected() -> None:
    with pytest.raises(ArtifactValidationError, match="ambiguous"):
        validate_mps_runtime_guards(
            machine="arm64",
            mps_built=True,
            mps_available=True,
            environ={"PYTORCH_ENABLE_MPS_FALLBACK": "sometimes"},
        )


def test_only_mps_oom_retries_in_order() -> None:
    mps_error = RuntimeError("MPS backend out of memory: failed to allocate")
    assert is_mps_out_of_memory(mps_error)
    assert not is_mps_out_of_memory(RuntimeError("CUDA out of memory"))
    assert not is_mps_out_of_memory(MemoryError("out of memory"))
    assert classify_mps_failure(mps_error) is MPSRunStatus.BLOCKED_RESOURCE
    assert (
        classify_mps_failure(RuntimeError("CUDA out of memory"))
        is MPSRunStatus.FAILED_RUNTIME
    )
    assert should_retry_attempt(
        batch_size=256, category="mps_out_of_memory", failure_stage="attribution"
    )
    assert should_retry_attempt(
        batch_size=128, category="mps_out_of_memory", failure_stage="attribution"
    )
    assert not should_retry_attempt(
        batch_size=64, category="mps_out_of_memory", failure_stage="attribution"
    )
    assert not should_retry_attempt(
        batch_size=256, category="mps_out_of_memory", failure_stage="loading"
    )
    assert not should_retry_attempt(
        batch_size=256, category="failed_runtime", failure_stage="attribution"
    )
    assert batch_deviation(256) is None
    assert "MPS OOM" in batch_deviation(128)  # type: ignore[operator]


def test_sampled_telemetry_and_attempt_peaks_are_monotone() -> None:
    samples = [
        MPSTelemetrySample(10, 20, 100, 30, "normal", 2, 1.0),
        MPSTelemetrySample(11, 25, 100, 40, "warning", 3, 2.0),
    ]
    aggregate = aggregate_mps_telemetry(samples)
    assert aggregate.peak_current_allocated_bytes == 11
    assert aggregate.peak_driver_allocated_bytes == 25
    assert aggregate.peak_process_rss_bytes == 40
    attempt = aggregate_stage_attempt_telemetry({"attribution": aggregate})
    assert attempt["attempt_peak_driver_allocated_bytes"] == 25
    validate_attempt_peak_invariants(
        {"stage_peaks": {"a": 4, "b": 9}, "attempt_peak_driver_allocated_bytes": 9}
    )
    with pytest.raises(ArtifactValidationError):
        validate_attempt_peak_invariants(
            {"stage_peaks": {"a": 10}, "attempt_peak_driver_allocated_bytes": 9}
        )


def test_sparse_metadata_adapter_matches_cpu_torch_reference() -> None:
    metadata = make_sparse_coo_metadata(
        [[0, 0, 1], [0, 1, 1]], [1.5, 2.0, -3.0], [2, 2]
    )
    dense = sparse_coo_metadata_to_dense_cpu(metadata)
    try:
        import torch  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        assert dense == [[1.5, 2.0], [0.0, -3.0]]
    else:
        assert dense.device.type == "cpu"
        assert dense.tolist() == [[1.5, 2.0], [0.0, -3.0]]
    cpu_again = sparse_coo_to_dense_on_device(metadata, "cpu")
    if isinstance(dense, list):
        assert cpu_again == dense
    else:
        assert cpu_again.tolist() == dense.tolist()


def test_dense_sparse_boundary_is_numerically_equivalent_when_torch_exists() -> None:
    torch = pytest.importorskip("torch")
    device = (
        "mps"
        if bool(torch.backends.mps.is_built())
        and bool(torch.backends.mps.is_available())
        else "cpu"
    )
    source = torch.tensor(
        [[0.0, 1.5, 0.0], [-2.0, 0.0, 3.0]],
        device=device,
        dtype=torch.float16,
    )
    metadata, indices, values = _dense_to_cpu_sparse_metadata(source, torch)
    assert metadata.device.type == "cpu"
    assert indices.device.type == device
    reconstructed = torch.zeros_like(source)
    reconstructed.index_put_((indices[0], indices[1]), values, accumulate=True)
    assert torch.equal(reconstructed, source)


def test_exact_asset_allowlists_cover_only_consumed_files() -> None:
    assert MODEL_REQUIRED_FILES == MODEL_METADATA_FILES + MODEL_WEIGHT_FILES
    assert TRANSCODER_REQUIRED_FILES == (
        TRANSCODER_METADATA_FILES + TRANSCODER_WEIGHT_FILES
    )
    assert MODEL_WEIGHT_FILES == (
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    )
    assert (
        tuple(f"layer_{layer}.safetensors" for layer in range(26))
        == TRANSCODER_WEIGHT_FILES
    )


def test_snapshot_manifest_hashes_contained_hf_symlinks(tmp_path: Path) -> None:
    revision = "a" * 40
    cache = tmp_path / "hub"
    repository = cache / "models--example--asset"
    snapshot = repository / "snapshots" / revision
    blobs = repository / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    (snapshot / "config.json").write_bytes(b"{}\n")
    weight_content = b"verified fake safetensors bytes"
    weight_digest = hashlib.sha256(weight_content).hexdigest()
    blob = blobs / weight_digest
    blob.write_bytes(weight_content)
    weight = snapshot / "weight.safetensors"
    weight.symlink_to(blob)

    manifest = _validate_required_snapshot(
        snapshot,
        cache_root=cache,
        identifier="example/asset",
        revision=revision,
        required_files=("config.json", "weight.safetensors"),
        role="offline fake",
    )
    assert [item["path"] for item in manifest] == [
        "config.json",
        "weight.safetensors",
    ]
    assert manifest[1]["sha256"] == weight_digest
    assert all(not Path(str(item["path"])).is_absolute() for item in manifest)

    blob.write_bytes(b"content changed after the blob name was chosen")
    with pytest.raises(RuntimeError, match="blob identity"):
        _validate_required_snapshot(
            snapshot,
            cache_root=cache,
            identifier="example/asset",
            revision=revision,
            required_files=("config.json", "weight.safetensors"),
            role="offline fake",
        )

    weight.unlink()
    escaped = tmp_path / "escaped.safetensors"
    escaped.write_bytes(weight_content)
    weight.symlink_to(escaped)
    with pytest.raises(RuntimeError, match="escapes"):
        _validate_required_snapshot(
            snapshot,
            cache_root=cache,
            identifier="example/asset",
            revision=revision,
            required_files=("config.json", "weight.safetensors"),
            role="offline fake",
        )


def test_snapshot_override_requires_canonical_external_hf_cache_layout(
    tmp_path: Path,
) -> None:
    revision = "a" * 40
    canonical = tmp_path / "hub" / "models--example--asset" / "snapshots" / revision
    canonical.mkdir(parents=True)
    assert (
        _cache_root_for_override(canonical, identifier="example/asset")
        == tmp_path / "hub"
    )

    arbitrary = tmp_path / "copied-snapshot" / revision
    arbitrary.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="canonical Hugging Face cache"):
        _cache_root_for_override(arbitrary, identifier="example/asset")

    wrong_repository = (
        tmp_path / "hub" / "models--other--asset" / "snapshots" / revision
    )
    wrong_repository.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="exact repository"):
        _cache_root_for_override(wrong_repository, identifier="example/asset")


def test_partial_local_snapshot_forces_exact_online_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = "a" * 40
    cache = tmp_path / "hub"
    snapshot = cache / "models--example--asset" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    calls: list[bool] = []

    def snapshot_download(**kwargs: Any) -> str:
        local_only = bool(kwargs["local_files_only"])
        calls.append(local_only)
        if not local_only:
            for name in kwargs["allow_patterns"]:
                (snapshot / name).write_bytes(b"pinned phase content")
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    resolved = _download_snapshot_phase(
        identifier="example/asset",
        revision=revision,
        allow_patterns=("weight.safetensors",),
        cache_root=cache,
        allow_download=True,
    )
    assert resolved == snapshot
    assert calls == [True, False]
    assert (snapshot / "weight.safetensors").is_file()

    (snapshot / "weight.safetensors").unlink()
    calls.clear()
    with pytest.raises(RuntimeError, match=r"local.*phase is unavailable"):
        _download_snapshot_phase(
            identifier="example/asset",
            revision=revision,
            allow_patterns=("weight.safetensors",),
            cache_root=cache,
            allow_download=False,
        )
    assert calls == [True]


def test_asset_manifest_is_json_only_and_contains_observed_totals() -> None:
    model_files = [{"path": "config.json", "size_bytes": 2, "sha256": "a" * 64}]
    transcoder_files = [{"path": "config.yaml", "size_bytes": 3, "sha256": "b" * 64}]
    manifest = _build_asset_manifest(model_files, transcoder_files)
    assert manifest["model"]["file_count"] == 1
    assert manifest["model"]["total_bytes"] == 2
    assert manifest["transcoder"]["total_bytes"] == 3
    assert manifest["project_external_cache"] is True
    assert "snapshot_path" not in repr(manifest)


class _FakeScalar:
    def __init__(self, value: int | bool) -> None:
        self.value = value

    def all(self) -> _FakeScalar:
        return self

    def item(self) -> int | bool:
        return self.value


class _FakeTensor:
    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        length: int,
        nonzero: int = 1,
        minimum: int = 0,
        maximum: int = 0,
    ) -> None:
        self.shape = shape
        self.ndim = len(shape)
        self.length = length
        self.nonzero = nonzero
        self.minimum = minimum
        self.maximum = maximum

    def __len__(self) -> int:
        return self.length

    def detach(self) -> _FakeTensor:
        return self

    def numel(self) -> int:
        result = 1
        for value in self.shape:
            result *= value
        return result

    def min(self) -> _FakeScalar:
        return _FakeScalar(self.minimum)

    def max(self) -> _FakeScalar:
        return _FakeScalar(self.maximum)


class _FakeTorch:
    @staticmethod
    def isfinite(_value: Any) -> _FakeScalar:
        return _FakeScalar(True)

    @staticmethod
    def count_nonzero(value: _FakeTensor) -> _FakeScalar:
        return _FakeScalar(value.nonzero)

    @staticmethod
    def unique(value: _FakeTensor) -> _FakeTensor:
        return value


def test_graph_summary_uses_observed_dimensions_and_exact_counts() -> None:
    graph = SimpleNamespace(
        adjacency_matrix=_FakeTensor(shape=(21, 21), length=21, nonzero=7),
        selected_features=_FakeTensor(shape=(2,), length=2, minimum=3, maximum=8),
        active_features=_FakeTensor(shape=(50, 3), length=50),
        activation_values=_FakeTensor(shape=(50,), length=50),
        logit_targets=list(range(10)),
        logit_probabilities=_FakeTensor(shape=(10,), length=10),
        n_pos=3,
        cfg=SimpleNamespace(n_layers=2),
    )
    summary = _summarize_graph(graph, _FakeTorch, require_ten_logits=True)
    assert summary == {
        "finite": True,
        "node_count": 21,
        "adjacency_shape": [21, 21],
        "active_feature_count": 50,
        "selected_feature_count": 2,
        "error_node_count": 6,
        "input_node_count": 3,
        "logit_node_count": 10,
        "edge_count": 7,
    }
    graph.logit_targets = list(range(9))
    graph.logit_probabilities = _FakeTensor(shape=(9,), length=9)
    with pytest.raises(ArtifactValidationError, match="10 logits"):
        _summarize_graph(graph, _FakeTorch, require_ten_logits=True)


class _CaptureValue:
    def __init__(self, label: str) -> None:
        self.label = label

    def detach(self) -> _CaptureValue:
        return self

    def clone(self) -> _CaptureValue:
        return _CaptureValue(self.label)


class _FakeInterventionModel:
    feature_output_hook = "hook_out"

    def __init__(self) -> None:
        self.before = _CaptureValue("before")
        self.after = _CaptureValue("after")

    def _get_feature_intervention_hooks(self, *_args: Any, **_kwargs: Any) -> Any:
        target = "blocks.20.hook_out"

        def calculate_delta_hook(activations: Any, hook: Any) -> Any:
            del hook
            return activations

        def intervention_hook(activations: Any, hook: Any) -> Any:
            del activations, hook
            return self.after

        return (
            [(target, calculate_delta_hook), (target, intervention_hook)],
            [],
            [],
        )

    def feature_intervention(self) -> Any:
        hooks, _, _ = self._get_feature_intervention_hooks()
        value = self.before
        for _, hook in hooks:
            replacement = hook(value, hook=None)
            if replacement is not None:
                value = replacement
        return value


def test_real_intervention_hook_is_instrumented_and_restored() -> None:
    model = _FakeInterventionModel()
    original = model._get_feature_intervention_hooks
    with _capture_feature_intervention_write(model, layer=20) as captured:
        result = model.feature_intervention()
    assert result.label == "after"
    assert captured["before"].label == "before"
    assert captured["after"].label == "after"
    assert model._get_feature_intervention_hooks == original


def test_sparse_attribution_adapter_installs_and_restores(monkeypatch: Any) -> None:
    root = types.ModuleType("circuit_tracer")
    attribution = types.ModuleType("circuit_tracer.attribution")
    attribute_module = types.ModuleType(
        "circuit_tracer.attribution.attribute_transformerlens"
    )
    context_module = types.ModuleType(
        "circuit_tracer.attribution.context_transformerlens"
    )

    def original_partial(*_args: Any, **_kwargs: Any) -> str:
        return "partial"

    def original_batch(*_args: Any, **_kwargs: Any) -> str:
        return "batch"

    class AttributionContext:
        compute_batch = original_batch

    attribute_module.compute_partial_influences = original_partial  # type: ignore[attr-defined]
    attribution.attribute_transformerlens = attribute_module  # type: ignore[attr-defined]
    context_module.AttributionContext = AttributionContext  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "circuit_tracer", root)
    monkeypatch.setitem(sys.modules, "circuit_tracer.attribution", attribution)
    monkeypatch.setitem(
        sys.modules,
        "circuit_tracer.attribution.attribute_transformerlens",
        attribute_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "circuit_tracer.attribution.context_transformerlens",
        context_module,
    )

    class Transcoders:
        def compute_attribution_components(self, *_args: Any) -> str:
            return "components"

    model = SimpleNamespace(transcoders=Transcoders())
    original_components = model.transcoders.compute_attribution_components
    with _mps_sparse_attribution_adapter(model) as usage:
        assert usage == {
            "component_calls": 0,
            "batch_calls": 0,
            "partial_calls": 0,
        }
        assert model.transcoders.compute_attribution_components != original_components
        assert AttributionContext.compute_batch is not original_batch
        assert attribute_module.compute_partial_influences is not original_partial
    assert model.transcoders.compute_attribution_components == original_components
    assert AttributionContext.compute_batch is original_batch
    assert attribute_module.compute_partial_influences is original_partial
