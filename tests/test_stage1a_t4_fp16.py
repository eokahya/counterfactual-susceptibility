from __future__ import annotations

import json
import math
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STAGE1A_SCRIPTS = REPOSITORY_ROOT / "scripts/stage1a"
if str(STAGE1A_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STAGE1A_SCRIPTS))

from run_stage1a_t4_fp16 import _write_manifest  # noqa: E402
from validate_t4_fp16_artifacts import (  # noqa: E402
    BUNDLE_PREFIX,
    validate_t4_artifact_directory,
    validate_t4_small_bundle,
    write_t4_checksums,
)

from cfsus.reproduction.artifacts import (  # noqa: E402
    REDACTED,
    ArtifactValidationError,
    make_artifact_envelope,
    write_json_atomic,
)
from cfsus.reproduction.config import Stage1AConfigError  # noqa: E402
from cfsus.reproduction.t4_fp16 import (  # noqa: E402
    MAX_BUNDLE_MEMBER_BYTES,
    OOM_BATCH_SEQUENCE,
    T4_CLAIM_BOUNDARY,
    T4_SMALL_FILES,
    T4RunStatus,
    batch_deviation,
    classify_t4_failure,
    is_cuda_out_of_memory,
    sanitize_exception_message,
    should_retry_attempt,
    validate_t4_fp16_mapping,
    validate_t4_run_manifest,
)


def _valid_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_name": "stage1a_gemma2_2b_t4_fp16_hardware_adaptation",
        "reproduction_class": "hardware_adapted_fp16",
        "project_base_commit": "13b42a5debe38def14f173530bcbc81ca3f8440e",
        "reference_dtype": "bfloat16",
        "execution_dtype": "float16",
        "reference_status": "pending",
        "claim_boundary": T4_CLAIM_BOUNDARY,
        "environment": {
            "python": "3.11",
            "pytorch": "2.6.0",
            "cuda_wheel": "cu124",
        },
        "upstream": {
            "repository": "https://github.com/decoderesearch/circuit-tracer",
            "revision": "8f1e2438df612464e229e44c4a00ff637bf9379b",
        },
        "model": {
            "identifier": "google/gemma-2-2b",
            "revision": "c5ebcd40d208330abc697524c919956e692655cf",
            "snapshot_path": (
                "results/generated/stage1a_t4_fp16/assets/google-gemma-2-2b"
            ),
        },
        "transcoder": {
            "identifier": "mwhanna/gemma-scope-transcoders",
            "revision": "bd5773156dea09893636c801df1237d0410307d2",
            "snapshot_path": (
                "results/generated/stage1a_t4_fp16/assets/"
                "mwhanna-gemma-scope-transcoders"
            ),
        },
        "runtime": {
            "backend": "transformerlens",
            "device": "cuda",
            "dtype": "float16",
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
        },
        "oom_retry": {
            "batch_sizes": [256, 128, 64],
            "trigger": "cuda_out_of_memory_only",
            "fresh_process": True,
            "clear_cuda_cache_between_attempts": True,
        },
        "artifacts": {
            "raw_graph": ("results/generated/stage1a_t4_fp16/attribution_graph.pt"),
            "environment_manifest": (
                "results/stage1a_t4_fp16/environment_manifest.json"
            ),
            "asset_manifest": "results/stage1a_t4_fp16/asset_manifest.json",
            "attribution_summary": ("results/stage1a_t4_fp16/attribution_summary.json"),
            "intervention_summary": (
                "results/stage1a_t4_fp16/intervention_summary.json"
            ),
            "semantics_summary": ("results/stage1a_t4_fp16/semantics_summary.json"),
            "checksums": "results/stage1a_t4_fp16/checksums.sha256",
            "run_manifest": (
                "results/stage1a_t4_fp16/stage1a_t4_fp16_run_manifest.json"
            ),
        },
    }


def _valid_manifest() -> dict[str, Any]:
    check_names = {
        "immutable_assets_loaded",
        "model_parameter_samples_finite",
        "thresholds_finite",
        "baseline_logits_finite",
        "cached_values_finite",
        "attribution_values_finite",
        "intervention_values_finite",
        "baseline_repeat_within_tolerance",
        "noop_within_tolerance",
        "jumprelu_semantics_passed",
        "desired_value_mapping_passed",
        "artifact_validation_passed",
        "attribution_graph_nonempty",
        "intervention_completed",
        "semantics_completed",
    }
    return {
        "schema_version": 1,
        "status": "completed_hardware_adapted_fp16",
        "reproduction_class": "hardware_adapted_fp16",
        "claim_boundary": T4_CLAIM_BOUNDARY,
        "project": {
            "base_commit": "13b42a5debe38def14f173530bcbc81ca3f8440e",
            "execution_commit": "1" * 40,
            "dirty": False,
        },
        "upstream": {
            "repository": "https://github.com/decoderesearch/circuit-tracer",
            "revision": "8f1e2438df612464e229e44c4a00ff637bf9379b",
        },
        "model": {
            "identifier": "google/gemma-2-2b",
            "revision": "c5ebcd40d208330abc697524c919956e692655cf",
        },
        "transcoder": {
            "identifier": "mwhanna/gemma-scope-transcoders",
            "revision": "bd5773156dea09893636c801df1237d0410307d2",
        },
        "runtime": {
            "backend": "transformerlens",
            "device": "cuda",
            "gpu_name": "Tesla T4",
            "compute_capability": [7, 5],
            "torch_version": "2.6.0+cu124",
            "torch_cuda_version": "12.4",
            "reference_dtype": "bfloat16",
            "execution_dtype": "float16",
            "bf16_supported": False,
        },
        "attribution": {
            "attempted_batch_sizes": [256],
            "selected_batch_size": 256,
            "batch_deviation": None,
        },
        "retry_history": [
            {
                "batch_size": 256,
                "outcome": "completed",
                "category": "completed",
                "exception_type": None,
                "message": "completed",
                "failure_stage": None,
                "peak_memory_bytes": 1,
                "wall_seconds": 1.0,
                "cleanup_succeeded": True,
            }
        ],
        "timings": {
            "attempt_wall_seconds": [1.0],
            "attempt_peak_memory_bytes": [1],
        },
        "checks": {**dict.fromkeys(check_names, True), "nonfinite_count": 0},
        "artifacts": {
            name: {"size_bytes": 1, "sha256": "a" * 64}
            for name in T4_SMALL_FILES
            if name != "stage1a_t4_fp16_run_manifest.json"
        },
        "readiness": {
            "stage1b_engineering_readiness": True,
            "stage1b_empirical_claim_readiness": False,
        },
        "bf16_reference": {
            "dtype": "bfloat16",
            "status": "pending",
            "statement": "Native-BF16 reference reproduction remains pending.",
        },
    }


def test_t4_config_is_separate_strict_and_pinned() -> None:
    config = validate_t4_fp16_mapping(_valid_config())

    assert config.runtime.device == "cuda"
    assert config.runtime.dtype == "float16"
    assert config.reference_dtype == "bfloat16"
    assert config.oom_retry.batch_sizes == OOM_BATCH_SEQUENCE
    assert config.artifacts.run_manifest.endswith("stage1a_t4_fp16_run_manifest.json")


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("runtime", "dtype", "bfloat16"),
        ("runtime", "device", "mps"),
        ("model", "revision", "main"),
        ("model", "revision", "1" * 40),
        ("transcoder", "revision", "0" * 40),
        ("attribution", "prompt", "changed"),
        ("attribution", "batch_size", 128),
        ("attribution", "offload", "cpu"),
        ("intervention", "alphas", [0.0, 1.0]),
    ],
)
def test_t4_config_rejects_mutable_or_changed_scientific_inputs(
    section: str, field: str, value: object
) -> None:
    config = _valid_config()
    config[section][field] = value

    with pytest.raises(Stage1AConfigError):
        validate_t4_fp16_mapping(config)


def test_t4_config_rejects_overclaim_and_unsafe_paths() -> None:
    overclaim = _valid_config()
    overclaim["experiment_name"] = "stage1a exact reproduction"
    with pytest.raises(Stage1AConfigError):
        validate_t4_fp16_mapping(overclaim)

    unsafe = _valid_config()
    unsafe["artifacts"]["raw_graph"] = "/tmp/graph.pt"
    with pytest.raises(Stage1AConfigError, match="under"):
        validate_t4_fp16_mapping(unsafe)


def test_cuda_oom_classifier_accepts_only_cuda_specific_failures() -> None:
    cuda_oom_type = type(
        "OutOfMemoryError", (RuntimeError,), {"__module__": "torch.cuda"}
    )
    assert is_cuda_out_of_memory(cuda_oom_type("allocation failed"))
    assert is_cuda_out_of_memory(RuntimeError("CUDA out of memory. Tried 2 GiB"))
    assert not is_cuda_out_of_memory(MemoryError("CPU allocation failed"))
    assert not is_cuda_out_of_memory(RuntimeError("out of memory"))
    assert not is_cuda_out_of_memory(RuntimeError("CUDA illegal memory access"))

    wrapper = RuntimeError("wrapped")
    wrapper.__cause__ = RuntimeError("CUDA error: out of memory")
    assert is_cuda_out_of_memory(wrapper)


def test_failure_classification_is_fail_closed_and_non_oom_does_not_retry() -> None:
    assert classify_t4_failure(RuntimeError("CUDA out of memory")) is (
        T4RunStatus.BLOCKED_RESOURCE
    )
    assert classify_t4_failure(RuntimeError("non-finite logits")) is (
        T4RunStatus.FAILED_PRECISION
    )
    assert classify_t4_failure(AssertionError("asset mismatch")) is (
        T4RunStatus.FAILED_RUNTIME
    )
    assert OOM_BATCH_SEQUENCE == (256, 128, 64)
    assert should_retry_attempt(
        batch_size=256,
        category="cuda_out_of_memory",
        failure_stage="attribution",
    )
    assert not should_retry_attempt(
        batch_size=128,
        category="failed_precision",
        failure_stage="attribution",
    )
    assert not should_retry_attempt(
        batch_size=128,
        category="cuda_out_of_memory",
        failure_stage="runtime_loading",
    )
    assert not should_retry_attempt(
        batch_size=64,
        category="cuda_out_of_memory",
        failure_stage="attribution",
    )


def test_batch_deviation_is_recorded_only_for_reduced_batches() -> None:
    assert batch_deviation(256) is None
    assert "no bitwise-equivalence claim" in str(batch_deviation(128))
    assert "64" in str(batch_deviation(64))


@pytest.mark.parametrize("status", [status.value for status in T4RunStatus])
def test_manifest_accepts_only_declared_terminal_statuses(status: str) -> None:
    manifest = _valid_manifest()
    manifest["status"] = status
    completed = status == T4RunStatus.COMPLETED
    manifest["readiness"]["stage1b_engineering_readiness"] = completed
    if not completed:
        manifest["attribution"]["selected_batch_size"] = None
        manifest["retry_history"][0].update(
            {
                "outcome": "failed",
                "category": status,
                "exception_type": "RuntimeError",
                "failure_stage": "runtime_loading",
                "message": "sanitized failure",
            }
        )
    if status == T4RunStatus.PREPARED:
        manifest["attribution"]["attempted_batch_sizes"] = []
        manifest["retry_history"] = []
        manifest["timings"] = {
            "attempt_wall_seconds": [],
            "attempt_peak_memory_bytes": [],
        }
        manifest["checks"] = {
            name: (0 if name == "nonfinite_count" else False)
            for name in manifest["checks"]
        }
        manifest["artifacts"] = {}

    validate_t4_run_manifest(manifest)


def test_manifest_rejects_overclaim_nonfinite_and_false_readiness() -> None:
    invalid_status = _valid_manifest()
    invalid_status["status"] = "completed"
    with pytest.raises(ArtifactValidationError, match="terminal status"):
        validate_t4_run_manifest(invalid_status)

    nonfinite = _valid_manifest()
    nonfinite["checks"]["nonfinite_count"] = 1
    with pytest.raises(ArtifactValidationError, match="finite selected batch"):
        validate_t4_run_manifest(nonfinite)

    empirical = _valid_manifest()
    empirical["readiness"]["stage1b_empirical_claim_readiness"] = True
    with pytest.raises(ArtifactValidationError, match="must be False"):
        validate_t4_run_manifest(empirical)

    nan_timing = _valid_manifest()
    nan_timing["timings"]["bad"] = math.nan
    with pytest.raises(ArtifactValidationError, match="finite"):
        validate_t4_run_manifest(nan_timing)


def test_manifest_requires_cuda_oom_attribution_stage_for_retry() -> None:
    manifest = _valid_manifest()
    manifest["attribution"] = {
        "attempted_batch_sizes": [256, 128],
        "selected_batch_size": 128,
        "batch_deviation": batch_deviation(128),
    }
    manifest["retry_history"] = [
        {
            "batch_size": 256,
            "outcome": "failed",
            "category": "cuda_out_of_memory",
            "exception_type": "OutOfMemoryError",
            "failure_stage": "attribution",
            "message": "CUDA out of memory",
            "peak_memory_bytes": 1,
            "wall_seconds": 1.0,
            "cleanup_succeeded": True,
        },
        {
            "batch_size": 128,
            "outcome": "completed",
            "category": "completed",
            "exception_type": None,
            "failure_stage": None,
            "message": "completed",
            "peak_memory_bytes": 1,
            "wall_seconds": 1.0,
            "cleanup_succeeded": True,
        },
    ]
    manifest["timings"] = {
        "attempt_wall_seconds": [1.0, 1.0],
        "attempt_peak_memory_bytes": [1, 1],
    }
    validate_t4_run_manifest(manifest)

    manifest["retry_history"][0]["failure_stage"] = "runtime_loading"
    with pytest.raises(ArtifactValidationError, match="attribution-stage"):
        validate_t4_run_manifest(manifest)


def test_sanitized_cuda_error_does_not_leak_token_or_private_path() -> None:
    # commit-safety: allow-test-fixture
    token = "hf_" + "z" * 24
    private_path = "/" + "Users/alice/cache"
    error = RuntimeError(f"CUDA out of memory token={token} {private_path}")
    message = sanitize_exception_message(error)

    assert message == REDACTED
    assert token not in message
    assert "alice" not in message


def _write_bundle(path: Path, names: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, json.dumps({"safe": True}))


def test_t4_bundle_allowlist_and_traversal_rules(tmp_path: Path) -> None:
    expected_names = [f"{BUNDLE_PREFIX}{name}" for name in sorted(T4_SMALL_FILES)]
    valid = tmp_path / "valid.zip"
    _write_bundle(valid, expected_names)
    assert set(validate_t4_small_bundle(valid)) == set(expected_names)

    raw = tmp_path / "raw.zip"
    _write_bundle(raw, [*expected_names, f"{BUNDLE_PREFIX}graph.pt"])
    with pytest.raises(ArtifactValidationError, match="allowlist"):
        validate_t4_small_bundle(raw)

    traversal = tmp_path / "traversal.zip"
    _write_bundle(traversal, [*expected_names[:-1], "../escape.json"])
    with pytest.raises(ArtifactValidationError):
        validate_t4_small_bundle(traversal)

    oversized = tmp_path / "oversized.zip"
    with zipfile.ZipFile(oversized, "w") as archive:
        for name in expected_names:
            content = (
                b"x" * (MAX_BUNDLE_MEMBER_BYTES + 1)
                if name.endswith("environment_manifest.json")
                else b"{}"
            )
            archive.writestr(name, content)
    with pytest.raises(ArtifactValidationError, match="size limit"):
        validate_t4_small_bundle(oversized)

    symlink = tmp_path / "symlink.zip"
    with zipfile.ZipFile(symlink, "w") as archive:
        for name in expected_names:
            info = zipfile.ZipInfo(name)
            info.external_attr = (0o120777 << 16) if name == expected_names[0] else 0
            archive.writestr(info, b"{}")
    with pytest.raises(ArtifactValidationError, match="unsafe member"):
        validate_t4_small_bundle(symlink)


def test_t4_artifact_validator_rejects_nonfinite_summary(tmp_path: Path) -> None:
    directory = tmp_path / "results/stage1a_t4_fp16"
    directory.mkdir(parents=True)
    record = {
        "schema_version": 1,
        "artifact_type": "attribution_summary",
        "run_id": "t4-test",
        "status": "completed",
        "provenance": {},
        "payload": {"nonfinite_count": 1, "value": math.nan},
        "warnings": [],
        "deviations": [],
    }
    (directory / "attribution_summary.json").write_text(
        json.dumps(record), encoding="utf-8"
    )

    with pytest.raises(ArtifactValidationError, match="finite"):
        validate_t4_artifact_directory(
            directory,
            require_run_manifest=False,
            require_complete=False,
        )


def test_completed_t4_artifact_set_validates_end_to_end(tmp_path: Path) -> None:
    directory = tmp_path / "results/stage1a_t4_fp16"
    directory.mkdir(parents=True)
    provenance = {
        "upstream_revision": "8f1e2438df612464e229e44c4a00ff637bf9379b",
        "model_identifier": "google/gemma-2-2b",
        "model_revision": "c5ebcd40d208330abc697524c919956e692655cf",
        "transcoder_identifier": "mwhanna/gemma-scope-transcoders",
        "transcoder_revision": "bd5773156dea09893636c801df1237d0410307d2",
        "backend": "transformerlens",
        "device": "cuda",
        "dtype": "float16",
        "reproduction_class": "hardware_adapted_fp16",
        "reference_dtype": "bfloat16",
        "execution_dtype": "float16",
        "reference_status": "pending",
        "project_commit": "1" * 40,
        "project_dirty": False,
        "asset_integrity": {
            "verification": "exact_file_content_hashes_matched",
        },
        "parameter_finiteness_sample": {"passed": True},
        "threshold_finiteness": {"passed": True},
        "gpu": {
            "name": "Tesla T4",
            "compute_capability": [7, 5],
            "bf16_supported": False,
            "torch_version": "2.6.0+cu124",
            "torch_cuda_version": "12.4",
        },
    }
    common_boundary = T4_CLAIM_BOUNDARY
    artifacts = {
        "environment_manifest.json": make_artifact_envelope(
            artifact_type="environment_manifest",
            run_id="t4-environment",
            status="observed",
            provenance={},
            payload={},
        ),
        "asset_manifest.json": make_artifact_envelope(
            artifact_type="asset_manifest",
            run_id="t4-assets",
            status="resolved",
            provenance={},
            payload={},
        ),
        "attribution_summary.json": make_artifact_envelope(
            artifact_type="attribution_summary",
            run_id="t4-attribution",
            status="completed",
            provenance=provenance,
            payload={
                "nonfinite_count": 0,
                "claim_boundary": common_boundary,
                "parameters": {"batch_size": 256},
                "graph": {"finite": True, "selected_feature_count": 1},
                "raw_validation": {"passed": True},
            },
        ),
        "intervention_summary.json": make_artifact_envelope(
            artifact_type="intervention_summary",
            run_id="t4-intervention",
            status="completed",
            provenance=provenance,
            payload={
                "nonfinite_count": 0,
                "claim_boundary": common_boundary,
                "baseline_noop_comparison": {"within_tolerance": True},
                "determinism": {"within_tolerance": True},
                "desired_values": [
                    {"alpha": 0.0},
                    {"alpha": 0.5},
                    {"alpha": 1.0},
                ],
            },
        ),
        "semantics_summary.json": make_artifact_envelope(
            artifact_type="semantics_summary",
            run_id="t4-semantics",
            status="completed",
            provenance=provenance,
            payload={
                "nonfinite_count": 0,
                "claim_boundary": common_boundary,
                "gate_check": {
                    "strict_greater_than": True,
                    "equality_inactive": True,
                },
                "intervention_value_check": {
                    "desired_values": [{}, {}, {}],
                },
            },
        ),
    }
    for name, artifact in artifacts.items():
        write_json_atomic(directory / name, artifact)
    write_t4_checksums(directory)
    _write_manifest(
        directory=directory,
        project_commit="1" * 40,
        project_dirty=False,
        status="completed_hardware_adapted_fp16",
        selected_batch=256,
        history=[
            {
                "batch_size": 256,
                "outcome": "completed",
                "category": "completed",
                "exception_type": None,
                "message": "completed",
                "failure_stage": None,
                "peak_memory_bytes": 1,
                "wall_seconds": 1.0,
                "cleanup_succeeded": True,
            }
        ],
        provenance=provenance,
        success=True,
    )

    assert set(validate_t4_artifact_directory(directory)) == T4_SMALL_FILES


def test_t4_notebook_is_output_free_compiles_and_invokes_tracked_runner() -> None:
    path = REPOSITORY_ROOT / "notebooks/stage1a_t4_fp16_reproduction_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    for index, cell in enumerate(notebook["cells"]):
        assert not cell.get("attachments")
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), f"{path}:cell-{index}", "exec")
    assert 'PROJECT_REF = "stage-1a-t4-fp16"' in source
    assert 'EXPECTED_PROJECT_COMMIT = ""' in source
    assert "run_stage1a_t4_fp16.py" in source
    assert "validate_t4_fp16_artifacts.py" in source
    assert "torch.float16" in source
    assert "torch.cuda.is_bf16_supported()" in source
    assert "/content/stage1a-t4-fp16-small-artifacts.zip" in source
    assert "torch.bfloat16" not in source


@pytest.mark.parametrize(
    "path",
    [
        "configs/stage1a_gemma2_2b_official_reproduction.yaml",
        "notebooks/stage1a_official_reproduction_colab.ipynb",
        "docs/STAGE_1A_REPORT.md",
    ],
)
def test_original_bf16_files_match_the_required_base_commit(path: str) -> None:
    expected = subprocess.run(
        ("git", "show", f"13b42a5debe38def14f173530bcbc81ca3f8440e:{path}"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout

    assert (REPOSITORY_ROOT / path).read_bytes() == expected
