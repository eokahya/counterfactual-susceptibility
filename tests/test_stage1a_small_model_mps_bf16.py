"""Offline policy tests for the isolated Stage 1A-S-BF16 runtime."""

from __future__ import annotations

import copy
import json
import os
import runpy
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from cfsus.reproduction.artifacts import ArtifactValidationError
from cfsus.reproduction.small_model_mps_bf16 import (
    ARTIFACT_ALLOWLIST,
    COMPLETED_STATUS,
    EXECUTION_BASE_COMMIT,
    EXPERIMENT_CLASS,
    assert_fallback_disabled,
    bf16_ulp,
    conservative_memory_feasibility,
    feature_selection_audit_from_graph,
    layerwise_jumprelu_reference,
    load_bf16_config,
    projected_graph_bytes,
    select_feature_from_graph,
    validate_projected_manifest,
    validate_small_artifact_directory,
    within_bf16_ulps,
)
from cfsus.reproduction.small_model_mps_bf16_artifacts import (
    _validate_intervention,
    scan_artifact_text,
    strict_json_load,
    validate_checksum_manifest,
    validate_peak_hierarchy,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage1a_gemma3_270m_mps_bf16_pilot.yaml"
PROJECTED = ROOT / "configs/stage1a_small_model_projected_download.json"
ARTIFACT_SCHEMA = ROOT / "configs/stage1a_small_model_mps_bf16_artifact_schema.json"
RUNTIME_ADAPTER = ROOT / "src/cfsus/reproduction/small_model_mps_bf16_runtime.py"
RUNNER = ROOT / "scripts/stage1a/run_stage1a_small_model_mps_bf16.py"
WORKER = ROOT / "scripts/stage1a/run_stage1a_small_model_mps_bf16_worker.py"


def test_bf16_config_is_exact_and_separate() -> None:
    pytest.importorskip("yaml")
    config = load_bf16_config(CONFIG)
    assert config["experiment_class"] == EXPERIMENT_CLASS
    assert config["completed_status"] == COMPLETED_STATUS
    assert config["runtime"]["dtype"] == "bfloat16"
    assert config["runtime"]["device"] == "mps"
    assert config["runtime"]["outer_autocast_allowed"] is False
    assert config["accepted"]["intervention_alphas"] == [0.0, 0.5, 1.0]
    assert (
        config["tolerances"]["baseline_noop_maximum_absolute_logit_difference"] == 0.0
    )


@pytest.mark.parametrize(
    ("section", "key", "bad_value"),
    [
        ("runtime", "device", "cuda"),
        ("runtime", "dtype", "float16"),
        ("runtime", "outer_autocast_allowed", True),
        ("model", "revision", "main"),
        ("transcoder", "revision", "main"),
        ("feature_selection", "manual_selection_allowed", True),
        ("accepted", "attribution_batch_sizes", [32, 16]),
        ("accepted", "freeze_attention", False),
        ("tolerances", "fp32_reference_norm_ratio_minimum", 0.90),
        ("tolerances", "fp32_reference_magnitude_ratio_maximum", 3.0),
        ("tolerances", "baseline_noop_maximum_absolute_logit_difference", 1.0),
    ],
)
def test_bf16_config_rejects_mutation(
    section: str, key: str, bad_value: object
) -> None:
    pytest.importorskip("yaml")
    import yaml

    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    candidate = copy.deepcopy(value)
    candidate[section][key] = bad_value
    from cfsus.reproduction.small_model_mps_bf16 import validate_bf16_config

    with pytest.raises(ArtifactValidationError):
        validate_bf16_config(candidate)


def test_projected_manifest_remains_exact() -> None:
    value = json.loads(PROJECTED.read_text(encoding="utf-8"))
    result = validate_projected_manifest(value)
    assert result["projected_total_bytes"] == 2_087_816_677


def test_bf16_artifact_schema_freezes_allowlist_and_claim_boundary() -> None:
    schema = json.loads(ARTIFACT_SCHEMA.read_text(encoding="utf-8"))
    properties = schema["properties"]
    assert set(properties["artifact_allowlist"]["const"]) == ARTIFACT_ALLOWLIST
    assert properties["completed_status"]["const"] == COMPLETED_STATUS
    assert (
        properties["provenance"]["const"]["execution_base_commit"]
        == EXECUTION_BASE_COMMIT
    )
    readiness = properties["readiness_on_success"]["const"]
    assert readiness["stage1b_engineering_readiness"] is True
    assert readiness["stage1b_empirical_claim_readiness"] is False
    assert readiness["paper_results_readiness"] is False


def test_fallback_detector_is_fail_closed_for_unknown_values() -> None:
    assert_fallback_disabled({})
    assert_fallback_disabled({"PYTORCH_ENABLE_MPS_FALLBACK": "false"})
    for value in ("1", "true", "maybe", "enabled"):
        with pytest.raises(ArtifactValidationError):
            assert_fallback_disabled({"PYTORCH_ENABLE_MPS_FALLBACK": value})


def test_bf16_ulp_and_known_overflow_envelope() -> None:
    assert bf16_ulp(69_120.0) == 512.0
    assert within_bf16_ulps(69_120.0, 68_928.0, 1)
    assert not within_bf16_ulps(70_000.0, 68_928.0, 1)


def test_normalized_l2_promotes_only_after_cpu_boundary() -> None:
    torch = pytest.importorskip("torch")
    from cfsus.reproduction.small_model_mps_bf16 import normalized_l2

    observed = torch.tensor([1.0, 2.0], dtype=torch.float32)
    reference = torch.tensor([1.0, 2.0], dtype=torch.float32)
    assert normalized_l2(observed, reference, torch) == 0.0


def test_loaded_jumprelu_reference_uses_each_layers_thresholds() -> None:
    torch = pytest.importorskip("torch")
    preactivations = torch.tensor([[[2.0, 2.0]], [[2.0, 2.0]]])
    thresholds = torch.tensor([[1.0, 3.0], [3.0, 1.0]])
    observed = layerwise_jumprelu_reference(preactivations, thresholds, torch)
    assert torch.equal(
        observed,
        torch.tensor([[[2.0, 0.0]], [[0.0, 2.0]]]),
    )


def test_memory_and_graph_projections_are_conservative() -> None:
    memory = conservative_memory_feasibility()
    assert memory.feasible
    assert memory.total_conservative_bytes < 24 * 1024**3
    assert memory.full_eager_plt_bytes > 700 * 1024**2
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


def test_feature_selection_uses_direct_effect_and_stable_tie_break() -> None:
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
    selected = select_feature_from_graph(graph, final_position=5)
    assert (selected.layer, selected.position, selected.feature) == (1, 5, 8)
    audit = feature_selection_audit_from_graph(
        graph, final_position=5, selection=selected
    )
    assert audit["candidate_count"] == 2
    assert audit["source_count"] == 2
    assert audit["excluded_counts"] == {
        "non_final_position": 0,
        "nonpositive_baseline": 0,
        "nonfinite_baseline": 0,
        "nonfinite_score": 0,
    }


def test_artifact_structure_rejects_extra_archive_and_hardlink(tmp_path: Path) -> None:
    directory = tmp_path / "bundle"
    directory.mkdir()
    for name in ARTIFACT_ALLOWLIST:
        (directory / name).write_text("{}\n", encoding="utf-8")
    validate_small_artifact_directory(directory)
    extra = directory / "payload.zip"
    extra.write_bytes(b"zip")
    with pytest.raises(ArtifactValidationError):
        validate_small_artifact_directory(directory)
    extra.unlink()
    alias = tmp_path / "alias"
    os.link(directory / "attempts.json", alias)
    try:
        with pytest.raises(ArtifactValidationError, match="hardlinked"):
            validate_small_artifact_directory(directory)
    finally:
        alias.unlink()


def test_bf16_artifact_json_and_checksum_parsers_fail_closed(
    tmp_path: Path,
) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text('{"value": NaN}\n', encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="non-finite"):
        strict_json_load(bad_json)
    checksum = tmp_path / "checksums.sha256"
    checksum.write_text(f"{'0' * 64}  ../escape.json\n", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="malformed"):
        validate_checksum_manifest(tmp_path)


def test_bf16_artifact_text_scan_rejects_secret_markers(tmp_path: Path) -> None:
    for name in ARTIFACT_ALLOWLIST:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(
        '{"authorization:": "forbidden"}\n', encoding="utf-8"
    )
    with pytest.raises(ArtifactValidationError, match="secret"):
        scan_artifact_text(tmp_path)


def test_bf16_attempt_peak_hierarchy_is_fail_closed() -> None:
    telemetry = {
        "attempt_peaks": {
            "mps_current_bytes": 10,
            "mps_driver_bytes": 20,
            "process_rss_bytes": 30,
            "swap_used_bytes": 40,
            "swap_growth_bytes": 0,
            "minimum_available_memory_bytes": 8 * 1024**3,
        },
        "stage_peaks": {
            "stage": {
                "mps_current_bytes": 10,
                "mps_driver_bytes": 20,
                "process_rss_bytes": 30,
                "swap_used_bytes": 40,
                "swap_growth_bytes": 0,
                "minimum_available_memory_bytes": 8 * 1024**3,
            }
        },
        "sample_count": 1,
        "violations": [],
        "telemetry_failures": 0,
        "thermal_states": ["nominal"],
    }
    validate_peak_hierarchy(telemetry)
    broken = copy.deepcopy(telemetry)
    broken["stage_peaks"]["stage"]["mps_driver_bytes"] = 21
    with pytest.raises(ArtifactValidationError, match="dominate"):
        validate_peak_hierarchy(broken)


def test_intervention_validator_requires_explicit_maximum_absolute_difference() -> None:
    diagnostics = {
        "device": "mps:0",
        "dtype": "torch.bfloat16",
        "nan_count": 0,
        "positive_infinity_count": 0,
        "negative_infinity_count": 0,
        "nonfinite_count": 0,
    }
    compact = {"diagnostics": diagnostics}
    conditions = []
    for alpha, desired, maximum in ((0.0, 8.0, 0.0), (0.5, 4.0, 2.0), (1.0, 0.0, 4.0)):
        conditions.append(
            {
                "alpha": alpha,
                "baseline_activation": 8.0,
                "desired_absolute_activation": desired,
                "sent_absolute_activation": desired,
                "sent_device": "mps:0",
                "sent_dtype": "torch.bfloat16",
                "normalized_l2_from_baseline": 0.0 if alpha == 0.0 else 0.1,
                "maximum_absolute_logit_difference_from_baseline": maximum,
                "logits": compact,
            }
        )
    record = {
        "status": "passed",
        "intervention": {
            "freeze_attention": True,
            "runtime_monkeypatches": 0,
            "baseline_noop_normalized_l2_tolerance": 0.01,
            "baseline_noop_maximum_absolute_logit_difference_tolerance": 0.0,
            "raw_baseline": compact,
            "baseline": compact,
            "baseline_repeat": compact,
            "raw_to_frozen_baseline_normalized_l2": 0.0,
            "baseline_repeat_normalized_l2": 0.0,
            "raw_to_frozen_baseline_maximum_absolute_logit_difference": 0.0,
            "baseline_repeat_maximum_absolute_logit_difference": 0.0,
            "conditions": conditions,
        },
    }
    _validate_intervention(record)
    broken = copy.deepcopy(record)
    del broken["intervention"]["conditions"][0][
        "maximum_absolute_logit_difference_from_baseline"
    ]
    with pytest.raises(ArtifactValidationError, match="maximum absolute"):
        _validate_intervention(broken)


def test_snapshot_validator_rejects_symlink_escape(tmp_path: Path) -> None:
    from cfsus.reproduction.small_model_mps_bf16 import validate_snapshot_tree

    cache = tmp_path / "cache"
    snapshot = cache / "repo" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_text("secret", encoding="utf-8")
    (snapshot / "config.json").symlink_to(outside)
    with pytest.raises(ArtifactValidationError, match="escapes"):
        validate_snapshot_tree(
            snapshot=snapshot, cache_root=cache, expected_paths={"config.json"}
        )


def test_snapshot_validator_rejects_hardlink_and_empty_file(tmp_path: Path) -> None:
    from cfsus.reproduction.small_model_mps_bf16 import validate_snapshot_tree

    cache = tmp_path / "cache"
    snapshot = cache / "repo" / "snapshots" / ("b" * 40)
    snapshot.mkdir(parents=True)
    source = cache / "source"
    source.write_text("payload", encoding="utf-8")
    os.link(source, snapshot / "config.json")
    with pytest.raises(ArtifactValidationError, match="hardlinked"):
        validate_snapshot_tree(
            snapshot=snapshot, cache_root=cache, expected_paths={"config.json"}
        )
    (snapshot / "config.json").unlink()
    (snapshot / "config.json").write_bytes(b"")
    with pytest.raises(ArtifactValidationError, match="empty"):
        validate_snapshot_tree(
            snapshot=snapshot, cache_root=cache, expected_paths={"config.json"}
        )


def test_supervisor_kills_owned_process_group_on_safety() -> None:
    from cfsus.reproduction.small_model_mps_bf16 import supervise_process_group

    started = time.time()

    def unsafe_sample(_pid: int) -> dict[str, object]:
        return {"sampled_at_unix": time.time(), "violations": ["test_limit"]}

    outcome = supervise_process_group(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        timeout_seconds=5.0,
        sample_interval_seconds=0.01,
        sample_host=unsafe_sample,
        telemetry_failure_limit=1,
        terminate_grace_seconds=0.2,
        kill_grace_seconds=0.2,
        environment=os.environ,
    )
    assert outcome.safety_terminated is True
    assert outcome.timed_out is False
    assert outcome.termination_signal is not None
    assert outcome.finished_at_unix >= started


def test_supervisor_timeout_and_telemetry_failure_are_fail_closed() -> None:
    from cfsus.reproduction.small_model_mps_bf16 import supervise_process_group

    timeout = supervise_process_group(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        timeout_seconds=0.05,
        sample_interval_seconds=0.01,
        sample_host=lambda _pid: {"violations": []},
        telemetry_failure_limit=2,
        terminate_grace_seconds=0.2,
        kill_grace_seconds=0.2,
        environment=os.environ,
    )
    assert timeout.timed_out is True

    def broken_sample(_pid: int) -> dict[str, object]:
        raise RuntimeError("synthetic telemetry failure")

    telemetry = supervise_process_group(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        timeout_seconds=5.0,
        sample_interval_seconds=0.01,
        sample_host=broken_sample,
        telemetry_failure_limit=2,
        terminate_grace_seconds=0.2,
        kill_grace_seconds=0.2,
        environment=os.environ,
    )
    assert telemetry.safety_terminated is True
    assert telemetry.telemetry_failures == 2


def test_bf16_runtime_adapter_does_not_mutate_upstream_classes() -> None:
    source = RUNTIME_ADAPTER.read_text(encoding="utf-8")
    assert "MethodType" not in source
    assert "compute_attribution_components =" not in source
    assert "compute_batch =" not in source
    assert "compute_partial_influences =" not in source
    assert 'runtime_monkeypatches": 0' in source


def test_loaded_semantics_keeps_inference_tensor_math_in_inference_mode() -> None:
    source = WORKER.read_text(encoding="utf-8")
    assert 'sampler.stage("loaded_semantics"), torch.inference_mode()' in source


def test_attribution_retry_requires_a_verified_oom() -> None:
    namespace = runpy.run_path(RUNNER, run_name="bf16_runner_test")
    failure_class = namespace["WorkerStageFailure"]
    is_verified = namespace["_is_verified_attribution_oom"]
    supervisor = {"safety_terminated": False, "timed_out": False}
    verified = failure_class(
        "smoke",
        {
            "error": {"message": "MPS backend out of memory"},
            "telemetry": {
                "stage_peaks": {"attribution": {}},
                "violations": [],
            },
        },
        supervisor,
    )
    assert is_verified(verified)
    generic = failure_class(
        "smoke",
        {
            "error": {"message": "unsupported operator"},
            "telemetry": {
                "stage_peaks": {"attribution": {}},
                "violations": [],
            },
        },
        supervisor,
    )
    assert not is_verified(generic)


@pytest.mark.skipif(
    os.environ.get("CFSUS_RUN_REAL_MPS_BF16_TESTS") != "1",
    reason="real MPS/BF16 tests are explicit opt-in",
)
def test_real_mps_bf16_overflow_and_sparse_boundary() -> None:
    torch = pytest.importorskip("torch")
    from cfsus.reproduction.small_model_mps_bf16 import (
        run_overflow_regression,
        validate_live_sparse_boundary,
    )

    assert torch.backends.mps.is_available()
    config = load_bf16_config(CONFIG)
    assert run_overflow_regression(torch, config["tolerances"])["passed"]
    assert validate_live_sparse_boundary(torch)["passed"]
