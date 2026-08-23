"""Offline policy and producer tests for the separate Stage 1A MPS runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts/stage1a"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mps_runtime as mps  # noqa: E402
import run_stage1a_mps_fp16 as runner  # noqa: E402
import run_stage1a_mps_fp16_worker as worker  # noqa: E402
import validate_mps_fp16_artifacts as validator  # noqa: E402

from cfsus.reproduction.config import (  # noqa: E402
    OFFICIAL_MODEL_ID,
    OFFICIAL_MODEL_REVISION,
    OFFICIAL_TRANSCODER_ID,
    OFFICIAL_TRANSCODER_REVISION,
    OFFICIAL_UPSTREAM_REVISION,
)

COMMIT = "1" * 40
TELEMETRY_METHOD = "sampled torch.mps counters plus RSS pressure swap"


def test_preserved_t4_verification_requires_exact_regular_file_hashes(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "stage1a_t4_fp16"
    directory.mkdir()
    artifact = directory / "artifact.json"
    artifact.write_bytes(b"preserved evidence\n")
    expected = {"artifact.json": runner._sha256_file(artifact)}
    runner._validate_preserved_t4(directory, expected)

    artifact.write_bytes(b"changed evidence\n")
    with pytest.raises(mps.MPSRuntimeError, match="content changed"):
        runner._validate_preserved_t4(directory, expected)


def test_protected_git_state_requires_branch_refs_and_base_ancestry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values: dict[tuple[str, ...], str] = {
        ("symbolic-ref", "--quiet", "HEAD"): (f"refs/heads/{runner.EXPECTED_BRANCH}"),
        ("merge-base", runner.PROJECT_BASE_COMMIT, "HEAD"): runner.PROJECT_BASE_COMMIT,
        ("ls-files", "--", runner.PRESERVED_T4_INDEX_PATHSPEC): "",
        ("ls-files", "-v", "-z"): "H README.md\0",
        ("ls-files", "-f", "-z"): "H README.md\0",
        ("ls-files", "-z"): "README.md\0",
        ("for-each-ref", "--format=%(refname)", "refs/replace"): "",
        (
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            "info/grafts",
        ): str((ROOT / ".git/info/grafts").resolve()),
    }
    values.update(
        {
            ("show-ref", "--verify", "--hash", reference): expected
            for reference, expected in runner.PROTECTED_REFS.items()
        }
    )
    monkeypatch.setattr(runner, "_git_output", lambda *args: values[args])
    runner._validate_protected_git_state()

    main_reference = "refs/heads/main"
    values[("show-ref", "--verify", "--hash", main_reference)] = "0" * 40
    with pytest.raises(mps.MPSRuntimeError, match="protected Git ref changed"):
        runner._validate_protected_git_state()

    values[("show-ref", "--verify", "--hash", main_reference)] = runner.PROTECTED_REFS[
        main_reference
    ]
    values[("ls-files", "--", runner.PRESERVED_T4_INDEX_PATHSPEC)] = (
        "Results/Stage1A_T4_FP16/run_manifest.json"
    )
    with pytest.raises(mps.MPSRuntimeError, match="unexpectedly tracked"):
        runner._validate_protected_git_state()

    values[("ls-files", "--", runner.PRESERVED_T4_INDEX_PATHSPEC)] = ""
    values[("ls-files", "-v", "-z")] = "h scripts/stage1a/runner.py\0"
    with pytest.raises(mps.MPSRuntimeError, match="assume-unchanged"):
        runner._validate_protected_git_state()

    values[("ls-files", "-v", "-z")] = "S scripts/stage1a/runner.py\0"
    with pytest.raises(mps.MPSRuntimeError, match="assume-unchanged"):
        runner._validate_protected_git_state()

    values[("ls-files", "-v", "-z")] = "H README.md\0"
    values[("for-each-ref", "--format=%(refname)", "refs/replace")] = (
        "refs/replace/1111111111111111111111111111111111111111"
    )
    with pytest.raises(mps.MPSRuntimeError, match="replacement refs"):
        runner._validate_protected_git_state()


def test_git_proof_uses_unambiguous_refs_and_case_insensitive_t4_pathspec() -> None:
    assert all(reference.startswith("refs/") for reference in runner.PROTECTED_REFS)
    assert runner.PRESERVED_T4_INDEX_PATHSPEC == (
        ":(top,icase,literal)results/stage1a_t4_fp16"
    )


def test_git_graft_proof_rejects_nonempty_or_unsafe_grafts(tmp_path: Path) -> None:
    grafts = tmp_path / "linked-git-dir/info/grafts"
    grafts.parent.mkdir(parents=True)
    runner._validate_no_legacy_grafts(grafts)
    grafts.write_text("legacy graft\n", encoding="utf-8")
    with pytest.raises(mps.MPSRuntimeError, match="grafts"):
        runner._validate_no_legacy_grafts(grafts)


def test_t4_index_proof_rejects_physical_unicode_aliases(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    preserved = repository / "results/stage1a_t4_fp16"
    preserved.mkdir(parents=True)
    original = preserved / "artifact.json"
    original.write_text("preserved\n", encoding="utf-8")
    alias = repository / "re\u017fults/\u017ftage1a_t4_fp16/artifact.json"
    alias.parent.mkdir(parents=True, exist_ok=True)
    if not alias.exists():
        alias.hardlink_to(original)
    with pytest.raises(mps.MPSRuntimeError, match="unexpectedly tracked"):
        runner._validate_preserved_t4_index_aliases(
            f"{alias.relative_to(repository).as_posix()}\0",
            repository_root=repository,
            directory=preserved,
        )
    unicode_directory_alias = alias.parent
    if not unicode_directory_alias.samefile(preserved):
        unicode_directory_alias = repository / "tracked-gitlink"
        unicode_directory_alias.symlink_to(preserved, target_is_directory=True)
    with pytest.raises(mps.MPSRuntimeError, match="unexpectedly tracked"):
        runner._validate_preserved_t4_index_aliases(
            f"{unicode_directory_alias.relative_to(repository).as_posix()}\0",
            repository_root=repository,
            directory=preserved,
        )
    unicode_ancestor_alias = repository / "re\u017fults"
    canonical_ancestor = repository / "results"
    if not unicode_ancestor_alias.samefile(canonical_ancestor):
        unicode_ancestor_alias = repository / "tracked-parent-gitlink"
        unicode_ancestor_alias.symlink_to(canonical_ancestor, target_is_directory=True)
    with pytest.raises(mps.MPSRuntimeError, match="unexpectedly tracked"):
        runner._validate_preserved_t4_index_aliases(
            f"{unicode_ancestor_alias.relative_to(repository).as_posix()}\0",
            repository_root=repository,
            directory=preserved,
        )


def test_external_xet_cache_rejects_a_symlink_into_the_project(tmp_path: Path) -> None:
    cache = tmp_path / "external-cache"
    cache.mkdir()
    (cache / "xet").symlink_to(ROOT, target_is_directory=True)
    with pytest.raises(mps.MPSRuntimeError, match="cache subdirectory is unsafe"):
        runner._validate_external_cache_subdirectory(cache, "xet")


def test_external_cache_tree_rejects_nested_redirects(tmp_path: Path) -> None:
    cache = tmp_path / "external-cache"
    blob = cache / "hub/blobs/blob"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"verified blob")
    snapshot = cache / "hub/snapshots/revision"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").symlink_to(blob)
    (cache / "xet").mkdir()
    runner._validate_external_cache_tree(cache)

    (cache / "hub/.locks").symlink_to(ROOT, target_is_directory=True)
    with pytest.raises(mps.MPSRuntimeError, match="cache symlink"):
        runner._validate_external_cache_tree(cache)


def test_external_cache_tree_fails_closed_on_unreadable_directories(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "external-cache"
    unreadable = cache / "xet/hidden"
    unreadable.mkdir(parents=True)
    unreadable.chmod(0o333)
    try:
        with pytest.raises(mps.MPSRuntimeError, match="could not be traversed"):
            runner._validate_external_cache_tree(cache)
    finally:
        unreadable.chmod(0o700)


def test_external_cache_rejects_an_apfs_unicode_alias_into_the_project() -> None:
    canonical = ROOT / "results/generated"
    alias = Path(str(canonical).replace("/Users/", "/U\u017fers/", 1))
    if not alias.exists() or not alias.samefile(canonical):
        alias = canonical
    with pytest.raises(mps.MPSRuntimeError, match="physically overlaps"):
        runner._validate_physical_project_separation(alias)


def test_publication_state_rejects_a_mid_run_head_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "_validate_source_checkout",
        lambda: ("0" * 40, False),
    )
    with pytest.raises(mps.MPSRuntimeError, match="HEAD changed during MPS execution"):
        runner._revalidate_publication_state(COMMIT)


def test_mps_oom_classifier_is_narrow() -> None:
    assert mps.is_mps_out_of_memory(RuntimeError("MPS out of memory while allocating"))
    assert not mps.is_mps_out_of_memory(RuntimeError("out of memory"))
    assert not mps.is_mps_out_of_memory(RuntimeError("CUDA out of memory"))
    assert not mps.is_mps_out_of_memory(MemoryError("CPU allocation failed"))


def test_sanitized_mps_error_preserves_safe_diagnostics() -> None:
    diagnostic = "Expected scalar_type Float for MPS index_put"

    assert mps.sanitize_error(RuntimeError(diagnostic)) == diagnostic


@pytest.mark.parametrize(
    "credential",
    (
        "password=hunter2",
        "token=opaque123",
        "authorization: Basic opaque123",
        "api_key=short-secret",
        "Bearer tiny-secret",
    ),
)
def test_sanitized_mps_error_redacts_generic_credentials(credential: str) -> None:
    message = mps.sanitize_error(RuntimeError(f"MPS failure {credential}"))

    assert message == "[REDACTED]"
    assert credential not in message


def test_sanitized_mps_error_redacts_tokens_and_private_paths() -> None:
    token = "hf_" + "z" * 24
    private_path = "/" + "Users/alice/cache"
    error = RuntimeError(f"MPS failure {token} {private_path}")

    message = mps.sanitize_error(error)

    assert message == "[REDACTED]"
    assert token not in message
    assert "alice" not in message


def test_worker_failure_report_keeps_the_sanitized_leaf_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generated = tmp_path / "generated"
    attempt = generated / "attempt-256-test"
    report_path = attempt / "attempt_report.json"
    diagnostic = "Expected scalar_type Float for MPS index_put"
    leaf = RuntimeError(diagnostic)
    wrapper = ValueError("runtime wrapper")
    wrapper.__cause__ = leaf

    monkeypatch.setattr(worker, "_generated_root", lambda: generated.resolve())
    monkeypatch.setattr(worker, "_load_config", lambda _path: {})
    monkeypatch.setattr(worker, "fallback_enabled", lambda: False)
    monkeypatch.setattr(
        "run_stage1a_mps_fp16_worker.importlib.import_module",
        lambda _name: SimpleNamespace(mps=None),
    )

    def fail_loading(_config: dict[str, Any], _args: Any) -> Any:
        raise wrapper

    def run_stage(
        name: str,
        *,
        torch_getter: Any,
        function: Any,
        stage_records: dict[str, dict[str, Any]],
    ) -> Any:
        del torch_getter
        stage_records[name] = {
            "sample_count": 2,
            "peak_mps_current_allocated_bytes": 0,
            "peak_mps_driver_allocated_bytes": 1,
            "peak_mps_recommended_max_bytes": 2,
            "peak_process_rss_bytes": 3,
            "peak_swap_used_bytes": 0,
        }
        return function()

    monkeypatch.setattr(worker, "_load_bundle", fail_loading)
    monkeypatch.setattr(worker, "_run_stage", run_stage)

    code = worker.main(
        [
            "--config",
            str(tmp_path / "config.yaml"),
            "--batch-size",
            "256",
            "--attempt-directory",
            str(attempt),
            "--attempt-report",
            str(report_path),
        ]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert code == 1
    assert report["exception_type"] == "RuntimeError"
    assert report["message"] == diagnostic
    assert report["diagnostic_redacted"] is True


def test_only_attribution_mps_oom_retries_in_declared_sequence() -> None:
    assert mps.should_retry_mps_attempt(
        batch_size=256,
        category="mps_out_of_memory",
        failure_stage="attribution",
    )
    assert mps.should_retry_mps_attempt(
        batch_size=128,
        category="mps_out_of_memory",
        failure_stage="attribution",
    )
    assert not mps.should_retry_mps_attempt(
        batch_size=64,
        category="mps_out_of_memory",
        failure_stage="attribution",
    )
    assert not mps.should_retry_mps_attempt(
        batch_size=256, category="failed_runtime", failure_stage="attribution"
    )
    assert not mps.should_retry_mps_attempt(
        batch_size=256,
        category="mps_out_of_memory",
        failure_stage="runtime_loading",
    )
    with pytest.raises(ValueError):
        mps.should_retry_mps_attempt(
            batch_size=32,
            category="mps_out_of_memory",
            failure_stage="attribution",
        )


def test_attempt_peak_dominates_stage_peaks() -> None:
    assert mps.peak_memory_bytes([10, 20, 15]) == 20
    assert mps.attempt_peak_at_least_stages(20, [10, 20, 15])
    assert not mps.attempt_peak_at_least_stages(19, [10, 20, 15])


def test_fallback_and_token_helpers_expose_booleans_only() -> None:
    environment = {
        "PYTORCH_ENABLE_MPS_FALLBACK": "true",
        "HF_TOKEN": "not-serialized",
    }
    assert mps.fallback_enabled(environment)
    assert mps.secure_hf_token_present(environment) is True
    assert isinstance(mps.secure_hf_token_present(environment), bool)


def test_snapshot_manifest_rejects_escape_and_hash_mismatch(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    payload = snapshot / "config.json"
    payload.write_text("{}", encoding="utf-8")
    manifest = [mps.manifest_file(payload, root=snapshot)]
    mps.validate_snapshot_containment(snapshot, manifest)
    manifest[0]["sha256"] = "0" * 64
    with pytest.raises(mps.MPSRuntimeError):
        mps.validate_snapshot_containment(snapshot, manifest)


def test_memory_gate_records_required_sparse_metadata_deviation() -> None:
    candidate = mps.evaluate_memory_feasibility(
        {
            "model_bytes": runner.MODEL_METADATA_BYTES,
            "transcoder_bytes": runner.TRANSCODER_METADATA_BYTES,
        },
        physical_memory_bytes=32 * 1024**3,
        pressure="normal",
        swap_used_bytes=0,
    )
    assert candidate["status"] == "feasible_with_explicit_execution_deviation"
    assert candidate["resource_status"] == "feasible"
    assert candidate["estimated_peak_bytes"] == sum(
        candidate["estimate_components"].values()
    )
    assert candidate["scientific_parameters_changed"] is False
    assert "CPU sparse COO metadata" in candidate["execution_deviations"][0]
    blocked = mps.evaluate_memory_feasibility(
        {"model_bytes": 1, "transcoder_bytes": 1},
        physical_memory_bytes=32 * 1024**3,
        pressure="critical",
    )
    assert blocked["status"] == "blocked"
    unknown = mps.evaluate_memory_feasibility(
        {"model_bytes": 1, "transcoder_bytes": 1},
        physical_memory_bytes=32 * 1024**3,
        pressure=None,
        swap_used_bytes=None,
    )
    assert unknown["status"] == "blocked"
    assert unknown["downloads_authorized"] is False


def test_observed_identical_loading_plan_fails_closed() -> None:
    candidate = mps.evaluate_memory_feasibility(
        {
            "model_bytes": runner.MODEL_METADATA_BYTES,
            "transcoder_bytes": runner.TRANSCODER_METADATA_BYTES,
        },
        physical_memory_bytes=32 * 1024**3,
        pressure="normal",
        swap_used_bytes=512 * 1024**2,
        observed_loading=runner.OBSERVED_RUNTIME_LOADING_STOP,
    )

    assert candidate["status"] == "blocked"
    assert candidate["downloads_authorized"] is False
    assert candidate["reason"] == (
        "observed identical runtime-loading plan exceeds the conservative budget"
    )
    assert candidate["effective_peak_bytes"] == 40_032_174_080
    assert candidate["empirical_loading_observation"] == (
        runner.OBSERVED_RUNTIME_LOADING_STOP
    )


def test_malformed_empirical_loading_observation_fails_closed() -> None:
    candidate = mps.evaluate_memory_feasibility(
        {"model_bytes": 1, "transcoder_bytes": 1},
        physical_memory_bytes=32 * 1024**3,
        pressure="normal",
        swap_used_bytes=0,
        observed_loading={"loading_plan_id": "incomplete"},
    )

    assert candidate["status"] == "blocked"
    assert candidate["downloads_authorized"] is False
    assert candidate["empirical_loading_observation"] is None


def test_main_stops_before_cache_or_worker_for_observed_loading_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generated = tmp_path / "generated"
    monkeypatch.setattr(runner, "GENERATED_DIRECTORY", generated)
    monkeypatch.setattr(runner, "_load_config", lambda _path: {})
    monkeypatch.setattr(runner, "_validate_source_checkout", lambda: (COMMIT, False))
    monkeypatch.setattr(runner, "_preflight", lambda _path: {"status": "passed"})
    monkeypatch.setattr(
        runner,
        "_memory_gate",
        lambda: {
            "status": "blocked",
            "downloads_authorized": False,
            "reason": (
                "observed identical runtime-loading plan exceeds the conservative "
                "budget"
            ),
        },
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("blocked main reached cache or worker handling")

    monkeypatch.setattr(runner, "_validate_external_cache_subdirectory", forbidden)
    monkeypatch.setattr(runner, "_validate_external_cache_tree", forbidden)
    monkeypatch.setattr(runner, "_run_attempt", forbidden)

    code = runner.main(["--config", str(tmp_path / "config.yaml")])

    assert code == 2
    reports = list(generated.glob("preflight-*/feasibility_report.json"))
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert payload["downloads_authorized"] is False


def test_memory_pressure_classifies_observed_free_percentage() -> None:
    def run(command: tuple[str, ...], **_kwargs: Any) -> SimpleNamespace:
        if command == ("memory_pressure", "-Q"):
            return SimpleNamespace(
                returncode=0,
                stdout="System-wide memory free percentage: 66%\n",
            )
        return SimpleNamespace(returncode=1, stdout="")

    assert mps._memory_pressure(run) == "normal"


def test_attempt_cadence_does_not_double_count_stage_boundaries() -> None:
    records: dict[str, dict[str, Any]] = {}
    start = 1.0
    for index, name in enumerate(sorted(validator.EXPECTED_ATTEMPT_STAGES)):
        duration = 1.2 if index == 0 else 0.2
        sample_count = 3 if index == 0 else 2
        records[name] = {
            "started_at_unix": start,
            "finished_at_unix": start + duration,
            "wall_seconds": duration,
            "sample_count": sample_count,
            "sampling_interval_seconds": worker.SAMPLING_INTERVAL_SECONDS,
            "target_sampling_interval_seconds": worker.SAMPLING_INTERVAL_SECONDS,
            "sampling_method": "periodic_boundary_and_interval_samples",
            "peak_mps_current_allocated_bytes": 10,
            "peak_mps_driver_allocated_bytes": 20,
            "peak_mps_recommended_max_bytes": 100,
            "peak_process_rss_bytes": 50,
            "peak_swap_used_bytes": 0,
            "system_memory_pressures": ["normal"],
            "samples": [],
        }
        start += duration
    evidence = worker._memory_evidence(records)
    assert evidence["timing"]["sample_count"] == 3
    assert evidence["sampling_interval_seconds"] == worker.SAMPLING_INTERVAL_SECONDS


def test_preflight_unlinks_stale_output_and_scrubs_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "preflight_summary.json"
    destination.write_text('{"status":"stale"}', encoding="utf-8")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0")

    def fake_run(
        _command: tuple[str, ...],
        *,
        cwd: Path,
        env: dict[str, str],
        check: bool,
    ) -> SimpleNamespace:
        assert cwd == ROOT
        assert check is False
        assert not destination.exists()
        assert "PYTORCH_ENABLE_MPS_FALLBACK" not in env
        assert "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in env
        destination.write_text(
            json.dumps({"status": "passed", "probe_status": "passed"}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    report = runner._preflight(destination, runner=fake_run)
    assert report["status"] == "passed"


def _timing(
    start: float,
    finish: float,
    *,
    current: int,
    driver: int,
    rss: int,
    swap: int,
    sample_count: int = 4,
) -> dict[str, Any]:
    return {
        "started_at_unix": start,
        "finished_at_unix": finish,
        "wall_seconds": finish - start,
        "sampling_method": TELEMETRY_METHOD,
        "sampling_interval_seconds": 0.1,
        "sample_count": sample_count,
        "mps_current_allocated_peak_bytes": current,
        "mps_driver_allocated_peak_bytes": driver,
        "mps_recommended_max_bytes": 1_000,
        "process_rss_peak_bytes": rss,
        "memory_pressure_states": ["normal"],
        "swap_used_peak_bytes": swap,
    }


def _peaks(timing: dict[str, Any]) -> dict[str, int]:
    return {key: int(timing[key]) for key in validator.PEAK_KEYS}


def _asset(
    identifier: str,
    revision: str,
    pins: dict[str, tuple[int, str]],
) -> dict[str, Any]:
    files = [
        {
            "path": name,
            "size_bytes": size,
            "sha256": digest,
        }
        for name, (size, digest) in sorted(pins.items())
    ]
    return {
        "identifier": identifier,
        "revision": revision,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(size for size, _digest in pins.values()),
        "complete": True,
        "snapshot_containment_verified": True,
        "offline_ready": True,
    }


def _raw_preflight() -> dict[str, Any]:
    primitive = {
        name: {
            "attempted": True,
            "passed": True,
            "device": "mps",
            "dtype": "float16",
            "cpu_reference_passed": True,
        }
        for name in validator.EXPECTED_OPERATOR_NAMES
    }
    operations = {
        name: {
            "attempted": True,
            "passed": True,
            "device": "mps",
            "dtype": "float16",
            "cpu_reference_passed": True,
        }
        for name in validator.EXPECTED_PREFLIGHT_OPERATIONS
    }
    operations["operators"]["operations"] = primitive
    operations["strict_jumprelu"].update(strict_gate_equal=True, equality_inactive=True)
    operations["sparse_metadata_boundary"].update(
        cpu_metadata_explicit=True,
        replacement_boundary_passed=True,
        dense_scientific_device="mps",
    )
    operations["disk_offload_safetensors"].update(
        upstream_disk_offload_helper_tested=True
    )
    return {
        "status": "passed",
        "probe_status": "passed",
        "environment": {
            "python": "3.11.13",
            "system": "Darwin",
            "architecture": "arm64",
            "torch_version": "2.6.0",
            "mps_built": True,
            "mps_available": True,
            "fallback_enabled": False,
            "fallback_used": False,
            "fallback_env_value_present": False,
            "high_watermark_override": None,
        },
        "checks": {name: True for name in validator.EXPECTED_PREFLIGHT_CHECKS},
        "operations": operations,
        "tolerances": {
            "absolute": 0.005,
            "relative": 0.002,
            "finite_required": True,
        },
        "large_assets_downloaded": False,
        "scientific_model_result": False,
    }


def _raw_attempt(attempt_directory: Path) -> dict[str, Any]:
    semantics_timing = _timing(1.0, 2.0, current=20, driver=40, rss=100, swap=5)
    intervention_timing = _timing(2.0, 3.0, current=30, driver=50, rss=110, swap=6)
    attribution_timing = _timing(3.0, 4.0, current=40, driver=80, rss=120, swap=7)
    memory_timing = _timing(
        1.0,
        4.0,
        current=40,
        driver=80,
        rss=120,
        swap=7,
        sample_count=12,
    )
    stage_peaks = {
        "runtime_loading": _peaks(semantics_timing),
        "model_only_forward": _peaks(semantics_timing),
        "semantics": _peaks(semantics_timing),
        "intervention": _peaks(intervention_timing),
        "attribution": _peaks(attribution_timing),
        "cleanup": _peaks(attribution_timing),
    }
    attempt_peaks = _peaks(memory_timing)
    payloads: dict[str, dict[str, Any]] = {
        "model_smoke.json": {
            "progressive": {"passed": True},
            "post_load_passed": True,
        },
        "environment.json": {
            "platform": {
                "system": "Darwin",
                "machine": "arm64",
                "hardware_family": "Apple M2 Max",
                "physical_memory_bytes": 32 * 1024**3,
            },
            "python": {"version": "3.11.13", "architecture": "arm64"},
            "packages": {
                "torch": "2.6.0",
                "transformer_lens": "3.2.1",
                "circuit_tracer": "0.5.2",
                "circuit_tracer_revision": OFFICIAL_UPSTREAM_REVISION,
                "circuit_tracer_vcs_url": worker.CIRCUIT_TRACER_REPOSITORY,
                "circuit_tracer_record_hashes_verified": 42,
            },
            "runtime": {
                "backend": "transformerlens",
                "accelerator_backend": "mps",
                "device": "mps",
                "dtype": "float16",
                "execution_class": validator.EXECUTION_CLASS,
                "offload": "disk",
            },
            "mps": {
                "built": True,
                "available": True,
                "allocation_probe": {
                    "success": True,
                    "device": "mps",
                    "dtype": "float16",
                    "finite": True,
                },
            },
            "fallback_enabled": False,
            "fallback_used": False,
            "fallback_env_value_present": False,
            "high_watermark_override": None,
            "memory_guardrails_preserved": True,
            "pip_check": "passed",
            "lock_match": "exact",
            "lock_path": "environments/stage1a_mps/requirements-lock.txt",
            "lock_sha256": validator.EXPECTED_LOCK_SHA256,
        },
        "asset.json": {
            "verification": "exact_file_content_hashes_matched",
            "immutable_revisions_only": True,
            "project_external_cache": True,
            "unmanifested_file_count": 0,
            "assets": {
                "model": _asset(
                    OFFICIAL_MODEL_ID,
                    OFFICIAL_MODEL_REVISION,
                    validator.MODEL_FILE_PINS,
                ),
                "transcoder": _asset(
                    OFFICIAL_TRANSCODER_ID,
                    OFFICIAL_TRANSCODER_REVISION,
                    validator.TRANSCODER_FILE_PINS,
                ),
            },
        },
        "semantics.json": {
            "loaded_runtime": {
                "passed": True,
                "model_loaded": True,
                "transcoder_loaded": True,
                "model_device": "mps",
                "transcoder_device": "mps",
                "model_dtype": "float16",
                "transcoder_dtype": "float16",
                "model_only_forward_passed": True,
                "model_only_forward": {
                    "passed": True,
                    "prompt": "The capital of state containing Dallas is",
                    "token_count": 8,
                    "logits_shape": [1, 8, 256_000],
                    "tokenizer_revision": OFFICIAL_MODEL_REVISION,
                    "device": "mps",
                    "dtype": "float16",
                    "finite": True,
                    "completed_before_transcoder_load": True,
                },
                "output_finite": True,
                "fallback_used": False,
            },
            "preactivation": {
                "verified": True,
                "threshold_retrieved": True,
                "definition": "F.linear(feature_input, W_enc, b_enc)",
                "bias_convention": "b_enc is included; b_dec is excluded",
                "cache_shape": [26, 8, 16_384],
                "projection_absolute_tolerance": 0.005,
            },
            "gate_check": {
                "rule": "z if z > threshold else 0",
                "strict_greater_than": True,
                "equality_inactive": True,
                "equality_probe_maximum_absolute_output": 0.0,
                "absolute_tolerance": 0.005,
                "active_example": {
                    "layer": 20,
                    "position": -1,
                    "feature_id": 1,
                    "preactivation": 1.1,
                    "threshold": 1.0,
                    "post_gate_activation": 1.1,
                    "active": True,
                    "signed_margin": 0.1,
                },
                "inactive_example": {
                    "layer": 20,
                    "position": -1,
                    "feature_id": 2,
                    "preactivation": 1.0,
                    "threshold": 1.0,
                    "post_gate_activation": 0.0,
                    "active": False,
                    "signed_margin": 0.0,
                },
                "official_intervention_source": {
                    "layer": 20,
                    "position": -1,
                    "feature_id": 341,
                    "preactivation": 10.0,
                    "threshold": 1.0,
                    "post_gate_activation": 10.0,
                    "active": True,
                    "signed_margin": 9.0,
                },
            },
            "intervention_value_check": {
                "passed": True,
                "formula": "(1-alpha)*baseline_activation",
                "alphas": [0.0, 0.5, 1.0],
            },
            "feature": {
                "layer": 20,
                "position": -1,
                "feature_id": 341,
                "baseline_activation": 10.0,
            },
            "baseline_repeat_error": 0.001,
            "projection_discrepancy": 0.001,
            "gate_discrepancy": 0.0,
            "nonfinite_count": 0,
            "timing": semantics_timing,
        },
        "intervention.json": {
            "parameters": {
                "prompt": "Hecho: Michael Jordan juega al",
                "feature": {"layer": 20, "position": -1, "feature_id": 341},
                "alphas": [0.0, 0.5, 1.0],
                "freeze_attention": True,
                "constrained_layers": None,
            },
            "baseline_activation_captured": True,
            "baseline_activation": 10.0,
            "baseline_repeat_error": 0.001,
            "baseline_repeat_max_combined_tolerance_ratio": 0.05,
            "baseline_noop_comparison": {
                "within_tolerance": True,
                "max_abs_error": 0.001,
                "max_rel_error": 0.0001,
                "max_combined_tolerance_ratio": 0.05,
                "absolute_tolerance": 0.02,
                "relative_tolerance": 0.002,
            },
            "desired_values": [
                {
                    "alpha": alpha,
                    "expected_activation": expected,
                    "observed_activation": expected,
                    "absolute_error": 0.0,
                    "within_tolerance": True,
                    "output_finite": True,
                }
                for alpha, expected in ((0.0, 10.0), (0.5, 5.0), (1.0, 0.0))
            ],
            "outputs_finite": True,
            "same_assets_and_runtime": True,
            "nonfinite_count": 0,
            "timing": intervention_timing,
        },
        "attribution.json": {
            "parameters": {
                "prompt": "The capital of state containing Dallas is",
                "max_n_logits": 10,
                "desired_logit_probability": 0.95,
                "max_feature_nodes": 8192,
                "offload": "disk",
            },
            "accepted_batch_size": 256,
            "graph": {
                "finite": True,
                "node_count": 18,
                "selected_feature_count": 3,
                "active_feature_count": 3,
                "edge_count": 5,
                "logit_node_count": 10,
                "input_node_count": 4,
                "error_node_count": 1,
                "adjacency_shape": [18, 18],
            },
            "raw_validation": {"passed": True, "raw_graph_committed": False},
            "nonfinite_count": 0,
            "timing": attribution_timing,
        },
        "memory.json": {
            "nonfinite_count": 0,
            "timing": memory_timing,
            "telemetry_method": TELEMETRY_METHOD,
            "sampling_interval_seconds": memory_timing["sampling_interval_seconds"],
            "stages": {},
            "attempt_peaks": attempt_peaks,
        },
    }
    attempt_directory.mkdir(parents=True)
    for filename, payload in payloads.items():
        (attempt_directory / filename).write_text(
            json.dumps(payload, sort_keys=True), encoding="utf-8"
        )
    return {
        "schema_version": 1,
        "attempt_id": attempt_directory.name,
        "batch_size": 256,
        "outcome": "completed",
        "category": "completed",
        "exception_type": None,
        "message": "MPS attempt completed with all checks passing.",
        "failure_stage": None,
        "wall_seconds": 3.0,
        "cleanup_succeeded": True,
        "fresh_process": True,
        "oom_confirmed": False,
        "oom_classifier_match": False,
        "retry_eligible": False,
        "diagnostic_redacted": True,
        "sample_count": 12,
        "stage_peaks": stage_peaks,
        "attempt_peaks": attempt_peaks,
        "process_exit_code": 0,
    }


def test_fake_producer_promotes_only_a_strictly_validated_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_preflight = runner.PREFLIGHT_OUTPUT
    before = real_preflight.read_bytes() if real_preflight.is_file() else None
    result_directory = tmp_path / "results/stage1a_mps_fp16"
    generated_directory = tmp_path / "results/generated/stage1a_mps_fp16"
    monkeypatch.setattr(runner, "RESULT_DIRECTORY", result_directory)
    monkeypatch.setattr(runner, "GENERATED_DIRECTORY", generated_directory)
    monkeypatch.setattr(
        runner,
        "PREFLIGHT_OUTPUT",
        result_directory / "preflight/preflight_summary.json",
    )
    authentication_home = tmp_path / "authentication-home"
    monkeypatch.setenv("HF_HOME", str(authentication_home))
    monkeypatch.setenv("HF_XET_CACHE", str(ROOT / "unsafe-inherited-xet-cache"))
    monkeypatch.setenv("HF_XET_LOG_DEST", str(ROOT / "results/generated/xet.log"))
    monkeypatch.setenv("HF_XET_DATA_STAGING_SUBDIR", "unsafe-inherited-staging")
    source_validation_calls = 0

    def fake_validate_source_checkout() -> tuple[str, bool]:
        nonlocal source_validation_calls
        source_validation_calls += 1
        return COMMIT, False

    monkeypatch.setattr(
        runner,
        "_validate_source_checkout",
        fake_validate_source_checkout,
    )
    monkeypatch.setattr(runner, "_load_config", lambda _path: {})
    monkeypatch.setattr(runner, "_preflight", lambda _output=None: _raw_preflight())
    monkeypatch.setattr(
        runner,
        "_memory_gate",
        lambda: {
            "status": "feasible_with_explicit_execution_deviation",
            "downloads_authorized": True,
            "physical_memory_bytes": 32 * 1024**3,
            "pressure": "normal",
            "swap_used_bytes": 512 * 1024**2,
            "snapshot_sizes": {
                "model_bytes": validator.MODEL_REQUIRED_BYTES,
                "transcoder_bytes": validator.TRANSCODER_REQUIRED_BYTES,
            },
            "estimate_components": {
                "model_resident_estimate_bytes": int(
                    validator.MODEL_REQUIRED_BYTES * 0.60
                ),
                "transcoder_resident_estimate_bytes": (
                    validator.TRANSCODER_REQUIRED_BYTES
                ),
                "temporary_and_system_headroom_bytes": 6 * 1024**3,
            },
            "conservative_budget_bytes": 26 * 1024**3,
            "estimated_peak_bytes": (
                int(validator.MODEL_REQUIRED_BYTES * 0.60)
                + validator.TRANSCODER_REQUIRED_BYTES
                + 6 * 1024**3
            ),
        },
    )

    def fake_attempt(
        _args: Any, batch_size: int, _environment: dict[str, str]
    ) -> tuple[dict[str, Any], Path]:
        assert batch_size == 256
        assert _environment["HF_HOME"] == str(authentication_home)
        assert _environment["HF_HUB_CACHE"] == str(
            (tmp_path / "external-cache" / "hub").resolve()
        )
        assert _environment["HF_XET_CACHE"] == str(
            (tmp_path / "external-cache" / "xet").resolve()
        )
        assert "HF_XET_LOG_DEST" not in _environment
        assert "HF_XET_DATA_STAGING_SUBDIR" not in _environment
        directory = generated_directory / "attempt-256-fake"
        return _raw_attempt(directory), directory

    monkeypatch.setattr(runner, "_run_attempt", fake_attempt)
    code = runner.main(
        [
            "--config",
            str(ROOT / "configs/stage1a_gemma2_2b_mps_fp16_reproduction.yaml"),
            "--hf-cache",
            str(tmp_path / "external-cache"),
        ]
    )
    assert code == 0
    validator.validate_mps_artifact_directory(result_directory, require_complete=True)
    manifest = json.loads(
        (result_directory / runner.RUN_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["status"] == validator.COMPLETED_STATUS
    assert manifest["reproduction_class"] == validator.REPRODUCTION_CLASS
    assert source_validation_calls == 4
    after = real_preflight.read_bytes() if real_preflight.is_file() else None
    assert after == before
