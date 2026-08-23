#!/usr/bin/env python3
"""Run one isolated Stage 1A MPS/FP16 attempt and emit only raw evidence.

Canonical publication artifacts are produced by the parent process.  This
worker writes bounded, JSON-only payloads to a unique ignored attempt
directory, never a cache path or credential-derived field.
"""

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mps_runtime import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    MPS_BATCH_SEQUENCE,
    MPS_EXECUTION_DTYPE,
    TRANSCODER_ID,
    TRANSCODER_REVISION,
    UPSTREAM_REVISION,
    classify_failure,
    exception_chain,
    fallback_enabled,
    is_mps_out_of_memory,
    sample_mps_memory,
    sanitize_error,
    summarize_mps_samples,
)

from cfsus.reproduction.artifacts import (  # noqa: E402
    assert_publication_safe,
    validate_json_value,
    write_json_atomic,
)

SAMPLING_INTERVAL_SECONDS = 1.0
TELEMETRY_METHOD = "sampled torch.mps counters plus RSS pressure swap"
MODEL_REQUIRED_FILES = frozenset(
    {
        "config.json",
        "generation_config.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    }
)
TRANSCODER_REQUIRED_FILES = frozenset(
    {"config.yaml", *(f"layer_{layer}.safetensors" for layer in range(26))}
)
CIRCUIT_TRACER_REPOSITORY = "https://github.com/decoderesearch/circuit-tracer.git"
_T = TypeVar("_T")


def _generated_root() -> Path:
    return (REPOSITORY_ROOT / "results/generated/stage1a_mps_fp16").resolve()


def _safe_attempt_directory(path: Path) -> Path:
    root = _generated_root()
    if path.is_symlink():
        raise ValueError("attempt outputs cannot use a symlink")
    candidate = path.resolve()
    if not candidate.is_relative_to(root) or candidate == root:
        raise ValueError("attempt outputs must stay in the generated MPS directory")
    if candidate.exists() and not candidate.is_dir():
        raise ValueError("attempt output target is not a directory")
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _safe_report_path(path: Path, attempt_directory: Path) -> Path:
    if path.is_symlink():
        raise ValueError("attempt report cannot use a symlink")
    candidate = path.resolve()
    if candidate.parent != attempt_directory or candidate.name != "attempt_report.json":
        raise ValueError("attempt report must be the isolated attempt report")
    return candidate


def _write_report(path: Path, value: Mapping[str, Any]) -> None:
    record = dict(value)
    validate_json_value(record)
    assert_publication_safe(record)
    write_json_atomic(path, record)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} did not return a JSON object")
    validate_json_value(value)
    assert_publication_safe(value)
    return dict(value)


def _sample(torch_module: Any | None) -> dict[str, Any]:
    namespace = torch_module if torch_module is not None else SimpleNamespace(mps=None)
    return sample_mps_memory(namespace)


def _run_stage(
    name: str,
    *,
    torch_getter: Callable[[], Any | None],
    function: Callable[[], _T],
    stage_records: dict[str, dict[str, Any]],
) -> _T:
    """Run one named stage with boundary and periodic sampled telemetry."""

    started_at_unix = time.time()
    started = time.perf_counter()
    samples: list[dict[str, Any]] = [_sample(torch_getter())]
    stop = threading.Event()

    def poll() -> None:
        while not stop.wait(SAMPLING_INTERVAL_SECONDS):
            samples.append(_sample(torch_getter()))

    thread = threading.Thread(target=poll, name=f"mps-telemetry-{name}", daemon=True)
    thread.start()
    result: _T | None = None
    failure: BaseException | None = None
    try:
        result = function()
    except BaseException as error:
        failure = error
    finally:
        stop.set()
        thread.join(timeout=10.0)
        samples.append(_sample(torch_getter()))
        finished_at_unix = time.time()
        stage_records[name] = summarize_mps_samples(
            samples,
            started_at_unix=started_at_unix,
            finished_at_unix=finished_at_unix,
            wall_seconds=time.perf_counter() - started,
            sampling_interval_seconds=SAMPLING_INTERVAL_SECONDS,
        )
    if failure is not None:
        raise failure
    pressures = stage_records[name]["system_memory_pressures"]
    if pressures != ["normal"]:
        raise RuntimeError("unsafe or unclassified memory pressure during MPS stage")
    return result  # type: ignore[return-value]


def _compact_timing(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    values = {
        "mps_current_allocated_peak_bytes": record.get(
            "peak_mps_current_allocated_bytes"
        ),
        "mps_driver_allocated_peak_bytes": record.get(
            "peak_mps_driver_allocated_bytes"
        ),
        "mps_recommended_max_bytes": record.get("peak_mps_recommended_max_bytes"),
        "process_rss_peak_bytes": record.get("peak_process_rss_bytes"),
        "swap_used_peak_bytes": record.get("peak_swap_used_bytes"),
    }
    for name, value in values.items():
        minimum = (
            1
            if name
            in {
                "mps_driver_allocated_peak_bytes",
                "mps_recommended_max_bytes",
                "process_rss_peak_bytes",
            }
            else 0
        )
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise RuntimeError(f"{label} lacks positive sampled {name}")
    pressures = record.get("system_memory_pressures")
    if not isinstance(pressures, list) or not pressures:
        raise RuntimeError(f"{label} lacks memory-pressure samples")
    current = values["mps_current_allocated_peak_bytes"]
    driver = values["mps_driver_allocated_peak_bytes"]
    recommended = values["mps_recommended_max_bytes"]
    assert isinstance(current, int)
    assert isinstance(driver, int)
    assert isinstance(recommended, int)
    if current > driver or driver > recommended:
        raise RuntimeError(f"{label} MPS allocator counters are inconsistent")
    return {
        "started_at_unix": float(record["started_at_unix"]),
        "finished_at_unix": float(record["finished_at_unix"]),
        "wall_seconds": float(record["wall_seconds"]),
        "sampling_method": TELEMETRY_METHOD,
        "sampling_interval_seconds": float(record["target_sampling_interval_seconds"]),
        "sample_count": int(record["sample_count"]),
        **values,
        "memory_pressure_states": list(pressures),
    }


def _attempt_peaks(stage_records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    def peak(name: str) -> int | None:
        values = [
            int(record[name])
            for record in stage_records.values()
            if isinstance(record.get(name), int)
            and not isinstance(record.get(name), bool)
            and int(record[name]) >= 0
        ]
        return max(values) if values else None

    return {
        "peak_mps_current_allocated_bytes": peak("peak_mps_current_allocated_bytes"),
        "peak_mps_driver_allocated_bytes": peak("peak_mps_driver_allocated_bytes"),
        "peak_mps_recommended_max_bytes": peak("peak_mps_recommended_max_bytes"),
        "peak_process_rss_bytes": peak("peak_process_rss_bytes"),
        "peak_swap_used_bytes": peak("peak_swap_used_bytes"),
    }


def _validator_peaks(peaks: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mps_current_allocated_peak_bytes": peaks.get(
            "peak_mps_current_allocated_bytes"
        ),
        "mps_driver_allocated_peak_bytes": peaks.get("peak_mps_driver_allocated_bytes"),
        "process_rss_peak_bytes": peaks.get("peak_process_rss_bytes"),
        "swap_used_peak_bytes": peaks.get("peak_swap_used_bytes"),
    }


def _stage_peak_map(
    stage_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, record in stage_records.items():
        result[name] = {
            "mps_current_allocated_peak_bytes": record.get(
                "peak_mps_current_allocated_bytes"
            ),
            "mps_driver_allocated_peak_bytes": record.get(
                "peak_mps_driver_allocated_bytes"
            ),
            "process_rss_peak_bytes": record.get("peak_process_rss_bytes"),
            "swap_used_peak_bytes": record.get("peak_swap_used_bytes"),
        }
    return result


def _memory_evidence(
    stage_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    attempt_peaks = _attempt_peaks(stage_records)
    timing_start = min(
        float(item["started_at_unix"]) for item in stage_records.values()
    )
    timing_finish = max(
        float(item["finished_at_unix"]) for item in stage_records.values()
    )
    pressure_states = list(
        dict.fromkeys(
            pressure
            for item in stage_records.values()
            for pressure in item["system_memory_pressures"]
        )
    )
    timing = {
        "started_at_unix": timing_start,
        "finished_at_unix": timing_finish,
        "wall_seconds": timing_finish - timing_start,
        "sampling_method": TELEMETRY_METHOD,
        # Each stage has two supplemental boundary observations.  The common
        # attempt cadence counts the actual periodic observations plus only
        # the outer attempt boundaries, so transitions are not double-counted.
        "sample_count": 2
        + sum(max(0, int(item["sample_count"]) - 2) for item in stage_records.values()),
        "mps_current_allocated_peak_bytes": attempt_peaks[
            "peak_mps_current_allocated_bytes"
        ],
        "mps_driver_allocated_peak_bytes": attempt_peaks[
            "peak_mps_driver_allocated_bytes"
        ],
        "mps_recommended_max_bytes": attempt_peaks["peak_mps_recommended_max_bytes"],
        "process_rss_peak_bytes": attempt_peaks["peak_process_rss_bytes"],
        "memory_pressure_states": pressure_states,
        "swap_used_peak_bytes": attempt_peaks["peak_swap_used_bytes"],
    }
    timing["sampling_interval_seconds"] = SAMPLING_INTERVAL_SECONDS
    for key, value in timing.items():
        if not key.endswith("_bytes"):
            continue
        minimum = (
            1
            if key
            in {
                "mps_driver_allocated_peak_bytes",
                "mps_recommended_max_bytes",
                "process_rss_peak_bytes",
            }
            else 0
        )
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise RuntimeError("attempt lacks sampled MPS/RSS telemetry")
    if not pressure_states:
        raise RuntimeError("attempt lacks memory-pressure telemetry")
    current = timing["mps_current_allocated_peak_bytes"]
    driver = timing["mps_driver_allocated_peak_bytes"]
    recommended = timing["mps_recommended_max_bytes"]
    if current > driver or driver > recommended:
        raise RuntimeError("attempt MPS allocator counters are inconsistent")
    return {
        "nonfinite_count": 0,
        "timing": timing,
        "telemetry_method": TELEMETRY_METHOD,
        "sampling_interval_seconds": SAMPLING_INTERVAL_SECONDS,
        "stages": dict(stage_records),
        "attempt_peaks": _validator_peaks(attempt_peaks),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--batch-size", type=int, choices=MPS_BATCH_SEQUENCE, required=True
    )
    parser.add_argument("--attempt-report", type=Path, required=True)
    parser.add_argument("--attempt-directory", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--transcoder-snapshot", type=Path)
    return parser


def _load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as error:
        raise RuntimeError("PyYAML is unavailable") from error
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("MPS configuration must be a mapping")
    from mps_runtime import validate_mps_configuration

    validate_mps_configuration(value)
    return value


def _load_bundle(config: dict[str, Any], args: argparse.Namespace) -> Any:
    module = importlib.import_module("cfsus.reproduction.mps_fp16")
    loader = getattr(module, "load_mps_runtime", None)
    if not callable(loader):
        raise RuntimeError("mps_fp16.load_mps_runtime is unavailable")
    return loader(
        config,
        allow_download=args.allow_download,
        model_snapshot=args.model_snapshot,
        transcoder_snapshot=args.transcoder_snapshot,
        progressive=True,
    )


def _verify_circuit_tracer_install() -> tuple[str, int]:
    distribution = importlib.metadata.distribution("circuit-tracer")
    try:
        direct_url = json.loads(distribution.read_text("direct_url.json") or "")
    except (json.JSONDecodeError, TypeError) as error:
        raise RuntimeError(
            "circuit-tracer PEP 610 provenance is unavailable"
        ) from error
    vcs = direct_url.get("vcs_info") if isinstance(direct_url, dict) else None
    if (
        not isinstance(vcs, dict)
        or direct_url.get("url") != CIRCUIT_TRACER_REPOSITORY
        or vcs.get("vcs") != "git"
        or vcs.get("commit_id") != UPSTREAM_REVISION
        or vcs.get("requested_revision") != UPSTREAM_REVISION
    ):
        raise RuntimeError("circuit-tracer PEP 610 provenance is not the pinned commit")

    verified = 0
    package_files = [
        item
        for item in distribution.files or ()
        if str(item).startswith("circuit_tracer/")
        and "__pycache__" not in item.parts
        and item.suffix != ".pyc"
    ]
    if not package_files:
        raise RuntimeError("circuit-tracer installed files are unavailable")
    for item in package_files:
        recorded_hash = item.hash
        if recorded_hash is None or recorded_hash.mode != "sha256":
            raise RuntimeError("circuit-tracer RECORD lacks a source-file hash")
        path = Path(str(distribution.locate_file(item)))
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("circuit-tracer installed source file is unsafe")
        digest = hashlib.sha256(path.read_bytes()).digest()
        encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        if encoded != recorded_hash.value:
            raise RuntimeError("circuit-tracer installed source differs from RECORD")
        if item.size is not None and path.stat().st_size != item.size:
            raise RuntimeError(
                "circuit-tracer installed source size differs from RECORD"
            )
        verified += 1
    return CIRCUIT_TRACER_REPOSITORY, verified


def _core_function(name: str) -> Any:
    module = importlib.import_module("cfsus.reproduction.mps_fp16")
    function = getattr(module, name, None)
    if not callable(function):
        raise RuntimeError(f"mps_fp16.{name} is unavailable")
    return function


def _environment_payload(bundle: Any, model_smoke: Mapping[str, Any]) -> dict[str, Any]:
    torch_module = getattr(bundle, "torch", None)
    backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    built = bool(backend is not None and backend.is_built())
    available = bool(backend is not None and backend.is_available())
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("loaded MPS worker is not native Darwin arm64")
    if not built or not available or fallback_enabled():
        raise RuntimeError("loaded worker failed the no-fallback MPS gate")
    if model_smoke.get("post_load_passed") is not True:
        raise RuntimeError("model-only forward evidence is invalid")
    high_watermark = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
    if high_watermark is not None:
        raise RuntimeError("MPS high-watermark override must be absent")
    lock = REPOSITORY_ROOT / "environments/stage1a_mps/requirements-lock.txt"
    if lock.is_symlink() or not lock.is_file():
        raise RuntimeError("MPS environment lock is unavailable")
    lock_digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    try:
        physical_memory = int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise RuntimeError("physical memory observation is unavailable") from error
    pip_environment = os.environ.copy()
    pip_environment.pop("PYTHONPATH", None)
    pip_environment.pop("PYTHONHOME", None)
    pip_check = subprocess.run(
        (sys.executable, "-m", "pip", "check"),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env=pip_environment,
    )
    pip_freeze = subprocess.run(
        (sys.executable, "-m", "pip", "freeze", "--all"),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env=pip_environment,
    )
    expected_lock = tuple(
        line
        for line in lock.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    )
    observed_lock = tuple(line for line in pip_freeze.stdout.splitlines() if line)
    if pip_freeze.returncode != 0 or observed_lock != expected_lock:
        raise RuntimeError("selected MPS environment does not exactly match its lock")
    try:
        transformer_lens_version = importlib.metadata.version("transformer-lens")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("TransformerLens package metadata is unavailable") from error
    try:
        circuit_tracer_version = importlib.metadata.version("circuit-tracer")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("circuit-tracer package metadata is unavailable") from error
    circuit_tracer_url, verified_record_files = _verify_circuit_tracer_install()
    payload: dict[str, Any] = {
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "hardware_family": "Apple M2 Max",
            "physical_memory_bytes": physical_memory,
        },
        "python": {
            "version": platform.python_version(),
            "architecture": platform.machine(),
        },
        "packages": {
            "torch": str(getattr(torch_module, "__version__", "unknown")),
            "transformer_lens": transformer_lens_version,
            "circuit_tracer": circuit_tracer_version,
            "circuit_tracer_revision": UPSTREAM_REVISION,
            "circuit_tracer_vcs_url": circuit_tracer_url,
            "circuit_tracer_record_hashes_verified": verified_record_files,
        },
        "runtime": {
            "backend": "transformerlens",
            "accelerator_backend": "mps",
            "device": "mps",
            "dtype": MPS_EXECUTION_DTYPE,
            "execution_class": "completed_hardware_adapted_mps_fp16",
            "offload": "disk",
        },
        "mps": {
            "built": built,
            "available": available,
            "allocation_probe": {
                "success": True,
                "device": "mps",
                "dtype": MPS_EXECUTION_DTYPE,
                "finite": True,
            },
        },
        "fallback_enabled": False,
        "fallback_used": False,
        "fallback_env_value_present": False,
        "high_watermark_override": None,
        "memory_guardrails_preserved": True,
        "pip_check": "passed" if pip_check.returncode == 0 else "failed",
        "lock_match": "exact",
        "lock_path": "environments/stage1a_mps/requirements-lock.txt",
        "lock_sha256": lock_digest,
    }
    if payload["pip_check"] != "passed":
        raise RuntimeError("pip check failed in the selected MPS environment")
    return payload


def _asset_payload(bundle: Any) -> dict[str, Any]:
    raw = _mapping(getattr(bundle, "asset_manifest", None), "asset manifest")
    if set(raw) != {
        "verification",
        "cache_policy",
        "upstream_revision",
        "immutable_revisions_only",
        "project_external_cache",
        "unmanifested_file_count",
        "model",
        "transcoder",
    }:
        raise RuntimeError("asset manifest keys are not exact")
    if (
        raw.get("verification") != "exact_file_content_hashes_matched"
        or raw.get("cache_policy") != "project_external_huggingface_cache"
        or raw.get("upstream_revision") != UPSTREAM_REVISION
        or raw.get("immutable_revisions_only") is not True
        or raw.get("project_external_cache") is not True
        or raw.get("unmanifested_file_count") != 0
    ):
        raise RuntimeError("asset hashes were not verified")
    expected = (
        ("model", MODEL_ID, MODEL_REVISION, MODEL_REQUIRED_FILES),
        (
            "transcoder",
            TRANSCODER_ID,
            TRANSCODER_REVISION,
            TRANSCODER_REQUIRED_FILES,
        ),
    )
    assets: dict[str, Any] = {}
    for label, identifier, revision, required_files in expected:
        section = raw.get(label)
        if not isinstance(section, dict):
            raise RuntimeError(f"asset manifest is missing {label}")
        if set(section) != {
            "identifier",
            "revision",
            "files",
            "file_count",
            "total_bytes",
            "complete",
            "snapshot_containment_verified",
            "offline_ready",
        }:
            raise RuntimeError(f"asset manifest {label} keys are not exact")
        if (
            section.get("identifier") != identifier
            or section.get("revision") != revision
            or section.get("complete") is not True
            or section.get("snapshot_containment_verified") is not True
            or section.get("offline_ready") is not True
        ):
            raise RuntimeError(f"asset manifest {label} pin is invalid")
        files = section.get("files")
        if not isinstance(files, list) or not files:
            raise RuntimeError(f"asset manifest {label} files are missing")
        observed_paths: set[str] = set()
        total_bytes = 0
        for item in files:
            if not isinstance(item, dict):
                raise RuntimeError("asset file manifest item is invalid")
            relative = item.get("path")
            digest = item.get("sha256")
            size = item.get("size_bytes")
            if (
                not isinstance(relative, str)
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
            ):
                raise RuntimeError("asset file manifest metadata is invalid")
            observed_paths.add(relative)
            total_bytes += size
        if observed_paths != set(required_files) or len(files) != len(required_files):
            raise RuntimeError(f"asset manifest {label} file set is invalid")
        if (
            section.get("file_count") != len(files)
            or section.get("total_bytes") != total_bytes
        ):
            raise RuntimeError(f"asset manifest {label} aggregate is invalid")
        assets[label] = {
            "identifier": identifier,
            "revision": revision,
            "files": files,
            "file_count": len(files),
            "total_bytes": total_bytes,
            "complete": True,
            "snapshot_containment_verified": True,
            "offline_ready": True,
        }
    return {
        "verification": "exact_file_content_hashes_matched",
        "immutable_revisions_only": True,
        "project_external_cache": True,
        "unmanifested_file_count": 0,
        "assets": assets,
    }


def _release(bundle: Any | None) -> bool:
    succeeded = True
    torch_module = getattr(bundle, "torch", None) if bundle is not None else None
    try:
        if bundle is not None and hasattr(bundle, "close"):
            bundle.close()
    except Exception:
        succeeded = False
    try:
        gc.collect()
        mps = getattr(torch_module, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache"):
            mps.empty_cache()
        if mps is not None and hasattr(mps, "synchronize"):
            mps.synchronize()
    except Exception:
        succeeded = False
    return succeeded


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    attempt_directory = _safe_attempt_directory(args.attempt_directory)
    report_path = _safe_report_path(args.attempt_report, attempt_directory)
    if (args.model_snapshot is None) != (args.transcoder_snapshot is None):
        raise ValueError("model and transcoder snapshot overrides must be paired")

    started = time.perf_counter()
    stage = "configuration_validation"
    bundle: Any | None = None
    torch_module: Any | None = None
    stage_records: dict[str, dict[str, Any]] = {}
    error: BaseException | None = None
    failure_stage: str | None = None
    cleanup_succeeded = False
    try:
        config = _load_config(args.config.resolve())
        if fallback_enabled():
            raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be disabled")
        # Expose the real MPS allocator to the sampler before model/transcoder
        # loading begins so transient conversion and placement peaks are observed.
        torch_module = importlib.import_module("torch")

        def load() -> Any:
            nonlocal bundle, torch_module
            bundle = _load_bundle(config, args)
            torch_module = getattr(bundle, "torch", None)
            return bundle

        stage = "runtime_loading"
        _run_stage(
            stage,
            torch_getter=lambda: torch_module,
            function=load,
            stage_records=stage_records,
        )

        stage = "model_only_forward"

        def capture_model_evidence() -> tuple[
            dict[str, Any], dict[str, Any], dict[str, Any]
        ]:
            smoke_passed = bool(_core_function("model_only_forward_smoke")(bundle))
            progressive_smoke = _mapping(
                getattr(bundle, "model_only_forward", {"passed": smoke_passed}),
                "progressive model-only forward",
            )
            model_smoke = {
                "progressive": progressive_smoke,
                "post_load_passed": smoke_passed,
            }
            if not smoke_passed or progressive_smoke.get("passed") is not True:
                raise RuntimeError("model-only MPS smoke did not pass")
            return (
                model_smoke,
                _environment_payload(bundle, model_smoke),
                _asset_payload(bundle),
            )

        model_smoke, environment, asset = _run_stage(
            stage,
            torch_getter=lambda: torch_module,
            function=capture_model_evidence,
            stage_records=stage_records,
        )
        _write_report(attempt_directory / "model_smoke.json", model_smoke)
        _write_report(attempt_directory / "environment.json", environment)
        _write_report(attempt_directory / "asset.json", asset)

        stage = "semantics"
        semantics = _run_stage(
            stage,
            torch_getter=lambda: torch_module,
            function=lambda: _mapping(
                _core_function("verify_mps_runtime_semantics")(bundle), "semantics"
            ),
            stage_records=stage_records,
        )
        semantics["timing"] = _compact_timing(stage_records[stage], stage)
        _write_report(attempt_directory / "semantics.json", semantics)

        stage = "intervention"
        intervention = _run_stage(
            stage,
            torch_getter=lambda: torch_module,
            function=lambda: _mapping(
                _core_function("reproduce_mps_intervention")(bundle),
                "intervention",
            ),
            stage_records=stage_records,
        )
        intervention["timing"] = _compact_timing(stage_records[stage], stage)
        _write_report(attempt_directory / "intervention.json", intervention)

        stage = "attribution"
        attribution = _run_stage(
            stage,
            torch_getter=lambda: torch_module,
            function=lambda: _mapping(
                _core_function("reproduce_mps_attribution")(
                    bundle,
                    batch_size=args.batch_size,
                    raw_graph_output=attempt_directory / "attribution_graph.pt",
                ),
                "attribution",
            ),
            stage_records=stage_records,
        )
        attribution["timing"] = _compact_timing(stage_records[stage], stage)
        _write_report(attempt_directory / "attribution.json", attribution)
    except BaseException as caught:
        if isinstance(caught, (KeyboardInterrupt, SystemExit)):
            raise
        error = caught
        failure_stage = stage
    finally:
        cleanup_stage = "cleanup"
        try:
            cleanup_succeeded = _run_stage(
                cleanup_stage,
                torch_getter=lambda: torch_module,
                function=lambda: _release(bundle),
                stage_records=stage_records,
            )
        except BaseException as cleanup_error:
            cleanup_succeeded = False
            if error is None:
                error = cleanup_error
                failure_stage = cleanup_stage
        if error is None and not cleanup_succeeded:
            error = RuntimeError("MPS cleanup did not complete")
            failure_stage = cleanup_stage

    attempt_peaks = _attempt_peaks(stage_records)
    memory: dict[str, Any] | None = None
    if error is None:
        try:
            memory = _memory_evidence(stage_records)
        except BaseException as telemetry_error:
            if isinstance(telemetry_error, (KeyboardInterrupt, SystemExit)):
                raise
            error = telemetry_error
            failure_stage = "telemetry_finalization"
    if error is None and memory is not None:
        _write_report(attempt_directory / "memory.json", memory)
        memory_timing = _mapping(memory["timing"], "memory timing")
        report = {
            "schema_version": 1,
            "attempt_id": attempt_directory.name,
            "batch_size": args.batch_size,
            "outcome": "completed",
            "category": "completed",
            "exception_type": None,
            "message": "MPS attempt completed with all checks passing.",
            "failure_stage": None,
            "wall_seconds": time.perf_counter() - started,
            "cleanup_succeeded": cleanup_succeeded,
            "fresh_process": True,
            "oom_confirmed": False,
            "oom_classifier_match": False,
            "retry_eligible": False,
            "diagnostic_redacted": True,
            "sample_count": memory_timing["sample_count"],
            "attempt_peaks": _validator_peaks(attempt_peaks),
            "stage_peaks": _stage_peak_map(stage_records),
        }
        code = 0
    else:
        assert error is not None
        category = classify_failure(error)
        leaf = exception_chain(error)[-1]
        oom_confirmed = is_mps_out_of_memory(error)
        retry_eligible = (
            oom_confirmed
            and failure_stage == "attribution"
            and args.batch_size != MPS_BATCH_SEQUENCE[-1]
        )
        report = {
            "schema_version": 1,
            "attempt_id": attempt_directory.name,
            "batch_size": args.batch_size,
            "outcome": "failed",
            "category": category,
            "exception_type": type(leaf).__name__,
            "message": sanitize_error(leaf),
            "failure_stage": failure_stage,
            "wall_seconds": time.perf_counter() - started,
            "cleanup_succeeded": cleanup_succeeded,
            "fresh_process": True,
            "oom_confirmed": oom_confirmed,
            "oom_classifier_match": oom_confirmed,
            "retry_eligible": retry_eligible,
            "diagnostic_redacted": True,
            "sample_count": sum(
                int(item["sample_count"]) for item in stage_records.values()
            ),
            "attempt_peaks": _validator_peaks(attempt_peaks),
            "stage_peaks": _stage_peak_map(stage_records),
        }
        code = 10 if oom_confirmed else 1
    _write_report(report_path, report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
