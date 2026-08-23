#!/usr/bin/env python3
"""Execute one isolated Stage 1A T4/FP16 scientific attempt."""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from preflight import collect_report  # noqa: E402
from reproduce_attribution import (  # noqa: E402
    RuntimeBundle,
    _mapping,
    load_runtime,
    load_yaml,
    repository_root,
    reproduce_attribution,
)
from reproduce_intervention import reproduce_intervention  # noqa: E402
from verify_runtime_semantics import verify_runtime_semantics  # noqa: E402


def _safe_report_path(path: Path) -> Path:
    candidate = path.resolve()
    allowed = (repository_root() / "results/generated/stage1a_t4_fp16").resolve()
    if not candidate.is_relative_to(allowed):
        raise ValueError("attempt report must stay in the T4 generated directory")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def _peak_cuda_bytes(torch: Any | None) -> int | None:
    if torch is None:
        return None
    try:
        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except Exception:
        return None
    return None


def _record_peak_cuda_bytes(observations: list[int], torch: Any | None) -> None:
    peak = _peak_cuda_bytes(torch)
    if peak is not None:
        observations.append(peak)


def _maximum_peak_cuda_bytes(observations: list[int]) -> int | None:
    return max(observations) if observations else None


def _release(bundle: RuntimeBundle | None) -> bool:
    succeeded = True
    torch = bundle.torch if bundle is not None else sys.modules.get("torch")
    if bundle is not None:
        try:
            del bundle.model
        except Exception:
            succeeded = False
    try:
        gc.collect()
    except Exception:
        succeeded = False
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            succeeded = False
    return succeeded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(256, 128, 64), required=True)
    parser.add_argument("--attempt-report", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--transcoder-snapshot", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    from cfsus.reproduction.artifacts import write_json_atomic
    from cfsus.reproduction.t4_fp16 import (
        classify_t4_failure,
        exception_chain,
        is_cuda_out_of_memory,
        sanitize_exception_message,
        validate_t4_fp16_mapping,
    )

    args = build_parser().parse_args(argv)
    report_path = _safe_report_path(args.attempt_report)
    bundle: RuntimeBundle | None = None
    error: BaseException | None = None
    stage = "configuration_validation"
    started = time.perf_counter()
    peak_memory_observations: list[int] = []
    report: dict[str, Any]
    try:
        config = load_yaml(args.config.resolve())
        validate_t4_fp16_mapping(config)
        if (args.model_snapshot is None) != (args.transcoder_snapshot is None):
            raise ValueError("snapshot overrides must be supplied together")
        stage = "runtime_loading"
        print("Worker entering immutable runtime loading", flush=True)
        bundle = load_runtime(
            config,
            allow_download=False,
            model_snapshot=args.model_snapshot,
            transcoder_snapshot=args.transcoder_snapshot,
        )
        _record_peak_cuda_bytes(peak_memory_observations, bundle.torch)
        print("Worker completed immutable runtime loading", flush=True)
        artifacts = _mapping(config.get("artifacts"), "artifacts")
        stage = "environment_observation"
        write_json_atomic(
            repository_root() / str(artifacts["environment_manifest"]),
            collect_report(
                model_snapshot_present=True,
                transcoder_snapshot_present=True,
            ),
        )
        stage = "runtime_semantics"
        verify_runtime_semantics(bundle)
        _record_peak_cuda_bytes(peak_memory_observations, bundle.torch)
        stage = "intervention"
        reproduce_intervention(bundle)
        _record_peak_cuda_bytes(peak_memory_observations, bundle.torch)
        stage = "attribution"
        reproduce_attribution(bundle, batch_size=args.batch_size)
        _record_peak_cuda_bytes(peak_memory_observations, bundle.torch)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        error = exc
    finally:
        _record_peak_cuda_bytes(
            peak_memory_observations,
            bundle.torch if bundle is not None else sys.modules.get("torch"),
        )
        peak_memory = _maximum_peak_cuda_bytes(peak_memory_observations)
        cleanup_succeeded = _release(bundle)
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

    if error is None:
        report = {
            "schema_version": 1,
            "batch_size": args.batch_size,
            "outcome": "completed",
            "category": "completed",
            "exception_type": None,
            "message": "Attempt completed with all scientific checks passing.",
            "failure_stage": None,
            "peak_memory_bytes": peak_memory,
            "wall_seconds": time.perf_counter() - started,
            "cleanup_succeeded": cleanup_succeeded,
            "runtime_provenance": bundle.provenance if bundle is not None else {},
        }
        exit_code = 0
    else:
        status = classify_t4_failure(error)
        cuda_oom = is_cuda_out_of_memory(error)
        category = "cuda_out_of_memory" if cuda_oom else status.value
        diagnostic_error = exception_chain(error)[-1]
        exception_type = type(diagnostic_error).__name__
        report = {
            "schema_version": 1,
            "batch_size": args.batch_size,
            "outcome": "failed",
            "category": category,
            "exception_type": exception_type,
            "message": sanitize_exception_message(diagnostic_error),
            "failure_stage": stage,
            "peak_memory_bytes": peak_memory,
            "wall_seconds": time.perf_counter() - started,
            "cleanup_succeeded": cleanup_succeeded,
            "runtime_provenance": bundle.provenance if bundle is not None else {},
        }
        exit_code = 10 if category == "cuda_out_of_memory" else 1
    write_json_atomic(report_path, report)
    print(
        f"T4 attempt batch={args.batch_size} outcome={report['outcome']} "
        f"category={report['category']}"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
