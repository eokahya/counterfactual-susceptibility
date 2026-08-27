#!/usr/bin/env python3
"""Thin Stage 1D orchestration over the accepted Stage 1C-v4 production core."""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfsus.mps_telemetry import MPSTelemetrySampler  # noqa: E402
from cfsus.reproduction.small_model_mps_bf16 import (  # noqa: E402
    assert_fallback_disabled,
)
from cfsus.stage1b_runtime import (  # noqa: E402
    build_mps_bf16_replacement,
    resolve_offline_snapshots,
)
from cfsus.stage1c_v3.execution_journal import (  # noqa: E402
    CanonicalExecutionJournal,
)
from cfsus.stage1c_v3.intervention_runtime import (  # noqa: E402
    Stage1CInterventionBackend,
)
from cfsus.stage1c_v3.serialization import (  # noqa: E402
    read_json_strict,
    write_json_new,
)
from cfsus.stage1d.artifacts import (  # noqa: E402
    build_records,
    publish_records,
    read_completed_journal,
)
from cfsus.stage1d.config import (  # noqa: E402
    CONFIG_PATH,
    EXPERIMENT_CLASS,
    load_stage1d_config,
)
from cfsus.stage1d.execution import execute_frozen_pairs  # noqa: E402
from cfsus.stage1d.prediction_runtime import build_prediction_manifest  # noqa: E402
from cfsus.stage1d.preflight import runtime_identity, verify_git  # noqa: E402
from cfsus.stage1d.protocol import (  # noqa: E402
    publish_protocol_manifest,
    sha256_file,
)
from cfsus.stage1d.rehearsal import (  # noqa: E402
    run_rehearsal,
    validate_rehearsal,
)
from cfsus.stage1d.validation import (  # noqa: E402
    validate_bundle,
    validate_prediction,
    validate_protocol,
)

CREDENTIAL_VARIABLES = {
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "GITHUB_TOKEN",
    "GH_TOKEN",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "protocol-manifest",
            "prediction-worker",
            "assemble-prediction",
            "rehearse",
            "validate-rehearsal",
            "intervention-worker",
            "assemble",
            "validate",
        ),
    )
    parser.add_argument("--config", type=Path, default=ROOT / CONFIG_PATH)
    parser.add_argument("--hf-cache", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--protocol-commit")
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--prediction-freeze-commit")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--attempt-lock", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--emergency-output", type=Path)
    return parser


def _dict(path: Path, label: str) -> dict[str, Any]:
    value = read_json_strict(path)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _required(value: Path | str | None, label: str) -> Any:
    if value is None:
        raise RuntimeError(f"{label} is required")
    return value


def _safe_environment_boundary() -> None:
    assert_fallback_disabled()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for key in CREDENTIAL_VARIABLES:
        os.environ.pop(key, None)
    for key in tuple(os.environ):
        if key.startswith("GIT_"):
            os.environ.pop(key, None)


def _verify_assets(cache: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage1a/verify_small_model_mps_bf16_assets.py"),
            "--hf-cache",
            str(cache),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
        env={
            key: value
            for key, value in os.environ.items()
            if key not in CREDENTIAL_VARIABLES
        }
        | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    record = json.loads(result.stdout)
    if record.get("status") != "verified":
        raise RuntimeError("immutable asset byte verification failed")
    return {
        "status": "verified",
        "actual_total_bytes": record["actual_total_bytes"],
        "model_total_bytes": record["model"]["total_bytes"],
        "transcoder_total_bytes": record["transcoder"]["total_bytes"],
        "exact_allowlist_hashes_verified": True,
        "download_performed": False,
        "network_accessed": False,
    }


def _load_runtime(
    cache: Path, config: dict[str, Any], emergency: Path
) -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any]]:
    identity = runtime_identity(cache, ROOT)
    assets = _verify_assets(cache)
    import nnsight  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    import transformers  # type: ignore[import-not-found]

    if (
        not torch.backends.mps.is_built()
        or not torch.backends.mps.is_available()
        or torch.is_autocast_enabled()
    ):
        raise RuntimeError("native MPS/BF16 runtime is unavailable")
    identity.update(
        {
            "mps_built": True,
            "mps_available": True,
            "outer_autocast_enabled": False,
            "nnsight": nnsight.__version__,
            "transformers": transformers.__version__,
            "torch": str(torch.__version__),
            "circuit-tracer": importlib.metadata.version("circuit-tracer"),
            "device": "mps:0",
            "dtype": "torch.bfloat16",
            "metadata_device": "cpu",
        }
    )
    model_snapshot, transcoder_snapshot = resolve_offline_snapshots(cache, ROOT)
    sampler = MPSTelemetrySampler(torch, config["safety_limits"], emergency)
    with sampler.stage("replacement_runtime_loading"):
        model, module_guard = build_mps_bf16_replacement(
            model_snapshot, transcoder_snapshot, torch
        )
    return (
        model,
        torch,
        sampler,
        identity,
        {"assets": assets, "module_guard": module_guard},
    )


def _finish_sampler(sampler: Any, *, already_finished: bool) -> dict[str, Any] | None:
    if already_finished:
        return None
    with contextlib.suppress(Exception):
        return cast(dict[str, Any], sampler.finish())
    return None


def _prediction_worker(args: argparse.Namespace) -> None:
    cache = cast(Path, _required(args.hf_cache, "--hf-cache"))
    protocol_path = cast(Path, _required(args.protocol, "--protocol"))
    output = cast(Path, _required(args.output, "--output"))
    emergency = cast(Path, _required(args.emergency_output, "--emergency-output"))
    config = load_stage1d_config(args.config)
    git = verify_git(ROOT)
    protocol = _dict(protocol_path, "protocol manifest")
    validate_protocol(ROOT, protocol, config)
    if protocol["protocol_commit"] != git["head"]:
        raise RuntimeError("prediction must run at the pushed protocol-freeze commit")
    model: Any = None
    sampler: Any = None
    sampler_finished = False
    try:
        model, torch, sampler, environment, runtime_evidence = _load_runtime(
            cache, config, emergency
        )
        prediction = build_prediction_manifest(
            model,
            torch,
            sampler,
            config,
            protocol_manifest=protocol,
            git_identity=git,
        )
        telemetry = sampler.finish()
        sampler_finished = True
        if telemetry["violations"] or telemetry["telemetry_failures"]:
            raise RuntimeError("prediction telemetry contains a safety failure")
        result = {
            "schema_version": 1,
            "artifact_type": "stage1d_prediction_worker",
            "status": "passed",
            "prediction_manifest": prediction,
            "git": git,
            "environment": environment,
            "runtime_evidence": runtime_evidence,
            "telemetry": telemetry,
            "evaluation_source_suppression_api_calls": 0,
        }
        write_json_new(output, result)
    finally:
        if sampler is not None:
            _finish_sampler(sampler, already_finished=sampler_finished)
        if model is not None:
            del model
        gc.collect()
        if "torch" in locals():
            with contextlib.suppress(Exception):
                torch.mps.empty_cache()


def _assemble_prediction(args: argparse.Namespace) -> None:
    protocol = _dict(cast(Path, _required(args.protocol, "--protocol")), "protocol")
    worker = _dict(cast(Path, _required(args.worker, "--worker")), "worker")
    output = cast(Path, _required(args.output, "--output"))
    config = load_stage1d_config(args.config)
    git = verify_git(ROOT)
    validate_protocol(ROOT, protocol, config)
    if (
        worker.get("artifact_type") != "stage1d_prediction_worker"
        or worker.get("status") != "passed"
    ):
        raise RuntimeError("prediction worker did not pass")
    prediction = _object(worker.get("prediction_manifest"), "prediction")
    validate_prediction(prediction, config, protocol)
    if prediction["prediction_execution_commit"] != git["head"]:
        raise RuntimeError("prediction worker commit differs")
    write_json_new(output, prediction)
    print(json.dumps({"status": "passed", "sha256": sha256_file(output)}))


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _rehearse(args: argparse.Namespace) -> None:
    cache = cast(Path, _required(args.hf_cache, "--hf-cache"))
    output = cast(Path, _required(args.output, "--output"))
    journal = cast(Path, _required(args.journal, "--journal"))
    emergency = cast(Path, _required(args.emergency_output, "--emergency-output"))
    prediction_commit = str(
        _required(args.prediction_freeze_commit, "--prediction-freeze-commit")
    )
    config = load_stage1d_config(args.config)
    verify_git(ROOT, expected_head=prediction_commit)
    model: Any = None
    sampler: Any = None
    sampler_finished = False
    try:
        model, torch, sampler, _, _ = _load_runtime(cache, config, emergency)
        run_rehearsal(
            repository=ROOT,
            model=model,
            torch=torch,
            sampler=sampler,
            journal_path=journal,
            output=output,
        )
        telemetry = sampler.finish()
        sampler_finished = True
        if telemetry["violations"] or telemetry["telemetry_failures"]:
            raise RuntimeError("rehearsal telemetry contains a safety failure")
    finally:
        if sampler is not None:
            _finish_sampler(sampler, already_finished=sampler_finished)
        if model is not None:
            del model
        gc.collect()
        if "torch" in locals():
            with contextlib.suppress(Exception):
                torch.mps.empty_cache()


def _verify_tracked_prediction(path: Path) -> None:
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"HEAD:{relative}"],
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout
    if tracked != path.read_bytes():
        raise RuntimeError(
            "working prediction manifest differs from tracked HEAD bytes"
        )


def _intervention_worker(args: argparse.Namespace) -> None:
    cache = cast(Path, _required(args.hf_cache, "--hf-cache"))
    protocol_path = cast(Path, _required(args.protocol, "--protocol"))
    prediction_path = cast(Path, _required(args.prediction, "--prediction"))
    output = cast(Path, _required(args.output, "--output"))
    journal_path = cast(Path, _required(args.journal, "--journal"))
    attempt_lock = cast(Path, _required(args.attempt_lock, "--attempt-lock"))
    emergency = cast(Path, _required(args.emergency_output, "--emergency-output"))
    prediction_commit = str(
        _required(args.prediction_freeze_commit, "--prediction-freeze-commit")
    )
    config = load_stage1d_config(args.config)
    git = verify_git(ROOT, expected_head=prediction_commit)
    protocol = _dict(protocol_path, "protocol")
    prediction = _dict(prediction_path, "prediction")
    validate_protocol(ROOT, protocol, config)
    validate_prediction(prediction, config, protocol)
    _verify_tracked_prediction(prediction_path)
    prediction_sha = sha256_file(prediction_path)
    if journal_path.exists() or attempt_lock.exists() or output.exists():
        raise RuntimeError("canonical output, journal, and attempt lock must be new")
    pair_ids = tuple(
        pair["pair_id"]
        for prompt in prediction["prompts"]
        for pair in prompt["execution_pairs"]
    )
    model: Any = None
    sampler: Any = None
    sampler_finished = False
    with CanonicalExecutionJournal(
        journal_path,
        attempt_lock,
        frozen_pair_ids=pair_ids,
        pre_intervention_commit=prediction_commit,
        prediction_manifest_sha256=prediction_sha,
        experiment_class=EXPERIMENT_CLASS,
        attempt_boundary=config["intervention"]["attempt_boundary"],
        attempt_lock_artifact_type="stage1d_local_scientific_attempt_lock",
    ) as journal:
        try:
            model, torch, sampler, environment, runtime_evidence = _load_runtime(
                cache, config, emergency
            )
            sweeps, calls = execute_frozen_pairs(
                model=model,
                torch=torch,
                sampler=sampler,
                prompts=prediction["prompts"],
                journal=journal,
                backend_factory=Stage1CInterventionBackend,
                maximum_bisection_steps=int(
                    config["schedules"]["maximum_bisection_steps"]
                ),
            )
            telemetry = sampler.finish()
            sampler_finished = True
            if telemetry["violations"] or telemetry["telemetry_failures"]:
                raise RuntimeError("canonical telemetry contains a safety failure")
            result = {
                "schema_version": 1,
                "artifact_type": "stage1d_intervention_worker",
                "status": "passed",
                "prediction_manifest_sha256": prediction_sha,
                "prediction_freeze_commit": prediction_commit,
                "pre_run_commit": prediction_commit,
                "canonical_attempt_count": 1,
                "scientific_retry_count": 0,
                "instrumented_evaluation_api_calls": calls,
                "completed_in_memory_sweep_count": len(sweeps),
                "in_memory_sweeps_publication_trusted": False,
                "journal_sha256": sha256_file(journal_path),
                "environment": environment,
                "runtime_evidence": runtime_evidence,
                "telemetry": telemetry,
                "git": git,
            }
            write_json_new(output, result)
        finally:
            if sampler is not None:
                _finish_sampler(sampler, already_finished=sampler_finished)
            if model is not None:
                del model
            gc.collect()
            if "torch" in locals():
                with contextlib.suppress(Exception):
                    torch.mps.empty_cache()


def _assemble(args: argparse.Namespace) -> None:
    protocol = _dict(cast(Path, _required(args.protocol, "--protocol")), "protocol")
    prediction = _dict(
        cast(Path, _required(args.prediction, "--prediction")), "prediction"
    )
    worker = _dict(cast(Path, _required(args.worker, "--worker")), "worker")
    journal_path = cast(Path, _required(args.journal, "--journal"))
    output = cast(Path, _required(args.output_dir, "--output-dir"))
    config = load_stage1d_config(args.config)
    validate_protocol(ROOT, protocol, config)
    validate_prediction(prediction, config, protocol)
    if (
        worker.get("artifact_type") != "stage1d_intervention_worker"
        or worker.get("status") != "passed"
    ):
        raise RuntimeError("canonical intervention worker did not pass")
    if worker.get("journal_sha256") != sha256_file(journal_path):
        raise RuntimeError("canonical journal digest differs from worker evidence")
    points = read_completed_journal(journal_path)
    records = build_records(
        protocol=protocol,
        prediction=prediction,
        worker=worker,
        points=points,
        config=config,
    )
    publish_records(output, records)
    print(json.dumps(validate_bundle(ROOT, output), sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _safe_environment_boundary()
    if args.action == "protocol-manifest":
        output = cast(Path, _required(args.output, "--output"))
        commit = str(_required(args.protocol_commit, "--protocol-commit"))
        publish_protocol_manifest(ROOT, output, protocol_commit=commit)
    elif args.action == "prediction-worker":
        _prediction_worker(args)
    elif args.action == "assemble-prediction":
        _assemble_prediction(args)
    elif args.action == "rehearse":
        _rehearse(args)
    elif args.action == "validate-rehearsal":
        result = validate_rehearsal(
            cast(Path, _required(args.output, "--output")),
            cast(Path, _required(args.journal, "--journal")),
        )
        print(json.dumps(result, sort_keys=True))
    elif args.action == "intervention-worker":
        _intervention_worker(args)
    elif args.action == "assemble":
        _assemble(args)
    elif args.action == "validate":
        output = cast(Path, _required(args.output_dir, "--output-dir"))
        print(json.dumps(validate_bundle(ROOT, output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
