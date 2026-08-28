#!/usr/bin/env python3
"""Thin Stage 1G orchestration over the accepted production path."""

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
from cfsus.stage1c_v3.execution_journal import CanonicalExecutionJournal  # noqa: E402
from cfsus.stage1c_v3.prediction import canonical_v3_pair_id  # noqa: E402
from cfsus.stage1c_v3.serialization import (  # noqa: E402
    read_json_strict,
    write_json_new,
)
from cfsus.stage1d.preflight import runtime_identity  # noqa: E402
from cfsus.stage1d.protocol import sha256_file  # noqa: E402
from cfsus.stage1g import (  # noqa: E402
    CONFIG_PATH,
    EXPERIMENT_CLASS,
    RUNTIME_FINGERPRINT,
    Stage1GPredictionBackend,
    build_prediction_manifest,
    build_protocol_manifest,
    build_records,
    execute_frozen_pairs,
    load_config,
    publish_records,
    read_completed_journal,
    run_output_sensitivity_validation,
    validate_bundle,
    validate_output_sensitivity,
    validate_prediction,
    validate_protocol,
    verify_git,
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
            "validate-prediction",
            "real-rehearsal",
            "validate-real-rehearsal",
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
    parser.add_argument("--sensitivity-output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--emergency-output", type=Path)
    return parser


def _required(value: Any, label: str) -> Any:
    if value is None:
        raise RuntimeError(f"{label} is required")
    return value


def _dict(path: Path, label: str) -> dict[str, Any]:
    value = read_json_strict(path)
    if type(value) is not dict:
        raise RuntimeError(f"{label} must be an object")
    return cast(dict[str, Any], value)


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


def _finish_sampler(sampler: Any, *, already_finished: bool) -> None:
    if not already_finished:
        with contextlib.suppress(Exception):
            sampler.finish()


def _prediction_worker(args: argparse.Namespace) -> None:
    cache = cast(Path, _required(args.hf_cache, "--hf-cache"))
    protocol_path = cast(Path, _required(args.protocol, "--protocol"))
    output = cast(Path, _required(args.output, "--output"))
    emergency = cast(Path, _required(args.emergency_output, "--emergency-output"))
    config = load_config(args.config)
    git = verify_git(ROOT)
    protocol = _dict(protocol_path, "protocol manifest")
    validate_protocol(ROOT, protocol, config)
    if protocol["protocol_commit"] != git["head"]:
        raise RuntimeError("prediction must run at pushed Stage 1G protocol commit")
    model: Any = None
    sampler: Any = None
    finished = False
    try:
        model, torch, sampler, environment, runtime_evidence = _load_runtime(
            cache, config, emergency
        )
        sensitivity = run_output_sensitivity_validation(model, torch, sampler, config)
        validate_output_sensitivity(sensitivity, config)
        prediction = build_prediction_manifest(
            model,
            torch,
            sampler,
            config,
            protocol=protocol,
            sensitivity_validation=sensitivity,
            git=git,
        )
        telemetry = sampler.finish()
        finished = True
        if telemetry["violations"] or telemetry["telemetry_failures"]:
            raise RuntimeError(
                "Stage 1G prediction telemetry contains a safety failure"
            )
        write_json_new(
            output,
            {
                "schema_version": 1,
                "artifact_type": "stage1g_prediction_worker",
                "status": "passed",
                "output_sensitivity_validation": sensitivity,
                "prediction_manifest": prediction,
                "git": git,
                "environment": environment,
                "runtime_evidence": runtime_evidence,
                "telemetry": telemetry,
                "fresh_scientific_intervention_api_calls": 0,
            },
        )
    finally:
        if sampler is not None:
            _finish_sampler(sampler, already_finished=finished)
        if model is not None:
            del model
        gc.collect()
        if "torch" in locals():
            with contextlib.suppress(Exception):
                torch.mps.empty_cache()


def _assemble_prediction(args: argparse.Namespace) -> None:
    protocol = _dict(cast(Path, _required(args.protocol, "--protocol")), "protocol")
    worker = _dict(cast(Path, _required(args.worker, "--worker")), "worker")
    prediction_output = cast(Path, _required(args.output, "--output"))
    sensitivity_output = cast(
        Path, _required(args.sensitivity_output, "--sensitivity-output")
    )
    config = load_config(args.config)
    git = verify_git(ROOT)
    validate_protocol(ROOT, protocol, config)
    if (
        worker.get("artifact_type") != "stage1g_prediction_worker"
        or worker.get("status") != "passed"
    ):
        raise RuntimeError("Stage 1G prediction worker did not pass")
    sensitivity = cast(dict[str, Any], worker["output_sensitivity_validation"])
    prediction = cast(dict[str, Any], worker["prediction_manifest"])
    validate_output_sensitivity(sensitivity, config)
    validate_prediction(prediction, config, protocol)
    if prediction["prediction_execution_commit"] != git["head"]:
        raise RuntimeError("Stage 1G prediction worker commit differs")
    write_json_new(sensitivity_output, sensitivity)
    write_json_new(prediction_output, prediction)
    print(
        json.dumps(
            {
                "status": "passed",
                "prediction_sha256": sha256_file(prediction_output),
                "sensitivity_sha256": sha256_file(sensitivity_output),
            },
            sort_keys=True,
        )
    )


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
            "working Stage 1G prediction differs from tracked HEAD bytes"
        )


def _real_rehearsal(args: argparse.Namespace) -> None:
    cache = cast(Path, _required(args.hf_cache, "--hf-cache"))
    prediction_path = cast(Path, _required(args.prediction, "--prediction"))
    output = cast(Path, _required(args.output, "--output"))
    journal_path = cast(Path, _required(args.journal, "--journal"))
    emergency = cast(Path, _required(args.emergency_output, "--emergency-output"))
    commit = str(_required(args.prediction_freeze_commit, "--prediction-freeze-commit"))
    config = load_config(args.config)
    verify_git(ROOT, expected_head=commit)
    prediction = _dict(prediction_path, "prediction")
    scientific_ids = {
        pair["pair_id"]
        for prompt in prediction["prompts"]
        for pair in prompt["execution_pairs"]
    }
    settings = config["rehearsal"]
    model: Any = None
    sampler: Any = None
    finished = False
    try:
        model, torch, sampler, _, _ = _load_runtime(cache, config, emergency)
        answer_id, answer_error = __import__(
            "cfsus.stage1g", fromlist=["_continuation_token"]
        )._continuation_token(
            model.tokenizer, settings["real_prompt"], settings["real_answer"]
        )
        contrast_id, contrast_error = __import__(
            "cfsus.stage1g", fromlist=["_continuation_token"]
        )._continuation_token(
            model.tokenizer, settings["real_prompt"], settings["real_contrast"]
        )
        if answer_error or contrast_error or answer_id is None or contrast_id is None:
            raise RuntimeError("Stage 1G rehearsal continuation tokens differ")
        token_ids = [
            int(item)
            for item in model.ensure_tokenized(settings["real_prompt"])
            .detach()
            .cpu()
            .tolist()
        ]
        position = len(token_ids) - 1
        backend = Stage1GPredictionBackend(
            model,
            prompt=settings["real_prompt"],
            prompt_id=settings["real_prompt_id"],
            torch=torch,
        )
        active = backend.collect_active_sources(
            groups=tuple((layer, position) for layer in range(18)),
            chunk_size=int(config["scanner"]["canonical_chunk_size"]),
        )
        candidates = [
            (source, target)
            for target in active
            for source in active
            if source.feature.layer < target.feature.layer
            and source.feature.position <= target.feature.position
        ]
        candidates.sort(
            key=lambda item: (
                -item[1].activation,
                item[0].activation,
                item[0].feature,
                item[1].feature,
            )
        )
        if not candidates:
            raise RuntimeError("Stage 1G real rehearsal has no causal active pair")
        source, target = candidates[0]
        pair_id = canonical_v3_pair_id(
            source=source.feature,
            target=target.feature,
            runtime_fingerprint=RUNTIME_FINGERPRINT,
            prompt_id=settings["real_prompt_id"],
            seed="stage1g-real-multiedit-rehearsal-v1",
            experiment_class="stage1g_real_multiedit_rehearsal",
        )
        if pair_id in scientific_ids:
            raise RuntimeError("Stage 1G rehearsal overlaps a scientific pair")
        pair = {
            "pair_id": pair_id,
            "prompt_id": settings["real_prompt_id"],
            "source": {
                "layer": source.feature.layer,
                "position": source.feature.position,
                "feature_id": source.feature.feature_id,
            },
            "target": {
                "layer": target.feature.layer,
                "position": target.feature.position,
                "feature_id": target.feature.feature_id,
            },
            "method_memberships": ["B"],
            "panel_ranks": {"B": 1},
            "answer_token_id": answer_id,
            "contrast_token_id": contrast_id,
        }
        prompt = {
            "id": settings["real_prompt_id"],
            "text": settings["real_prompt"],
            "token_ids": token_ids,
            "answer_token_id": answer_id,
            "contrast_token_id": contrast_id,
            "execution_pairs": [pair],
        }
        with CanonicalExecutionJournal(
            journal_path,
            None,
            frozen_pair_ids=(pair_id,),
            pre_intervention_commit=commit,
            prediction_manifest_sha256=sha256_file(prediction_path),
            experiment_class="stage1g_real_multiedit_rehearsal",
            attempt_boundary="engineering_rehearsal_not_scientific_attempt",
            attempt_lock_artifact_type="stage1g_rehearsal_no_lock",
        ) as journal:
            sweeps, calls = execute_frozen_pairs(
                model=model,
                torch=torch,
                sampler=sampler,
                prompts=[prompt],
                journal=journal,
                allow_active_target=True,
                force_all_conditions=True,
            )
        telemetry = sampler.finish()
        finished = True
        if telemetry["violations"] or telemetry["telemetry_failures"]:
            raise RuntimeError("Stage 1G rehearsal telemetry contains a safety failure")
        write_json_new(
            output,
            {
                "schema_version": 1,
                "artifact_type": "stage1g_real_multiedit_rehearsal",
                "status": "passed",
                "scientific_attempt_consumed": False,
                "scientific_pair_overlap": False,
                "pair_id": pair_id,
                "call_count": calls,
                "sweeps": sweeps,
                "journal_sha256": sha256_file(journal_path),
                "telemetry": telemetry,
            },
        )
    finally:
        if sampler is not None:
            _finish_sampler(sampler, already_finished=finished)
        if model is not None:
            del model
        gc.collect()
        if "torch" in locals():
            with contextlib.suppress(Exception):
                torch.mps.empty_cache()


def _validate_real_rehearsal(args: argparse.Namespace) -> None:
    output = _dict(cast(Path, _required(args.output, "--output")), "rehearsal")
    journal_path = cast(Path, _required(args.journal, "--journal"))
    points = read_completed_journal(journal_path)
    conditions = {point["condition"] for point in points}
    if (
        output.get("artifact_type") != "stage1g_real_multiedit_rehearsal"
        or output.get("status") != "passed"
        or output.get("scientific_attempt_consumed") is not False
        or output.get("scientific_pair_overlap") is not False
        or output.get("call_count") != len(points)
        or len(points) != 5
        or output.get("journal_sha256") != sha256_file(journal_path)
        or conditions
        != {
            "baseline_noop",
            "baseline_repeat",
            "source_full_ablation",
            "source_ablation_target_clamp",
            "target_only_injection",
        }
    ):
        raise RuntimeError("Stage 1G real rehearsal evidence differs")
    print(json.dumps({"status": "passed", "call_count": len(points)}, sort_keys=True))


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
    config = load_config(args.config)
    git = verify_git(ROOT, expected_head=prediction_commit)
    protocol = _dict(protocol_path, "protocol")
    prediction = _dict(prediction_path, "prediction")
    validate_protocol(ROOT, protocol, config)
    pairs = validate_prediction(prediction, config, protocol)
    _verify_tracked_prediction(prediction_path)
    if prediction["status"] != "prediction_frozen_ready_for_commit" or not pairs:
        raise RuntimeError("Stage 1G canonical panel is not intervention-ready")
    prediction_sha = sha256_file(prediction_path)
    if journal_path.exists() or attempt_lock.exists() or output.exists():
        raise RuntimeError("Stage 1G canonical output, journal, and lock must be new")
    model: Any = None
    sampler: Any = None
    finished = False
    with CanonicalExecutionJournal(
        journal_path,
        attempt_lock,
        frozen_pair_ids=tuple(pairs),
        pre_intervention_commit=prediction_commit,
        prediction_manifest_sha256=prediction_sha,
        experiment_class=EXPERIMENT_CLASS,
        attempt_boundary=config["intervention"]["attempt_boundary"],
        attempt_lock_artifact_type="stage1g_local_scientific_attempt_lock",
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
            )
            telemetry = sampler.finish()
            finished = True
            if telemetry["violations"] or telemetry["telemetry_failures"]:
                raise RuntimeError(
                    "Stage 1G canonical telemetry contains a safety failure"
                )
            write_json_new(
                output,
                {
                    "schema_version": 1,
                    "artifact_type": "stage1g_intervention_worker",
                    "status": "passed",
                    "prediction_manifest_sha256": prediction_sha,
                    "prediction_freeze_commit": prediction_commit,
                    "pre_run_commit": prediction_commit,
                    "canonical_attempt_count": 1,
                    "scientific_retry_count": 0,
                    "instrumented_intervention_api_calls": calls,
                    "completed_in_memory_sweep_count": len(sweeps),
                    "in_memory_sweeps_publication_trusted": False,
                    "journal_sha256": sha256_file(journal_path),
                    "environment": environment,
                    "runtime_evidence": runtime_evidence,
                    "telemetry": telemetry,
                    "git": git,
                },
            )
        finally:
            if sampler is not None:
                _finish_sampler(sampler, already_finished=finished)
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
    sensitivity = _dict(
        cast(Path, _required(args.sensitivity_output, "--sensitivity-output")),
        "sensitivity",
    )
    worker = _dict(cast(Path, _required(args.worker, "--worker")), "worker")
    journal = cast(Path, _required(args.journal, "--journal"))
    output = cast(Path, _required(args.output_dir, "--output-dir"))
    config = load_config(args.config)
    validate_protocol(ROOT, protocol, config)
    validate_output_sensitivity(sensitivity, config)
    validate_prediction(prediction, config, protocol)
    if (
        worker.get("artifact_type") != "stage1g_intervention_worker"
        or worker.get("status") != "passed"
    ):
        raise RuntimeError("Stage 1G intervention worker did not pass")
    if worker.get("journal_sha256") != sha256_file(journal):
        raise RuntimeError("Stage 1G journal digest differs from worker evidence")
    points = read_completed_journal(journal)
    if worker.get("instrumented_intervention_api_calls") != len(points):
        raise RuntimeError("Stage 1G worker call count differs from journal")
    records = build_records(
        protocol=protocol,
        sensitivity=sensitivity,
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
        verify_git(ROOT)
        value = build_protocol_manifest(
            ROOT,
            protocol_commit=str(_required(args.protocol_commit, "--protocol-commit")),
        )
        write_json_new(cast(Path, _required(args.output, "--output")), value)
    elif args.action == "prediction-worker":
        _prediction_worker(args)
    elif args.action == "assemble-prediction":
        _assemble_prediction(args)
    elif args.action == "validate-prediction":
        config = load_config(args.config)
        protocol = _dict(cast(Path, _required(args.protocol, "--protocol")), "protocol")
        prediction = _dict(
            cast(Path, _required(args.prediction, "--prediction")), "prediction"
        )
        validate_protocol(ROOT, protocol, config)
        pairs = validate_prediction(prediction, config, protocol)
        print(
            json.dumps({"status": "passed", "pair_count": len(pairs)}, sort_keys=True)
        )
    elif args.action == "real-rehearsal":
        _real_rehearsal(args)
    elif args.action == "validate-real-rehearsal":
        _validate_real_rehearsal(args)
    elif args.action == "intervention-worker":
        _intervention_worker(args)
    elif args.action == "assemble":
        _assemble(args)
    elif args.action == "validate":
        print(
            json.dumps(
                validate_bundle(
                    ROOT, cast(Path, _required(args.output_dir, "--output-dir"))
                ),
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
