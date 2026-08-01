#!/usr/bin/env python3
"""Run the complete pinned Stage 1A reproduction with fail-closed prechecks."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from preflight import collect_report  # noqa: E402
from reproduce_attribution import (  # noqa: E402
    RuntimeBundle,
    Stage1ABlocked,
    _mapping,
    load_runtime,
    load_yaml,
    repository_root,
    reproduce_attribution,
    validate_official_config,
)
from reproduce_intervention import reproduce_intervention  # noqa: E402
from resolve_assets import ResolutionError, resolve_assets  # noqa: E402
from validate_artifacts import (  # noqa: E402
    regenerate_checksums,
    validate_present_artifacts,
    verify_checksums,
)
from verify_runtime_semantics import verify_runtime_semantics  # noqa: E402

MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
TRANSCODER_REVISION = "bd5773156dea09893636c801df1237d0410307d2"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
FAILURE_DIAGNOSTIC = (
    repository_root() / "results/generated/stage1a/failure_diagnostic.json"
)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise Stage1ABlocked("asset manifest contains a duplicate JSON key")
        value[key] = item
    return value


def _validate_rich_asset_manifest(path: Path) -> None:
    """Validate and preserve the tracked exact-file metadata for offline use."""

    from cfsus.reproduction.artifacts import (
        ArtifactValidationError,
        validate_artifact_envelope,
    )

    if path.is_symlink() or not path.is_file():
        raise Stage1ABlocked("the rich asset manifest is missing or is a symlink")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object
        )
        validate_artifact_envelope(value, expected_type="asset_manifest")
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ArtifactValidationError,
    ) as exc:
        raise Stage1ABlocked(
            f"the rich asset manifest is invalid: {type(exc).__name__}"
        ) from exc
    if not isinstance(value, dict):
        raise Stage1ABlocked("the rich asset manifest must be a JSON object")
    provenance = value.get("provenance")
    payload = value.get("payload")
    if not isinstance(provenance, dict) or not isinstance(payload, dict):
        raise Stage1ABlocked("the rich asset manifest has invalid metadata sections")
    expected_pins = {
        "model_repo_id": "google/gemma-2-2b",
        "model_revision": MODEL_REVISION,
        "transcoder_repo_id": "mwhanna/gemma-scope-transcoders",
        "transcoder_revision": TRANSCODER_REVISION,
    }
    if any(provenance.get(key) != item for key, item in expected_pins.items()):
        raise Stage1ABlocked("the rich asset manifest has different immutable pins")

    assets = payload.get("assets")
    if not isinstance(assets, dict):
        raise Stage1ABlocked("the rich asset manifest has no exact-file asset metadata")
    requirements = {
        "model": (
            "google/gemma-2-2b",
            MODEL_REVISION,
            {
                "config.json",
                "model-00001-of-00003.safetensors",
                "model-00002-of-00003.safetensors",
                "model-00003-of-00003.safetensors",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
            },
        ),
        "selected_transcoder": (
            "mwhanna/gemma-scope-transcoders",
            TRANSCODER_REVISION,
            {"config.yaml", *(f"layer_{layer}.safetensors" for layer in range(26))},
        ),
    }
    for key, (repo_id, revision, required_names) in requirements.items():
        asset = assets.get(key)
        if (
            not isinstance(asset, dict)
            or asset.get("repo_id") != repo_id
            or asset.get("verified_revision") != revision
        ):
            raise Stage1ABlocked(f"rich metadata for {key} has a pin mismatch")
        files = asset.get("files")
        if not isinstance(files, list):
            raise Stage1ABlocked(f"rich metadata for {key} has no file inventory")
        by_name = {item.get("name"): item for item in files if isinstance(item, dict)}
        if not required_names.issubset(by_name):
            raise Stage1ABlocked(f"rich metadata for {key} omits required files")
        for name in required_names:
            record = by_name[name]
            size = record.get("size_bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise Stage1ABlocked(f"rich metadata for {key} has an invalid size")
            if (
                name.endswith(".safetensors")
                and SHA256_PATTERN.fullmatch(str(record.get("lfs_sha256"))) is None
            ):
                raise Stage1ABlocked(
                    f"rich metadata for {key} has an invalid weight hash"
                )


def _write_metadata_artifacts(
    config: dict[str, Any],
    *,
    allow_download: bool,
    model_snapshot: Path | None,
    transcoder_snapshot: Path | None,
) -> None:
    """Resolve prerequisites and emit only sanitized metadata artifacts."""

    from cfsus.reproduction.artifacts import write_json_atomic

    artifacts = _mapping(config.get("artifacts"), "artifacts")
    preflight_report = collect_report()
    write_json_atomic(
        repository_root() / str(artifacts["environment_manifest"]),
        preflight_report,
    )

    offline_overrides = (
        model_snapshot is not None
        and transcoder_snapshot is not None
        and not allow_download
    )
    if offline_overrides:
        assert model_snapshot is not None
        assert transcoder_snapshot is not None
        if not model_snapshot.is_dir() or not transcoder_snapshot.is_dir():
            raise Stage1ABlocked("an explicit immutable snapshot directory is missing")
        _validate_rich_asset_manifest(
            repository_root() / str(artifacts["asset_manifest"])
        )
        return

    requested = ("model", "transcoder") if allow_download else ()
    try:
        asset_report = resolve_assets(requested)
    except ResolutionError as exc:
        raise Stage1ABlocked(
            f"immutable asset resolution failed: {type(exc).__name__}"
        ) from exc
    write_json_atomic(
        repository_root() / str(artifacts["asset_manifest"]), asset_report
    )
    if asset_report["status"] not in {"resolved", "completed"}:
        resolution_status = asset_report["payload"].get("resolution_status")
        raise Stage1ABlocked(
            f"immutable assets are unavailable ({resolution_status}); "
            "no scientific run started"
        )


def _release_runtime(bundle: RuntimeBundle | None) -> bool:
    """Release model references and accelerator caches without masking failures."""

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
    if torch is None:
        return succeeded
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        succeeded = False
    try:
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache"):
            mps.empty_cache()
    except Exception:
        succeeded = False
    return succeeded


def _failure_fields(error: Exception, stage: str) -> tuple[dict[str, str], int]:
    chain: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and all(current is not item for item in chain):
        chain.append(current)
        current = current.__cause__ or current.__context__
    out_of_memory = any(
        isinstance(item, MemoryError)
        or type(item).__name__ in {"OutOfMemoryError", "OutOfMemoryException"}
        for item in chain
    )
    if out_of_memory:
        category = "out_of_memory"
        exception_type = "OutOfMemoryError"
        status = "failed"
        exit_code = 1
    elif isinstance(error, Stage1ABlocked):
        category = "prerequisite_blocked"
        exception_type = "Stage1ABlocked"
        status = "blocked"
        exit_code = 2
    elif isinstance(error, RuntimeError):
        category = "runtime_failure"
        exception_type = "RuntimeError"
        status = "failed"
        exit_code = 1
    else:
        category = "unexpected_failure"
        exception_type = "UnexpectedError"
        status = "failed"
        exit_code = 1
    return {
        "category": category,
        "exception_type": exception_type,
        "failure_stage": stage,
        "status": status,
    }, exit_code


def _write_failure_diagnostic(
    fields: dict[str, str], *, cleanup_succeeded: bool
) -> bool:
    """Atomically preserve a publication-safe diagnostic without error text."""

    from cfsus.reproduction.artifacts import (
        make_artifact_envelope,
        write_json_atomic,
    )

    record = make_artifact_envelope(
        artifact_type="failure_diagnostic",
        run_id="stage1a-orchestrator-failure",
        status=fields["status"],
        provenance={
            "model_revision": MODEL_REVISION,
            "runner": "scripts/stage1a/run_stage1a.py",
            "transcoder_revision": TRANSCODER_REVISION,
        },
        payload={
            "category": fields["category"],
            "cleanup_attempted": True,
            "cleanup_succeeded": cleanup_succeeded,
            "exception_type": fields["exception_type"],
            "failure_stage": fields["failure_stage"],
            "message_and_traceback_omitted": True,
        },
    )
    try:
        write_json_atomic(FAILURE_DIAGNOSTIC, record)
    except Exception:
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=repository_root()
        / "configs/stage1a_gemma2_2b_official_reproduction.yaml",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit exact-revision asset downloads during resolution only.",
    )
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--transcoder-snapshot", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle: RuntimeBundle | None = None
    failure: tuple[dict[str, str], int] | None = None
    stage = "configuration_validation"
    try:
        config = load_yaml(args.config.resolve())
        validate_official_config(config)
        if (args.model_snapshot is None) != (args.transcoder_snapshot is None):
            raise Stage1ABlocked(
                "model and transcoder snapshot overrides must be supplied together"
            )
        if args.allow_download and args.model_snapshot is not None:
            raise Stage1ABlocked(
                "--allow-download cannot be combined with explicit snapshot overrides"
            )
        stage = "metadata_resolution"
        _write_metadata_artifacts(
            config,
            allow_download=args.allow_download,
            model_snapshot=args.model_snapshot,
            transcoder_snapshot=args.transcoder_snapshot,
        )
        stage = "runtime_loading"
        bundle = load_runtime(
            config,
            allow_download=args.allow_download,
            model_snapshot=args.model_snapshot,
            transcoder_snapshot=args.transcoder_snapshot,
        )
        from cfsus.reproduction.artifacts import write_json_atomic

        stage = "environment_observation"
        artifacts = _mapping(config.get("artifacts"), "artifacts")
        write_json_atomic(
            repository_root() / str(artifacts["environment_manifest"]),
            collect_report(
                model_snapshot_present=True,
                transcoder_snapshot_present=True,
            ),
        )
        stage = "runtime_semantics"
        verify_runtime_semantics(bundle)
        stage = "official_intervention"
        reproduce_intervention(bundle)
        stage = "official_attribution"
        reproduce_attribution(bundle)
        artifact_directory = repository_root() / "results/stage1a"
        checksum_path = repository_root() / str(artifacts["checksums"])
        stage = "artifact_validation"
        validate_present_artifacts(artifact_directory, strict_payloads=True)
        stage = "checksum_regeneration"
        regenerate_checksums(artifact_directory, checksum_path)
        stage = "checksum_verification"
        verify_checksums(artifact_directory, checksum_path)
    except Exception as exc:
        failure = _failure_fields(exc, stage)
    finally:
        cleanup_succeeded = _release_runtime(bundle)
    if failure is not None:
        fields, exit_code = failure
        diagnostic_written = _write_failure_diagnostic(
            fields,
            cleanup_succeeded=cleanup_succeeded,
        )
        label = "BLOCKED" if fields["status"] == "blocked" else "FAILED"
        diagnostic_status = "written" if diagnostic_written else "write_failed"
        print(
            f"{label}: {fields['exception_type']} during {fields['failure_stage']}; "
            f"sanitized diagnostic {diagnostic_status}",
            file=sys.stderr,
        )
        return exit_code
    print("Stage 1A pinned attribution, intervention, and semantics run completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
