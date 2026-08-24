#!/usr/bin/env python3
"""Download only the immutable Stage 1A-S model and selected PLT subset."""

from __future__ import annotations

import argparse
import json
import stat
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cfsus.reproduction.artifacts import sha256_file, write_json_atomic  # noqa: E402
from cfsus.reproduction.small_model_mps_fp16 import (  # noqa: E402
    MODEL_IDENTIFIER,
    MODEL_REVISION,
    PROJECTED_MANIFEST,
    TRANSCODER_IDENTIFIER,
    TRANSCODER_REVISION,
    assert_fallback_disabled,
    load_small_model_config,
    validate_projected_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/stage1a_small_model_mps_fp16_pilot.yaml",
    )
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def _validate_external_cache(path: Path) -> Path:
    if not path.is_absolute():
        raise RuntimeError("Hugging Face cache must be an absolute path")
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError("Hugging Face cache is not a safe directory")
    resolved = path.resolve(strict=True)
    if resolved == REPOSITORY_ROOT or resolved.is_relative_to(REPOSITORY_ROOT):
        raise RuntimeError("Hugging Face cache must remain outside the repository")
    return resolved


def _regular_or_snapshot_symlink(path: Path) -> Path:
    if path.is_symlink():
        target = path.resolve(strict=True)
        if not target.is_file():
            raise RuntimeError("snapshot symlink target is not a regular file")
        return target
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("snapshot contains a special file")
    return path


def _verify_snapshot(
    snapshot: Path, expected_paths: set[str], expected_revision: str
) -> list[dict[str, Any]]:
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise RuntimeError("downloaded snapshot is missing or unsafe")
    if snapshot.name != expected_revision:
        raise RuntimeError("downloaded snapshot directory is not the exact revision")
    observed: set[str] = set()
    files: list[dict[str, Any]] = []
    for candidate in snapshot.rglob("*"):
        if candidate.is_dir():
            continue
        relative = candidate.relative_to(snapshot).as_posix()
        target = _regular_or_snapshot_symlink(candidate)
        observed.add(relative)
        files.append(
            {
                "path": relative,
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    if observed != expected_paths:
        raise RuntimeError(
            "snapshot allowlist mismatch: "
            f"missing={sorted(expected_paths - observed)}, "
            f"extra={sorted(observed - expected_paths)}"
        )
    return sorted(files, key=lambda item: str(item["path"]))


def main() -> int:
    arguments = _parser().parse_args()
    if not arguments.allow_download:
        raise RuntimeError("asset download requires explicit --allow-download")
    assert_fallback_disabled()
    config = load_small_model_config(arguments.config)
    with (REPOSITORY_ROOT / PROJECTED_MANIFEST).open(encoding="utf-8") as stream:
        projected = validate_projected_manifest(json.load(stream))
    cache = _validate_external_cache(arguments.hf_cache)

    from huggingface_hub import (  # type: ignore[import-not-found]
        HfApi,
        snapshot_download,
    )

    api = HfApi()
    if api.model_info(MODEL_IDENTIFIER, revision=MODEL_REVISION).sha != MODEL_REVISION:
        raise RuntimeError("model immutable metadata could not be revalidated")
    if (
        api.model_info(TRANSCODER_IDENTIFIER, revision=TRANSCODER_REVISION).sha
        != TRANSCODER_REVISION
    ):
        raise RuntimeError("transcoder immutable metadata could not be revalidated")

    model_patterns = list(config["model"]["allow_patterns"])
    transcoder_patterns = list(config["transcoder"]["allow_patterns"])
    model_snapshot = Path(
        snapshot_download(
            MODEL_IDENTIFIER,
            revision=MODEL_REVISION,
            allow_patterns=model_patterns,
            cache_dir=cache,
            token=True,
        )
    )
    transcoder_snapshot = Path(
        snapshot_download(
            TRANSCODER_IDENTIFIER,
            revision=TRANSCODER_REVISION,
            allow_patterns=transcoder_patterns,
            cache_dir=cache,
        )
    )
    model_files = _verify_snapshot(model_snapshot, set(model_patterns), MODEL_REVISION)
    transcoder_files = _verify_snapshot(
        transcoder_snapshot, set(transcoder_patterns), TRANSCODER_REVISION
    )
    actual_total = sum(int(item["bytes"]) for item in [*model_files, *transcoder_files])
    if actual_total != projected["projected_total_bytes"]:
        raise RuntimeError("actual asset bytes do not match the projected manifest")
    record = {
        "schema_version": 1,
        "artifact_type": "stage1a_small_model_mps_downloaded_assets",
        "status": "verified",
        "cache_location": "project_external_hugging_face_cache",
        "model": {
            "identifier": MODEL_IDENTIFIER,
            "revision": MODEL_REVISION,
            "files": model_files,
            "total_bytes": sum(int(item["bytes"]) for item in model_files),
        },
        "transcoder": {
            "identifier": TRANSCODER_IDENTIFIER,
            "revision": TRANSCODER_REVISION,
            "subfolder": config["transcoder"]["subfolder"],
            "files": transcoder_files,
            "total_bytes": sum(int(item["bytes"]) for item in transcoder_files),
        },
        "actual_total_bytes": actual_total,
        "projected_total_bytes": projected["projected_total_bytes"],
        "authentication_used": True,
        "authentication_value_recorded": False,
        "full_repository_downloaded": False,
    }
    output = arguments.output
    if output is not None:
        if not output.is_absolute():
            output = REPOSITORY_ROOT / output
        generated = (
            REPOSITORY_ROOT / config["artifacts"]["generated_directory"]
        ).resolve()
        if not output.parent.resolve().is_relative_to(generated):
            raise RuntimeError(
                "asset manifest must remain under the generated directory"
            )
        write_json_atomic(output, record)
    print(json.dumps(record, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
