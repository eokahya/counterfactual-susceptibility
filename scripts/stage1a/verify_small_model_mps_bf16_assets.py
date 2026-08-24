#!/usr/bin/env python3
"""Verify existing immutable Stage 1A-S-BF16 snapshots without downloading."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cfsus.reproduction.artifacts import write_json_atomic  # noqa: E402
from cfsus.reproduction.small_model_mps_bf16 import (  # noqa: E402
    CONFIG_PATH,
    LAYER_COUNT,
    MODEL_IDENTIFIER,
    MODEL_REVISION,
    PROJECTED_MANIFEST,
    TRANSCODER_IDENTIFIER,
    TRANSCODER_REVISION,
    TRANSCODER_SUBFOLDER,
    assert_fallback_disabled,
    load_bf16_config,
    validate_projected_manifest,
    validate_snapshot_tree,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / CONFIG_PATH)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def _repo_directory(identifier: str) -> str:
    return "models--" + identifier.replace("/", "--")


def _snapshot(cache: Path, identifier: str, revision: str) -> Path:
    candidate = cache / _repo_directory(identifier) / "snapshots" / revision
    if candidate.name != revision or not candidate.is_dir() or candidate.is_symlink():
        raise RuntimeError("exact immutable snapshot is missing or unsafe")
    return candidate


def _expected_hashes(files: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(item["path"]): str(item["lfs_sha256"])
        for item in files
        if "lfs_sha256" in item
    }


def _verify_lfs_hashes(
    records: list[dict[str, Any]], expected: dict[str, str], label: str
) -> None:
    observed = {str(item["path"]): str(item["sha256"]) for item in records}
    for path, digest in expected.items():
        if observed.get(path) != digest:
            raise RuntimeError(f"{label} LFS SHA mismatch for {path}")


def _header_summary(model_snapshot: Path, transcoder_snapshot: Path) -> dict[str, Any]:
    from safetensors import safe_open

    model_counts: Counter[str] = Counter()
    model_file = model_snapshot / "model.safetensors"
    with safe_open(model_file, framework="pt", device="cpu") as handle:
        for key in handle.keys():  # noqa: SIM118 - safe_open is not iterable
            model_counts[str(handle.get_slice(key).get_dtype())] += 1
    if model_counts != Counter({"BF16": 236}):
        raise RuntimeError(f"model safetensors dtype schema changed: {model_counts}")

    expected_keys = {
        "W_dec",
        "W_enc",
        "activation_function.threshold",
        "b_dec",
        "b_enc",
    }
    plt_counts: Counter[str] = Counter()
    per_layer: list[dict[str, Any]] = []
    root = transcoder_snapshot / TRANSCODER_SUBFOLDER
    for layer in range(LAYER_COUNT):
        path = root / f"layer_{layer}.safetensors"
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if keys != expected_keys:
                raise RuntimeError(f"PLT layer {layer} tensor keys changed")
            dtypes = {key: str(handle.get_slice(key).get_dtype()) for key in keys}
            shapes = {key: list(handle.get_slice(key).get_shape()) for key in keys}
        if set(dtypes.values()) != {"F32"}:
            raise RuntimeError(f"PLT layer {layer} is not all-FP32 on disk")
        plt_counts.update(dtypes.values())
        per_layer.append(
            {
                "layer": layer,
                "tensor_count": len(keys),
                "storage_dtype": "F32",
                "encoder_shape": shapes["W_enc"],
                "decoder_shape": shapes["W_dec"],
                "threshold_shape": shapes["activation_function.threshold"],
            }
        )
    return {
        "model": {"tensor_count": 236, "storage_dtype_counts": dict(model_counts)},
        "transcoder": {
            "layer_count": LAYER_COUNT,
            "tensor_count": sum(plt_counts.values()),
            "storage_dtype_counts": dict(plt_counts),
            "runtime_conversion_target": "torch.bfloat16",
            "layers": per_layer,
        },
    }


def main() -> int:
    arguments = _parser().parse_args()
    assert_fallback_disabled()
    config = load_bf16_config(arguments.config)
    cache = arguments.hf_cache
    if not cache.is_absolute() or cache.is_symlink() or not cache.is_dir():
        raise RuntimeError("Hugging Face cache must be an existing absolute directory")
    cache = cache.resolve(strict=True)
    if cache == REPOSITORY_ROOT or cache.is_relative_to(REPOSITORY_ROOT):
        raise RuntimeError("asset cache must remain project-external")
    with (REPOSITORY_ROOT / PROJECTED_MANIFEST).open(encoding="utf-8") as stream:
        projected = validate_projected_manifest(json.load(stream))

    model_snapshot = _snapshot(cache, MODEL_IDENTIFIER, MODEL_REVISION)
    transcoder_snapshot = _snapshot(cache, TRANSCODER_IDENTIFIER, TRANSCODER_REVISION)
    model_expected = set(config["model"]["allow_patterns"])
    transcoder_expected = set(config["transcoder"]["allow_patterns"])
    model_records = validate_snapshot_tree(
        snapshot=model_snapshot, cache_root=cache, expected_paths=model_expected
    )
    transcoder_records = validate_snapshot_tree(
        snapshot=transcoder_snapshot,
        cache_root=cache,
        expected_paths=transcoder_expected,
    )
    _verify_lfs_hashes(
        model_records,
        _expected_hashes(projected["model"]["files"]),
        "model",
    )
    _verify_lfs_hashes(
        transcoder_records,
        _expected_hashes(projected["transcoder"]["files"]),
        "transcoder",
    )
    model_total = sum(int(item["bytes"]) for item in model_records)
    transcoder_total = sum(int(item["bytes"]) for item in transcoder_records)
    combined = model_total + transcoder_total
    if model_total != 575_454_257:
        raise RuntimeError("model snapshot byte total changed")
    if transcoder_total != 1_512_362_420:
        raise RuntimeError("transcoder snapshot byte total changed")
    if combined != 2_087_816_677 or combined != projected["projected_total_bytes"]:
        raise RuntimeError("combined immutable asset bytes changed")

    model_config = json.loads(
        (model_snapshot / "config.json").read_text(encoding="utf-8")
    )
    if (
        model_config.get("torch_dtype") != "bfloat16"
        or model_config.get("num_hidden_layers") != 18
        or model_config.get("hidden_size") != 640
        or model_config.get("vocab_size") != 262_144
    ):
        raise RuntimeError("model config identity changed")
    record = {
        "schema_version": 1,
        "artifact_type": "stage1a_small_model_mps_bf16_asset_manifest",
        "status": "verified",
        "download_performed": False,
        "network_accessed": False,
        "authentication_used": False,
        "authentication_value_recorded": False,
        "cache_location": "project_external_hugging_face_cache",
        "model": {
            "identifier": MODEL_IDENTIFIER,
            "revision": MODEL_REVISION,
            "consumed_path_identity": (
                f"{_repo_directory(MODEL_IDENTIFIER)}/snapshots/{MODEL_REVISION}"
            ),
            "files": model_records,
            "total_bytes": model_total,
        },
        "transcoder": {
            "identifier": TRANSCODER_IDENTIFIER,
            "revision": TRANSCODER_REVISION,
            "subfolder": TRANSCODER_SUBFOLDER,
            "consumed_path_identity": (
                f"{_repo_directory(TRANSCODER_IDENTIFIER)}/snapshots/"
                f"{TRANSCODER_REVISION}"
            ),
            "files": transcoder_records,
            "total_bytes": transcoder_total,
        },
        "actual_total_bytes": combined,
        "projected_total_bytes": projected["projected_total_bytes"],
        "safetensors_headers": _header_summary(model_snapshot, transcoder_snapshot),
        "full_repository_downloaded": False,
        "other_widths_consumed": False,
        "feature_visualization_consumed": False,
    }
    if arguments.output is not None:
        output = arguments.output
        if not output.is_absolute():
            output = REPOSITORY_ROOT / output
        generated_root = (
            REPOSITORY_ROOT / config["artifacts"]["generated_directory"]
        ).resolve()
        if not output.parent.resolve().is_relative_to(generated_root):
            raise RuntimeError("asset output must remain under generated directory")
        write_json_atomic(output, record)
    print(json.dumps(record, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
