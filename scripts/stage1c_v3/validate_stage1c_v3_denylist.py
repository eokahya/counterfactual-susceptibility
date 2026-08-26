#!/usr/bin/env python3
"""Independent validator for the authenticated Stage 1C-v3 exact-pair denylist."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn

ROOT = Path(__file__).resolve().parents[2]
SOURCE_RELATIVE = Path(
    "results/stage1c_first_prospective_prediction/prediction_manifest.json"
)
DENYLIST_RELATIVE = Path("configs/stage1c_v3_v1_exact_pair_denylist.json")
SOURCE_SHA256 = "43cf17f3f87ff97f9fa2aa6b827c84416add5dced2824b69c057d99a5f2b882a"
SOURCE_BLOB_SHA1 = "847e9c3389097b529c0ac2861b3d519afe18d050"
SOURCE_FREEZE_COMMIT = "6ec950d93fe1215fdcfee68c87e1f58a23a78ae8"
DENYLIST_SHA256 = "ee31e29e2eb2be5aa5cbf72b95d75ea275098592eb54106f20aa7b3ba87405ad"
PAIR_COUNT = 28
FORBIDDEN_OUTCOME_KEYS = frozenset(
    {
        "actual_bf16_value_passed",
        "intervention_sweeps",
        "observed_alpha",
        "observed_crossing",
        "observed_outcome",
        "realized_suppression",
        "scientific_outcome",
        "target_activation_after_intervention",
    }
)


class ValidationError(ValueError):
    pass


def fail(message: str) -> NoReturn:
    raise ValidationError(message)


def _strict_json(data: bytes, label: str) -> Any:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValidationError(f"{label} is not strict UTF-8") from error

    def reject_constant(value: str) -> Any:
        fail(f"{label} contains non-finite {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                fail(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValidationError(f"{label} is invalid JSON") from error


def _regular_bytes(path: Path, label: str, maximum: int) -> bytes:
    try:
        info = path.lstat()
    except OSError as error:
        raise ValidationError(f"{label} is unreadable") from error
    if (
        path.is_symlink()
        or not path.is_file()
        or info.st_nlink != 1
        or not 0 < info.st_size <= maximum
    ):
        fail(f"{label} is not a bounded single-link regular file")
    return path.read_bytes()


def _walk_no_outcomes(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                fail(f"non-string key at {path}")
            normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
            if normalized in FORBIDDEN_OUTCOME_KEYS:
                fail(f"historical outcome field is forbidden: {path}.{key}")
            _walk_no_outcomes(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _walk_no_outcomes(item, f"{path}[{index}]")


def _feature(value: Any, label: str) -> tuple[int, int, int]:
    if not isinstance(value, dict) or set(value) != {
        "layer",
        "position",
        "feature_id",
    }:
        fail(f"{label} is not a strict FeatureRef")
    raw = value["layer"], value["position"], value["feature_id"]
    if any(isinstance(item, bool) or not isinstance(item, int) for item in raw):
        fail(f"{label} is not integer-valued")
    layer, position, feature_id = raw
    if not 0 <= layer < 18 or position < 0 or not 0 <= feature_id < 16_384:
        fail(f"{label} is out of range")
    return raw


def extract_pairs(manifest: Any) -> list[dict[str, dict[str, int]]]:
    if not isinstance(manifest, dict):
        fail("historical prediction manifest is not an object")
    _walk_no_outcomes(manifest)
    identity = {
        "schema_version": 1,
        "artifact_type": "stage1c_prediction_manifest",
        "experiment_class": "stage1c_first_prospective_prediction",
        "branch": "stage-1c-first-prospective-prediction",
        "base_commit": "efbf70a7e462e640a0e1819a93f3b92727bbd193",
    }
    if any(manifest.get(key) != value for key, value in identity.items()):
        fail("historical prediction identity differs")
    guards = manifest.get("prediction_only_guards")
    if (
        not isinstance(guards, dict)
        or guards.get("source_suppression_api_calls") != 0
        or guards.get("prior_inactive_target_outcome_read") is not False
    ):
        fail("historical prediction is not baseline-only")
    groups = manifest.get("selected_groups")
    counts = {"primary": 12, "near_boundary": 8, "directional": 8}
    if not isinstance(groups, dict) or set(groups) != set(counts):
        fail("historical selected groups differ")
    keys: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    for group in ("primary", "near_boundary", "directional"):
        rows = groups[group]
        if not isinstance(rows, list) or len(rows) != counts[group]:
            fail(f"historical {group} count differs")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                fail(f"historical {group}[{index}] is not an object")
            keys.append(
                (
                    _feature(row.get("source"), f"{group}[{index}].source"),
                    _feature(row.get("target"), f"{group}[{index}].target"),
                )
            )
    keys.sort()
    if len(keys) != PAIR_COUNT or len(set(keys)) != PAIR_COUNT:
        fail("historical exact-pair count or uniqueness differs")
    return [
        {
            "source": {
                "layer": source[0],
                "position": source[1],
                "feature_id": source[2],
            },
            "target": {
                "layer": target[0],
                "position": target[1],
                "feature_id": target[2],
            },
        }
        for source, target in keys
    ]


def validate_values(source: bytes, denylist: bytes) -> dict[str, Any]:
    source_sha = hashlib.sha256(source).hexdigest()
    source_blob = hashlib.sha1(
        f"blob {len(source)}\0".encode("ascii") + source
    ).hexdigest()
    if source_sha != SOURCE_SHA256 or source_blob != SOURCE_BLOB_SHA1:
        fail("historical source authentication failed")
    denylist_sha = hashlib.sha256(denylist).hexdigest()
    if denylist_sha != DENYLIST_SHA256:
        fail("sanitized denylist SHA-256 differs")
    manifest = _strict_json(source, "historical prediction manifest")
    value = _strict_json(denylist, "sanitized historical denylist")
    _walk_no_outcomes(value)
    pairs = extract_pairs(manifest)
    expected = {
        "artifact_type": "stage1c_v3_historical_exact_pair_denylist",
        "pair_count": PAIR_COUNT,
        "pairs": pairs,
        "schema_version": 1,
        "source_manifest": {
            "artifact_type": "stage1c_prediction_manifest",
            "branch": "stage-1c-first-prospective-prediction",
            "experiment_class": "stage1c_first_prospective_prediction",
            "freeze_commit": SOURCE_FREEZE_COMMIT,
            "git_blob_sha1": SOURCE_BLOB_SHA1,
            "path": SOURCE_RELATIVE.as_posix(),
            "schema_version": 1,
            "sha256": SOURCE_SHA256,
        },
    }
    if value != expected:
        fail("sanitized denylist does not exactly match source extraction")
    return {
        "denylist_sha256": denylist_sha,
        "pair_count": len(pairs),
        "source_manifest_sha256": source_sha,
        "status": "passed",
    }


def _verify_git_source() -> None:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "ls-tree",
            SOURCE_FREEZE_COMMIT,
            "--",
            SOURCE_RELATIVE.as_posix(),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    expected = f"100644 blob {SOURCE_BLOB_SHA1}\t{SOURCE_RELATIVE.as_posix()}"
    if result.stdout.rstrip("\n") != expected:
        fail("historical source is not the authenticated frozen Git blob")


def main() -> int:
    try:
        _verify_git_source()
        result = validate_values(
            _regular_bytes(ROOT / SOURCE_RELATIVE, "historical source", 2**21),
            _regular_bytes(ROOT / DENYLIST_RELATIVE, "denylist", 2**17),
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, subprocess.SubprocessError, ValidationError) as error:
        print(f"denylist validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
