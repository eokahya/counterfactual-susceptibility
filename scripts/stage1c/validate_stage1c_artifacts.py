#!/usr/bin/env python3
"""Standalone, fail-closed validator for the Stage 1C compact bundle.

This file intentionally has no project imports.  It is an independent
recomputation boundary: the experiment may produce JSON, but it cannot make
the validator accept its own interpretation of that JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
import struct
import sys
import zipfile
from itertools import pairwise
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any, NoReturn

ALLOWLIST = (
    "run_manifest.json",
    "asset_manifest.json",
    "environment_manifest.json",
    "prediction_manifest.json",
    "intervention_sweeps.json",
    "crossing_summary.json",
    "local_linearity_summary.json",
    "memory_timing_summary.json",
    "attempts.json",
    "checksums.sha256",
)
JSON_NAMES = frozenset(ALLOWLIST[:-1])
MAX_FILE = 2 * 1024 * 1024
MAX_BUNDLE = 5 * 1024 * 1024
SHA40 = re.compile(r"\A[0-9a-f]{40}\Z")
SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
CHECKSUM = re.compile(r"\A([0-9a-f]{64})  ([A-Za-z0-9_.-]+)\Z")
FORBIDDEN_KEYS = (
    "model_weight",
    "transcoder_weight",
    "cache",
    "raw_graph",
    "adjacency",
    "full_derivative",
    "dense_preactivation",
    "dense_activation",
    "gradient_tensor",
    "tokenizer_payload",
    "secret",
    "private_absolute_path",
    "private_path",
)
FORBIDDEN_SUFFIXES = frozenset(
    {
        ".bin",
        ".ckpt",
        ".gguf",
        ".h5",
        ".hdf5",
        ".joblib",
        ".npy",
        ".npz",
        ".onnx",
        ".parquet",
        ".pickle",
        ".pkl",
        ".pt",
        ".pth",
        ".safetensors",
        ".tflite",
        ".arrow",
        ".zarr",
        ".tar",
        ".gz",
        ".zip",
    }
)
SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    re.compile(r"(?i)https?://[^/@\s:]+:[^/@\s]+@"),
)
PRIVATE_PATH_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9_])(?:file://)?/Users/[^\s\"']+"),
    re.compile(r"(?<![A-Za-z0-9_])(?:file://)?/home/[^\s\"']+"),
    re.compile(r"(?<![A-Za-z0-9_])/(?:private/)?var/(?:folders|tmp)/[^\s\"']+"),
    re.compile(r"(?<![A-Za-z0-9_])(?:\$HOME|~)/(?:\.cache|Library/Caches)[^\s\"']*"),
)
FORBIDDEN_PREDICTION_KEYS = frozenset(
    {
        "observed",
        "observed_outcome",
        "observed_crossing",
        "realized_suppression",
        "actual_bf16_value_passed",
        "target_activation_after_intervention",
        "intervention_sweeps",
        "scientific_outcome",
    }
)


class ValidationError(ValueError):
    pass


def fail(message: str) -> NoReturn:
    raise ValidationError(message)


def strict_json(raw: bytes, name: str) -> dict[str, Any]:
    def constant(value: str) -> NoReturn:
        fail(f"non-finite JSON constant in {name}: {value}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                fail(f"duplicate JSON key in {name}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"), parse_constant=constant, object_pairs_hook=unique
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid UTF-8 JSON: {name}") from exc
    if not isinstance(value, dict):
        fail(f"{name} root must be an object")
    scan_value(value)
    return value


def scan_value(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        fail(f"non-finite value at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                fail(f"non-string key at {path}")
            normalized = key.casefold().replace("-", "_")
            if any(
                fragment in normalized for fragment in FORBIDDEN_KEYS
            ) and item not in (
                False,
                0,
                "forbidden",
                "none",
                "not_persisted",
            ):
                fail(f"forbidden payload key at {path}.{key}")
            scan_value(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            scan_value(item, f"{path}[{index}]")
    elif not (value is None or isinstance(value, (str, int, bool, float))):
        fail(f"unsupported JSON value at {path}")


def safe_bytes(path: Path, *, maximum: int = MAX_FILE) -> bytes:
    st = path.lstat()
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        fail(f"not a single-link regular file: {path.name}")
    if st.st_size > maximum:
        fail(f"file exceeds size cap: {path.name}")
    raw = path.read_bytes()
    if len(raw) != st.st_size or b"\0" in raw:
        fail(f"unstable or binary artifact: {path.name}")
    text = raw.decode("utf-8")
    patterns = (*SECRET_PATTERNS, *PRIVATE_PATH_PATTERNS)
    if any(pattern.search(text) for pattern in patterns):
        fail(f"secret or private path in {path.name}")
    return raw


def load_bundle(directory: Path) -> dict[str, dict[str, Any]]:
    if directory.is_symlink() or not directory.is_dir():
        fail("bundle must be a real directory")
    observed = {item.name for item in directory.iterdir()}
    if observed != set(ALLOWLIST):
        fail("bundle differs from exact allowlist")
    payload: dict[str, bytes] = {}
    total = 0
    for name in ALLOWLIST:
        maximum = 64 * 1024 if name == "checksums.sha256" else MAX_FILE
        raw = safe_bytes(directory / name, maximum=maximum)
        payload[name] = raw
        total += len(raw)
    if total > MAX_BUNDLE:
        fail("bundle exceeds total size cap")
    entries: dict[str, str] = {}
    for line in payload["checksums.sha256"].decode("utf-8").splitlines():
        match = CHECKSUM.fullmatch(line)
        if match is None:
            fail("malformed checksum line")
        digest, name = match.groups()
        if name in entries:
            fail("duplicate checksum entry")
        entries[name] = digest
    if set(entries) != set(JSON_NAMES):
        fail("checksum coverage differs from allowlist")
    for name, digest in entries.items():
        if hashlib.sha256(payload[name]).hexdigest() != digest:
            fail(f"checksum mismatch: {name}")
    return {name: strict_json(payload[name], name) for name in JSON_NAMES}


def scan_prediction(value: dict[str, Any]) -> None:
    def walk(item: Any, path: str = "$") -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                normalized = key.casefold().replace("-", "_")
                if normalized in FORBIDDEN_PREDICTION_KEYS:
                    fail(f"intervention field in prediction manifest: {path}.{key}")
                walk(child, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")

    walk(value)
    if value.get("status") != "prediction_frozen_ready_for_commit":
        fail("prediction manifest is not frozen-ready")
    if value.get("base_commit") != "efbf70a7e462e640a0e1819a93f3b92727bbd193":
        fail("prediction manifest base commit differs")
    if value.get("branch") != "stage-1c-first-prospective-prediction":
        fail("prediction manifest branch differs")
    if value.get("prompt") != {
        "id": "pilot",
        "text": "The capital of France is",
        "token_ids": [2, 818, 5279, 529, 7001, 563],
    }:
        fail("prediction prompt or token identity changed")
    runtime = value.get("runtime_identity")
    if not isinstance(runtime, dict) or any(
        runtime.get(key) != expected
        for key, expected in (
            ("backend", "nnsight"),
            ("device", "mps:0"),
            ("dtype", "torch.bfloat16"),
            ("model_revision", "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"),
            (
                "transcoder_revision",
                "fada11860ac1d337c1e41e9da308798405b94c8e",
            ),
            (
                "transcoder_subfolder",
                "transcoder_all/width_16k_l0_small",
            ),
            ("upstream_revision", "8f1e2438df612464e229e44c4a00ff637bf9379b"),
        )
    ):
        fail("prediction immutable runtime identity changed")
    protocol = value.get("protocol")
    if not isinstance(protocol, dict):
        fail("prediction protocol is missing")
    if protocol.get("scoring") != {
        "epsilon": 1.0e-12,
        "crossing_tolerance": 1.0e-9,
        "pair_seed": "stage1c-first-prospective-v1",
    }:
        fail("prediction scoring protocol changed")
    scanner = protocol.get("scanner")
    if not isinstance(scanner, dict) or any(
        scanner.get(key) != expected
        for key, expected in (
            ("selected_layers", list(range(18))),
            ("selected_positions", [1, 2, 3, 4, 5]),
            ("feature_width", 16_384),
            ("dense_oracle_chunk_size", 16_384),
            ("canonical_chunk_size", 1_024),
            ("top_k_per_group", 8),
            ("global_top_k", 128),
        )
    ):
        fail("prediction scanner protocol changed")
    sources = protocol.get("source_pool")
    if not isinstance(sources, dict) or any(
        sources.get(key) != expected
        for key, expected in (
            ("require_positive_activation", True),
            ("require_strictly_earlier_layer", True),
            ("require_causal_position", True),
            ("raw_graph_input", "forbidden"),
            ("maximum_active_sources", 10_000),
        )
    ):
        fail("prediction source-pool protocol changed")
    responses = protocol.get("responses")
    if not isinstance(responses, dict) or any(
        responses.get(key) != expected
        for key, expected in (
            (
                "method",
                "target_encoder_reverse_vjp_many_source_contraction",
            ),
            ("graph_edge_input", "forbidden"),
            ("target_batch_size", 8),
            ("maximum_eligible_pairs", 500_000),
        )
    ):
        fail("prediction targeted-response protocol changed")
    if protocol.get("schedule") != {
        "coarse_alphas": [0.0, 0.25, 0.5, 0.75, 1.0],
        "alpha_hat_offset": 0.015625,
        "maximum_bisection_steps": 8,
        "deduplicate_applied_bf16": True,
    }:
        fail("prediction intervention schedule changed")
    selection = protocol.get("selection")
    expected_selection = {
        "primary_maximum": 12,
        "near_boundary_maximum": 8,
        "directional_maximum": 8,
        "maximum_per_target": 1,
        "maximum_primary_per_source": 2,
        "primary_order": [
            "susceptibility_desc",
            "alpha_hat_asc",
            "target",
            "source",
        ],
        "near_order": ["distance_above_one_asc", "target", "source"],
        "directional_order": [
            "movement_over_margin_desc",
            "target",
            "source",
        ],
        "prefer_unused_control_targets": True,
        "control_overlap_fallback": "deterministic_after_unique_exhausted",
    }
    if selection != expected_selection:
        fail("prediction pair-selection protocol changed")
    if protocol.get("analysis") != {
        "minimum_nonzero_points": 3,
        "movement_sign_agreement_minimum": 0.80,
        "median_movement_sne_maximum": 0.50,
        "p95_movement_sne_maximum": 1.00,
        "critical_bracket_distance_maximum": 0.125,
        "undefined_metric_policy": "null_with_reason",
    }:
        fail("prediction analysis thresholds changed")
    if protocol.get("intervention_regime") != {
        "source_count": 1,
        "mapping": "desired=(1-alpha)*baseline",
        "freeze_attention": True,
        "constrained_layers": None,
        "target_clamp_allowed": False,
        "canonical_attempts": 1,
    }:
        fail("prediction intervention regime changed")
    calibration = value.get("baseline_pools", {}).get(
        "many_source_vjp_engineering_calibration"
    )
    if not isinstance(calibration, dict) or any(
        calibration.get(key) != expected
        for key, expected in (
            ("pair_count", 4),
            ("comparison", "exact_bf16_identity"),
            ("passed", True),
            ("inactive_target_intervention_calls", 0),
        )
    ):
        fail("many-source VJP engineering calibration did not pass")
    guards = value.get("prediction_only_guards")
    if guards != {
        "source_suppression_api_calls": 0,
        "prior_inactive_target_outcome_read": False,
        "intervention_worker_imported": False,
        "raw_graph_read": False,
        "raw_adjacency_read": False,
    }:
        fail("prediction-only guard record is invalid")
    hashes = value.get("protocol_file_sha256")
    if (
        not isinstance(hashes, dict)
        or len(hashes) != 16
        or any(
            not isinstance(name, str)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not isinstance(digest, str)
            or SHA256.fullmatch(digest) is None
            for name, digest in hashes.items()
        )
    ):
        fail("prediction protocol hash set is invalid")


def bf16_round(value: float) -> float:
    """Round a finite Python scalar through IEEE float32 to BF16 RNE."""

    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    lower = bits & 0xFFFF
    upper = bits >> 16
    if lower > 0x8000 or (lower == 0x8000 and upper & 1):
        upper += 1
    return float(struct.unpack(">f", struct.pack(">I", upper << 16))[0])


def feature_key(value: Any) -> tuple[int, int, int]:
    if not isinstance(value, dict) or set(value) != {"layer", "position", "feature_id"}:
        fail("invalid feature reference")
    result = tuple(value[key] for key in ("layer", "position", "feature_id"))
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in result
    ):
        fail("invalid feature coordinate")
    return result  # type: ignore[return-value]


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in range(start, end):
            result[order[index]] = rank
        start = end
    return result


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_rank = ranks(left)
    right_rank = ranks(right)
    left_mean = sum(left_rank) / len(left_rank)
    right_mean = sum(right_rank) / len(right_rank)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_rank, right_rank, strict=True)
    )
    denominator = math.sqrt(
        sum((item - left_mean) ** 2 for item in left_rank)
        * sum((item - right_mean) ** 2 for item in right_rank)
    )
    return None if denominator == 0.0 else numerator / denominator


def scientific_outcome(rows: list[dict[str, Any]]) -> str:
    primary = [item for item in rows if item["group"] == "primary"]
    if not primary:
        return "no_eligible_pairs"
    supporting = sum(bool(item["supporting_primary"]) for item in primary)
    if supporting == 0:
        return "not_supported"
    discrepancy = (
        supporting != len(primary)
        or any(item["directional_control_violation"] for item in rows)
        or any(item["near_boundary_control_crossing"] for item in rows)
        or any(item["nonmonotonic_gate"] for item in rows)
    )
    return "mixed" if discrepancy else "supported"


def number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        fail(f"{label} must be finite numeric")
    return float(value)


def recompute_pairs(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    groups = prediction.get("selected_groups")
    if not isinstance(groups, dict) or set(groups) != {
        "primary",
        "near_boundary",
        "directional",
    }:
        fail("prediction selected_groups missing")
    all_pairs: list[dict[str, Any]] = []
    eps = number(
        prediction.get("protocol", {}).get("scoring", {}).get("epsilon"),
        "epsilon",
    )
    group_rows: dict[str, list[dict[str, Any]]] = {}
    for group in ("primary", "near_boundary", "directional"):
        rows = groups.get(group, [])
        if not isinstance(rows, list):
            fail(f"selected group {group} is not a list")
        group_rows[group] = rows
        for row in rows:
            if not isinstance(row, dict):
                fail("pair row is not an object")
            if row.get("group") != group:
                fail("pair group label differs from its frozen group")
            source = feature_key(row.get("source"))
            target = feature_key(row.get("target"))
            if source[0] >= target[0] or source[1] > target[1]:
                fail("pair violates causal order")
            a = number(row.get("source_activation"), "source activation")
            z = number(row.get("target_preactivation"), "target preactivation")
            tau = number(row.get("target_threshold"), "target threshold")
            margin = number(row.get("margin"), "margin")
            response = number(row.get("targeted_response"), "targeted response")
            q = -a * response
            expected_margin = tau - z
            eps = number(
                prediction.get("protocol", {}).get("scoring", {}).get("epsilon"),
                "epsilon",
            )
            tolerance = number(
                prediction.get("protocol", {})
                .get("scoring", {})
                .get("crossing_tolerance"),
                "crossing tolerance",
            )
            if margin != expected_margin or number(row.get("q"), "q") != q:
                fail("pair score recomputation mismatch")
            if margin < 0.0:
                fail("inactive target has negative margin")
            score = q / (margin + eps)
            if number(row.get("susceptibility"), "susceptibility") != score:
                fail("susceptibility mismatch")
            alpha = row.get("predicted_alpha_star")
            if q > 0:
                if alpha is None or number(alpha, "alpha") != margin / q:
                    fail("predicted alpha mismatch")
            elif alpha is not None:
                fail("non-positive q has a predicted alpha")
            if abs(margin) <= tolerance:
                expected_status = "boundary_ambiguous"
            elif q <= 0.0:
                expected_status = "not_crossing"
            elif q - margin > tolerance:
                expected_status = "definitely_crossing"
            elif abs(q - margin) <= tolerance:
                expected_status = "boundary_ambiguous"
            else:
                expected_status = "not_crossing"
            if row.get("predicted_status") != expected_status:
                fail("predicted crossing status mismatch")
            runtime_fingerprint = (
                "gemma3-270m@9b0cfec892e2/plt@fada11860ac1/"
                "circuit-tracer@8f1e2438df61/nnsight/mps/bf16/stage1c"
            )
            pair_payload = {
                "prompt_id": prediction.get("prompt", {}).get("id"),
                "runtime_fingerprint": runtime_fingerprint,
                "seed": prediction.get("protocol", {})
                .get("scoring", {})
                .get("pair_seed"),
                "source": list(source),
                "target": list(target),
            }
            expected_id = hashlib.sha256(
                json.dumps(pair_payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if row.get("pair_id") != expected_id:
                fail("selected pair ID recomputation mismatch")
            if group == "primary" and not (
                expected_status == "definitely_crossing"
                and alpha is not None
                and 0.0 < float(alpha) < 1.0
            ):
                fail("primary group contains an ineligible pair")
            if group == "near_boundary" and not (
                q > 0.0
                and expected_status == "not_crossing"
                and alpha is not None
                and float(alpha) > 1.0
                and margin - q > tolerance
            ):
                fail("near-boundary group contains an ineligible pair")
            if group == "directional" and not (
                q <= 0.0 and expected_status == "not_crossing" and margin > tolerance
            ):
                fail("directional group contains an ineligible pair")
            all_pairs.append(
                {**row, "group": group, "source_key": source, "target_key": target}
            )
    ids = [row.get("pair_id") for row in all_pairs]
    if any(
        not isinstance(item, str) or not SHA256.fullmatch(item) for item in ids
    ) or len(ids) != len(set(ids)):
        fail("selected pair IDs are invalid or duplicated")
    primary_rows = groups.get("primary", [])
    if len({feature_key(row.get("target")) for row in primary_rows}) != len(
        primary_rows
    ):
        fail("primary targets are not unique")
    primary_source_counts: dict[tuple[int, int, int], int] = {}
    for row in primary_rows:
        source = feature_key(row.get("source"))
        primary_source_counts[source] = primary_source_counts.get(source, 0) + 1
    if any(count > 2 for count in primary_source_counts.values()):
        fail("primary source diversity cap is violated")
    limits = prediction.get("protocol", {}).get("selection", {})
    maximums = {
        "primary": limits.get("primary_maximum"),
        "near_boundary": limits.get("near_boundary_maximum"),
        "directional": limits.get("directional_maximum"),
    }
    for group, rows in group_rows.items():
        if len(rows) > int(maximums[group]):
            fail(f"selected {group} group exceeds its cap")
        primary_targets = {
            feature_key(item.get("target")) for item in group_rows["primary"]
        }
        near_targets = {
            feature_key(item.get("target")) for item in group_rows["near_boundary"]
        }
        for left, right in pairwise(rows):
            if group == "primary":
                left_key = (
                    -number(left["susceptibility"], "susceptibility"),
                    number(left["predicted_alpha_star"], "predicted alpha"),
                    feature_key(left["target"]),
                    feature_key(left["source"]),
                )
                right_key = (
                    -number(right["susceptibility"], "susceptibility"),
                    number(right["predicted_alpha_star"], "predicted alpha"),
                    feature_key(right["target"]),
                    feature_key(right["source"]),
                )
            elif group == "near_boundary":
                left_key = (
                    feature_key(left["target"]) in primary_targets,
                    number(left["predicted_alpha_star"], "predicted alpha") - 1.0,
                    feature_key(left["target"]),
                    feature_key(left["source"]),
                )
                right_key = (
                    feature_key(right["target"]) in primary_targets,
                    number(right["predicted_alpha_star"], "predicted alpha") - 1.0,
                    feature_key(right["target"]),
                    feature_key(right["source"]),
                )
            else:
                used_targets = primary_targets | near_targets
                left_key = (
                    feature_key(left["target"]) in used_targets,
                    -abs(number(left["q"], "q"))
                    / (number(left["margin"], "margin") + number(eps, "epsilon")),
                    feature_key(left["target"]),
                    feature_key(left["source"]),
                )
                right_key = (
                    feature_key(right["target"]) in used_targets,
                    -abs(number(right["q"], "q"))
                    / (number(right["margin"], "margin") + number(eps, "epsilon")),
                    feature_key(right["target"]),
                    feature_key(right["source"]),
                )
            if left_key > right_key:
                fail(f"selected {group} ordering is not frozen")
    schedule = prediction.get("protocol", {}).get("schedule", {})
    coarse = {float(item) for item in schedule.get("coarse_alphas", [])}
    offset = float(schedule.get("alpha_hat_offset"))
    for row in all_pairs:
        expected = set(coarse)
        alpha = row.get("predicted_alpha_star")
        if alpha is not None and 0.0 <= float(alpha) <= 1.0:
            value = float(alpha)
            expected.update(
                {
                    max(0.0, min(1.0, value - offset)),
                    value,
                    max(0.0, min(1.0, value + offset)),
                }
            )
        observed = tuple(float(item) for item in row.get("requested_alphas", []))
        if observed != tuple(sorted(expected)):
            fail("requested intervention schedule differs from frozen schedule")
    audit = prediction.get("selection_audit")
    if not isinstance(audit, dict) or any(
        audit.get(key) != expected
        for key, expected in (
            ("primary_count", len(group_rows["primary"])),
            ("near_boundary_count", len(group_rows["near_boundary"])),
            ("directional_count", len(group_rows["directional"])),
            ("groups_disjoint", True),
            ("primary_target_unique", True),
            ("primary_source_cap", 2),
        )
    ):
        fail("prediction selection audit differs from selected groups")
    return all_pairs


def validate_sweeps(
    prediction: dict[str, Any],
    sweeps: dict[str, Any],
    analyses: dict[str, Any],
) -> None:
    rows = sweeps.get("pairs")
    if not isinstance(rows, list):
        fail("intervention sweeps.pairs is missing")
    selected = {row["pair_id"]: row for row in recompute_pairs(prediction)}
    primary_count = len(prediction["selected_groups"]["primary"])
    expected_sweep_ids = set() if primary_count == 0 else set(selected)
    if set(item.get("pair_id") for item in rows) != expected_sweep_ids:
        fail("sweep pair IDs differ from prediction")
    for item in rows:
        pair = selected[item["pair_id"]]
        points = item.get("points")
        if not isinstance(points, list) or not points:
            fail("empty sweep")
        seen: set[float] = set()
        requested_seen: set[float] = set()
        for point in points:
            if not isinstance(point, dict):
                fail("invalid sweep point")
            realized = number(point.get("realized_suppression"), "realized suppression")
            requested = number(point.get("requested_alpha"), "requested alpha")
            desired = number(point.get("desired_high_precision"), "desired activation")
            applied = number(
                point.get("actual_bf16_value_passed"), "actual BF16 activation"
            )
            baseline = number(pair["source_activation"], "baseline activation")
            if desired != (1.0 - requested) * baseline:
                fail("desired source mapping mismatch")
            if applied != bf16_round(desired):
                fail("actual BF16 source value differs from independent rounding")
            if realized != 1.0 - applied / baseline:
                fail("realized suppression mismatch")
            if realized in seen:
                fail("duplicate applied suppression")
            seen.add(realized)
            requests = point.get("requested_mappings")
            if not isinstance(requests, list) or not requests:
                fail("point lacks requested BF16 mappings")
            if point.get("collapsed_request_count") != len(requests):
                fail("collapsed BF16 request count is invalid")
            if (
                point.get("source_value_device") != "mps:0"
                or point.get("source_value_dtype") != "torch.bfloat16"
            ):
                fail("applied source value left MPS/BF16")
            representative_request = min(
                requests,
                key=lambda item: number(item.get("requested_alpha"), "requested alpha"),
            )
            if requested != number(
                representative_request.get("requested_alpha"),
                "representative requested alpha",
            ) or desired != number(
                representative_request.get("desired_high_precision"),
                "representative desired activation",
            ):
                fail("top-level requested mapping is not the representative")
            for request in requests:
                request_alpha = number(
                    request.get("requested_alpha"), "requested alpha"
                )
                request_desired = number(
                    request.get("desired_high_precision"), "desired activation"
                )
                request_applied = number(
                    request.get("actual_bf16_value_passed"),
                    "actual BF16 activation",
                )
                request_realized = number(
                    request.get("realized_suppression"), "realized suppression"
                )
                if request_desired != (1.0 - request_alpha) * baseline:
                    fail("requested desired activation mapping mismatch")
                if request_applied != bf16_round(request_desired):
                    fail("requested BF16 value differs from independent rounding")
                if request_realized != 1.0 - request_applied / baseline:
                    fail("requested realized suppression mismatch")
                if request_applied != applied or request_realized != realized:
                    fail("requested mapping differs from applied point")
                requested_seen.add(request_alpha)
            z = number(point.get("target_preactivation"), "target z")
            tau = number(point.get("target_threshold"), "target tau")
            active = point.get("target_active")
            if not isinstance(active, bool) or active != (z > tau):
                fail("strict gate mismatch")
            if tau != number(pair["target_threshold"], "frozen target threshold"):
                fail("target threshold changed during intervention")
            activation = number(point.get("target_activation"), "target activation")
            if activation != (z if active else 0.0):
                fail("target post-gate activation differs from strict JumpReLU")
            if point.get("loaded_gate") != "a=z*1[z>tau]":
                fail("loaded gate identity changed")
            if point.get("threshold_equality_activity") != "inactive":
                fail("threshold equality policy changed")
            if point.get("target_clamped") is not False:
                fail("target was clamped")
            if point.get("freeze_attention") is not True:
                fail("attention freeze regime changed")
            if point.get("constrained_layers") is not None:
                fail("constrained-layer regime changed")
            predicted_z = number(
                point.get("predicted_target_preactivation"), "predicted target z"
            )
            expected_predicted_z = number(
                pair["target_preactivation"], "baseline z"
            ) + realized * number(pair["q"], "q")
            if predicted_z != expected_predicted_z:
                fail("local predicted target preactivation mismatch")
            predicted_active = point.get("predicted_target_active")
            if not isinstance(predicted_active, bool) or predicted_active != (
                predicted_z > number(pair["target_threshold"], "target threshold")
            ):
                fail("local predicted gate mismatch")
            predicted_activation = number(
                point.get("predicted_target_activation"), "predicted target activation"
            )
            expected_activation = predicted_z if predicted_active else 0.0
            if predicted_activation != expected_activation:
                fail("local predicted activation mismatch")
            absolute_error = number(
                point.get("target_preactivation_absolute_error"), "absolute error"
            )
            if absolute_error != abs(z - predicted_z):
                fail("target preactivation absolute error mismatch")
            sne = number(
                point.get("target_preactivation_symmetric_normalized_error"),
                "target preactivation error",
            )
            denominator = abs(z) + abs(predicted_z)
            expected_sne = (
                0.0 if denominator == 0.0 else 2.0 * abs(z - predicted_z) / denominator
            )
            if sne != expected_sne:
                fail("target preactivation normalized error mismatch")
        frozen_requested = {
            number(value, "frozen requested alpha")
            for value in pair.get("requested_alphas", [])
        }
        if not frozen_requested.issubset(requested_seen):
            fail("frozen requested schedule is incomplete after BF16 deduplication")
        base_points = [point for point in points if point.get("bisection_step") is None]
        refined: list[dict[str, Any]] = list(base_points)
        bisection_points = sorted(
            (point for point in points if point.get("bisection_step") is not None),
            key=lambda point: int(point["bisection_step"]),
        )
        for step, point in enumerate(bisection_points):
            if point.get("bisection_step") != step:
                fail("bisection step numbering is not canonical")
            ordered_refined = sorted(
                refined,
                key=lambda item: number(
                    item.get("realized_suppression"), "realized suppression"
                ),
            )
            bracket = next(
                (
                    (left, right)
                    for left, right in pairwise(ordered_refined)
                    if left.get("target_active") is False
                    and right.get("target_active") is True
                ),
                None,
            )
            if bracket is None:
                fail("bisection point exists without an inactive-active bracket")
            lower, upper = bracket
            expected_request = (
                number(lower["realized_suppression"], "lower realized")
                + number(upper["realized_suppression"], "upper realized")
            ) / 2.0
            if number(point.get("requested_alpha"), "bisection request") != (
                expected_request
            ):
                fail("bisection request is not the deterministic midpoint")
            refined.append(point)
        if len(bisection_points) > 8:
            fail("bisection exceeds the frozen step cap")
    analysis_rows = analyses.get("pairs")
    if not isinstance(analysis_rows, list):
        fail("crossing summary pair analyses missing")
    expected_analysis: dict[str, dict[str, Any]] = {}
    for item in rows:
        pair_id = item.get("pair_id")
        points = sorted(
            item["points"],
            key=lambda point: number(
                point.get("realized_suppression"), "realized suppression"
            ),
        )
        if points[0].get("target_active") is not False:
            fail("baseline target is not inactive")
        if number(points[0].get("realized_suppression"), "baseline suppression") != 0.0:
            fail("sweep does not start at zero realized suppression")
        if number(points[0].get("target_preactivation"), "baseline target z") != (
            number(pair["target_preactivation"], "predicted baseline target z")
        ):
            fail("intervention baseline differs from frozen prediction")
        full = next(
            (
                point
                for point in points
                if number(point.get("realized_suppression"), "realized suppression")
                == 1.0
            ),
            None,
        )
        if full is None:
            fail("sweep lacks exact full ablation")
        bracket = None
        for left, right in pairwise(points):
            if (
                left.get("target_active") is False
                and right.get("target_active") is True
            ):
                bracket = {
                    "lower_realized_suppression": number(
                        left.get("realized_suppression"), "lower bracket"
                    ),
                    "upper_realized_suppression": number(
                        right.get("realized_suppression"), "upper bracket"
                    ),
                }
                break
        baseline_z = number(pair["target_preactivation"], "baseline target z")
        q = number(pair["q"], "q")
        errors: list[float] = []
        signs: list[bool] = []
        for point in points[1:]:
            realized = number(point["realized_suppression"], "realized suppression")
            predicted_delta = realized * q
            observed_delta = (
                number(point["target_preactivation"], "target z") - baseline_z
            )
            denominator = abs(predicted_delta) + abs(observed_delta)
            errors.append(
                0.0
                if denominator == 0.0
                else 2.0 * abs(predicted_delta - observed_delta) / denominator
            )
            signs.append(
                observed_delta > 0.0
                if predicted_delta > 0.0
                else observed_delta < 0.0
                if predicted_delta < 0.0
                else observed_delta == 0.0
            )
        alpha = pair.get("predicted_alpha_star")
        distance = None
        if alpha is not None and bracket is not None:
            alpha_value = number(alpha, "predicted alpha")
            lower = bracket["lower_realized_suppression"]
            upper = bracket["upper_realized_suppression"]
            distance = (
                0.0
                if lower <= alpha_value <= upper
                else min(abs(alpha_value - lower), abs(alpha_value - upper))
            )
        analysis_config = prediction.get("protocol", {}).get("analysis", {})
        sign_agreement = sum(signs) / len(signs) if signs else None
        median_error = median(errors) if errors else None
        p95_error = (
            sorted(errors)[max(1, math.ceil(0.95 * len(errors))) - 1]
            if errors
            else None
        )
        local_pass = (
            len(points) - 1 >= int(analysis_config["minimum_nonzero_points"])
            and sign_agreement is not None
            and sign_agreement
            >= float(analysis_config["movement_sign_agreement_minimum"])
            and median_error is not None
            and median_error <= float(analysis_config["median_movement_sne_maximum"])
            and p95_error is not None
            and p95_error <= float(analysis_config["p95_movement_sne_maximum"])
            and distance is not None
            and distance <= float(analysis_config["critical_bracket_distance_maximum"])
        )
        full_delta = (
            number(full["target_preactivation"], "full-ablation target z") - baseline_z
        )
        active_seen = False
        nonmonotonic = False
        for point in points:
            if point["target_active"]:
                active_seen = True
            elif active_seen:
                nonmonotonic = True
        group = str(pair["group"])
        full_crossing = bool(full["target_active"])
        expected_analysis[str(pair_id)] = {
            "pair_id": pair_id,
            "group": group,
            "point_count": len(points),
            "nonzero_point_count": len(points) - 1,
            "predicted_full_ablation_crossing": (
                pair.get("predicted_status") == "definitely_crossing"
            ),
            "observed_full_ablation_crossing": bool(full["target_active"]),
            "observed_critical_bracket": bracket,
            "critical_bracket_distance": distance,
            "predicted_alpha_star": pair.get("predicted_alpha_star"),
            "movement_sign_agreement": sign_agreement,
            "median_movement_symmetric_normalized_error": median_error,
            "p95_movement_symmetric_normalized_error": p95_error,
            "full_ablation_observed_movement": full_delta,
            "local_calibration_passed": local_pass,
            "supporting_primary": (
                group == "primary" and full_crossing and full_delta > 0.0 and local_pass
            ),
            "directional_control_violation": (
                group == "directional" and full_delta > 0.0
            ),
            "near_boundary_control_crossing": (
                group == "near_boundary"
                and any(bool(point["target_active"]) for point in points)
            ),
            "nonmonotonic_gate": nonmonotonic,
        }
        bisection_steps = [
            point.get("bisection_step")
            for point in points
            if point.get("bisection_step") is not None
        ]
        if len(bisection_steps) > 8 or sorted(bisection_steps) != list(
            range(len(bisection_steps))
        ):
            fail("bisection steps are not deterministic or exceed the cap")
    if {item.get("pair_id") for item in analysis_rows} != set(expected_analysis):
        fail("crossing analysis pair IDs differ from sweeps")
    for row in analysis_rows:
        expected = expected_analysis[str(row.get("pair_id"))]
        for key, value in expected.items():
            if row.get(key) != value:
                fail(f"crossing analysis mismatch: {key}")
    computed = list(expected_analysis.values())
    primary = [item for item in computed if item["group"] == "primary"]
    near = [item for item in computed if item["group"] == "near_boundary"]
    directional = [item for item in computed if item["group"] == "directional"]
    primary_crossings = sum(
        bool(item["observed_full_ablation_crossing"]) for item in primary
    )
    near_crossings = sum(bool(item["near_boundary_control_crossing"]) for item in near)
    directional_violations = sum(
        bool(item["directional_control_violation"]) for item in directional
    )
    critical_pairs = [
        item
        for item in primary
        if item["observed_critical_bracket"] is not None
        and item["predicted_alpha_star"] is not None
    ]
    critical_observed = [
        (
            item["observed_critical_bracket"]["lower_realized_suppression"]
            + item["observed_critical_bracket"]["upper_realized_suppression"]
        )
        / 2.0
        for item in critical_pairs
    ]
    critical_predicted = [
        float(item["predicted_alpha_star"]) for item in critical_pairs
    ]
    bracket_distances = [
        float(item["critical_bracket_distance"])
        for item in primary
        if item["critical_bracket_distance"] is not None
    ]
    primary_median_errors = [
        float(item["median_movement_symmetric_normalized_error"])
        for item in primary
        if item["median_movement_symmetric_normalized_error"] is not None
    ]
    primary_p95_errors = [
        float(item["p95_movement_symmetric_normalized_error"])
        for item in primary
        if item["p95_movement_symmetric_normalized_error"] is not None
    ]
    aggregate = {
        "primary_pair_count": len(primary),
        "primary_full_ablation_crossing_count": primary_crossings,
        "primary_full_ablation_crossing_precision": (
            primary_crossings / len(primary) if primary else None
        ),
        "primary_precision_undefined_reason": None if primary else "no_primary_pairs",
        "supporting_primary_count": sum(
            bool(item["supporting_primary"]) for item in primary
        ),
        "near_boundary_pair_count": len(near),
        "near_boundary_crossing_count": near_crossings,
        "near_boundary_crossing_fraction": near_crossings / len(near) if near else None,
        "near_boundary_fraction_undefined_reason": (
            None if near else "no_near_boundary_controls"
        ),
        "directional_pair_count": len(directional),
        "directional_violation_count": directional_violations,
        "directional_violation_fraction": (
            directional_violations / len(directional) if directional else None
        ),
        "directional_fraction_undefined_reason": (
            None if directional else "no_directional_controls"
        ),
        "critical_suppression_spearman_pair_count": len(critical_pairs),
        "primary_bracket_distance_count": len(bracket_distances),
        "primary_bracket_distance_median": (
            median(bracket_distances) if bracket_distances else None
        ),
        "primary_bracket_distance_p95": (
            sorted(bracket_distances)[
                max(1, math.ceil(0.95 * len(bracket_distances))) - 1
            ]
            if bracket_distances
            else None
        ),
        "primary_bracket_distance_undefined_reason": (
            None if bracket_distances else "no_observed_primary_crossing_brackets"
        ),
        "primary_pair_median_movement_sne_median": (
            median(primary_median_errors) if primary_median_errors else None
        ),
        "primary_pair_median_movement_sne_p95": (
            sorted(primary_median_errors)[
                max(1, math.ceil(0.95 * len(primary_median_errors))) - 1
            ]
            if primary_median_errors
            else None
        ),
        "primary_pair_movement_error_undefined_reason": (
            None if primary_median_errors else "no_primary_movement_errors"
        ),
        "primary_pair_p95_movement_sne_median": (
            median(primary_p95_errors) if primary_p95_errors else None
        ),
    }
    correlation = spearman(critical_predicted, critical_observed)
    aggregate["critical_suppression_spearman"] = correlation
    aggregate["critical_suppression_spearman_undefined_reason"] = (
        None
        if correlation is not None
        else "fewer_than_two_nonconstant_observed_crossings"
    )
    supplied_aggregate = analyses.get("aggregate")
    if supplied_aggregate is not None:
        if not isinstance(supplied_aggregate, dict):
            fail("crossing aggregate is not an object")
        for key, value in aggregate.items():
            if supplied_aggregate.get(key) != value:
                fail(f"crossing aggregate mismatch: {key}")
    outcome = scientific_outcome(computed)
    if analyses.get("scientific_outcome") != outcome:
        fail("scientific outcome classification mismatch")


def validate_zip(path: Path) -> None:
    if path.suffix.lower() != ".zip":
        fail("hostile archive test requires a ZIP")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > 128:
                fail("ZIP has too many members")
            names: set[str] = set()
            expanded = 0
            for info in infos:
                name = info.filename.replace("\\", "/")
                pure = PurePosixPath(name)
                if name in names or pure.is_absolute() or ".." in pure.parts:
                    fail("ZIP path traversal or duplicate member")
                if any(
                    name.casefold().endswith(suffix) for suffix in FORBIDDEN_SUFFIXES
                ):
                    fail("ZIP contains a forbidden artifact extension")
                names.add(name)
                mode = (info.external_attr >> 16) & 0xFFFF
                if (
                    stat.S_ISLNK(mode)
                    or stat.S_ISCHR(mode)
                    or stat.S_ISBLK(mode)
                    or stat.S_ISFIFO(mode)
                ):
                    fail("ZIP contains a link or special file")
                expanded += max(0, info.file_size)
                if info.file_size > MAX_FILE or expanded > MAX_BUNDLE:
                    fail("ZIP expanded size exceeds cap")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValidationError("invalid ZIP bundle") from exc


def validate_bundle(
    directory: Path, execution_commit: str | None = None
) -> dict[str, Any]:
    if execution_commit is not None and not SHA40.fullmatch(execution_commit):
        fail("execution commit is not a SHA-1")
    artifacts = load_bundle(directory)
    prediction = artifacts["prediction_manifest.json"]
    scan_prediction(prediction)
    if execution_commit is not None:
        run = artifacts["run_manifest.json"]
        if run.get("execution_commit") != execution_commit:
            fail("run manifest execution commit mismatch")
    recompute_pairs(prediction)
    validate_sweeps(
        prediction,
        artifacts["intervention_sweeps.json"],
        artifacts["crossing_summary.json"],
    )
    sweep_rows = artifacts["intervention_sweeps.json"].get("pairs", [])
    errors = [
        number(
            point.get("target_preactivation_symmetric_normalized_error"),
            "point error",
        )
        for item in sweep_rows
        for point in item.get("points", [])
    ]
    local = artifacts["local_linearity_summary.json"]
    if local.get("point_count") != len(errors):
        fail("local-linearity point count mismatch")
    expected_median = median(errors) if errors else None
    expected_p95 = (
        sorted(errors)[max(1, math.ceil(0.95 * len(errors))) - 1] if errors else None
    )
    if local.get("median_symmetric_normalized_error") != expected_median:
        fail("local-linearity median mismatch")
    if local.get("p95_symmetric_normalized_error") != expected_p95:
        fail("local-linearity p95 mismatch")
    primary_count = len(prediction.get("selected_groups", {}).get("primary", []))
    attempts = artifacts["attempts.json"]
    expected_intervention_attempts = 0 if primary_count == 0 else 1
    if attempts.get("intervention_attempts") != expected_intervention_attempts:
        fail("canonical attempt count violates no-primary invariant")
    if attempts.get("prediction_attempts") != 1:
        fail("prediction attempt count is invalid")
    if attempts.get("scientific_retry_count") != 0:
        fail("scientific retry count is nonzero")
    if attempts.get("intervention_required") is not (primary_count > 0):
        fail("attempt intervention-required flag is invalid")
    if primary_count == 0:
        if artifacts["intervention_sweeps.json"].get("pairs") != []:
            fail("no-primary run contains intervention sweeps")
        if artifacts["crossing_summary.json"].get("scientific_outcome") != (
            "no_eligible_pairs"
        ):
            fail("no-primary run has an invalid scientific outcome")
    run = artifacts["run_manifest.json"]
    if run.get("branch") != "stage-1c-first-prospective-prediction":
        fail("run manifest branch identity mismatch")
    if run.get("base_commit") != "efbf70a7e462e640a0e1819a93f3b92727bbd193":
        fail("run manifest base identity mismatch")
    if not SHA40.fullmatch(str(run.get("execution_commit", ""))):
        fail("run manifest execution identity is invalid")
    if run.get("pre_intervention_commit") != run.get("execution_commit"):
        fail("pre-intervention and execution commits differ")
    prediction_digest = hashlib.sha256(
        (directory / "prediction_manifest.json").read_bytes()
    ).hexdigest()
    if run.get("prediction_manifest_sha256") != prediction_digest:
        fail("run prediction-manifest digest differs from artifact bytes")
    claim = run.get("claim_boundary")
    if claim != {
        "behavioral_importance_result": "none",
        "mediation_result": "none",
        "official_bf16_reproduction": "pending",
        "reference_clt_reproduction": "pending",
        "paper_results_readiness": False,
    }:
        fail("claim boundary changed")
    valid_statuses = {
        "completed_stage1c_first_prospective_prediction",
        "failed_runtime",
        "blocked",
    }
    if run.get("status") not in valid_statuses:
        fail("invalid run status")
    if run.get("verdict") != run.get("status"):
        fail("run verdict and status differ")
    if run.get("scientific_outcome") != artifacts["crossing_summary.json"].get(
        "scientific_outcome"
    ):
        fail("run and crossing scientific outcomes differ")
    outcome = str(run.get("scientific_outcome"))
    aggregate = artifacts["crossing_summary.json"].get("aggregate", {})
    primary_crossings = int(aggregate.get("primary_full_ablation_crossing_count", -1))
    expected_readiness = {
        "stage1b_measurement_primitives": "completed",
        "stage1c_first_prediction": "completed",
        "stage1c_scientific_outcome": outcome,
        "counterfactual_susceptibility_result": (
            "preliminary_single_prompt"
            if outcome in {"supported", "mixed"}
            else "negative_single_prompt"
            if outcome == "not_supported"
            else "none"
        ),
        "gate_crossing_result": (
            "prospective_single_prompt" if primary_crossings > 0 else "none"
        ),
        "behavioral_importance_result": "none",
        "mediation_result": "none",
        "official_bf16_reproduction": "pending",
        "reference_clt_reproduction": "pending",
        "paper_results_readiness": False,
    }
    if run.get("readiness") != expected_readiness:
        fail("outcome-sensitive readiness fields are invalid")
    if run.get("fresh_canonical_run") is not (primary_count > 0):
        fail("fresh canonical run flag violates no-primary policy")
    if run.get("intervention_required") is not (primary_count > 0):
        fail("run intervention-required flag violates no-primary policy")
    if run.get("scientific_retry_count") != 0:
        fail("run scientific retry count is nonzero")
    calls = run.get("canonical_source_suppression_api_calls")
    if not isinstance(calls, int) or calls < 0 or (primary_count == 0 and calls != 0):
        fail("canonical source-suppression API call count is invalid")
    if primary_count > 0 and calls < len(sweep_rows):
        fail("canonical source-suppression API call count is too small")
    if run.get("secondary_regime") != {
        "status": "not_implemented",
        "result": None,
    }:
        fail("secondary regime record is invalid")
    memory = artifacts["memory_timing_summary.json"]
    for phase in ("prediction_supervisor", "intervention_supervisor"):
        supervisor = memory.get(phase)
        if not isinstance(supervisor, dict) or any(
            supervisor.get(key) != expected
            for key, expected in (
                ("returncode", 0),
                ("timed_out", False),
                ("safety_terminated", False),
                ("telemetry_failures", 0),
            )
        ):
            fail(f"{phase} safety record did not pass")
    environment = artifacts["environment_manifest.json"]
    accelerator = environment.get("accelerator", {})
    if (
        environment.get("status") != "passed"
        or environment.get("execution_commit") != run.get("execution_commit")
        or accelerator.get("device") != "mps:0"
        or accelerator.get("dtype") != "torch.bfloat16"
        or accelerator.get("fallback_variable_present") is not False
        or accelerator.get("outer_autocast_enabled") is not False
        or accelerator.get("scientific_tensor_device") != "mps"
    ):
        fail("environment runtime identity or fallback policy is invalid")
    asset = artifacts["asset_manifest.json"]
    if any(
        asset.get(key) != expected
        for key, expected in (
            ("status", "verified"),
            ("download_performed", False),
            ("network_accessed", False),
            ("authentication_used", False),
            ("authentication_value_recorded", False),
            ("exact_allowlist_hashes_verified", True),
            ("actual_total_bytes", 2_087_816_677),
        )
    ):
        fail("immutable asset manifest is invalid")
    return {
        "status": "passed",
        "artifact_count": len(ALLOWLIST),
        "verdict": run.get("verdict", run.get("status")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--zip", dest="zip_path", type=Path)
    parser.add_argument("--execution-commit")
    args = parser.parse_args(argv)
    try:
        if args.zip_path is not None:
            validate_zip(args.zip_path)
            print(json.dumps({"status": "passed", "zip": str(args.zip_path)}))
            return 0
        if args.bundle is None:
            parser.error("--bundle is required unless --zip is supplied")
        print(
            json.dumps(
                validate_bundle(args.bundle, args.execution_commit),
                sort_keys=True,
                allow_nan=False,
            )
        )
    except ValidationError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
