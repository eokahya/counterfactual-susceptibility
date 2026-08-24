"""Fail-closed validation for compact Stage 1B artifact bundles."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn

from cfsus.reproduction.artifacts import (
    ArtifactValidationError,
    assert_publication_safe,
    sha256_file,
)
from cfsus.responses.validation import (
    compute_local_response_metrics,
    symmetric_normalized_error,
    validate_pair_distribution,
)
from cfsus.stage1b import (
    BASE_COMMIT,
    BRANCH,
    COMPLETED_STATUS,
    MODEL_REVISION,
    TRANSCODER_REVISION,
    TRANSCODER_SUBFOLDER,
    UPSTREAM_REVISION,
)
from cfsus.types import (
    ActivePairReference,
    FeatureRef,
    LocalResponseEstimate,
    NearThresholdCandidate,
)

ARTIFACT_ALLOWLIST = frozenset(
    {
        "run_manifest.json",
        "asset_manifest.json",
        "environment_manifest.json",
        "scanner_oracle_summary.json",
        "near_threshold_candidates.json",
        "local_response_validation_summary.json",
        "local_response_validation_pairs.json",
        "memory_timing_summary.json",
        "attempts.json",
        "checksums.sha256",
    }
)
JSON_FILES = ARTIFACT_ALLOWLIST - {"checksums.sha256"}
MAXIMUM_TOTAL_BYTES = 5 * 1024 * 1024
MAXIMUM_JSON_BYTES = 2 * 1024 * 1024
MAXIMUM_CHECKSUM_BYTES = 64 * 1024
SHA40_RE = re.compile(r"\A[0-9a-f]{40}\Z")
SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
CHECKSUM_RE = re.compile(r"\A([0-9a-f]{64})  ([A-Za-z0-9_.-]+)\Z")
FORBIDDEN_KEY_FRAGMENTS = (
    "adjacency",
    "full_activation",
    "complete_activation",
    "full_preactivation",
    "dense_preactivation",
    "gradient_tensor",
    "raw_graph",
    "model_weight",
    "transcoder_weight",
)
FORBIDDEN_TEXT_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"),
    re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    re.compile(r"(?<![A-Za-z0-9_])(?:file://)?/Users/"),
    re.compile(r"(?<![A-Za-z0-9_])(?:file://)?/home/"),
    re.compile(r"(?<![A-Za-z0-9_])(?:file://)?/private/var/"),
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPOSITORY_ROOT / "configs/stage1b_measurement_primitives_gemma3_270m_mps_bf16.yaml"
)
SCHEMA_PATH = (
    REPOSITORY_ROOT / "configs/stage1b_measurement_primitives_artifact_schema.json"
)


def _fail(message: str) -> NoReturn:
    raise ArtifactValidationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        _fail(f"{label} keys differ from the frozen schema")


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return int(value)


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric")
    result = float(value)
    if not (float("-inf") < result < float("inf")):
        _fail(f"{label} must be finite")
    return result


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{label} must be a non-empty string")
    return value


def strict_json_bytes(data: bytes, *, label: str) -> dict[str, Any]:
    """Parse one finite JSON object from already-hashed bytes; reject duplicates."""

    def reject_constant(value: str) -> NoReturn:
        _fail(f"non-finite JSON constant is forbidden: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(f"invalid JSON artifact {label}") from error
    result = _mapping(value, label)
    assert_publication_safe(result)
    _scan_forbidden_keys(result)
    return result


def _scan_forbidden_keys(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = key.casefold().replace("-", "_")
            if any(fragment in normalized for fragment in FORBIDDEN_KEY_FRAGMENTS):
                _fail(f"forbidden payload key at {path}.{key}")
            _scan_forbidden_keys(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_forbidden_keys(item, path=f"{path}[{index}]")


def validate_bundle_structure(directory: Path) -> dict[str, bytes]:
    """Read each exact regular file once and enforce the total bundle cap."""

    if directory.is_symlink() or not directory.is_dir():
        _fail("artifact directory must be a real directory")
    observed = {path.name for path in directory.iterdir()}
    if observed != ARTIFACT_ALLOWLIST:
        _fail("artifact directory differs from the exact allowlist")
    payloads: dict[str, bytes] = {}
    total = 0
    for name in sorted(ARTIFACT_ALLOWLIST):
        path = directory / name
        stat = path.lstat()
        if path.is_symlink() or not path.is_file() or stat.st_nlink != 1:
            _fail(f"artifact {name} is not a single-link regular file")
        maximum = (
            MAXIMUM_CHECKSUM_BYTES if name == "checksums.sha256" else MAXIMUM_JSON_BYTES
        )
        if stat.st_size > maximum:
            _fail(f"artifact {name} exceeds its size cap")
        data = path.read_bytes()
        if len(data) != stat.st_size or b"\x00" in data:
            _fail(f"artifact {name} changed during read or contains binary data")
        try:
            text = data.decode("utf-8")
        except UnicodeError as error:
            raise ArtifactValidationError(f"artifact {name} is not UTF-8") from error
        if any(pattern.search(text) for pattern in FORBIDDEN_TEXT_PATTERNS):
            _fail(f"artifact {name} contains a secret or private path")
        payloads[name] = data
        total += len(data)
    if total >= MAXIMUM_TOTAL_BYTES:
        _fail("artifact bundle is not below the 5 MiB cap")
    return payloads


def validate_checksums(payloads: Mapping[str, bytes]) -> dict[str, str]:
    """Require exact checksum coverage over the same immutable byte snapshots."""

    entries: dict[str, str] = {}
    try:
        lines = payloads["checksums.sha256"].decode("utf-8").splitlines()
    except UnicodeError as error:
        raise ArtifactValidationError("checksum manifest is not UTF-8") from error
    for line in lines:
        match = CHECKSUM_RE.fullmatch(line)
        if match is None:
            _fail("checksum manifest line is malformed")
        digest, name = match.groups()
        if name in entries:
            _fail("checksum manifest contains a duplicate filename")
        entries[name] = digest
    if set(entries) != JSON_FILES:
        _fail("checksum coverage differs from the JSON allowlist")
    for name, expected in entries.items():
        if hashlib.sha256(payloads[name]).hexdigest() != expected:
            _fail(f"checksum mismatch for {name}")
    return entries


def load_bundle(directory: Path) -> dict[str, dict[str, Any]]:
    """Return strict parsed JSON after structural and checksum validation."""

    payloads = validate_bundle_structure(directory)
    validate_checksums(payloads)
    return {name: strict_json_bytes(payloads[name], label=name) for name in JSON_FILES}


def _feature(value: Any, label: str) -> FeatureRef:
    mapping = _mapping(value, label)
    _exact_keys(mapping, {"layer", "position", "feature_id"}, label)
    feature = FeatureRef(
        _integer(mapping["layer"], f"{label}.layer"),
        _integer(mapping["position"], f"{label}.position"),
        _integer(mapping["feature_id"], f"{label}.feature_id"),
    )
    if feature.layer >= 18 or feature.position not in {1, 2, 3, 4, 5}:
        _fail(f"{label} is outside the frozen layer/position domain")
    if feature.feature_id >= 16_384:
        _fail(f"{label} feature ID is outside the frozen PLT width")
    return feature


def _validate_run_manifest(value: dict[str, Any], execution_commit: str) -> None:
    _exact_keys(
        value,
        {
            "schema_version",
            "artifact_type",
            "status",
            "branch",
            "base_commit",
            "execution_commit",
            "fresh_canonical_run",
            "scientific_retry_count",
            "config_sha256",
            "artifact_schema_sha256",
            "runtime_identity",
            "claim_boundary",
        },
        "run manifest",
    )
    for key, expected in (
        ("schema_version", 1),
        ("artifact_type", "stage1b_measurement_primitives_run_manifest"),
        ("status", COMPLETED_STATUS),
        ("branch", BRANCH),
        ("base_commit", BASE_COMMIT),
        ("execution_commit", execution_commit),
        ("fresh_canonical_run", True),
        ("scientific_retry_count", 0),
        ("config_sha256", sha256_file(CONFIG_PATH)),
        ("artifact_schema_sha256", sha256_file(SCHEMA_PATH)),
    ):
        if value.get(key) != expected:
            _fail(f"run manifest {key} is invalid")
    identity = _mapping(value["runtime_identity"], "runtime identity")
    for key, expected in (
        ("backend", "nnsight"),
        ("device", "mps:0"),
        ("dtype", "torch.bfloat16"),
        ("upstream_revision", UPSTREAM_REVISION),
        ("model_revision", MODEL_REVISION),
        ("transcoder_revision", TRANSCODER_REVISION),
        ("transcoder_subfolder", TRANSCODER_SUBFOLDER),
        ("prompt_id", "pilot"),
    ):
        if identity.get(key) != expected:
            _fail(f"runtime identity {key} is invalid")
    claim = _mapping(value["claim_boundary"], "claim boundary")
    expected_claim = {
        "stage1b_measurement_primitives": "completed",
        "stage1c_first_prediction_readiness": True,
        "stage1b_empirical_claim_readiness": False,
        "counterfactual_susceptibility_result": "none",
        "gate_crossing_result": "none",
        "behavioral_importance_result": "none",
        "mediation_result": "none",
        "official_bf16_reproduction": "pending",
        "reference_clt_reproduction": "pending",
        "paper_results_readiness": False,
    }
    if claim != expected_claim:
        _fail("run manifest claim boundary is invalid")


def _validate_asset_manifest(value: dict[str, Any]) -> None:
    _exact_keys(
        value,
        {
            "schema_version",
            "artifact_type",
            "status",
            "download_performed",
            "network_accessed",
            "authentication_used",
            "authentication_value_recorded",
            "exact_allowlist_hashes_verified",
            "full_repository_downloaded",
            "other_widths_consumed",
            "feature_visualization_consumed",
            "actual_total_bytes",
            "model",
            "transcoder",
        },
        "asset manifest",
    )
    for key, expected in (
        ("schema_version", 1),
        ("artifact_type", "stage1b_measurement_primitives_asset_manifest"),
        ("status", "verified"),
        ("download_performed", False),
        ("network_accessed", False),
        ("authentication_used", False),
        ("authentication_value_recorded", False),
        ("exact_allowlist_hashes_verified", True),
        ("full_repository_downloaded", False),
        ("other_widths_consumed", False),
        ("feature_visualization_consumed", False),
        ("actual_total_bytes", 2_087_816_677),
    ):
        if value.get(key) != expected:
            _fail(f"asset manifest {key} is invalid")
    model = _mapping(value["model"], "asset model")
    _exact_keys(model, {"identifier", "revision", "total_bytes"}, "asset model")
    if model != {
        "identifier": "google/gemma-3-270m",
        "revision": MODEL_REVISION,
        "total_bytes": 575_454_257,
    }:
        _fail("asset model identity is invalid")
    transcoder = _mapping(value["transcoder"], "asset transcoder")
    _exact_keys(
        transcoder,
        {"identifier", "revision", "subfolder", "layer_count", "total_bytes"},
        "asset transcoder",
    )
    if transcoder != {
        "identifier": "mwhanna/gemma-scope-2-270m-pt",
        "revision": TRANSCODER_REVISION,
        "subfolder": TRANSCODER_SUBFOLDER,
        "layer_count": 18,
        "total_bytes": 1_512_362_420,
    }:
        _fail("asset transcoder identity is invalid")


def _validate_environment_manifest(
    value: dict[str, Any], execution_commit: str
) -> None:
    _exact_keys(
        value,
        {
            "schema_version",
            "artifact_type",
            "status",
            "execution_commit",
            "platform",
            "packages",
            "accelerator",
            "privacy",
        },
        "environment manifest",
    )
    for key, expected in (
        ("schema_version", 1),
        ("artifact_type", "stage1b_measurement_primitives_environment"),
        ("status", "passed"),
        ("execution_commit", execution_commit),
    ):
        if value.get(key) != expected:
            _fail(f"environment manifest {key} is invalid")
    platform_record = _mapping(value["platform"], "environment platform")
    _exact_keys(
        platform_record,
        {"system", "machine", "python", "host_class"},
        "environment platform",
    )
    if platform_record != {
        "system": "Darwin",
        "machine": "arm64",
        "python": "3.11.13",
        "host_class": "Apple M2 Max 32 GiB unified memory",
    }:
        _fail("environment platform identity is invalid")
    packages = _mapping(value["packages"], "environment packages")
    if packages != {
        "circuit-tracer": "0.5.2",
        "nnsight": "0.6.1",
        "torch": "2.6.0",
        "transformers": "4.57.3",
    }:
        _fail("environment package lock is invalid")
    accelerator = _mapping(value["accelerator"], "environment accelerator")
    if accelerator != {
        "device": "mps:0",
        "dtype": "torch.bfloat16",
        "mps_built": True,
        "mps_available": True,
        "mps_bfloat16_probe": "passed",
        "fallback_variable_present": False,
        "outer_autocast_enabled": False,
        "scientific_tensor_device": "mps",
        "graph_metadata_device": "cpu",
    }:
        _fail("environment accelerator evidence is invalid")
    privacy = _mapping(value["privacy"], "environment privacy")
    if privacy != {
        "network_accessed": False,
        "credential_values_read": False,
        "secret_values_recorded": False,
        "private_paths_recorded": False,
    }:
        _fail("environment privacy evidence is invalid")


def _validate_candidates(value: dict[str, Any], config: Mapping[str, Any]) -> None:
    _exact_keys(
        value,
        {"schema_version", "artifact_type", "candidate_count", "candidates"},
        "near-threshold candidates",
    )
    if value.get("schema_version") != 1 or value.get("artifact_type") != (
        "stage1b_measurement_primitives_near_threshold_candidates"
    ):
        _fail("near-threshold candidate artifact identity is invalid")
    rows = value.get("candidates")
    if not isinstance(rows, list) or len(rows) > int(config["scanner"]["global_top_k"]):
        _fail("near-threshold candidate table is oversized")
    if value.get("candidate_count") != len(rows):
        _fail("near-threshold candidate count is inconsistent")
    candidates: list[NearThresholdCandidate] = []
    for row in rows:
        item = _mapping(row, "candidate row")
        _exact_keys(
            item,
            {
                "feature",
                "preactivation",
                "activation",
                "threshold",
                "margin",
                "device",
                "dtype",
                "activity",
            },
            "candidate row",
        )
        if (
            item["activity"] != "inactive"
            or item["device"] != "mps:0"
            or item["dtype"] != "torch.bfloat16"
        ):
            _fail("candidate activity is not inactive")
        candidates.append(
            NearThresholdCandidate(
                feature=_feature(item["feature"], "candidate feature"),
                preactivation=_number(item["preactivation"], "candidate preactivation"),
                activation=_number(item["activation"], "candidate activation"),
                threshold=_number(item["threshold"], "candidate threshold"),
                margin=_number(item["margin"], "candidate margin"),
                device=_text(item["device"], "candidate device"),
                dtype=_text(item["dtype"], "candidate dtype"),
            )
        )
    if tuple(item.sort_key for item in candidates) != tuple(
        sorted(item.sort_key for item in candidates)
    ):
        _fail("near-threshold candidates are not in frozen order")
    if len({item.feature for item in candidates}) != len(candidates):
        _fail("near-threshold candidates contain duplicates")


def _parse_pair_rows(
    value: dict[str, Any], config: Mapping[str, Any]
) -> tuple[tuple[ActivePairReference, ...], tuple[LocalResponseEstimate, ...]]:
    _exact_keys(
        value,
        {"schema_version", "artifact_type", "pair_count", "pairs"},
        "local-response pairs",
    )
    if value.get("schema_version") != 1 or value.get("artifact_type") != (
        "stage1b_measurement_primitives_local_response_validation_pairs"
    ):
        _fail("local-response pair artifact identity is invalid")
    rows = value.get("pairs")
    expected_ids = config["responses"]["canonical_pair_ids"]
    if not isinstance(rows, list) or len(rows) != 64:
        _fail("local-response pair table must contain exactly 64 rows")
    if value.get("pair_count") != 64:
        _fail("local-response pair count is invalid")
    if [row.get("pair_id") for row in rows if isinstance(row, Mapping)] != expected_ids:
        _fail("local-response pair IDs differ from the frozen canonical order")
    references: list[ActivePairReference] = []
    estimates: list[LocalResponseEstimate] = []
    for row in rows:
        item = _mapping(row, "local-response pair row")
        _exact_keys(
            item,
            {
                "pair_id",
                "source",
                "target",
                "source_activation",
                "target_preactivation",
                "raw_edge",
                "targeted_response",
                "reconstructed_edge",
                "symmetric_normalized_error",
                "device",
                "dtype",
                "method",
                "convention",
                "graph_edge_used",
            },
            "local-response pair row",
        )
        source = _feature(item["source"], "pair source")
        target = _feature(item["target"], "pair target")
        if item["graph_edge_used"] is not False:
            _fail("targeted-response row claims graph-edge use")
        if (
            item["device"] != "mps:0"
            or item["dtype"] != "torch.bfloat16"
            or item["method"] != config["responses"]["method"]
            or item["convention"] != config["responses"]["convention"]
        ):
            _fail("targeted-response runtime identity is invalid")
        reference = ActivePairReference(
            pair_id=_text(item["pair_id"], "pair ID"),
            source=source,
            target=target,
            source_activation=_number(
                item["source_activation"], "pair source activation"
            ),
            raw_edge=_number(item["raw_edge"], "pair raw edge"),
        )
        estimate = LocalResponseEstimate(
            source=source,
            target=target,
            source_activation=_number(
                item["source_activation"], "pair source activation"
            ),
            target_preactivation=_number(
                item["target_preactivation"], "pair target preactivation"
            ),
            response=_number(item["targeted_response"], "targeted response"),
            device=_text(item["device"], "pair device"),
            dtype=_text(item["dtype"], "pair dtype"),
            method=_text(item["method"], "pair method"),
            convention=_text(item["convention"], "pair convention"),
            graph_edge_used=False,
        )
        rebuilt = estimate.source_activation * estimate.response
        if _number(item["reconstructed_edge"], "reconstructed edge") != rebuilt:
            _fail("serialized reconstructed edge is inconsistent")
        error = symmetric_normalized_error(reference.raw_edge, rebuilt)
        if (
            _number(
                item["symmetric_normalized_error"],
                "symmetric normalized error",
            )
            != error
        ):
            _fail("serialized pair error is inconsistent")
        references.append(reference)
        estimates.append(estimate)
    validate_pair_distribution(
        references,
        minimum_pairs=64,
        minimum_target_layers=6,
        minimum_target_positions=3,
        require_both_signs=True,
    )
    endpoint_records = [
        {
            "pair_id": item.pair_id,
            "source": {
                "layer": item.source.layer,
                "position": item.source.position,
                "feature_id": item.source.feature_id,
            },
            "target": {
                "layer": item.target.layer,
                "position": item.target.position,
                "feature_id": item.target.feature_id,
            },
        }
        for item in references
    ]
    digest = hashlib.sha256(
        json.dumps(endpoint_records, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if digest != config["responses"]["canonical_endpoint_manifest_sha256"]:
        _fail("canonical endpoint digest differs from the frozen config")
    return tuple(references), tuple(estimates)


def _validate_response_summary(
    value: dict[str, Any], metrics: Any, config: Mapping[str, Any]
) -> None:
    required = {
        "schema_version",
        "artifact_type",
        "status",
        "edge_floor",
        "pair_count",
        "above_edge_floor_count",
        "spearman",
        "sign_agreement",
        "median_symmetric_normalized_error",
        "p95_symmetric_normalized_error",
        "method",
        "convention",
        "graph_edge_orientation",
        "graph_edge_used_by_targeted_path",
        "calibration_pair_ids_disjoint",
    }
    _exact_keys(value, required, "local-response summary")
    for key, expected in (
        ("schema_version", 1),
        (
            "artifact_type",
            "stage1b_measurement_primitives_local_response_validation_summary",
        ),
        ("edge_floor", config["responses"]["edge_floor"]),
    ):
        if value.get(key) != expected:
            _fail(f"local-response summary {key} is invalid")
    for metric_key in (
        "pair_count",
        "above_edge_floor_count",
        "spearman",
        "sign_agreement",
        "median_symmetric_normalized_error",
        "p95_symmetric_normalized_error",
    ):
        if value.get(metric_key) != getattr(metrics, metric_key):
            _fail(
                f"local-response summary {metric_key} was not independently reproduced"
            )
    tolerance = config["tolerances"]
    if metrics.spearman < float(tolerance["spearman_minimum"]):
        _fail("local-response Spearman is below the hard floor")
    if metrics.sign_agreement < float(tolerance["sign_agreement_minimum"]):
        _fail("local-response sign agreement is below the hard floor")
    if metrics.median_symmetric_normalized_error > float(
        tolerance["median_symmetric_normalized_error_maximum"]
    ):
        _fail("local-response median error exceeds the hard ceiling")
    if metrics.p95_symmetric_normalized_error > float(
        tolerance["p95_symmetric_normalized_error_maximum"]
    ):
        _fail("local-response p95 error exceeds the hard ceiling")
    for summary_key, summary_expected in (
        ("status", "passed"),
        ("method", config["responses"]["method"]),
        ("convention", config["responses"]["convention"]),
        ("graph_edge_orientation", "target_row_source_column"),
        ("graph_edge_used_by_targeted_path", False),
        ("calibration_pair_ids_disjoint", True),
    ):
        if value.get(summary_key) != summary_expected:
            _fail(f"local-response summary {summary_key} is invalid")


def _validate_scanner_summary(
    value: dict[str, Any], config: Mapping[str, Any], candidate_count: int
) -> None:
    _exact_keys(
        value,
        {
            "schema_version",
            "artifact_type",
            "status",
            "group_count",
            "selected_layers",
            "selected_positions",
            "feature_width",
            "chunk_sizes",
            "dense_oracle_chunk_size",
            "canonical_chunk_size",
            "top_k_per_group",
            "global_top_k",
            "exact_candidate_identity_and_order",
            "bounded_oracle_recall",
            "candidate_count",
            "maximum_retained_candidates",
            "persisted_dense_arrays",
            "loaded_gate",
            "threshold_equality_activity",
            "device",
            "dtype",
        },
        "scanner oracle summary",
    )
    scanner = config["scanner"]
    expected = {
        "schema_version": 1,
        "artifact_type": "stage1b_measurement_primitives_scanner_oracle_summary",
        "status": "passed",
        "group_count": 90,
        "selected_layers": scanner["selected_layers"],
        "selected_positions": scanner["selected_positions"],
        "feature_width": 16_384,
        "chunk_sizes": [257, 1024, 4096],
        "dense_oracle_chunk_size": 16_384,
        "canonical_chunk_size": 1024,
        "top_k_per_group": 8,
        "global_top_k": 128,
        "exact_candidate_identity_and_order": True,
        "bounded_oracle_recall": 1.0,
        "candidate_count": candidate_count,
        "persisted_dense_arrays": False,
        "loaded_gate": "a=z*1[z>tau]",
        "threshold_equality_activity": "inactive",
        "device": "mps:0",
        "dtype": "torch.bfloat16",
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            _fail(f"scanner oracle summary {key} is invalid")
    maximum_retained = _integer(
        value["maximum_retained_candidates"],
        "scanner maximum retained candidates",
    )
    if maximum_retained > 8 + 4096:
        _fail("scanner retained more than one chunk plus top-K")


def _validate_memory_summary(value: dict[str, Any], config: Mapping[str, Any]) -> None:
    _exact_keys(
        value,
        {
            "schema_version",
            "artifact_type",
            "status",
            "safety_limits",
            "worker",
            "supervisor",
        },
        "memory summary",
    )
    for key, expected in (
        ("schema_version", 1),
        ("artifact_type", "stage1b_measurement_primitives_memory_timing"),
        ("status", "passed"),
        ("safety_limits", config["safety_limits"]),
    ):
        if value.get(key) != expected:
            _fail(f"memory summary {key} is invalid")
    worker = _mapping(value["worker"], "worker telemetry")
    required_worker = {
        "started_at_unix",
        "finished_at_unix",
        "sample_count",
        "sampling_interval_seconds",
        "attempt_peaks",
        "stage_peaks",
        "thermal_states",
        "violations",
        "telemetry_failures",
    }
    _exact_keys(worker, required_worker, "worker telemetry")
    if worker["violations"] != [] or worker["telemetry_failures"] != 0:
        _fail("worker telemetry contains a failure")
    if worker["thermal_states"] != ["nominal"]:
        _fail("worker thermal evidence is not nominal")
    _integer(worker["sample_count"], "worker sample count", minimum=1)
    if worker["sampling_interval_seconds"] != 1.0:
        _fail("worker sampling interval changed")
    peaks = _mapping(worker["attempt_peaks"], "worker attempt peaks")
    required_peaks = {
        "mps_current_bytes",
        "mps_driver_bytes",
        "process_rss_bytes",
        "swap_used_bytes",
        "swap_growth_bytes",
        "minimum_available_memory_bytes",
    }
    _exact_keys(peaks, required_peaks, "worker attempt peaks")
    limits = config["safety_limits"]
    if _integer(peaks["mps_driver_bytes"], "MPS driver peak") > int(
        limits["maximum_mps_driver_bytes"]
    ):
        _fail("MPS driver peak exceeds the frozen cap")
    if _integer(peaks["process_rss_bytes"], "process RSS peak") > int(
        limits["maximum_process_rss_bytes"]
    ):
        _fail("process RSS peak exceeds the frozen cap")
    if _integer(peaks["swap_growth_bytes"], "swap growth") > int(
        limits["maximum_swap_growth_bytes"]
    ):
        _fail("swap growth exceeds the frozen cap")
    if _integer(
        peaks["minimum_available_memory_bytes"], "minimum available memory"
    ) < int(limits["minimum_available_memory_bytes"]):
        _fail("available memory crossed the frozen floor")
    stages = _mapping(worker["stage_peaks"], "worker stage peaks")
    required_stages = {
        "worker_start",
        "replacement_runtime_loading",
        "scanner_dense_oracle",
        "scanner_chunk_257",
        "scanner_chunk_1024",
        "scanner_chunk_4096",
        "ephemeral_graph_reference",
        "targeted_vjp_canonical",
    }
    if set(stages) != required_stages:
        _fail("worker stage telemetry coverage is invalid")
    for stage_name, stage_value in stages.items():
        stage = _mapping(stage_value, f"worker stage {stage_name}")
        _exact_keys(stage, required_peaks, f"worker stage {stage_name}")
        for peak_key in (
            "mps_current_bytes",
            "mps_driver_bytes",
            "process_rss_bytes",
            "swap_used_bytes",
            "swap_growth_bytes",
        ):
            if _integer(stage[peak_key], f"{stage_name}.{peak_key}") > _integer(
                peaks[peak_key], f"attempt.{peak_key}"
            ):
                _fail("stage peak exceeds the corresponding attempt peak")
        if _integer(
            stage["minimum_available_memory_bytes"],
            f"{stage_name}.minimum_available_memory_bytes",
        ) < _integer(
            peaks["minimum_available_memory_bytes"],
            "attempt.minimum_available_memory_bytes",
        ):
            _fail("stage available-memory minimum is below the attempt minimum")
    supervisor = _mapping(value["supervisor"], "supervisor telemetry")
    required_supervisor = {
        "returncode",
        "timed_out",
        "safety_terminated",
        "termination_signal",
        "telemetry_failures",
        "sample_count",
        "peak_process_group_rss_bytes",
        "minimum_available_memory_bytes",
        "peak_swap_growth_bytes",
        "thermal_states",
        "started_at_unix",
        "finished_at_unix",
    }
    _exact_keys(supervisor, required_supervisor, "supervisor telemetry")
    for key, expected in (
        ("returncode", 0),
        ("timed_out", False),
        ("safety_terminated", False),
        ("termination_signal", None),
        ("telemetry_failures", 0),
        ("thermal_states", ["nominal"]),
    ):
        if supervisor.get(key) != expected:
            _fail(f"supervisor telemetry {key} is invalid")
    _integer(supervisor["sample_count"], "supervisor sample count", minimum=1)
    if _integer(
        supervisor["peak_process_group_rss_bytes"], "supervisor RSS peak"
    ) > int(limits["maximum_process_rss_bytes"]):
        _fail("supervisor RSS peak exceeds the frozen cap")
    if _integer(supervisor["peak_swap_growth_bytes"], "supervisor swap growth") > int(
        limits["maximum_swap_growth_bytes"]
    ):
        _fail("supervisor swap growth exceeds the frozen cap")
    if _integer(
        supervisor["minimum_available_memory_bytes"],
        "supervisor minimum available memory",
    ) < int(limits["minimum_available_memory_bytes"]):
        _fail("supervisor available memory crossed the frozen floor")
    worker_started = _number(worker["started_at_unix"], "worker start time")
    worker_finished = _number(worker["finished_at_unix"], "worker finish time")
    supervisor_started = _number(supervisor["started_at_unix"], "supervisor start time")
    supervisor_finished = _number(
        supervisor["finished_at_unix"], "supervisor finish time"
    )
    if not (
        supervisor_started <= worker_started < worker_finished <= supervisor_finished
    ):
        _fail("worker/supervisor timing order is invalid")


def _validate_attempts(value: dict[str, Any]) -> None:
    _exact_keys(
        value,
        {
            "schema_version",
            "artifact_type",
            "status",
            "scientific_retry_count",
            "attempts",
        },
        "attempts",
    )
    for key, expected in (
        ("schema_version", 1),
        ("artifact_type", "stage1b_measurement_primitives_attempts"),
        ("status", "passed"),
        ("scientific_retry_count", 0),
    ):
        if value.get(key) != expected:
            _fail(f"attempts {key} is invalid")
    attempts = value["attempts"]
    if not isinstance(attempts, list) or len(attempts) != 1:
        _fail("canonical attempt history must contain exactly one attempt")
    attempt = _mapping(attempts[0], "canonical attempt")
    expected_attempt = {
        "mode": "canonical",
        "fresh_process": True,
        "worker_status": "passed",
        "supervisor_returncode": 0,
        "timed_out": False,
        "safety_terminated": False,
        "telemetry_failures": 0,
        "calibration_artifact_read": False,
    }
    if attempt != expected_attempt:
        _fail("canonical attempt record is invalid")


def validate_stage1b_artifacts(
    directory: Path,
    *,
    config: Mapping[str, Any],
    execution_commit: str,
) -> dict[str, Any]:
    """Validate the complete canonical bundle and recompute scientific metrics."""

    if SHA40_RE.fullmatch(execution_commit) is None:
        _fail("execution commit must be an exact 40-character SHA")
    if config.get("phase") != "canonical_frozen":
        _fail("artifact validation requires a canonical-frozen config")
    artifacts = load_bundle(directory)
    _validate_run_manifest(artifacts["run_manifest.json"], execution_commit)
    _validate_asset_manifest(artifacts["asset_manifest.json"])
    _validate_environment_manifest(
        artifacts["environment_manifest.json"], execution_commit
    )
    _validate_candidates(artifacts["near_threshold_candidates.json"], config)
    references, estimates = _parse_pair_rows(
        artifacts["local_response_validation_pairs.json"], config
    )
    edge_floor = float(config["responses"]["edge_floor"])
    metrics = compute_local_response_metrics(
        references, estimates, edge_floor=edge_floor
    )
    _validate_response_summary(
        artifacts["local_response_validation_summary.json"], metrics, config
    )
    candidate_count = artifacts["near_threshold_candidates.json"]["candidate_count"]
    _validate_scanner_summary(
        artifacts["scanner_oracle_summary.json"], config, candidate_count
    )
    _validate_memory_summary(artifacts["memory_timing_summary.json"], config)
    _validate_attempts(artifacts["attempts.json"])
    return {
        "status": "passed",
        "artifact_count": len(ARTIFACT_ALLOWLIST),
        "pair_count": metrics.pair_count,
        "candidate_count": artifacts["near_threshold_candidates.json"][
            "candidate_count"
        ],
        "verdict": COMPLETED_STATUS,
    }


__all__ = [
    "ARTIFACT_ALLOWLIST",
    "MAXIMUM_TOTAL_BYTES",
    "load_bundle",
    "strict_json_bytes",
    "validate_bundle_structure",
    "validate_checksums",
    "validate_stage1b_artifacts",
]
