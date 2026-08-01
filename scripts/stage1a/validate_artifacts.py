#!/usr/bin/env python3
"""Validate present Stage 1A summaries and their checksum manifest.

Absent empirical summaries are not fabricated. Every JSON artifact that is
present must use the strict Stage 1A envelope, contain publication-safe finite
data, and appear exactly once in the deterministic checksum manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
DEFAULT_ARTIFACT_DIRECTORY = REPOSITORY_ROOT / "results" / "stage1a"
DEFAULT_CHECKSUM_NAME = "checksums.sha256"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cfsus.reproduction.artifacts import (  # noqa: E402
    ArtifactValidationError,
    assert_publication_safe,
    build_checksum_manifest,
    validate_artifact_envelope,
    verify_checksum_manifest,
    write_checksum_manifest_atomic,
)

EXPECTED_TYPE_BY_FILENAME = {
    "asset_manifest.json": "asset_manifest",
    "attribution_summary.json": "attribution_summary",
    "colab_handoff_manifest.json": "colab_handoff_manifest",
    "environment_manifest.json": "environment_manifest",
    "intervention_summary.json": "intervention_summary",
    "semantics_summary.json": "semantics_summary",
}

_SHA40 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_UPSTREAM_COMMIT = "8f1e2438df612464e229e44c4a00ff637bf9379b"
_MODEL_ID = "google/gemma-2-2b"
_MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
_TRANSCODER_ID = "mwhanna/gemma-scope-transcoders"
_TRANSCODER_REVISION = "bd5773156dea09893636c801df1237d0410307d2"
_ATTRIBUTION_PROMPT = "The capital of state containing Dallas is"
_INTERVENTION_PROMPT = "Hecho: Michael Jordan juega al"


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{label} must be a JSON object")
    return value


def _list(value: object, label: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = "a non-empty" if nonempty else "a"
        raise ArtifactValidationError(f"{label} must be {qualifier} JSON array")
    return value


def _require_keys(value: dict[str, Any], required: set[str], label: str) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ArtifactValidationError(
            f"{label} is missing required keys: {', '.join(missing)}"
        )


def _string(value: object, label: str, *, expected: str | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactValidationError(f"{label} must be a non-empty string")
    if expected is not None and value != expected:
        raise ArtifactValidationError(f"{label} must equal {expected!r}")
    return value


def _boolean(value: object, label: str, *, expected: bool | None = None) -> bool:
    if not isinstance(value, bool):
        raise ArtifactValidationError(f"{label} must be a boolean")
    if expected is not None and value is not expected:
        raise ArtifactValidationError(f"{label} must be {expected}")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ArtifactValidationError(
            f"{label} must be an integer greater than or equal to {minimum}"
        )
    return value


def _number(value: object, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactValidationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ArtifactValidationError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ArtifactValidationError(f"{label} must be at least {minimum}")
    return result


def _sha(value: object, label: str, *, expected: str | None = None) -> str:
    revision = _string(value, label)
    if _SHA40.fullmatch(revision) is None:
        raise ArtifactValidationError(f"{label} must be a lowercase 40-character SHA")
    if expected is not None and revision != expected:
        raise ArtifactValidationError(f"{label} does not match the required revision")
    return revision


def _sha256(value: object, label: str) -> str:
    digest = _string(value, label)
    if _SHA256.fullmatch(digest) is None:
        raise ArtifactValidationError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _token_sequence(payload: dict[str, Any], label: str) -> None:
    token_ids = _list(payload.get("token_ids"), f"{label}.token_ids", nonempty=True)
    for index, token_id in enumerate(token_ids):
        _integer(token_id, f"{label}.token_ids[{index}]")
    tokens = payload.get("tokens")
    if tokens is not None:
        token_strings = _list(tokens, f"{label}.tokens", nonempty=True)
        if len(token_strings) != len(token_ids):
            raise ArtifactValidationError(
                f"{label}.tokens must align one-to-one with token_ids"
            )
        for index, token in enumerate(token_strings):
            _string(token, f"{label}.tokens[{index}]")


def _timing(value: object, label: str) -> None:
    timing = _mapping(value, label)
    _require_keys(
        timing,
        {"wall_seconds", "process_peak_rss_bytes", "cuda_peak_allocated_bytes"},
        label,
    )
    _number(timing["wall_seconds"], f"{label}.wall_seconds", minimum=0.0)
    _integer(
        timing["process_peak_rss_bytes"],
        f"{label}.process_peak_rss_bytes",
        minimum=1,
    )
    if timing["cuda_peak_allocated_bytes"] is not None:
        _integer(
            timing["cuda_peak_allocated_bytes"],
            f"{label}.cuda_peak_allocated_bytes",
        )


def _validate_runtime_provenance(record: dict[str, Any], label: str) -> None:
    provenance = _mapping(record["provenance"], f"{label} provenance")
    _require_keys(
        provenance,
        {
            "upstream_repository",
            "upstream_revision",
            "model_identifier",
            "model_revision",
            "transcoder_identifier",
            "transcoder_revision",
            "backend",
            "device",
            "dtype",
        },
        f"{label} provenance",
    )
    _sha(
        provenance["upstream_revision"],
        f"{label} provenance.upstream_revision",
        expected=_UPSTREAM_COMMIT,
    )
    _string(
        provenance["model_identifier"],
        f"{label} provenance.model_identifier",
        expected=_MODEL_ID,
    )
    _sha(
        provenance["model_revision"],
        f"{label} provenance.model_revision",
        expected=_MODEL_REVISION,
    )
    _string(
        provenance["transcoder_identifier"],
        f"{label} provenance.transcoder_identifier",
        expected=_TRANSCODER_ID,
    )
    _sha(
        provenance["transcoder_revision"],
        f"{label} provenance.transcoder_revision",
        expected=_TRANSCODER_REVISION,
    )
    _string(
        provenance["backend"], f"{label} provenance.backend", expected="transformerlens"
    )
    if provenance["device"] not in {"cuda", "mps"}:
        raise ArtifactValidationError(f"{label} provenance.device must be cuda or mps")
    _string(provenance["dtype"], f"{label} provenance.dtype", expected="bfloat16")


def _validate_environment(record: dict[str, Any]) -> None:
    if record["status"] != "observed":
        raise ArtifactValidationError("environment_manifest status must be observed")
    provenance = _mapping(record["provenance"], "environment provenance")
    _require_keys(
        provenance,
        {
            "code_commit",
            "environment_lock",
            "lock_format",
            "lock_provenance",
            "observation_scope",
            "stage",
            "upstream_commit",
        },
        "environment provenance",
    )
    if provenance["code_commit"] is not None:
        _sha(provenance["code_commit"], "environment provenance.code_commit")
    _string(provenance["stage"], "environment provenance.stage", expected="stage1a")
    _sha(
        provenance["upstream_commit"],
        "environment provenance.upstream_commit",
        expected=_UPSTREAM_COMMIT,
    )
    _string(provenance["environment_lock"], "environment provenance.environment_lock")

    payload = _mapping(record["payload"], "environment payload")
    _require_keys(
        payload,
        {
            "accelerators",
            "assets",
            "execution_policy",
            "offline_only",
            "packages",
            "platform",
            "privacy",
            "python",
        },
        "environment payload",
    )
    _boolean(payload["offline_only"], "environment payload.offline_only", expected=True)
    python = _mapping(payload["python"], "environment payload.python")
    _require_keys(python, {"implementation", "version"}, "environment payload.python")
    _string(python["implementation"], "environment payload.python.implementation")
    _string(python["version"], "environment payload.python.version")
    packages = _mapping(payload["packages"], "environment payload.packages")
    _require_keys(
        packages, {"circuit_tracer", "versions"}, "environment payload.packages"
    )
    versions = _mapping(packages["versions"], "environment payload.packages.versions")
    for distribution in ("circuit-tracer", "torch", "transformer-lens", "transformers"):
        _string(
            versions.get(distribution),
            f"environment payload.packages.versions.{distribution}",
        )
    circuit_tracer = _mapping(
        packages["circuit_tracer"], "environment payload.packages.circuit_tracer"
    )
    _string(
        circuit_tracer.get("version"),
        "environment payload.packages.circuit_tracer.version",
        expected="0.5.2",
    )
    direct_url = _mapping(
        circuit_tracer.get("direct_url"),
        "environment payload.packages.circuit_tracer.direct_url",
    )
    for key in ("commit_id", "requested_revision"):
        _sha(
            direct_url.get(key),
            f"environment payload.packages.circuit_tracer.direct_url.{key}",
            expected=_UPSTREAM_COMMIT,
        )
    _string(
        direct_url.get("vcs"),
        "environment payload.packages.circuit_tracer.direct_url.vcs",
        expected="git",
    )


def _validate_asset_manifest(record: dict[str, Any]) -> None:
    if record["status"] not in {"blocked", "failed", "resolved"}:
        raise ArtifactValidationError("asset_manifest has an invalid resolution status")
    provenance = _mapping(record["provenance"], "asset provenance")
    required_pins = {
        "upstream_commit": _UPSTREAM_COMMIT,
        "model_revision": _MODEL_REVISION,
        "transcoder_revision": _TRANSCODER_REVISION,
    }
    _require_keys(
        provenance,
        {
            "upstream_commit",
            "model_repo_id",
            "model_revision",
            "transcoder_repo_id",
            "transcoder_revision",
            "metadata_source",
        },
        "asset provenance",
    )
    for key, expected in required_pins.items():
        _sha(provenance[key], f"asset provenance.{key}", expected=expected)
    _string(
        provenance["model_repo_id"],
        "asset provenance.model_repo_id",
        expected=_MODEL_ID,
    )
    _string(
        provenance["transcoder_repo_id"],
        "asset provenance.transcoder_repo_id",
        expected=_TRANSCODER_ID,
    )

    payload = _mapping(record["payload"], "asset payload")
    _require_keys(
        payload,
        {
            "resolution_status",
            "source",
            "assets",
            "access_probe",
            "downloads_requested",
            "download_results",
            "safety",
            "transcoder_comparison",
        },
        "asset payload",
    )
    _string(payload["resolution_status"], "asset payload.resolution_status")
    assets = _mapping(payload["assets"], "asset payload.assets")
    for key, repo_id, revision in (
        ("model", _MODEL_ID, _MODEL_REVISION),
        ("selected_transcoder", _TRANSCODER_ID, _TRANSCODER_REVISION),
    ):
        asset = _mapping(assets.get(key), f"asset payload.assets.{key}")
        _require_keys(
            asset,
            {"repo_id", "verified_revision", "snapshot_file_bytes", "files"},
            f"asset payload.assets.{key}",
        )
        _string(
            asset["repo_id"], f"asset payload.assets.{key}.repo_id", expected=repo_id
        )
        _sha(
            asset["verified_revision"],
            f"asset payload.assets.{key}.verified_revision",
            expected=revision,
        )
        _integer(
            asset["snapshot_file_bytes"],
            f"asset payload.assets.{key}.snapshot_file_bytes",
            minimum=1,
        )
        files = _list(
            asset["files"], f"asset payload.assets.{key}.files", nonempty=True
        )
        names: set[str] = set()
        for index, file_record in enumerate(files):
            item = _mapping(file_record, f"asset payload.assets.{key}.files[{index}]")
            _require_keys(
                item,
                {"name", "size_bytes", "git_blob_id", "lfs_sha256"},
                f"asset payload.assets.{key}.files[{index}]",
            )
            name = _string(
                item["name"], f"asset payload.assets.{key}.files[{index}].name"
            )
            if name in names:
                raise ArtifactValidationError(
                    f"asset payload.assets.{key}.files contains duplicate names"
                )
            names.add(name)
            _integer(
                item["size_bytes"],
                f"asset payload.assets.{key}.files[{index}].size_bytes",
            )
            if name.endswith(".safetensors"):
                _sha256(
                    item["lfs_sha256"],
                    f"asset payload.assets.{key}.files[{index}].lfs_sha256",
                )
        required_names = (
            {
                "config.json",
                "model-00001-of-00003.safetensors",
                "model-00002-of-00003.safetensors",
                "model-00003-of-00003.safetensors",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
            }
            if key == "model"
            else {"config.yaml", *(f"layer_{layer}.safetensors" for layer in range(26))}
        )
        if not required_names.issubset(names):
            raise ArtifactValidationError(
                f"asset payload.assets.{key}.files omits required runtime files"
            )
    probe = _mapping(payload["access_probe"], "asset payload.access_probe")
    _string(
        probe.get("repo_id"), "asset payload.access_probe.repo_id", expected=_MODEL_ID
    )
    _sha(
        probe.get("revision"),
        "asset payload.access_probe.revision",
        expected=_MODEL_REVISION,
    )
    access_granted = _boolean(
        probe.get("access_granted"), "asset payload.access_probe.access_granted"
    )
    if record["status"] == "blocked" and access_granted:
        raise ArtifactValidationError(
            "blocked asset manifest cannot claim model access"
        )
    safety = _mapping(payload["safety"], "asset payload.safety")
    _boolean(
        safety.get("tokens_serialized"),
        "asset payload.safety.tokens_serialized",
        expected=False,
    )
    _boolean(
        safety.get("local_paths_serialized"),
        "asset payload.safety.local_paths_serialized",
        expected=False,
    )


def _validate_colab_handoff(record: dict[str, Any]) -> None:
    if record["status"] not in {"blocked", "resolved"}:
        raise ArtifactValidationError("colab_handoff_manifest has an invalid status")
    provenance = _mapping(record["provenance"], "Colab handoff provenance")
    _require_keys(
        provenance,
        {
            "project_repository",
            "project_ref",
            "upstream_repository",
            "upstream_revision",
        },
        "Colab handoff provenance",
    )
    _sha(
        provenance["upstream_revision"],
        "Colab handoff provenance.upstream_revision",
        expected=_UPSTREAM_COMMIT,
    )
    payload = _mapping(record["payload"], "Colab handoff payload")
    _require_keys(
        payload,
        {"assets", "environment_plan", "execution", "handoff_state"},
        "Colab handoff payload",
    )
    assets = _mapping(payload["assets"], "Colab handoff payload.assets")
    for key, repository, revision in (
        ("model", _MODEL_ID, _MODEL_REVISION),
        ("transcoder", _TRANSCODER_ID, _TRANSCODER_REVISION),
    ):
        asset = _mapping(assets.get(key), f"Colab handoff payload.assets.{key}")
        _string(
            asset.get("repository"),
            f"Colab handoff payload.assets.{key}.repository",
            expected=repository,
        )
        _sha(
            asset.get("revision"),
            f"Colab handoff payload.assets.{key}.revision",
            expected=revision,
        )
    execution = _mapping(payload["execution"], "Colab handoff payload.execution")
    _require_keys(
        execution,
        {
            "base_config",
            "entrypoint",
            "notebook",
            "repository_ref",
            "runtime_overrides",
        },
        "Colab handoff payload.execution",
    )
    _string(
        execution["entrypoint"],
        "Colab handoff payload.execution.entrypoint",
        expected="scripts/stage1a/run_stage1a.py",
    )
    _string(
        execution["notebook"],
        "Colab handoff payload.execution.notebook",
        expected="notebooks/stage1a_official_reproduction_colab.ipynb",
    )
    overrides = _mapping(
        execution["runtime_overrides"],
        "Colab handoff payload.execution.runtime_overrides",
    )
    _string(overrides.get("device"), "Colab handoff runtime device", expected="cuda")
    _string(
        overrides.get("attribution_offload"),
        "Colab handoff attribution offload",
        expected="disk",
    )


def _validate_attribution(record: dict[str, Any]) -> None:
    if record["status"] != "completed":
        raise ArtifactValidationError("attribution_summary status must be completed")
    _validate_runtime_provenance(record, "attribution")
    payload = _mapping(record["payload"], "attribution payload")
    _require_keys(
        payload,
        {
            "source_notebook",
            "prompt",
            "token_ids",
            "tokens",
            "parameters",
            "graph",
            "raw_validation",
            "logit_targets",
            "raw_artifact",
            "timing",
            "seed",
            "classification",
            "claim_boundary",
        },
        "attribution payload",
    )
    _string(
        payload["prompt"], "attribution payload.prompt", expected=_ATTRIBUTION_PROMPT
    )
    _token_sequence(payload, "attribution payload")
    parameters = _mapping(payload["parameters"], "attribution payload.parameters")
    _require_keys(
        parameters,
        {
            "max_n_logits",
            "desired_logit_prob",
            "max_feature_nodes",
            "batch_size",
            "offload",
        },
        "attribution payload.parameters",
    )
    if (
        parameters["max_n_logits"] != 10
        or parameters["desired_logit_prob"] != 0.95
        or parameters["max_feature_nodes"] != 8192
    ):
        raise ArtifactValidationError(
            "attribution payload parameters do not match the official target"
        )
    _integer(
        parameters["batch_size"], "attribution payload.parameters.batch_size", minimum=1
    )
    if parameters["offload"] not in {None, "cpu", "disk"}:
        raise ArtifactValidationError(
            "attribution payload.parameters.offload is invalid"
        )
    graph = _mapping(payload["graph"], "attribution payload.graph")
    _require_keys(
        graph,
        {
            "adjacency_shape",
            "active_feature_count",
            "selected_feature_count",
            "error_node_count",
            "input_node_count",
            "logit_node_count",
            "nonzero_edge_count",
            "finite",
        },
        "attribution payload.graph",
    )
    adjacency_shape = _list(
        graph["adjacency_shape"],
        "attribution payload.graph.adjacency_shape",
        nonempty=True,
    )
    for index, dimension in enumerate(adjacency_shape):
        _integer(
            dimension, f"attribution payload.graph.adjacency_shape[{index}]", minimum=1
        )
    for name in (
        "active_feature_count",
        "error_node_count",
        "input_node_count",
        "logit_node_count",
    ):
        _integer(graph[name], f"attribution payload.graph.{name}", minimum=1)
    _integer(
        graph["selected_feature_count"],
        "attribution payload.graph.selected_feature_count",
        minimum=1,
    )
    _integer(
        graph["nonzero_edge_count"],
        "attribution payload.graph.nonzero_edge_count",
        minimum=1,
    )
    _boolean(graph["finite"], "attribution payload.graph.finite", expected=True)
    raw_validation = _mapping(
        payload["raw_validation"], "attribution payload.raw_validation"
    )
    _boolean(
        raw_validation.get("passed"),
        "attribution payload.raw_validation.passed",
        expected=True,
    )
    raw = _mapping(payload["raw_artifact"], "attribution payload.raw_artifact")
    _require_keys(
        raw, {"path", "sha256", "size_bytes"}, "attribution payload.raw_artifact"
    )
    raw_path = _string(raw["path"], "attribution payload.raw_artifact.path")
    if not raw_path.startswith("results/generated/stage1a/") or not raw_path.endswith(
        ".pt"
    ):
        raise ArtifactValidationError(
            "attribution raw artifact must be an ignored Stage 1A .pt file"
        )
    _sha256(raw["sha256"], "attribution payload.raw_artifact.sha256")
    _integer(
        raw["size_bytes"], "attribution payload.raw_artifact.size_bytes", minimum=1
    )
    targets = _list(
        payload["logit_targets"], "attribution payload.logit_targets", nonempty=True
    )
    for index, target in enumerate(targets):
        item = _mapping(target, f"attribution payload.logit_targets[{index}]")
        _require_keys(
            item,
            {"token_id", "token", "probability_weight"},
            f"attribution payload.logit_targets[{index}]",
        )
        _integer(
            item["token_id"], f"attribution payload.logit_targets[{index}].token_id"
        )
        _string(item["token"], f"attribution payload.logit_targets[{index}].token")
        probability = _number(
            item["probability_weight"],
            f"attribution payload.logit_targets[{index}].probability_weight",
            minimum=0.0,
        )
        if probability > 1.0:
            raise ArtifactValidationError(
                "attribution probability weight must not exceed one"
            )
    _timing(payload["timing"], "attribution payload.timing")
    _integer(payload["seed"], "attribution payload.seed")
    _string(payload["classification"], "attribution payload.classification")
    _string(payload["claim_boundary"], "attribution payload.claim_boundary")


def _validate_intervention(record: dict[str, Any]) -> None:
    if record["status"] != "completed":
        raise ArtifactValidationError("intervention_summary status must be completed")
    _validate_runtime_provenance(record, "intervention")
    payload = _mapping(record["payload"], "intervention payload")
    _require_keys(
        payload,
        {
            "source_notebook",
            "prompt",
            "token_ids",
            "feature",
            "baseline_activation",
            "desired_values",
            "fixed_top_k_union_token_ids",
            "conditions",
            "baseline_noop_comparison",
            "determinism",
            "regime",
            "timing",
            "seed",
            "claim_boundary",
        },
        "intervention payload",
    )
    _string(
        payload["prompt"], "intervention payload.prompt", expected=_INTERVENTION_PROMPT
    )
    _token_sequence(payload, "intervention payload")
    feature = _mapping(payload["feature"], "intervention payload.feature")
    _require_keys(
        feature,
        {"layer", "requested_position", "resolved_position", "feature_id"},
        "intervention payload.feature",
    )
    if (feature["layer"], feature["requested_position"], feature["feature_id"]) != (
        20,
        -1,
        341,
    ):
        raise ArtifactValidationError(
            "intervention payload.feature is not (20, -1, 341)"
        )
    _integer(
        feature["resolved_position"], "intervention payload.feature.resolved_position"
    )
    baseline = _number(
        payload["baseline_activation"],
        "intervention payload.baseline_activation",
        minimum=0.0,
    )
    if baseline <= 0.0:
        raise ArtifactValidationError(
            "intervention baseline activation must be positive"
        )
    desired_values = _list(
        payload["desired_values"], "intervention payload.desired_values", nonempty=True
    )
    if len(desired_values) != 3:
        raise ArtifactValidationError(
            "intervention desired_values must contain three conditions"
        )
    for index, (item, expected_alpha) in enumerate(
        zip(desired_values, (0.0, 0.5, 1.0), strict=True)
    ):
        condition = _mapping(item, f"intervention payload.desired_values[{index}]")
        alpha = _number(
            condition.get("alpha"),
            f"intervention payload.desired_values[{index}].alpha",
        )
        desired = _number(
            condition.get("desired_post_gate_activation"),
            (
                "intervention payload.desired_values"
                f"[{index}].desired_post_gate_activation"
            ),
        )
        expected_desired = (1.0 - expected_alpha) * baseline
        if alpha != expected_alpha or not math.isclose(
            desired, expected_desired, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ArtifactValidationError(
                "intervention desired value does not match the alpha mapping"
            )
    fixed_ids = _list(
        payload["fixed_top_k_union_token_ids"],
        "intervention payload.fixed_top_k_union_token_ids",
        nonempty=True,
    )
    if fixed_ids != sorted(set(fixed_ids)):
        raise ArtifactValidationError(
            "intervention fixed token IDs must be unique and sorted"
        )
    conditions = _mapping(payload["conditions"], "intervention payload.conditions")
    _require_keys(
        conditions,
        {"baseline", "noop", "half_suppression", "full_ablation"},
        "intervention payload.conditions",
    )
    for name in ("baseline", "noop", "half_suppression", "full_ablation"):
        rows = _list(
            conditions[name], f"intervention payload.conditions.{name}", nonempty=True
        )
        if len(rows) != len(fixed_ids):
            raise ArtifactValidationError(
                f"intervention condition {name} does not align with fixed token IDs"
            )
        for index, row in enumerate(rows):
            item = _mapping(row, f"intervention payload.conditions.{name}[{index}]")
            _require_keys(
                item,
                {
                    "token_id",
                    "token",
                    "logit",
                    "probability",
                    "signed_logit_change_from_baseline",
                    "signed_probability_change_from_baseline",
                },
                f"intervention payload.conditions.{name}[{index}]",
            )
            if item["token_id"] != fixed_ids[index]:
                raise ArtifactValidationError(
                    f"intervention condition {name} token IDs do not match "
                    "the fixed union"
                )
            _string(
                item["token"], f"intervention payload.conditions.{name}[{index}].token"
            )
            _number(
                item["logit"], f"intervention payload.conditions.{name}[{index}].logit"
            )
            probability = _number(
                item["probability"],
                f"intervention payload.conditions.{name}[{index}].probability",
                minimum=0.0,
            )
            if probability > 1.0:
                raise ArtifactValidationError(
                    "intervention probability must not exceed one"
                )
            _number(
                item["signed_logit_change_from_baseline"],
                (
                    f"intervention payload.conditions.{name}[{index}]"
                    ".signed_logit_change_from_baseline"
                ),
            )
            _number(
                item["signed_probability_change_from_baseline"],
                (
                    f"intervention payload.conditions.{name}[{index}]"
                    ".signed_probability_change_from_baseline"
                ),
            )
    comparison = _mapping(
        payload["baseline_noop_comparison"],
        "intervention payload.baseline_noop_comparison",
    )
    _boolean(
        comparison.get("within_tolerance"),
        "intervention payload.baseline_noop_comparison.within_tolerance",
        expected=True,
    )
    determinism = _mapping(payload["determinism"], "intervention payload.determinism")
    _boolean(
        determinism.get("within_tolerance"),
        "intervention payload.determinism.within_tolerance",
        expected=True,
    )
    regime = _mapping(payload["regime"], "intervention payload.regime")
    _boolean(
        regime.get("freeze_attention"),
        "intervention payload.regime.freeze_attention",
        expected=True,
    )
    if regime.get("constrained_layers", "missing") is not None:
        raise ArtifactValidationError("intervention constrained_layers must be null")
    _timing(payload["timing"], "intervention payload.timing")
    _integer(payload["seed"], "intervention payload.seed")
    _string(payload["claim_boundary"], "intervention payload.claim_boundary")


def _validate_semantics(record: dict[str, Any]) -> None:
    if record["status"] != "completed":
        raise ArtifactValidationError("semantics_summary status must be completed")
    _validate_runtime_provenance(record, "semantics")
    payload = _mapping(record["payload"], "semantics payload")
    _require_keys(
        payload,
        {
            "prompt",
            "token_ids",
            "cache_shape",
            "cache_index_order",
            "cache_flags",
            "parameters",
            "preactivation_equation",
            "gate_check",
            "intervention_value_check",
            "timing",
            "seed",
            "claim_boundary",
        },
        "semantics payload",
    )
    _string(
        payload["prompt"], "semantics payload.prompt", expected=_INTERVENTION_PROMPT
    )
    _token_sequence(payload, "semantics payload")
    cache_shape = _list(payload["cache_shape"], "semantics payload.cache_shape")
    if len(cache_shape) != 3 or cache_shape[0] != 26 or cache_shape[2] != 16384:
        raise ArtifactValidationError(
            "semantics cache shape must be [26, positions, 16384]"
        )
    _integer(cache_shape[1], "semantics payload.cache_shape[1]", minimum=1)
    if payload["cache_index_order"] != ["layer", "token_position", "feature_id"]:
        raise ArtifactValidationError("semantics cache index order is invalid")
    cache_flags = _mapping(payload["cache_flags"], "semantics payload.cache_flags")
    _boolean(
        cache_flags.get("preactivation_apply_activation_function"),
        "semantics preactivation cache flag",
        expected=False,
    )
    _boolean(
        cache_flags.get("post_gate_apply_activation_function"),
        "semantics post-gate cache flag",
        expected=True,
    )
    _boolean(cache_flags.get("sparse"), "semantics sparse cache flag", expected=False)
    parameters = _mapping(payload["parameters"], "semantics payload.parameters")
    for name, expected in (
        ("layer_count", 26),
        ("d_model", 2304),
        ("d_transcoder", 16384),
    ):
        if parameters.get(name) != expected:
            raise ArtifactValidationError(
                f"semantics payload.parameters.{name} is invalid"
            )
    _string(
        parameters.get("activation_function"),
        "semantics payload.parameters.activation_function",
        expected="JumpReLU",
    )
    equation = _mapping(
        payload["preactivation_equation"], "semantics payload.preactivation_equation"
    )
    _boolean(
        equation.get("b_enc_included"), "semantics preactivation b_enc", expected=True
    )
    _boolean(
        equation.get("b_dec_included"), "semantics preactivation b_dec", expected=False
    )
    gate = _mapping(payload["gate_check"], "semantics payload.gate_check")
    _boolean(
        gate.get("strict_greater_than"),
        "semantics gate strict comparison",
        expected=True,
    )
    _boolean(gate.get("equality_inactive"), "semantics gate equality", expected=True)
    samples = _mapping(gate.get("samples"), "semantics payload.gate_check.samples")
    _require_keys(
        samples,
        {"active", "inactive", "closest_margin", "official_intervention_source"},
        "semantics payload.gate_check.samples",
    )
    for name, sample_value in samples.items():
        sample = _mapping(sample_value, f"semantics {name} sample")
        _require_keys(
            sample,
            {
                "layer",
                "position",
                "feature_id",
                "preactivation",
                "threshold",
                "post_gate_activation",
                "active",
                "signed_margin",
            },
            f"semantics {name} sample",
        )
        _integer(sample["layer"], f"semantics {name} sample.layer")
        _integer(sample["position"], f"semantics {name} sample.position")
        _integer(sample["feature_id"], f"semantics {name} sample.feature_id")
        _number(sample["preactivation"], f"semantics {name} sample.preactivation")
        _number(sample["threshold"], f"semantics {name} sample.threshold")
        _number(
            sample["post_gate_activation"],
            f"semantics {name} sample.post_gate_activation",
        )
        _boolean(sample["active"], f"semantics {name} sample.active")
        _number(sample["signed_margin"], f"semantics {name} sample.signed_margin")
    inactive = _mapping(samples["inactive"], "semantics inactive sample")
    _boolean(inactive.get("active"), "semantics inactive sample.active", expected=False)
    _number(inactive.get("preactivation"), "semantics inactive sample.preactivation")
    _number(inactive.get("threshold"), "semantics inactive sample.threshold")
    inactive_activation = _number(
        inactive.get("post_gate_activation"),
        "semantics inactive sample.post_gate_activation",
    )
    if inactive_activation != 0.0:
        raise ArtifactValidationError(
            "semantics inactive sample must have zero activation"
        )
    active = _mapping(samples["active"], "semantics active sample")
    _boolean(active.get("active"), "semantics active sample.active", expected=True)
    value_check = _mapping(
        payload["intervention_value_check"],
        "semantics payload.intervention_value_check",
    )
    _require_keys(
        value_check,
        {
            "upstream_argument",
            "project_mapping",
            "official_feature_baseline_activation",
            "alpha",
            "desired_noop_activation",
            "delta_logic_still_uses_post_gate_activation",
            "baseline_noop_maximum_absolute_logit_error",
            "noop_repeat_maximum_absolute_logit_error",
            "absolute_tolerance",
            "relative_tolerance",
        },
        "semantics payload.intervention_value_check",
    )
    _string(
        value_check.get("upstream_argument"),
        "semantics intervention upstream_argument",
        expected="absolute_desired_post_gate_activation",
    )
    _string(
        value_check["project_mapping"],
        "semantics intervention project_mapping",
        expected="desired = (1 - alpha) * baseline_activation",
    )
    baseline_activation = _number(
        value_check["official_feature_baseline_activation"],
        "semantics intervention baseline activation",
        minimum=0.0,
    )
    if baseline_activation <= 0.0 or value_check["alpha"] != 0.0:
        raise ArtifactValidationError(
            "semantics no-op must use alpha zero and an active source"
        )
    desired_noop = _number(
        value_check["desired_noop_activation"],
        "semantics intervention desired no-op activation",
    )
    if desired_noop != baseline_activation:
        raise ArtifactValidationError(
            "semantics desired no-op must equal baseline activation"
        )
    _boolean(
        value_check["delta_logic_still_uses_post_gate_activation"],
        "semantics intervention delta logic",
        expected=True,
    )
    _number(
        value_check.get("baseline_noop_maximum_absolute_logit_error"),
        "semantics baseline/no-op error",
        minimum=0.0,
    )
    _timing(payload["timing"], "semantics payload.timing")
    _integer(payload["seed"], "semantics payload.seed")
    _string(payload["claim_boundary"], "semantics payload.claim_boundary")


_PAYLOAD_VALIDATORS = {
    "asset_manifest": _validate_asset_manifest,
    "attribution_summary": _validate_attribution,
    "colab_handoff_manifest": _validate_colab_handoff,
    "environment_manifest": _validate_environment,
    "intervention_summary": _validate_intervention,
    "semantics_summary": _validate_semantics,
}


def validate_artifact_payload(value: object) -> None:
    """Validate required scientific/provenance fields for one known artifact type."""

    record = _mapping(value, "artifact")
    artifact_type = _string(record.get("artifact_type"), "artifact_type")
    validator = _PAYLOAD_VALIDATORS.get(artifact_type)
    if validator is None:
        raise ArtifactValidationError(
            f"unsupported Stage 1A artifact_type {artifact_type!r}"
        )
    validator(record)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ArtifactValidationError(f"artifact has duplicate JSON key {key!r}")
        value[key] = item
    return value


def _load_json(path: Path) -> object:
    if path.is_symlink() or not path.is_file():
        raise ArtifactValidationError(
            f"artifact must be a regular, non-symlink file: {path.name}"
        )
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(
            f"artifact is not readable strict UTF-8 JSON: {path.name}"
        ) from exc


def present_artifact_paths(directory: Path) -> tuple[Path, ...]:
    """Return every present JSON artifact in deterministic relative-path order."""

    if directory.is_symlink():
        raise ArtifactValidationError("artifact directory must not be a symlink")
    if not directory.exists():
        return ()
    if not directory.is_dir():
        raise ArtifactValidationError("artifact directory is not a directory")
    return tuple(
        sorted(
            directory.rglob("*.json"),
            key=lambda path: path.relative_to(directory).as_posix(),
        )
    )


def validate_present_artifacts(
    directory: Path,
    *,
    require_any: bool = True,
    strict_payloads: bool = False,
) -> tuple[str, ...]:
    """Validate present envelopes, optionally enforcing type-specific payloads."""

    paths = present_artifact_paths(directory)
    if require_any and not paths:
        raise ArtifactValidationError("no Stage 1A JSON artifacts are present")

    validated: list[str] = []
    for path in paths:
        relative = path.relative_to(directory).as_posix()
        value = _load_json(path)
        expected_type = EXPECTED_TYPE_BY_FILENAME.get(relative)
        if expected_type is None:
            raise ArtifactValidationError(
                f"unsupported Stage 1A artifact filename: {relative}"
            )
        validate_artifact_envelope(value, expected_type=expected_type)
        if strict_payloads:
            validate_artifact_payload(value)
        assert_publication_safe(value)
        validated.append(relative)
    return tuple(validated)


def checksum_targets(directory: Path, manifest: Path) -> tuple[Path, ...]:
    """Return every JSON artifact, rejecting unrelated result-directory files."""

    if not directory.exists():
        return ()
    targets: list[Path] = []
    for path in directory.rglob("*"):
        if path == manifest:
            continue
        if path.is_symlink():
            raise ArtifactValidationError(
                f"checksum target must not be a symlink: {path.name}"
            )
        if path.is_file():
            if path.suffix.lower() != ".json":
                raise ArtifactValidationError(
                    f"unsupported file in Stage 1A artifact directory: {path.name}"
                )
            targets.append(path)
    return tuple(
        sorted(targets, key=lambda path: path.relative_to(directory).as_posix())
    )


def regenerate_checksums(
    directory: Path,
    manifest: Path,
    *,
    checksum_root: Path = REPOSITORY_ROOT,
) -> str:
    """Atomically regenerate checksums for every present Stage 1A artifact."""

    targets = checksum_targets(directory, manifest)
    if not targets:
        raise ArtifactValidationError("cannot checksum an empty artifact directory")
    return write_checksum_manifest_atomic(manifest, targets, root=checksum_root)


def verify_checksums(
    directory: Path,
    manifest: Path,
    *,
    checksum_root: Path = REPOSITORY_ROOT,
) -> tuple[str, ...]:
    """Verify digest values, ordering, normalization, and complete coverage."""

    if manifest.is_symlink() or not manifest.is_file():
        raise ArtifactValidationError("checksum manifest is missing or is a symlink")
    targets = checksum_targets(directory, manifest)
    if not targets:
        raise ArtifactValidationError("checksum manifest has no artifact targets")

    verified = verify_checksum_manifest(manifest, root=checksum_root)
    expected = build_checksum_manifest(targets, root=checksum_root)
    try:
        actual = manifest.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ArtifactValidationError("checksum manifest is not UTF-8 text") from exc
    if actual != expected:
        raise ArtifactValidationError(
            "checksum manifest is incomplete, unsorted, or contains extra entries"
        )
    return verified


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIRECTORY,
    )
    parser.add_argument(
        "--checksums",
        type=Path,
        help=f"Checksum path (default: ARTIFACT_DIR/{DEFAULT_CHECKSUM_NAME}).",
    )
    parser.add_argument(
        "--write-checksums",
        action="store_true",
        help="Atomically regenerate the checksum manifest before verification.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Permit an absent artifact directory; no checksum is then required.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    directory = args.artifact_dir.resolve()
    manifest = (
        args.checksums.resolve()
        if args.checksums is not None
        else directory / DEFAULT_CHECKSUM_NAME
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "valid": False,
        "artifacts": [],
        "checksums": [],
        "errors": [],
    }
    try:
        artifacts = validate_present_artifacts(
            directory,
            require_any=not args.allow_empty,
            strict_payloads=True,
        )
        report["artifacts"] = list(artifacts)
        if artifacts:
            if args.write_checksums:
                regenerate_checksums(directory, manifest)
            report["checksums"] = list(verify_checksums(directory, manifest))
        elif manifest.exists():
            raise ArtifactValidationError(
                "checksum manifest exists without any JSON artifacts"
            )
        report["valid"] = True
    except (OSError, ArtifactValidationError) as exc:
        report["errors"] = [str(exc).replace(str(REPOSITORY_ROOT), ".")]

    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
