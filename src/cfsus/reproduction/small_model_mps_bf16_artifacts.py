"""Independent schema and semantic validation for Stage 1A-S-BF16 artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn

from cfsus.reproduction.artifacts import ArtifactValidationError, sha256_file
from cfsus.reproduction.small_model_mps_bf16 import (
    ARTIFACT_ALLOWLIST,
    BACKEND,
    COMPLETED_STATUS,
    ENVIRONMENT_LOCK,
    EXECUTION_BASE_COMMIT,
    EXPERIMENT_CLASS,
    MODEL_IDENTIFIER,
    MODEL_REVISION,
    TRANSCODER_IDENTIFIER,
    TRANSCODER_REVISION,
    TRANSCODER_SUBFOLDER,
    UPSTREAM_REVISION,
    validate_small_artifact_directory,
    within_bf16_ulps,
)

BRANCH = "stage-1a-small-model-mps-bf16"
JSON_FILES = ARTIFACT_ALLOWLIST - {"checksums.sha256"}
HEX40_RE = re.compile(r"[0-9a-f]{40}")
CHECKSUM_RE = re.compile(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)")
FORBIDDEN_TEXT = (
    "authorization:",
    "bearer ",
    "hf_token",
    "api_key",
    "access_token",
    "/users/",
    "-----begin private key-----",
)
PEAK_MAXIMUMS = {
    "mps_driver_bytes": 24 * 1024**3,
    "process_rss_bytes": 24 * 1024**3,
    "swap_growth_bytes": 4 * 1024**3,
}
MINIMUM_AVAILABLE = 4 * 1024**3


def _fail(message: str) -> NoReturn:
    raise ArtifactValidationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    return dict(value)


def strict_json_load(path: Path) -> dict[str, Any]:
    """Load one finite JSON object and reject non-standard constants."""

    def reject_constant(value: str) -> None:
        _fail(f"non-finite JSON constant is forbidden: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(f"invalid JSON artifact {path.name}") from error
    return _mapping(value, path.name)


def validate_checksum_manifest(directory: Path) -> dict[str, str]:
    """Require one exact digest for every non-manifest allowlisted artifact."""

    path = directory / "checksums.sha256"
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = CHECKSUM_RE.fullmatch(line)
        if match is None:
            _fail("checksum manifest line is malformed")
        digest, name = match.groups()
        if name in entries:
            _fail("checksum manifest contains a duplicate name")
        entries[name] = digest
    if set(entries) != JSON_FILES:
        _fail("checksum manifest coverage differs from the artifact allowlist")
    for name, expected in entries.items():
        if sha256_file(directory / name) != expected:
            _fail(f"checksum mismatch for {name}")
    return entries


def scan_artifact_text(directory: Path) -> None:
    """Reject secrets, absolute user paths, and binary content."""

    for name in ARTIFACT_ALLOWLIST:
        data = (directory / name).read_bytes()
        if b"\x00" in data:
            _fail("binary artifact content is forbidden")
        try:
            text = data.decode("utf-8").casefold()
        except UnicodeDecodeError as error:
            raise ArtifactValidationError("artifact is not UTF-8 text") from error
        if any(pattern in text for pattern in FORBIDDEN_TEXT):
            _fail(f"secret or private-path marker found in {name}")


def _require(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        _fail(f"{label} must be {expected!r}, got {value!r}")


def _finite_summary(value: Any, label: str) -> None:
    summary = _mapping(value, label)
    _require(summary.get("device"), "mps:0", f"{label}.device")
    _require(summary.get("dtype"), "torch.bfloat16", f"{label}.dtype")
    for key in (
        "nan_count",
        "positive_infinity_count",
        "negative_infinity_count",
        "nonfinite_count",
    ):
        _require(summary.get(key), 0, f"{label}.{key}")


def validate_peak_hierarchy(telemetry: Any) -> None:
    root = _mapping(telemetry, "telemetry")
    attempt = _mapping(root.get("attempt_peaks"), "attempt peaks")
    stages = _mapping(root.get("stage_peaks"), "stage peaks")
    if not stages or int(root.get("sample_count", 0)) < 1:
        _fail("telemetry lacks stages or samples")
    for stage_name, raw_stage in stages.items():
        stage = _mapping(raw_stage, f"stage peak {stage_name}")
        for metric in (
            "mps_current_bytes",
            "mps_driver_bytes",
            "process_rss_bytes",
            "swap_used_bytes",
            "swap_growth_bytes",
        ):
            if int(attempt.get(metric, -1)) < int(stage.get(metric, 0)):
                _fail("attempt maximum does not dominate every stage maximum")
        if int(attempt.get("minimum_available_memory_bytes", 0)) > int(
            stage.get("minimum_available_memory_bytes", 0)
        ):
            _fail("attempt memory minimum does not dominate every stage minimum")
    for metric, maximum in PEAK_MAXIMUMS.items():
        if int(attempt.get(metric, maximum + 1)) > maximum:
            _fail(f"telemetry exceeds {metric}")
    if int(attempt.get("minimum_available_memory_bytes", 0)) < MINIMUM_AVAILABLE:
        _fail("telemetry violates the available-memory reserve")
    if root.get("violations") or int(root.get("telemetry_failures", 0)) != 0:
        _fail("telemetry contains a safety or sampling failure")
    if not set(root.get("thermal_states", [])).issubset({"nominal", "fair"}):
        _fail("telemetry contains an unsafe thermal state")


def memory_entry(source: str, index: int, attempt: Any) -> dict[str, Any]:
    """Build the exact cross-file timing/peak projection for one attempt."""

    record = _mapping(attempt, "attempt")
    worker = _mapping(record.get("worker"), "attempt.worker")
    supervisor = _mapping(record.get("supervisor"), "attempt.supervisor")
    telemetry = _mapping(worker.get("telemetry"), "worker telemetry")
    return {
        "source": source,
        "attempt_index": index,
        "stage": record.get("stage"),
        "batch_size": record.get("batch_size"),
        "worker_status": worker.get("status"),
        "worker_started_at_unix": telemetry.get("started_at_unix"),
        "worker_finished_at_unix": telemetry.get("finished_at_unix"),
        "worker_attempt_peaks": telemetry.get("attempt_peaks"),
        "worker_stage_peaks": telemetry.get("stage_peaks"),
        "supervisor_started_at_unix": supervisor.get("started_at_unix"),
        "supervisor_finished_at_unix": supervisor.get("finished_at_unix"),
        "supervisor_peak_process_group_rss_bytes": supervisor.get(
            "peak_process_group_rss_bytes"
        ),
        "supervisor_peak_swap_growth_bytes": supervisor.get("peak_swap_growth_bytes"),
        "supervisor_thermal_states": supervisor.get("thermal_states"),
    }


def derive_memory_entries(attempts: Any) -> list[dict[str, Any]]:
    root = _mapping(attempts, "attempts artifact")
    entries: list[dict[str, Any]] = []
    for raw_set in root.get("attempt_sets", []):
        attempt_set = _mapping(raw_set, "attempt set")
        source = str(attempt_set.get("source"))
        record = _mapping(attempt_set.get("record"), "attempt set record")
        for index, attempt in enumerate(record.get("attempts", [])):
            entries.append(memory_entry(source, index, attempt))
    return entries


def _validate_environment(
    value: dict[str, Any], execution_commit: str | None, repository_root: Path
) -> None:
    _require(value.get("status"), "passed", "environment status")
    _require(value.get("branch"), BRANCH, "environment branch")
    commit = value.get("execution_commit")
    if not isinstance(commit, str) or HEX40_RE.fullmatch(commit) is None:
        _fail("execution commit is not an immutable SHA")
    if execution_commit is not None:
        _require(commit, execution_commit, "execution commit")
    environment = _mapping(value.get("environment"), "environment")
    for key, expected in (
        ("machine", "arm64"),
        ("python", "3.11.13"),
        ("torch", "2.6.0"),
        ("nnsight", "0.6.1"),
        ("circuit_tracer", "0.5.2"),
        ("transformers", "4.57.3"),
        ("mps_built", True),
        ("mps_available", True),
        ("fallback_variable_present", False),
        ("outer_autocast_enabled", False),
        ("upstream_commit", UPSTREAM_REVISION),
    ):
        _require(environment.get(key), expected, f"environment.{key}")
    observed_lock = hashlib.sha256(
        (repository_root / ENVIRONMENT_LOCK).read_bytes()
    ).hexdigest()
    _require(environment.get("lock_sha256"), observed_lock, "environment lock SHA")


def _validate_assets(value: dict[str, Any]) -> None:
    _require(value.get("status"), "verified", "asset status")
    _require(value.get("actual_total_bytes"), 2_087_816_677, "asset bytes")
    _require(value.get("download_performed"), False, "asset download flag")
    _require(value.get("network_accessed"), False, "asset network flag")
    model = _mapping(value.get("model"), "model asset")
    transcoder = _mapping(value.get("transcoder"), "transcoder asset")
    for mapping, identity, revision, count, size in (
        (model, MODEL_IDENTIFIER, MODEL_REVISION, 8, 575_454_257),
        (
            transcoder,
            TRANSCODER_IDENTIFIER,
            TRANSCODER_REVISION,
            19,
            1_512_362_420,
        ),
    ):
        _require(mapping.get("identifier"), identity, "asset identifier")
        _require(mapping.get("revision"), revision, "asset revision")
        _require(len(mapping.get("files", [])), count, "asset file count")
        _require(mapping.get("total_bytes"), size, "asset size")
    _require(
        model.get("consumed_path_identity"),
        f"models--google--gemma-3-270m/snapshots/{MODEL_REVISION}",
        "model consumed snapshot",
    )
    _require(
        transcoder.get("consumed_path_identity"),
        (f"models--mwhanna--gemma-scope-2-270m-pt/snapshots/{TRANSCODER_REVISION}"),
        "transcoder consumed snapshot",
    )
    _require(transcoder.get("subfolder"), TRANSCODER_SUBFOLDER, "transcoder subfolder")


def _validate_operator_probe(value: dict[str, Any]) -> None:
    _require(value.get("status"), "passed", "operator status")
    overflow = _mapping(value.get("overflow_regression"), "overflow")
    fp16 = _mapping(overflow.get("fp16"), "FP16 overflow")
    bf16 = _mapping(overflow.get("bf16"), "BF16 overflow recovery")
    reference = _mapping(overflow.get("fp32_reference"), "FP32 reference")
    _require(fp16.get("finite"), False, "FP16 overflow finite flag")
    _require(fp16.get("result"), "positive_infinity", "FP16 overflow result")
    _require(bf16.get("finite"), True, "BF16 recovery finite flag")
    _require(reference.get("finite"), True, "FP32 reference finite flag")
    if float(bf16.get("relative_error_against_fp32", math.inf)) > 0.005:
        _fail("BF16 overflow regression exceeds the frozen relative tolerance")
    operators = _mapping(value.get("operators"), "operators")
    required = {
        "tensor_add",
        "linear",
        "matmul",
        "batched_matmul",
        "embedding",
        "rmsnorm_source_path",
        "rotary_source_path",
        "attention_scaling",
        "attention_softmax_source_path",
        "masking",
        "gather",
        "scatter",
        "index_put_accumulate_false",
        "index_add",
        "nonzero",
        "topk",
        "sort",
        "threshold_comparison",
        "loaded_jumprelu_class",
        "retained_autograd",
        "jvp",
        "vjp",
    }
    if not required.issubset(operators):
        _fail("operator probe coverage is incomplete")
    for name in required - {"nonzero"}:
        _finite_summary(operators[name], f"operator {name}")
    sparse = _mapping(operators.get("sparse_metadata_adapter"), "sparse adapter")
    for key, expected in (
        ("passed", True),
        ("bit_exact_roundtrip", True),
        ("metadata_device", "cpu"),
        ("metadata_value_dtype", "torch.bfloat16"),
    ):
        _require(sparse.get(key), expected, f"sparse adapter {key}")


def _validate_model(value: dict[str, Any]) -> None:
    _require(value.get("status"), "passed", "model status")
    _require(value.get("fp32_diagnostic_process_overlap"), False, "process overlap")
    guard = _mapping(value.get("module_guard"), "model module guard")
    _require(guard.get("parameter_device"), "mps", "model parameter device")
    _require(
        guard.get("floating_parameter_dtype"),
        "torch.bfloat16",
        "model parameter dtype",
    )
    prompts = value.get("prompts", [])
    _require(
        [item.get("prompt_id") for item in prompts],
        ["bos_only", "hello", "pilot"],
        "model prompt order",
    )
    for prompt in prompts:
        if len(prompt.get("hidden_states", [])) != 19:
            _fail("model hidden-state coverage is incomplete")
        if len(prompt.get("decoder_layers", [])) != 18:
            _fail("model decoder-layer coverage is incomplete")
        for summary in prompt["hidden_states"]:
            _finite_summary(summary, "model hidden state")
        for layer in prompt["decoder_layers"]:
            for field in ("feedforward_residual", "feedforward_post_norm", "output"):
                _finite_summary(layer[field]["summary"], f"decoder layer {field}")
        _finite_summary(prompt["logits"]["diagnostics"], "model logits")
        _require(
            prompt.get("prior_fp16_failure_coordinate", {}).get("finite"),
            True,
            "prior FP16 failure coordinate",
        )


def _validate_fp32_reference(value: dict[str, Any]) -> None:
    _require(value.get("status"), "passed", "FP32 reference status")
    _require(
        value.get("execution_class"),
        "separate_cpu_fp32_diagnostic_only",
        "FP32 diagnostic class",
    )
    _require(value.get("accepted_execution_fallback"), False, "FP32 fallback flag")
    _require(value.get("ran_after_mps_process_exit"), True, "FP32 process order")
    for prompt in value.get("prompts", []):
        _require(prompt.get("passed"), True, "FP32 prompt comparison")
        _require(prompt.get("mps_bf16_finite"), True, "MPS comparison finite")
        _require(prompt.get("cpu_fp32_finite"), True, "CPU comparison finite")
    if len(value.get("prompts", [])) != 3:
        _fail("FP32 reference prompt coverage is incomplete")


def _validate_loaded(value: dict[str, Any]) -> None:
    one_layer = _mapping(value.get("one_layer"), "one-layer semantics")
    accepted = _mapping(value.get("accepted"), "accepted loaded semantics")
    for mapping in (one_layer, accepted):
        _require(mapping.get("threshold_equality_inactive"), True, "gate equality")
        _require(mapping.get("maximum_gate_discrepancy"), 0.0, "gate discrepancy")
        _require(mapping.get("nonfinite_count"), 0, "loaded nonfinite count")
    if int(one_layer.get("active_count_final_position", 0)) < 1:
        _fail("one-layer loaded semantics lacks an active feature")
    if int(one_layer.get("inactive_count_final_position", 0)) < 1:
        _fail("one-layer loaded semantics lacks an inactive feature")
    if float(accepted.get("selected_activation", 0.0)) <= 0.0:
        _fail("accepted selected feature is not baseline-active")
    _require(accepted.get("inactive_activation"), 0.0, "inactive activation")
    one_layer_guard = _mapping(one_layer.get("module_guard"), "one-layer module guard")
    _require(
        one_layer_guard.get("parameter_device"), "mps", "one-layer parameter device"
    )
    _require(
        one_layer_guard.get("floating_parameter_dtype"),
        "torch.bfloat16",
        "one-layer parameter dtype",
    )
    full_plt = _mapping(value.get("full_plt"), "full PLT")
    _require(full_plt.get("status"), "passed", "full PLT status")
    _require(full_plt.get("layer_count"), 18, "full PLT layer count")
    _require(full_plt.get("total_feature_width"), 294_912, "full PLT feature count")
    full_guard = _mapping(full_plt.get("module_guard"), "full PLT module guard")
    _require(full_guard.get("parameter_device"), "mps", "full PLT parameter device")
    _require(
        full_guard.get("floating_parameter_dtype"),
        "torch.bfloat16",
        "full PLT parameter dtype",
    )
    replacement = _mapping(value.get("replacement_runtime"), "replacement runtime")
    _require(replacement.get("status"), "passed", "replacement runtime status")
    _require(replacement.get("device"), "mps:0", "replacement runtime device")
    _require(replacement.get("dtype"), "torch.bfloat16", "replacement runtime dtype")
    replacement_guard = _mapping(
        replacement.get("module_guard"), "replacement module guard"
    )
    _require(
        replacement_guard.get("parameter_device"),
        "mps",
        "replacement parameter devices",
    )
    _require(
        replacement_guard.get("floating_parameter_dtype"),
        "torch.bfloat16",
        "replacement parameter dtypes",
    )


def _validate_attribution(value: dict[str, Any]) -> None:
    _require(value.get("status"), "passed", "attribution status")
    _require(value.get("profile"), "accepted", "attribution profile")
    if int(value.get("batch_size", 0)) not in {64, 32, 16}:
        _fail("accepted attribution batch is outside frozen policy")
    module_guard = _mapping(value.get("module_guard"), "attribution module guard")
    _require(
        module_guard.get("parameter_device"),
        "mps",
        "attribution parameter devices",
    )
    _require(
        module_guard.get("floating_parameter_dtype"),
        "torch.bfloat16",
        "attribution parameter dtypes",
    )
    graph = _mapping(value.get("attribution"), "attribution")
    for key in ("active_feature_count", "selected_feature_count", "nonzero_edge_count"):
        if int(graph.get(key, 0)) < 1:
            _fail(f"attribution {key} must be positive")
    _require(graph.get("nonfinite_count"), 0, "attribution nonfinite count")
    _require(graph.get("raw_graph_persisted"), False, "raw graph persistence")
    _require(graph.get("graph_metadata_device"), "cpu", "graph metadata device")
    _require(graph.get("scientific_tensor_device"), "mps", "scientific device")
    _require(
        graph.get("scientific_tensor_dtype"),
        "torch.bfloat16",
        "scientific dtype",
    )
    usage = _mapping(graph.get("adapter_usage"), "adapter usage")
    _require(usage.get("runtime_monkeypatches"), 0, "runtime monkeypatch count")
    if int(usage.get("component_calls", 0)) < 1 or int(usage.get("batch_calls", 0)) < 1:
        _fail("BF16 attribution adapter was not exercised")
    selection = _mapping(value.get("selection"), "selection")
    _require(
        selection.get("rule"),
        "highest_absolute_direct_contribution_to_baseline_top_logit_at_final_token",
        "selection rule",
    )
    if float(selection.get("baseline_activation", 0.0)) <= 0.0:
        _fail("selected feature is not baseline-active")
    audit = _mapping(value.get("selection_audit"), "selection audit")
    _require(audit.get("rule"), selection.get("rule"), "selection audit rule")
    _require(audit.get("raw_graph_persisted"), False, "selection audit graph policy")
    candidates = audit.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        _fail("selection audit has no candidates")
    _require(len(candidates), audit.get("candidate_count"), "selection candidate count")
    excluded = _mapping(audit.get("excluded_counts"), "selection exclusions")
    expected_exclusions = {
        "non_final_position",
        "nonpositive_baseline",
        "nonfinite_baseline",
        "nonfinite_score",
    }
    if set(excluded) != expected_exclusions or any(
        not isinstance(count, int) or count < 0 for count in excluded.values()
    ):
        _fail("selection exclusion accounting is invalid")
    _require(
        len(candidates) + sum(excluded.values()),
        audit.get("source_count"),
        "selection source coverage",
    )
    final_position = int(value.get("token_count", 0)) - 1
    normalized: list[tuple[float, int, int, int, float]] = []
    for candidate in candidates:
        record = _mapping(candidate, "selection candidate")
        score = float(record.get("score", math.nan))
        baseline = float(record.get("baseline_activation", math.nan))
        layer = int(record.get("layer", -1))
        position = int(record.get("position", -1))
        feature = int(record.get("feature", -1))
        if (
            not math.isfinite(score)
            or not math.isfinite(baseline)
            or score < 0.0
            or baseline <= 0.0
            or not 0 <= layer < 18
            or position != final_position
            or not 0 <= feature < 16_384
        ):
            _fail("selection candidate is outside the frozen domain")
        normalized.append((-score, layer, position, feature, baseline))
    negative_score, layer, position, feature, baseline = min(normalized)
    for key, expected in (
        ("layer", layer),
        ("position", position),
        ("feature", feature),
        ("baseline_activation", baseline),
        ("score", -negative_score),
    ):
        _require(selection.get(key), expected, f"independent selection {key}")


def _validate_intervention(value: dict[str, Any]) -> None:
    _require(value.get("status"), "passed", "intervention status")
    intervention = _mapping(value.get("intervention"), "intervention")
    _require(intervention.get("freeze_attention"), True, "freeze-attention policy")
    _require(intervention.get("runtime_monkeypatches"), 0, "runtime monkeypatches")
    tolerance = float(intervention.get("baseline_noop_normalized_l2_tolerance", -1))
    if tolerance != 0.01:
        _fail("no-op tolerance changed")
    maximum_absolute_tolerance = float(
        intervention.get(
            "baseline_noop_maximum_absolute_logit_difference_tolerance", math.nan
        )
    )
    if maximum_absolute_tolerance != 0.0:
        _fail("no-op maximum-absolute tolerance changed")
    for baseline_name in ("raw_baseline", "baseline", "baseline_repeat"):
        baseline_record = _mapping(intervention.get(baseline_name), baseline_name)
        _finite_summary(baseline_record["diagnostics"], f"{baseline_name} logits")
    for key in (
        "raw_to_frozen_baseline_normalized_l2",
        "baseline_repeat_normalized_l2",
    ):
        if float(intervention.get(key, math.inf)) > tolerance:
            _fail(f"intervention control {key} failed")
    for key in (
        "raw_to_frozen_baseline_maximum_absolute_logit_difference",
        "baseline_repeat_maximum_absolute_logit_difference",
    ):
        maximum_control = float(intervention.get(key, math.nan))
        if (
            not math.isfinite(maximum_control)
            or maximum_control > maximum_absolute_tolerance
        ):
            _fail(f"intervention control {key} failed")
    conditions = intervention.get("conditions", [])
    _require([item.get("alpha") for item in conditions], [0.0, 0.5, 1.0], "alphas")
    for condition in conditions:
        baseline = float(condition.get("baseline_activation", math.nan))
        alpha = float(condition.get("alpha", math.nan))
        desired = float(condition.get("desired_absolute_activation", math.nan))
        sent = float(condition.get("sent_absolute_activation", math.nan))
        reference = (1.0 - alpha) * baseline
        maximum_absolute = float(
            condition.get("maximum_absolute_logit_difference_from_baseline", math.nan)
        )
        if not math.isfinite(maximum_absolute) or maximum_absolute < 0.0:
            _fail("intervention maximum absolute difference is invalid")
        if not within_bf16_ulps(desired, reference, 1) or desired != sent:
            _fail("absolute BF16 intervention mapping is invalid")
        _require(condition.get("sent_device"), "mps:0", "intervention device")
        _require(condition.get("sent_dtype"), "torch.bfloat16", "intervention dtype")
        _finite_summary(condition["logits"]["diagnostics"], "intervention logits")
    noop = conditions[0]
    if float(noop.get("normalized_l2_from_baseline", math.inf)) > tolerance:
        _fail("no-op intervention differs from baseline")
    if (
        float(noop.get("maximum_absolute_logit_difference_from_baseline", math.inf))
        > maximum_absolute_tolerance
    ):
        _fail("no-op maximum absolute difference differs from baseline")


def _validate_attempts_and_memory(
    attempts: dict[str, Any], memory: dict[str, Any], execution_commit: str
) -> None:
    entries = derive_memory_entries(attempts)
    if entries != memory.get("attempts"):
        _fail("memory/timing summary does not exactly match attempt provenance")
    successful_accepted: list[dict[str, Any]] = []
    invalidated_accepted_count = 0
    allowed_dispositions = {
        "engineering_preflight",
        "invalidated_missing_required_maximum_absolute_difference",
        "canonical_accepted",
    }
    for raw_set in attempts.get("attempt_sets", []):
        attempt_set = _mapping(raw_set, "attempt set")
        source = attempt_set.get("source")
        disposition = attempt_set.get("disposition")
        if not isinstance(source, str) or disposition not in allowed_dispositions:
            _fail("attempt-set source or disposition is invalid")
        record = _mapping(attempt_set.get("record"), "attempt record")
        for attempt in record.get("attempts", []):
            worker = _mapping(attempt.get("worker"), "worker")
            if worker.get("status") == "passed":
                validate_peak_hierarchy(worker.get("telemetry"))
            if attempt.get("stage") == "accepted" and worker.get("status") == "passed":
                if (
                    source == "accepted/accepted_attempts.json"
                    and disposition == "canonical_accepted"
                ):
                    successful_accepted.append(worker)
                elif disposition == (
                    "invalidated_missing_required_maximum_absolute_difference"
                ):
                    invalidated_accepted_count += 1
                else:
                    _fail("accepted attempt disposition is invalid")
    if len(successful_accepted) != 1:
        _fail("exactly one accepted worker must pass")
    if invalidated_accepted_count != 1:
        _fail("the invalidated accepted attempt is not preserved exactly once")
    accepted = successful_accepted[0]
    git = _mapping(accepted.get("git"), "accepted Git identity")
    _require(git.get("execution_commit"), execution_commit, "accepted execution SHA")
    _require(git.get("branch"), BRANCH, "accepted branch")
    _require(git.get("working_tree_clean"), True, "accepted worktree cleanliness")
    runtime = _mapping(accepted.get("runtime"), "accepted runtime")
    for key, expected in (
        ("backend", BACKEND),
        ("device", "mps"),
        ("dtype", "bfloat16"),
        ("fallback_enabled", False),
        ("offline_execution", True),
    ):
        _require(runtime.get(key), expected, f"accepted runtime {key}")


def _validate_run_manifest(value: dict[str, Any], execution_commit: str) -> None:
    for key, expected in (
        ("verdict", COMPLETED_STATUS),
        ("experiment_class", EXPERIMENT_CLASS),
        ("branch", BRANCH),
        ("execution_commit", execution_commit),
        ("stage1b_engineering_readiness", True),
        ("stage1b_empirical_claim_readiness", False),
        ("official_bf16_reproduction", "pending"),
        ("reference_clt_reproduction", "pending"),
        ("counterfactual_susceptibility_result", "none"),
        ("paper_results_readiness", False),
        ("raw_graph_persisted", False),
        ("weights_or_cache_committed", False),
    ):
        _require(value.get(key), expected, f"run manifest {key}")
    provenance = _mapping(value.get("provenance"), "run provenance")
    for key, expected in (
        ("execution_base_commit", EXECUTION_BASE_COMMIT),
        ("upstream_revision", UPSTREAM_REVISION),
        ("model_identifier", MODEL_IDENTIFIER),
        ("model_revision", MODEL_REVISION),
        ("transcoder_identifier", TRANSCODER_IDENTIFIER),
        ("transcoder_revision", TRANSCODER_REVISION),
        ("transcoder_subfolder", TRANSCODER_SUBFOLDER),
        ("backend", BACKEND),
        ("device", "mps"),
        ("dtype", "bfloat16"),
    ):
        _require(provenance.get(key), expected, f"run provenance {key}")


def validate_bundle(
    directory: Path,
    *,
    repository_root: Path,
    execution_commit: str | None = None,
) -> dict[str, Any]:
    """Validate the complete small artifact bundle independently and fail closed."""

    validate_small_artifact_directory(directory)
    scan_artifact_text(directory)
    checksums = validate_checksum_manifest(directory)
    records = {name: strict_json_load(directory / name) for name in JSON_FILES}
    run = records["run_manifest.json"]
    observed_commit = run.get("execution_commit")
    if (
        not isinstance(observed_commit, str)
        or HEX40_RE.fullmatch(observed_commit) is None
    ):
        _fail("run manifest execution commit is invalid")
    if execution_commit is not None:
        _require(observed_commit, execution_commit, "requested execution commit")
    _validate_environment(
        records["environment_manifest.json"], observed_commit, repository_root
    )
    _validate_assets(records["asset_manifest.json"])
    _require(
        records["preflight_summary.json"].get("status"),
        "passed",
        "preflight status",
    )
    _validate_operator_probe(records["operator_probe_summary.json"])
    _validate_model(records["model_forward_summary.json"])
    _validate_fp32_reference(records["fp32_reference_summary.json"])
    _validate_loaded(records["loaded_semantics_summary.json"])
    _validate_attribution(records["attribution_summary.json"])
    _validate_intervention(records["intervention_summary.json"])
    _validate_attempts_and_memory(
        records["attempts.json"],
        records["memory_timing_summary.json"],
        observed_commit,
    )
    _validate_run_manifest(run, observed_commit)
    return {
        "status": "passed",
        "verdict": COMPLETED_STATUS,
        "execution_commit": observed_commit,
        "artifact_count": len(ARTIFACT_ALLOWLIST),
        "checksum_count": len(checksums),
    }


__all__ = [
    "BRANCH",
    "derive_memory_entries",
    "memory_entry",
    "scan_artifact_text",
    "strict_json_load",
    "validate_bundle",
    "validate_checksum_manifest",
    "validate_peak_hierarchy",
]
