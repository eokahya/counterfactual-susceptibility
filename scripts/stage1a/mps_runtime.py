"""Small, dependency-light safety helpers for the Stage 1A MPS runner.

This module intentionally contains no model imports.  The MPS runner uses it
for policy decisions and telemetry before importing TransformerLens or
``circuit-tracer``.  In particular, an MPS out-of-memory retry is fail-closed:
generic ``MemoryError`` and CPU allocation errors never authorize a retry.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import resource
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cfsus.reproduction.artifacts import REDACTED, redact_sensitive

MPS_BATCH_SEQUENCE = (256, 128, 64)
MPS_EXECUTION_DTYPE = "float16"
MPS_REPRODUCTION_CLASS = "hardware_adapted_mps_fp16"
MPS_COMPLETED_STATUS = "completed_hardware_adapted_mps_fp16"
MPS_CLAIM_BOUNDARY = (
    "Apple M2 Max/MPS FP16 hardware-adapted runtime using the pinned assets; "
    "the official native-BF16 reproduction and CUDA/T4 numerical equivalence "
    "remain pending."
)
EXPLICIT_CPU_SPARSE_DEVIATION = (
    "Explicit CPU sparse COO metadata adapter is required because native MPS "
    "sparse COO is unsupported; dense scientific tensors remain on MPS and "
    "scientific parameters are unchanged."
)
UPSTREAM_REVISION = "8f1e2438df612464e229e44c4a00ff637bf9379b"
MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
TRANSCODER_REVISION = "bd5773156dea09893636c801df1237d0410307d2"
MODEL_ID = "google/gemma-2-2b"
TRANSCODER_ID = "mwhanna/gemma-scope-transcoders"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)\b(?:token|access[_-]?token|api[_-]?key|auth[_-]?token|authorization|"
    r"bearer[_-]?token|cookie|credentials|github[_-]?token|hf[_-]?token|"
    r"id[_-]?token|password|passwd|private[_-]?token|refresh[_-]?token|"
    r"secret)\s*[:=]"
)
_BEARER_CREDENTIAL = re.compile(r"(?i)\bbearer\s+\S+")


class MPSRuntimeError(RuntimeError):
    """A fail-closed MPS policy or provenance error."""


def fallback_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return the fallback flag without exposing any other environment value."""

    values = os.environ if environ is None else environ
    return values.get("PYTORCH_ENABLE_MPS_FALLBACK", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def secure_hf_token_present(environ: Mapping[str, str] | None = None) -> bool:
    """Return only a boolean; the credential itself is never returned or logged."""

    values = os.environ if environ is None else environ
    return any(
        bool(values.get(name, "")) for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
    )


def validate_mps_configuration(config: Mapping[str, Any]) -> None:
    """Validate immutable scientific identity without importing model packages."""

    if config.get("reproduction_class") != MPS_REPRODUCTION_CLASS:
        raise MPSRuntimeError("configuration is not the separate MPS FP16 runtime")
    runtime = config.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("backend") != "transformerlens":
        raise MPSRuntimeError("MPS runtime requires the TransformerLens backend")
    if runtime.get("device") != "mps" or runtime.get("dtype") != MPS_EXECUTION_DTYPE:
        raise MPSRuntimeError("MPS runtime must use device=mps and dtype=float16")
    if (
        config.get("reference_dtype") != "bfloat16"
        or config.get("reference_status") != "pending"
    ):
        raise MPSRuntimeError("native BF16 reference must remain pending")
    upstream = config.get("upstream")
    model = config.get("model")
    transcoder = config.get("transcoder")
    if (
        not isinstance(upstream, Mapping)
        or upstream.get("revision") != UPSTREAM_REVISION
    ):
        raise MPSRuntimeError("upstream revision is not the exact authorized pin")
    if (
        not isinstance(model, Mapping)
        or model.get("identifier") != MODEL_ID
        or model.get("revision") != MODEL_REVISION
    ):
        raise MPSRuntimeError("model identity/revision is not the exact authorized pin")
    if (
        not isinstance(transcoder, Mapping)
        or transcoder.get("identifier") != TRANSCODER_ID
        or transcoder.get("revision") != TRANSCODER_REVISION
    ):
        raise MPSRuntimeError(
            "transcoder identity/revision is not the exact authorized pin"
        )
    attribution = config.get("attribution")
    if (
        not isinstance(attribution, Mapping)
        or attribution.get("prompt") != "The capital of state containing Dallas is"
    ):
        raise MPSRuntimeError("attribution prompt is not the fixed prompt")
    observed_attribution = {
        key: attribution.get(key)
        for key in ("max_n_logits", "desired_logit_probability", "max_feature_nodes")
    }
    if observed_attribution != {
        "max_n_logits": 10,
        "desired_logit_probability": 0.95,
        "max_feature_nodes": 8192,
    }:
        raise MPSRuntimeError("attribution scientific parameters differ")
    if attribution.get("batch_size") != 256 or attribution.get("offload") != "disk":
        raise MPSRuntimeError("attribution must begin at batch 256 with disk offload")
    intervention = config.get("intervention")
    if (
        not isinstance(intervention, Mapping)
        or intervention.get("prompt") != "Hecho: Michael Jordan juega al"
    ):
        raise MPSRuntimeError("intervention prompt is not the fixed prompt")
    if (
        intervention.get("feature") != {"layer": 20, "position": -1, "feature_id": 341}
        or intervention.get("alphas") != [0.0, 0.5, 1.0]
        or intervention.get("freeze_attention") is not True
        or intervention.get("constrained_layers") is not None
    ):
        raise MPSRuntimeError("intervention scientific parameters differ")
    retry = config.get("oom_retry")
    if (
        not isinstance(retry, Mapping)
        or tuple(retry.get("batch_sizes", ())) != MPS_BATCH_SEQUENCE
        or retry.get("trigger") != "mps_out_of_memory_only"
        or retry.get("fresh_process") is not True
    ):
        raise MPSRuntimeError(
            "MPS retry policy must be exactly 256,128,64 and fresh-process"
        )


def evaluate_memory_feasibility(
    snapshot_sizes: Mapping[str, int | None],
    *,
    physical_memory_bytes: int | None,
    pressure: str | None = None,
    swap_used_bytes: int | None = None,
    safety_fraction: float = 0.70,
    observed_loading: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Conservatively gate downloads and execution before loading weights."""

    model = snapshot_sizes.get("model_bytes")
    transcoder = snapshot_sizes.get("transcoder_bytes")
    model_bytes = model if isinstance(model, int) and model > 0 else None
    transcoder_bytes = (
        transcoder if isinstance(transcoder, int) and transcoder > 0 else None
    )
    known = model_bytes is not None and transcoder_bytes is not None
    # Snapshot bytes are not the resident FP16 footprint: the Gemma snapshot
    # contains serialized shards and the transcoder loader keeps a bounded
    # metadata/decoder working set.  Reserve six GiB for conversion, graph,
    # telemetry, and macOS/Codex overhead; this remains a pre-download gate,
    # not an observed peak claim.
    estimate_components: dict[str, int] | None = None
    estimate = None
    if model_bytes is not None and transcoder_bytes is not None:
        estimate_components = {
            "model_resident_estimate_bytes": int(model_bytes * 0.60),
            "transcoder_resident_estimate_bytes": transcoder_bytes,
            "temporary_and_system_headroom_bytes": 6 * 1024**3,
        }
        estimate = sum(estimate_components.values())
    physical_bytes = (
        physical_memory_bytes
        if isinstance(physical_memory_bytes, int) and physical_memory_bytes > 0
        else None
    )
    budget = (
        int(physical_bytes * safety_fraction) if physical_bytes is not None else None
    )
    empirical_loading: dict[str, Any] | None = None
    empirical_error: str | None = None
    if observed_loading is not None:
        required_observation_keys = {
            "loading_plan_id",
            "execution_commit",
            "attempt_report_sha256",
            "mps_current_allocated_peak_bytes",
            "mps_driver_allocated_peak_bytes",
            "swap_used_peak_bytes",
        }
        if set(observed_loading) != required_observation_keys:
            empirical_error = "empirical runtime-loading observation has invalid keys"
        else:
            plan_id = observed_loading.get("loading_plan_id")
            execution_commit = observed_loading.get("execution_commit")
            report_sha256 = observed_loading.get("attempt_report_sha256")
            current_peak = observed_loading.get("mps_current_allocated_peak_bytes")
            driver_peak = observed_loading.get("mps_driver_allocated_peak_bytes")
            observed_swap_peak = observed_loading.get("swap_used_peak_bytes")
            if (
                not isinstance(plan_id, str)
                or not plan_id
                or not isinstance(execution_commit, str)
                or re.fullmatch(r"[0-9a-f]{40}", execution_commit) is None
                or not isinstance(report_sha256, str)
                or _SHA256.fullmatch(report_sha256) is None
                or isinstance(current_peak, bool)
                or not isinstance(current_peak, int)
                or current_peak < 0
                or isinstance(driver_peak, bool)
                or not isinstance(driver_peak, int)
                or driver_peak < 0
                or isinstance(observed_swap_peak, bool)
                or not isinstance(observed_swap_peak, int)
                or observed_swap_peak < 0
                or current_peak > driver_peak
            ):
                empirical_error = "empirical runtime-loading observation is malformed"
            else:
                empirical_loading = dict(observed_loading)
    effective_peak = estimate
    if empirical_loading is not None:
        observed_driver_peak = int(empirical_loading["mps_driver_allocated_peak_bytes"])
        effective_peak = max(estimate or 0, observed_driver_peak)
    blocked_reason: str | None = None
    if not known:
        blocked_reason = "immutable snapshot sizes are unavailable"
    elif budget is None or physical_bytes is None or physical_bytes < 32 * 1024**3:
        blocked_reason = "host does not provide the required conservative 32 GiB budget"
    elif pressure != "normal":
        blocked_reason = "system memory pressure is unavailable or unsafe"
    elif isinstance(swap_used_bytes, bool) or not isinstance(swap_used_bytes, int):
        blocked_reason = "swap usage telemetry is unavailable"
    elif swap_used_bytes > 4 * 1024**3:
        blocked_reason = "swap usage exceeds the preflight safety threshold"
    elif empirical_error is not None:
        blocked_reason = empirical_error
    elif (
        empirical_loading is not None
        and effective_peak is not None
        and effective_peak > budget
    ):
        blocked_reason = (
            "observed identical runtime-loading plan exceeds the conservative budget"
        )
    elif (
        empirical_loading is not None
        and int(empirical_loading["swap_used_peak_bytes"]) > 4 * 1024**3
    ):
        blocked_reason = (
            "observed identical runtime-loading plan exceeded the swap safety threshold"
        )
    elif estimate is not None and estimate > budget:
        blocked_reason = (
            "conservative model/transcoder duplication estimate exceeds budget"
        )
    return {
        "status": (
            "blocked"
            if blocked_reason
            else "feasible_with_explicit_execution_deviation"
        ),
        "resource_status": "blocked" if blocked_reason else "feasible",
        "reason": blocked_reason,
        "physical_memory_bytes": physical_memory_bytes,
        "safety_fraction": safety_fraction,
        "conservative_budget_bytes": budget,
        "snapshot_sizes": {"model_bytes": model, "transcoder_bytes": transcoder},
        "estimate_components": estimate_components,
        "estimated_peak_bytes": estimate,
        "effective_peak_bytes": effective_peak,
        "empirical_loading_observation": empirical_loading,
        "pressure": pressure,
        "swap_used_bytes": swap_used_bytes,
        "downloads_authorized": blocked_reason is None,
        "native_mps_sparse_coo_supported": False,
        "scientific_parameters_changed": False,
        "execution_deviations": [EXPLICIT_CPU_SPARSE_DEVIATION],
    }


def require_mps(torch_module: Any, *, environ: Mapping[str, str] | None = None) -> Any:
    """Validate a real MPS backend and return ``torch.device('mps')``.

    No CPU fallback is enabled here.  The tiny allocation/matmul probe belongs
    to the caller because it is useful to persist as a separate preflight
    observation.
    """

    if fallback_enabled(environ):
        raise MPSRuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be disabled")
    backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    if (
        backend is None
        or not bool(backend.is_built())
        or not bool(backend.is_available())
    ):
        raise MPSRuntimeError("MPS is not built and available")
    return torch_module.device("mps")


def exception_chain(error: BaseException) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and all(current is not item for item in chain):
        chain.append(current)
        current = current.__cause__ or current.__context__
    return tuple(chain)


_MPS_OOM = re.compile(
    r"(?:mps|metal|mtl|apple)[^\n]{0,100}(?:out[ -]?of[ -]?memory|oom)|"
    r"(?:out[ -]?of[ -]?memory|oom)[^\n]{0,100}(?:mps|metal|mtl|apple)",
    re.IGNORECASE,
)


def is_mps_out_of_memory(error: BaseException) -> bool:
    """Recognize only explicitly MPS/Metal OOM failures."""

    for item in exception_chain(error):
        module = type(item).__module__.casefold()
        typename = type(item).__name__.casefold()
        message = str(item)
        if typename == "outofmemoryerror" and ("mps" in module or "metal" in module):
            return True
        if isinstance(item, (RuntimeError, MemoryError)) and _MPS_OOM.search(message):
            return True
    return False


# Short aliases keep the policy convenient for the isolated worker and for
# offline callers without duplicating the classifier implementation.
is_mps_oom = is_mps_out_of_memory


def classify_failure(error: BaseException) -> str:
    if is_mps_out_of_memory(error):
        return "mps_out_of_memory"
    text = " ".join(str(item) for item in exception_chain(error)).casefold()
    if any(token in text for token in ("non-finite", "nan", "infinite", "jumprelu")):
        return "failed_precision"
    if any(
        token in text
        for token in ("gated", "authentication", "permission", "403", "401")
    ):
        return "blocked_access"
    return "failed_runtime"


def sanitize_error(error: BaseException, *, limit: int = 240) -> str:
    """Make diagnostics one-line and credential/path-safe."""

    text = " ".join(str(error).split()) or type(error).__name__
    if _CREDENTIAL_ASSIGNMENT.search(text) or _BEARER_CREDENTIAL.search(text):
        return REDACTED
    redacted = redact_sensitive({"message": text})["message"]
    return redacted[:limit] if isinstance(redacted, str) else REDACTED


def should_retry_mps_attempt(
    *, batch_size: int, category: str, failure_stage: str | None = None
) -> bool:
    """Authorize only the next fresh process after a confirmed MPS OOM."""

    if batch_size not in MPS_BATCH_SEQUENCE:
        raise ValueError("batch size is outside the preregistered sequence")
    return (
        category == "mps_out_of_memory"
        and failure_stage == "attribution"
        and batch_size != MPS_BATCH_SEQUENCE[-1]
    )


def peak_memory_bytes(
    stage_peaks: Sequence[int | None], attempt_peak: int | None = None
) -> int | None:
    """Aggregate sampled peaks while preserving the attempt >= stage invariant."""

    values = [int(value) for value in stage_peaks if value is not None]
    if attempt_peak is not None:
        values.append(int(attempt_peak))
    return max(values) if values else None


def attempt_peak_at_least_stages(
    attempt_peak: int | None, stage_peaks: Sequence[int | None]
) -> bool:
    if attempt_peak is None:
        return not any(value is not None for value in stage_peaks)
    return all(value is None or attempt_peak >= value for value in stage_peaks)


def _max_rss_bytes() -> int | None:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (OSError, TypeError, ValueError):
        return None
    # macOS reports bytes; Linux reports KiB.
    return value if platform.system() == "Darwin" else value * 1024


def _current_rss_bytes(runner: Any = subprocess.run) -> int | None:
    output = _command(("ps", "-o", "rss=", "-p", str(os.getpid())), runner)
    if output:
        try:
            return int(output.strip()) * 1024
        except ValueError:
            pass
    return _max_rss_bytes()


def _command(command: tuple[str, ...], runner: Any = subprocess.run) -> str | None:
    try:
        result = runner(command, capture_output=True, text=True, check=False, timeout=2)
    except (OSError, subprocess.SubprocessError, TypeError):
        return None
    return result.stdout if getattr(result, "returncode", 1) == 0 else None


def _memory_pressure(runner: Any = subprocess.run) -> str | None:
    output = _command(("memory_pressure", "-Q"), runner)
    if output is None:
        output = _command(("memory_pressure",), runner)
    if output is None:
        return None
    lower = output.casefold()
    for marker in ("critical", "serious", "normal"):
        if marker in lower:
            return marker
    free = re.search(r"memory free percentage:\s*([0-9]{1,3})%", lower)
    if free is not None:
        percentage = int(free.group(1))
        if percentage >= 20:
            return "normal"
        if percentage >= 10:
            return "serious"
        return "critical"
    return "observed"


def _swap_used_bytes(runner: Any = subprocess.run) -> int | None:
    output = _command(("sysctl", "-n", "vm.swapusage"), runner)
    if not output:
        return None
    match = re.search(r"used\s*=\s*([0-9.]+)([KMGT])", output, re.IGNORECASE)
    if not match:
        return None
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    return int(float(match.group(1)) * multipliers[match.group(2).upper()])


def sample_mps_memory(
    torch_module: Any,
    *,
    runner: Any = subprocess.run,
    now: Any = time.time,
) -> dict[str, Any]:
    """Capture sampled MPS counters and host pressure without CUDA fields."""

    mps = getattr(torch_module, "mps", None)

    def counter(name: str) -> int | None:
        method = getattr(mps, name, None)
        try:
            value = method() if callable(method) else None
            return int(value) if value is not None else None
        except (RuntimeError, TypeError, ValueError):
            return None

    return {
        "sampled_at_unix": float(now()),
        "mps_current_allocated_bytes": counter("current_allocated_memory"),
        "mps_driver_allocated_bytes": counter("driver_allocated_memory"),
        "mps_recommended_max_bytes": counter("recommended_max_memory"),
        "process_rss_bytes": _current_rss_bytes(runner),
        "system_memory_pressure": _memory_pressure(runner),
        "swap_used_bytes": _swap_used_bytes(runner),
        "sampling_method": "torch.mps.counters+ps_rss+memory_pressure+sysctl",
    }


def summarize_mps_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    started_at_unix: float,
    finished_at_unix: float,
    wall_seconds: float,
    sampling_interval_seconds: float,
) -> dict[str, Any]:
    """Aggregate one named stage without upgrading samples into exact peaks."""

    if not samples:
        raise MPSRuntimeError("a telemetry stage must contain at least one sample")

    def peak(name: str) -> int | None:
        values = [
            int(item[name])
            for item in samples
            if isinstance(item.get(name), int)
            and not isinstance(item.get(name), bool)
            and int(item[name]) >= 0
        ]
        return max(values) if values else None

    pressures = list(
        dict.fromkeys(
            str(item["system_memory_pressure"])
            for item in samples
            if isinstance(item.get("system_memory_pressure"), str)
        )
    )
    observed_interval = (
        float(finished_at_unix - started_at_unix) / (len(samples) - 1)
        if len(samples) > 1
        else float(sampling_interval_seconds)
    )
    return {
        "started_at_unix": float(started_at_unix),
        "finished_at_unix": float(finished_at_unix),
        "wall_seconds": float(wall_seconds),
        "sample_count": len(samples),
        "sampling_interval_seconds": observed_interval,
        "target_sampling_interval_seconds": float(sampling_interval_seconds),
        "sampling_method": "periodic_boundary_and_interval_samples",
        "peak_mps_current_allocated_bytes": peak("mps_current_allocated_bytes"),
        "peak_mps_driver_allocated_bytes": peak("mps_driver_allocated_bytes"),
        "peak_mps_recommended_max_bytes": peak("mps_recommended_max_bytes"),
        "peak_process_rss_bytes": peak("process_rss_bytes"),
        "peak_swap_used_bytes": peak("swap_used_bytes"),
        "system_memory_pressures": pressures,
        "samples": [dict(item) for item in samples],
    }


mps_memory_sample = sample_mps_memory


def manifest_file(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    """Hash one regular file; paths are relative to a caller-selected root."""

    if path.is_symlink() or not path.is_file():
        raise MPSRuntimeError("asset manifest requires regular files")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    relative = path.name if root is None else path.relative_to(root).as_posix()
    return {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def validate_snapshot_containment(
    snapshot: Path, manifest: Sequence[Mapping[str, Any]]
) -> None:
    """Reject escaping symlinks, unmanifested entries, and malformed hashes."""

    root = snapshot.resolve()
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise MPSRuntimeError("snapshot is not a real directory")
    expected: dict[str, Mapping[str, Any]] = {}
    for item in manifest:
        name = item.get("path")
        digest = item.get("sha256")
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
        ):
            raise MPSRuntimeError("asset manifest path escapes snapshot")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise MPSRuntimeError("asset manifest hash is malformed")
        if name in expected:
            raise MPSRuntimeError("asset manifest contains duplicate paths")
        expected[name] = item
    for path in snapshot.rglob("*"):
        relative = path.relative_to(snapshot).as_posix()
        if path.is_symlink():
            if not path.resolve().is_relative_to(root):
                raise MPSRuntimeError("snapshot symlink escapes root")
            raise MPSRuntimeError("snapshot symlinks are not accepted")
        if path.is_file() and relative not in expected:
            raise MPSRuntimeError(f"snapshot contains unmanifested file: {relative}")
    for name, item in expected.items():
        path = snapshot / name
        observed = manifest_file(path, root=snapshot)
        if (
            observed["size_bytes"] != item.get("size_bytes")
            or observed["sha256"] != item["sha256"]
        ):
            raise MPSRuntimeError(f"snapshot manifest mismatch: {name}")


def explicit_cpu_sparse_metadata(
    indices: Any, values: Any, size: Sequence[int], *, torch_module: Any
) -> dict[str, Any]:
    """Represent sparse graph metadata on CPU while dense tensors remain on MPS.

    This is an execution-only adapter for the pinned upstream SparseMPS gap;
    it never moves dense scientific activations to CPU.
    """

    cpu = getattr(torch_module, "device", lambda name: name)("cpu")
    index_cpu = (
        indices.detach().to(device=cpu) if hasattr(indices, "detach") else indices
    )
    value_cpu = values.detach().to(device=cpu) if hasattr(values, "detach") else values
    return {
        "indices": index_cpu,
        "values": value_cpu,
        "size": tuple(int(x) for x in size),
        "device": "cpu",
        "purpose": "sparse_metadata_only",
    }


def validate_cpu_sparse_equivalence(
    mps_dense: Any,
    cpu_dense: Any,
    *,
    atol: float = 5e-3,
    rtol: float = 2e-3,
) -> bool:
    """Validate the small metadata adapter against a CPU dense reference."""

    if tuple(mps_dense.shape) != tuple(cpu_dense.shape):
        return False
    try:
        import torch  # type: ignore[import-not-found]

        return bool(
            torch.allclose(
                mps_dense.detach().float().cpu(),
                cpu_dense.detach().float().cpu(),
                atol=atol,
                rtol=rtol,
            )
        )
    except (ImportError, RuntimeError, TypeError):
        return False


def finite(value: Any, torch_module: Any) -> bool:
    try:
        return bool(torch_module.isfinite(value).all().item())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False
