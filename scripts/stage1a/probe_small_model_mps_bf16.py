#!/usr/bin/env python3
"""Bounded no-payload native-MPS/BF16 capability probe for Stage 1A-S-BF16."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cfsus.reproduction.artifacts import write_json_atomic  # noqa: E402
from cfsus.reproduction.small_model_mps_bf16 import (  # noqa: E402
    CONFIG_PATH,
    PROJECTED_MANIFEST,
    UPSTREAM_REVISION,
    assert_fallback_disabled,
    assert_mps_bf16_tensor,
    conservative_memory_feasibility,
    load_bf16_config,
    normalized_l2,
    run_overflow_regression,
    tensor_summary,
    validate_live_sparse_boundary,
    validate_projected_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / CONFIG_PATH)
    parser.add_argument("--output", type=Path)
    return parser


def _run_text(command: list[str]) -> str:
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
        env={
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        },
    )
    return result.stdout.strip()


def _package_vcs_commit() -> str:
    raw = importlib.metadata.distribution("circuit-tracer").read_text("direct_url.json")
    if raw is None:
        raise RuntimeError("circuit-tracer direct_url provenance is missing")
    commit = json.loads(raw).get("vcs_info", {}).get("commit_id")
    if commit != UPSTREAM_REVISION:
        raise RuntimeError("circuit-tracer installed commit does not match pin")
    return str(commit)


def _thermal_state() -> str:
    output = _run_text(["pmset", "-g", "therm"]).casefold()
    if "serious" in output or "critical" in output:
        return "serious_or_critical"
    if "no thermal warning level has been recorded" in output:
        return "nominal"
    if "fair" in output:
        return "fair"
    return "unknown"


def _swap_used_bytes() -> int:
    output = _run_text(["sysctl", "vm.swapusage"])
    marker = "used = "
    if marker not in output:
        raise RuntimeError("swap telemetry format is unavailable")
    value = output.split(marker, 1)[1].split("M", 1)[0].strip()
    return round(float(value) * 1024**2)


def _probe_tensor(name: str, value: Any, torch: Any) -> dict[str, Any]:
    assert_mps_bf16_tensor(value, torch, name)
    return tensor_summary(value, torch)


def _operator_probes(config: dict[str, Any]) -> dict[str, Any]:
    import torch
    from circuit_tracer.transcoder.activation_functions import JumpReLU

    device = torch.device("mps")
    results: dict[str, Any] = {}
    a = torch.tensor(
        [[0.25, -0.5, 1.0, 2.0], [-1.5, 0.75, 0.5, -0.25]],
        device=device,
        dtype=torch.bfloat16,
    )
    b = torch.tensor(
        [[0.5, -0.25, 1.0], [1.0, 0.5, -0.5], [-0.75, 1.25, 0.25], [0.5, 0.5, 0.5]],
        device=device,
        dtype=torch.bfloat16,
    )
    weight = torch.tensor(
        [[1.0, 0.5, -0.25, 0.75], [-0.5, 1.0, 0.5, 0.25], [0.25, -0.75, 1.0, 0.5]],
        device=device,
        dtype=torch.bfloat16,
    )
    bias = torch.tensor([0.25, -0.5, 0.75], device=device, dtype=torch.bfloat16)
    probes: dict[str, Callable[[], Any]] = {
        "tensor_add": lambda: a + torch.ones_like(a),
        "linear": lambda: torch.nn.functional.linear(a, weight, bias),
        "matmul": lambda: a @ b,
        "batched_matmul": lambda: torch.bmm(a.unsqueeze(0), b.unsqueeze(0)),
        "embedding": lambda: torch.nn.functional.embedding(
            torch.tensor([[0, 2, 1]], device=device, dtype=torch.long), weight
        ),
        "rmsnorm_source_path": lambda: (
            a.float() * torch.rsqrt(a.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        ).to(torch.bfloat16),
        "attention_scaling": lambda: (a @ a.T) * (4.0**-0.5),
        "attention_softmax_source_path": lambda: torch.nn.functional.softmax(
            a @ a.T, dim=-1, dtype=torch.float32
        ).to(torch.bfloat16),
        "masking": lambda: torch.where(a > 0, a, torch.zeros_like(a)),
        "gather": lambda: torch.gather(
            a, 1, torch.tensor([[0, 1], [2, 3]], device=device, dtype=torch.long)
        ),
        "scatter": lambda: torch.zeros_like(a).scatter(
            1,
            torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], device=device),
            a,
        ),
        "index_put_accumulate_false": lambda: torch.zeros_like(a).index_put_(
            (
                torch.tensor([0, 1], device=device),
                torch.tensor([1, 2], device=device),
            ),
            torch.tensor([1.0, 2.0], device=device, dtype=torch.bfloat16),
            accumulate=False,
        ),
        "index_add": lambda: torch.zeros_like(a).index_add_(
            0,
            torch.tensor([0, 1], device=device),
            torch.ones_like(a),
        ),
        "topk": lambda: torch.topk(a, 2).values,
        "sort": lambda: torch.sort(a).values,
        "threshold_comparison": lambda: a * (a > 0.5),
    }
    for name, function in probes.items():
        results[name] = _probe_tensor(name, function(), torch)

    nonzero = torch.nonzero(a, as_tuple=False)
    if nonzero.device.type != "mps" or nonzero.dtype != torch.long:
        raise RuntimeError("nonzero indices left MPS/long")
    results["nonzero"] = tensor_summary(nonzero, torch)

    positions = torch.arange(4, device=device, dtype=torch.long)
    if positions.device.type != "mps" or positions.dtype != torch.long:
        raise RuntimeError("integer arange left MPS")
    angle = torch.tensor([0.25, 0.5, 0.75, 1.0], device=device, dtype=torch.float32)
    rotary = (a * angle.cos().to(torch.bfloat16)) + (
        torch.flip(a, dims=(-1,)) * angle.sin().to(torch.bfloat16)
    )
    results["rotary_source_path"] = _probe_tensor("rotary_source_path", rotary, torch)

    threshold = torch.tensor([0.0, 0.5, 1.0, 1.5], device=device, dtype=torch.bfloat16)
    jump = JumpReLU(threshold)(a)
    results["loaded_jumprelu_class"] = _probe_tensor(
        "loaded_jumprelu_class", jump, torch
    )
    equality = JumpReLU(threshold)(threshold)
    if not bool(torch.equal(equality, torch.zeros_like(equality))):
        raise RuntimeError("JumpReLU equality is not inactive")
    results["jumprelu_equality_inactive"] = True

    grad_input = a.detach().clone().requires_grad_(True)
    retained = torch.autograd.grad(
        (grad_input * grad_input).sum(), grad_input, retain_graph=True
    )[0]
    results["retained_autograd"] = _probe_tensor("retained_autograd", retained, torch)

    def cubic(value: Any) -> Any:
        return value * value * value

    _, vjp_fn = torch.func.vjp(cubic, a)
    results["vjp"] = _probe_tensor("vjp", vjp_fn(torch.ones_like(a))[0], torch)
    _, tangent = torch.func.jvp(cubic, (a,), (torch.ones_like(a),))
    results["jvp"] = _probe_tensor("jvp", tangent, torch)

    try:
        sparse_native = a.to_sparse()
    except NotImplementedError as error:
        results["native_to_sparse"] = {
            "supported": False,
            "adapter_required": True,
            "error_type": type(error).__name__,
        }
    else:
        results["native_to_sparse"] = {
            "supported": True,
            "adapter_required": False,
            "value_dtype": str(sparse_native.values().dtype),
            "device": str(sparse_native.device),
        }
    results["sparse_metadata_adapter"] = validate_live_sparse_boundary(torch)

    cpu_reference = torch.nn.functional.linear(
        a.float().cpu(), weight.float().cpu(), bias.float().cpu()
    )
    observed = torch.nn.functional.linear(a, weight, bias)
    reference_error = normalized_l2(observed, cpu_reference, torch)
    if reference_error > float(
        config["tolerances"]["operator_reference_normalized_l2"]
    ):
        raise RuntimeError("BF16 linear probe diverged from FP32 reference")
    results["linear_fp32_reference_normalized_l2"] = reference_error
    return results


def _nnsight_probe() -> dict[str, Any]:
    import torch
    from nnsight import NNsight, save

    torch.manual_seed(0)
    module = torch.nn.Sequential(
        torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2)
    ).to(device="mps", dtype=torch.bfloat16)
    model = NNsight(module.eval())
    inputs = torch.ones(1, 4, device="mps", dtype=torch.bfloat16)
    with torch.inference_mode(), model.trace(inputs):
        baseline = save(model.output)
    with torch.inference_mode(), model.trace(inputs):
        model[0].output = torch.zeros_like(model[0].output)
        edited = save(model.output)
    assert_mps_bf16_tensor(baseline, torch, "NNsight baseline")
    assert_mps_bf16_tensor(edited, torch, "NNsight edited")
    effect = float(torch.max(torch.abs(baseline - edited)).item())
    if effect <= 0.0:
        raise RuntimeError("NNsight replacement had no effect")
    return {
        "passed": True,
        "baseline": tensor_summary(baseline, torch),
        "edited": tensor_summary(edited, torch),
        "maximum_absolute_effect": effect,
        "proxy_assignment_dtype_preserved": baseline.dtype == edited.dtype,
    }


def main() -> int:
    arguments = _parser().parse_args()
    started = time.time()
    assert_fallback_disabled()
    config = load_bf16_config(arguments.config)
    with (REPOSITORY_ROOT / PROJECTED_MANIFEST).open(encoding="utf-8") as stream:
        projected = validate_projected_manifest(json.load(stream))
    import torch

    if platform.machine() != "arm64" or sys.version_info[:2] != (3, 11):
        raise RuntimeError("probe requires native arm64 CPython 3.11")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not built and available")
    feasibility = conservative_memory_feasibility()
    if not feasibility.feasible:
        raise RuntimeError("conservative BF16 memory projection does not fit")
    overflow = run_overflow_regression(torch, config["tolerances"])
    operators = _operator_probes(config)
    nnsight = _nnsight_probe()
    thermal = _thermal_state()
    if thermal not in set(config["safety_limits"]["accepted_thermal_states"]):
        raise RuntimeError(f"unaccepted thermal state: {thermal}")
    lock = REPOSITORY_ROOT / config["artifacts"]["environment_lock"]
    record = {
        "schema_version": 1,
        "artifact_type": "stage1a_small_model_mps_bf16_operator_probe",
        "status": "passed",
        "execution_class": "preflight_only",
        "no_model_or_transcoder_payload_accessed": True,
        "environment": {
            "machine": platform.machine(),
            "python": platform.python_version(),
            "macos": platform.mac_ver()[0],
            "torch": torch.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "nnsight": importlib.metadata.version("nnsight"),
            "circuit_tracer": importlib.metadata.version("circuit-tracer"),
            "transformers": importlib.metadata.version("transformers"),
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
            "safetensors": importlib.metadata.version("safetensors"),
            "upstream_commit": _package_vcs_commit(),
            "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
            "fallback_variable_present": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
            is not None,
            "outer_autocast_enabled": bool(torch.is_autocast_enabled()),
            "source_mandated_internal_fp32": config["runtime"][
                "source_mandated_internal_fp32"
            ],
        },
        "projected_download_bytes": projected["projected_total_bytes"],
        "memory_projection": asdict(feasibility),
        "overflow_regression": overflow,
        "operators": operators,
        "nnsight_replacement": nnsight,
        "telemetry": {
            "swap_used_bytes": _swap_used_bytes(),
            "thermal_state": thermal,
            "mps_current_bytes": int(torch.mps.current_allocated_memory()),
            "mps_driver_bytes": int(torch.mps.driver_allocated_memory()),
        },
        "started_at_unix": started,
        "finished_at_unix": time.time(),
    }
    if arguments.output is not None:
        output = arguments.output
        if not output.is_absolute():
            output = REPOSITORY_ROOT / output
        generated_root = (
            REPOSITORY_ROOT / config["artifacts"]["generated_directory"]
        ).resolve()
        if not output.parent.resolve().is_relative_to(generated_root):
            raise RuntimeError("probe output must remain under generated directory")
        write_json_atomic(output, record)
    print(json.dumps(record, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
