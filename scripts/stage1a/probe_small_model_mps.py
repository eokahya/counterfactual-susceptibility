#!/usr/bin/env python3
"""No-download native-MPS operator and NNsight probe for Stage 1A-S."""

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
from cfsus.reproduction.small_model_mps_fp16 import (  # noqa: E402
    PROJECTED_MANIFEST,
    UPSTREAM_REVISION,
    assert_fallback_disabled,
    conservative_memory_feasibility,
    load_small_model_config,
    validate_live_sparse_boundary,
    validate_projected_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/stage1a_small_model_mps_fp16_pilot.yaml",
    )
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


def _package_vcs_commit() -> str:
    raw = importlib.metadata.distribution("circuit-tracer").read_text("direct_url.json")
    if raw is None:
        raise RuntimeError("circuit-tracer direct_url provenance is missing")
    value = json.loads(raw)
    commit = value.get("vcs_info", {}).get("commit_id")
    if commit != UPSTREAM_REVISION:
        raise RuntimeError("circuit-tracer installed commit does not match the pin")
    return str(commit)


def _tensor_record(value: Any) -> dict[str, Any]:
    import torch  # type: ignore[import-not-found]

    if not isinstance(value, torch.Tensor):
        raise TypeError("operator probe result must be a tensor")
    return {
        "device": str(value.device),
        "dtype": str(value.dtype),
        "finite": bool(torch.isfinite(value).all().item()),
        "shape": [int(size) for size in value.shape],
    }


def _operator_probes() -> dict[str, Any]:
    import torch

    device = torch.device("mps")
    torch.manual_seed(0)
    a = torch.randn(3, 4, device=device, dtype=torch.float16)
    b = torch.randn(4, 5, device=device, dtype=torch.float16)
    results: dict[str, Any] = {}

    probes: dict[str, Callable[[], Any]] = {
        "linear": lambda: torch.nn.functional.linear(
            a, torch.randn(6, 4, device=device, dtype=torch.float16)
        ),
        "matmul": lambda: a @ b,
        "batched_matmul": lambda: torch.bmm(
            torch.randn(2, 3, 4, device=device, dtype=torch.float16),
            torch.randn(2, 4, 5, device=device, dtype=torch.float16),
        ),
        "rms_norm": lambda: torch.nn.functional.rms_norm(a, (4,)),
        "rotary_sin_cos": lambda: (
            a * torch.cos(a) + torch.flip(a, (-1,)) * torch.sin(a)
        ),
        "softmax": lambda: a.softmax(-1),
        "topk": lambda: torch.topk(a, 2).values,
        "sort": lambda: torch.sort(a).values,
        "mask": lambda: torch.where(a > 0, a, torch.zeros_like(a)),
        "nonzero": lambda: (a > 0).nonzero(),
        "gather": lambda: torch.gather(
            a,
            1,
            torch.tensor([[0, 1], [1, 2], [2, 3]], device=device),
        ),
        "scatter": lambda: torch.zeros_like(a).scatter(
            1,
            torch.tensor([[0, 1, 2, 3]] * 3, device=device),
            a,
        ),
        "index_put": lambda: torch.zeros_like(a).index_put_(
            (
                torch.tensor([0, 1], device=device),
                torch.tensor([1, 2], device=device),
            ),
            torch.tensor([1.0, 2.0], device=device, dtype=torch.float16),
            accumulate=False,
        ),
        "index_add": lambda: torch.zeros_like(a).index_add_(
            0,
            torch.tensor([0, 2], device=device),
            torch.ones(2, 4, device=device, dtype=torch.float16),
        ),
        "threshold_equality": lambda: torch.where(a > a, a, torch.zeros_like(a)),
    }
    for name, probe in probes.items():
        record = _tensor_record(probe())
        if record["device"] != "mps:0" or not record["finite"]:
            raise RuntimeError(f"operator {name} left finite MPS execution")
        results[name] = record

    autograd_input = torch.randn(
        2, 3, device=device, dtype=torch.float16, requires_grad=True
    )
    gradient = torch.autograd.grad(
        (autograd_input * autograd_input).sum(), autograd_input, retain_graph=True
    )[0]
    results["autograd_retained"] = _tensor_record(gradient)

    def cubic(value: Any) -> Any:
        return value * value * value

    _, vjp_fn = torch.func.vjp(cubic, autograd_input.detach())
    results["vjp"] = _tensor_record(vjp_fn(torch.ones_like(autograd_input))[0])
    _, tangent = torch.func.jvp(
        cubic, (autograd_input.detach(),), (torch.ones_like(autograd_input),)
    )
    results["jvp"] = _tensor_record(tangent)

    try:
        a.to_sparse()
    except NotImplementedError as error:
        results["native_to_sparse"] = {
            "supported": False,
            "expected_gap": True,
            "error_type": type(error).__name__,
        }
    else:
        raise RuntimeError(
            "native MPS to_sparse unexpectedly changed; re-audit required"
        )
    results["sparse_metadata_adapter"] = validate_live_sparse_boundary(torch)
    return results


def _nnsight_assignment_probe() -> dict[str, Any]:
    import torch
    from nnsight import NNsight, save  # type: ignore[import-not-found]

    torch.manual_seed(0)
    module = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 2),
    ).to(device="mps", dtype=torch.float16)
    model = NNsight(module.eval())
    inputs = torch.ones(1, 4, device="mps", dtype=torch.float16)
    with torch.inference_mode(), model.trace(inputs):
        baseline = save(model.output)
    with torch.inference_mode(), model.trace(inputs):
        model[0].output = torch.zeros_like(model[0].output)
        edited = save(model.output)
    if baseline.device.type != "mps" or edited.device.type != "mps":
        raise RuntimeError("NNsight assignment left MPS")
    difference = float(torch.max(torch.abs(baseline - edited)).item())
    if difference <= 0.0:
        raise RuntimeError("NNsight assignment did not change the output")
    return {
        "passed": True,
        "baseline_device": str(baseline.device),
        "edited_device": str(edited.device),
        "maximum_absolute_effect": difference,
    }


def _attribution_component_adapter_probe() -> dict[str, Any]:
    import torch

    from cfsus.reproduction.small_model_mps_fp16 import (
        mps_compute_attribution_components,
    )

    class TinyTranscoder:
        def __init__(self, layer: int) -> None:
            generator = torch.Generator(device="cpu").manual_seed(layer + 31)
            self.W_enc = torch.randn(6, 4, generator=generator).to(
                device="mps", dtype=torch.float16
            )
            self.W_dec = torch.randn(6, 4, generator=generator).to(
                device="mps", dtype=torch.float16
            )
            self.b_enc = torch.linspace(-0.5, 0.5, 6).to(
                device="mps", dtype=torch.float16
            )
            self.b_dec = torch.zeros(4, device="mps", dtype=torch.float16)
            self.threshold = torch.linspace(-0.25, 0.75, 6).to(
                device="mps", dtype=torch.float16
            )
            self.W_skip = None
            self.d_model = 4

        def activation_function(self, value: Any) -> Any:
            return value * (value > self.threshold)

        def _get_decoder_vectors(self, features: Any) -> Any:
            return self.W_dec[features.to(device="mps")]

    class TinySet:
        d_transcoder = 6

        def __init__(self) -> None:
            self.layers = [TinyTranscoder(0), TinyTranscoder(1)]

        def __len__(self) -> int:
            return len(self.layers)

        def __iter__(self) -> Any:
            return iter(self.layers)

    transcoders = TinySet()
    inputs = torch.tensor(
        [
            [[0.5, -1.0, 0.25, 2.0], [1.0, 0.5, -0.5, 0.25]],
            [[-0.25, 0.75, 1.5, -1.0], [0.5, 0.25, 1.0, -0.5]],
        ],
        device="mps",
        dtype=torch.float16,
    )
    observed = mps_compute_attribution_components(transcoders, inputs)
    expected_activations = []
    expected_reconstruction = []
    for layer, transcoder in enumerate(transcoders):
        preactivation = torch.nn.functional.linear(
            inputs[layer], transcoder.W_enc, transcoder.b_enc
        )
        activation = transcoder.activation_function(preactivation)
        activation[0] = 0
        expected_activations.append(activation)
        expected_reconstruction.append(activation @ transcoder.W_dec + transcoder.b_dec)
    expected_dense = torch.stack(expected_activations)
    observed_dense = observed["activation_matrix"].to_dense().to(dtype=torch.float16)
    expected_reconstruction_tensor = torch.stack(expected_reconstruction)
    activation_error = float(
        torch.max(torch.abs(observed_dense - expected_dense.cpu())).item()
    )
    reconstruction_error = float(
        torch.max(
            torch.abs(observed["reconstruction"] - expected_reconstruction_tensor)
        ).item()
    )
    if activation_error != 0.0 or reconstruction_error > 0.005:
        raise RuntimeError("attribution component adapter is not equivalent")
    return {
        "passed": True,
        "activation_maximum_absolute_error": activation_error,
        "reconstruction_maximum_absolute_error": reconstruction_error,
        "activation_metadata_device": str(observed["activation_matrix"].device),
        "reconstruction_device": str(observed["reconstruction"].device),
        "active_feature_count": int(observed["activation_matrix"]._nnz()),
    }


def main() -> int:
    arguments = _parser().parse_args()
    started = time.time()
    assert_fallback_disabled()
    config = load_small_model_config(arguments.config)
    with (REPOSITORY_ROOT / PROJECTED_MANIFEST).open(encoding="utf-8") as stream:
        projected = validate_projected_manifest(json.load(stream))
    import torch

    if platform.machine() != "arm64" or sys.version_info[:2] != (3, 11):
        raise RuntimeError("probe requires native arm64 CPython 3.11")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not built and available")
    feasibility = conservative_memory_feasibility()
    if not feasibility.feasible:
        raise RuntimeError("conservative Stage 1A-S memory projection does not fit")
    operators = _operator_probes()
    nnsight_assignment = _nnsight_assignment_probe()
    attribution_component_adapter = _attribution_component_adapter_probe()
    lock = REPOSITORY_ROOT / config["artifacts"]["environment_lock"]
    record = {
        "schema_version": 1,
        "artifact_type": "stage1a_small_model_mps_operator_probe",
        "status": "passed",
        "execution_class": "preflight_only",
        "no_model_or_transcoder_payload_accessed": True,
        "environment": {
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "nnsight": importlib.metadata.version("nnsight"),
            "circuit_tracer": importlib.metadata.version("circuit-tracer"),
            "transformers": importlib.metadata.version("transformers"),
            "upstream_commit": _package_vcs_commit(),
            "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
            "fallback_variable_present": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
            is not None,
        },
        "projected_download_bytes": projected["projected_total_bytes"],
        "memory_projection": {
            key: value
            for key, value in asdict(feasibility).items()
            if isinstance(value, (bool, int))
        },
        "operators": operators,
        "nnsight_assignment": nnsight_assignment,
        "attribution_component_adapter": attribution_component_adapter,
        "telemetry": {
            "swap_used_bytes": _swap_used_bytes(),
            "thermal_state": _thermal_state(),
        },
        "started_at_unix": started,
        "finished_at_unix": time.time(),
    }
    if arguments.output is not None:
        output = arguments.output
        if not output.is_absolute():
            output = REPOSITORY_ROOT / output
        resolved_parent = output.parent.resolve()
        generated_root = (
            REPOSITORY_ROOT / config["artifacts"]["generated_directory"]
        ).resolve()
        if not resolved_parent.is_relative_to(generated_root):
            raise RuntimeError("probe output must remain under the generated directory")
        write_json_atomic(output, record)
    print(json.dumps(record, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
