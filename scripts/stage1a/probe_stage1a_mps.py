#!/usr/bin/env python3
"""Run a bounded, dependency-light Apple MPS FP16 preflight.

The probe intentionally does not import Gemma, TransformerLens, or the large
transcoder snapshots.  Every tensor used for an MPS test is created locally,
and every CPU transfer is an explicit, named boundary in the result.  The
probe never enables PyTorch's MPS fallback; a fallback setting is a failure.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import resource
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIRECTORY = REPOSITORY_ROOT / "results" / "stage1a_mps_fp16" / "preflight"
GENERATED_RESULT_DIRECTORY = (
    REPOSITORY_ROOT / "results" / "generated" / "stage1a_mps_fp16"
)
DEFAULT_OUTPUT = RESULT_DIRECTORY / "preflight_summary.json"
EXPECTED_TORCH_MAJOR_MINOR = (2, 6)
EXPECTED_DTYPE = "float16"
MPS_DEVICE = "mps"
MPS_ATOL = 5e-3
MPS_RTOL = 2e-3
MAX_OUTPUT_BYTES = 1_000_000
TRUTHY = frozenset({"1", "true", "yes", "on"})


def _safe_error(error: BaseException) -> str:
    """Return a bounded diagnostic without home paths or credential material."""

    message = " ".join(str(error).split())
    home = str(Path.home())
    if home:
        message = message.replace(home, "<HOME>")
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "TOKEN"):
        message = message.replace(os.environ.get(key, "\0"), "<REDACTED>")
    return message[:400] or type(error).__name__


def fallback_enabled(environ: dict[str, str] | None = None) -> bool:
    """Whether PyTorch's implicit MPS-to-CPU fallback was requested."""

    values = os.environ if environ is None else environ
    value = values.get("PYTORCH_ENABLE_MPS_FALLBACK", "")
    return value.strip().casefold() in TRUTHY


def _finite(value: Any, torch: Any) -> bool:
    try:
        return bool(torch.isfinite(value).all().item())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _same_device(value: Any, device: str) -> bool:
    """Match a tensor device by type, accepting PyTorch's indexed MPS spelling.

    PyTorch 2.6 can render a tensor allocated with ``device="mps"`` as
    ``mps:0``.  Comparing strings would therefore reject a valid MPS result.
    The type check remains explicit so an accidental CPU tensor can never pass
    an MPS check.
    """

    actual = getattr(value, "device", None)
    if actual is None:
        return False
    expected_text = str(device)
    expected_type, separator, expected_index_text = expected_text.partition(":")
    expected_index = int(expected_index_text) if separator else None
    actual_text = str(actual)
    actual_type: str | None
    actual_index: int | None
    if isinstance(actual, str):
        actual_type, actual_separator, actual_index_text = actual_text.partition(":")
        actual_index = None
        if actual_separator:
            try:
                actual_index = int(actual_index_text)
            except ValueError:
                return False
    else:
        raw_actual_type = getattr(actual, "type", None)
        actual_type = raw_actual_type if isinstance(raw_actual_type, str) else None
        raw_actual_index = getattr(actual, "index", None)
        actual_index = (
            raw_actual_index
            if isinstance(raw_actual_index, int)
            and not isinstance(raw_actual_index, bool)
            else None
        )
        if actual_type is None:
            actual_type, actual_separator, actual_index_text = actual_text.partition(
                ":"
            )
            if actual_separator and actual_index is None:
                try:
                    actual_index = int(actual_index_text)
                except ValueError:
                    return False
    if actual_type != expected_type:
        return False
    if expected_index is not None:
        return actual_index == expected_index
    if expected_type == "mps":
        return actual_index in (None, 0)
    return True


def _sync(torch: Any) -> None:
    mps = getattr(torch, "mps", None)
    synchronize = getattr(mps, "synchronize", None)
    if callable(synchronize):
        synchronize()


def _memory_snapshot(torch: Any | None) -> dict[str, int | None]:
    """Collect sampled MPS allocator values; exact CUDA peak semantics are not used."""

    result: dict[str, int | None] = {
        "current_allocated_bytes": None,
        "driver_allocated_bytes": None,
        "recommended_max_bytes": None,
        "process_max_rss_bytes": None,
    }
    if torch is not None:
        mps = getattr(torch, "mps", None)
        for key, name in (
            ("current_allocated_bytes", "current_allocated_memory"),
            ("driver_allocated_bytes", "driver_allocated_memory"),
            ("recommended_max_bytes", "recommended_max_memory"),
        ):
            function = getattr(mps, name, None)
            if callable(function):
                with contextlib.suppress(OSError, RuntimeError, TypeError, ValueError):
                    result[key] = int(function())
    try:
        max_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # macOS reports bytes, while Linux reports KiB.
        result["process_max_rss_bytes"] = (
            max_rss if platform.system() == "Darwin" else max_rss * 1024
        )
    except (OSError, ValueError):
        pass
    return result


def _record(
    name: str,
    function: Callable[[Any], dict[str, Any] | None],
    torch: Any,
    *,
    device: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "attempted": True,
        "device": device,
        "dtype": EXPECTED_DTYPE,
        "passed": False,
        "error": None,
        "error_type": None,
    }
    try:
        details = function(torch) or {}
        record.update(details)
        record["passed"] = details.get("cpu_reference_passed", True) is True
        if not record["passed"]:
            record["error"] = "MPS result differed from CPU reference"
    except (Exception, KeyboardInterrupt) as error:
        if isinstance(error, KeyboardInterrupt):
            raise
        record["error"] = _safe_error(error)
        record["error_type"] = type(error).__name__
    return record


def _cpu_reference_equal(torch: Any, observed: Any, reference: Any) -> bool:
    """Compare an MPS result with an independently computed CPU result."""

    observed_items = observed if isinstance(observed, tuple) else (observed,)
    reference_items = reference if isinstance(reference, tuple) else (reference,)
    if len(observed_items) != len(reference_items):
        return False
    for observed_item, reference_item in zip(
        observed_items, reference_items, strict=True
    ):
        observed_cpu = observed_item.detach().to(device="cpu")
        reference_cpu = reference_item.detach().to(device="cpu")
        if observed_cpu.shape != reference_cpu.shape:
            return False
        if observed_cpu.dtype.is_floating_point:
            if not bool(
                torch.allclose(
                    observed_cpu,
                    reference_cpu,
                    atol=MPS_ATOL,
                    rtol=MPS_RTOL,
                )
            ):
                return False
        elif not bool(torch.equal(observed_cpu, reference_cpu)):
            return False
    return True


def _operation_probe(torch: Any, device: str) -> dict[str, Any]:
    try:
        d = torch.device(device)
        base = torch.arange(16, device=d, dtype=torch.float16).reshape(4, 4) / 8
        weight = torch.eye(4, device=d, dtype=torch.float16)
        cpu_base = torch.arange(16, dtype=torch.float16).reshape(4, 4) / 8
        cpu_weight = torch.eye(4, dtype=torch.float16)
    except Exception as error:
        return {
            "attempted": True,
            "device": device,
            "dtype": EXPECTED_DTYPE,
            "passed": False,
            "cpu_reference_passed": False,
            "operation_count": 0,
            "operations": {},
            "error": _safe_error(error),
            "error_type": type(error).__name__,
        }
    records: dict[str, dict[str, Any]] = {}

    def run(
        name: str, function: Callable[[], Any], cpu_function: Callable[[], Any]
    ) -> Any | None:
        record: dict[str, Any] = {
            "attempted": True,
            "device": device,
            "dtype": EXPECTED_DTYPE,
            "passed": False,
            "cpu_reference_passed": False,
            "error": None,
            "error_type": None,
        }
        try:
            value = function()
            cpu_value = cpu_function()
            tensors = value if isinstance(value, tuple) else (value,)
            if any(not _same_device(item, device) for item in tensors):
                raise RuntimeError("operation produced an unexpected device")
            if any(not _finite(item, torch) for item in tensors):
                raise RuntimeError("operation produced a non-finite tensor")
            _sync(torch)
            record["cpu_reference_passed"] = _cpu_reference_equal(
                torch, value, cpu_value
            )
            if not record["cpu_reference_passed"]:
                raise RuntimeError("MPS result differed from CPU reference")
            record["passed"] = True
            return value
        except Exception as error:
            record["error"] = _safe_error(error)
            record["error_type"] = type(error).__name__
            return None
        finally:
            records[name] = record

    run("matmul", lambda: base @ weight, lambda: cpu_base @ cpu_weight)
    run(
        "einsum",
        lambda: torch.einsum("ij,jk->ik", base, weight),
        lambda: torch.einsum("ij,jk->ik", cpu_base, cpu_weight),
    )
    run(
        "softmax",
        lambda: torch.softmax(base, dim=-1),
        lambda: torch.softmax(cpu_base, dim=-1),
    )
    run(
        "layernorm",
        lambda: torch.nn.functional.layer_norm(base, (4,)),
        lambda: torch.nn.functional.layer_norm(cpu_base, (4,)),
    )
    top_result = run(
        "topk",
        lambda: torch.topk(base, k=2, dim=-1),
        lambda: torch.topk(cpu_base, k=2, dim=-1),
    )
    top_values, top_indices = (
        top_result if isinstance(top_result, tuple) else (None, None)
    )
    if top_values is not None and top_indices is not None:
        cpu_top_values, cpu_top_indices = torch.topk(cpu_base, k=2, dim=-1)
        run(
            "gather",
            lambda: torch.gather(base, 1, top_indices),
            lambda: torch.gather(cpu_base, 1, cpu_top_indices),
        )
        run(
            "scatter",
            lambda: torch.zeros_like(base).scatter_(1, top_indices, top_values),
            lambda: torch.zeros_like(cpu_base).scatter_(
                1, cpu_top_indices, cpu_top_values
            ),
        )
    else:
        records["gather"] = {
            "attempted": False,
            "passed": False,
            "cpu_reference_passed": False,
            "error": "topk prerequisite failed",
        }
        records["scatter"] = {
            "attempted": False,
            "passed": False,
            "cpu_reference_passed": False,
            "error": "topk prerequisite failed",
        }
    run(
        "index_add",
        lambda: torch.zeros_like(base).index_add_(
            0, torch.tensor([0, 1, 1, 3], device=d), base
        ),
        lambda: torch.zeros_like(cpu_base).index_add_(
            0, torch.tensor([0, 1, 1, 3]), cpu_base
        ),
    )
    run(
        "index_put",
        lambda: torch.zeros_like(base).index_put_(
            (torch.tensor([0, 2], device=d), torch.tensor([1, 3], device=d)),
            torch.tensor([1, 2], device=d, dtype=torch.float16),
        ),
        lambda: torch.zeros_like(cpu_base).index_put_(
            (torch.tensor([0, 2]), torch.tensor([1, 3])),
            torch.tensor([1, 2], dtype=torch.float16),
        ),
    )
    run(
        "where",
        lambda: torch.where(base > 0.5, base, torch.zeros_like(base)),
        lambda: torch.where(cpu_base > 0.5, cpu_base, torch.zeros_like(cpu_base)),
    )
    sorted_result = run(
        "sort",
        lambda: torch.sort(base.flatten()).values,
        lambda: torch.sort(cpu_base.flatten()).values,
    )
    run(
        "unique",
        lambda: torch.unique(torch.tensor([0, 1, 1, 2, 3], device=d)),
        lambda: torch.unique(torch.tensor([0, 1, 1, 2, 3])),
    )
    if sorted_result is not None:
        run(
            "searchsorted",
            lambda: torch.searchsorted(
                sorted_result, torch.tensor([0.5, 1.0], device=d)
            ),
            lambda: torch.searchsorted(
                torch.sort(cpu_base.flatten()).values, torch.tensor([0.5, 1.0])
            ),
        )
    else:
        records["searchsorted"] = {
            "attempted": False,
            "passed": False,
            "cpu_reference_passed": False,
            "error": "sort prerequisite failed",
        }
    passed = all(bool(record.get("passed")) for record in records.values())
    return {
        "attempted": True,
        "device": device,
        "dtype": EXPECTED_DTYPE,
        "passed": passed,
        "cpu_reference_passed": all(
            bool(record.get("cpu_reference_passed")) for record in records.values()
        ),
        "operation_count": len(records),
        "operations": records,
    }


def _transfer_probe(torch: Any, device: str) -> dict[str, Any]:
    def transfers(t: Any) -> dict[str, Any]:
        source = t.arange(8, dtype=t.float16).reshape(2, 4)
        moved = source.to(device=device)
        _sync(t)
        round_trip = moved.to(device="cpu")
        if not _same_device(moved, device) or str(round_trip.device) != "cpu":
            raise RuntimeError("explicit CPU/MPS transfer had an unexpected device")
        if not bool(t.allclose(round_trip, source, atol=MPS_ATOL, rtol=MPS_RTOL)):
            raise RuntimeError("CPU/MPS transfer exceeded tolerance")
        return {
            "round_trip_max_abs_error": float((round_trip - source).abs().max().item()),
            "cpu_reference_passed": True,
        }

    return _record("transfers", transfers, torch, device=device)


def _autograd_probe(torch: Any, device: str) -> dict[str, Any]:
    def autograd(t: Any) -> dict[str, Any]:
        x = t.ones((2, 4), device=device, dtype=t.float16, requires_grad=True)
        w = t.eye(4, device=device, dtype=t.float16)
        hook_seen: list[bool] = []

        def capture(gradient: Any) -> Any:
            hook_seen.append(_same_device(gradient, device) and _finite(gradient, t))
            return gradient

        x.register_hook(capture)
        loss = (x @ w).square().sum()
        loss.backward()
        _sync(t)
        if not hook_seen or not all(hook_seen) or x.grad is None:
            raise RuntimeError("MPS autograd hook did not receive a finite gradient")
        cpu_x = t.ones((2, 4), device="cpu", dtype=t.float16, requires_grad=True)
        cpu_w = t.eye(4, device="cpu", dtype=t.float16)
        (cpu_x @ cpu_w).square().sum().backward()
        cpu_reference_passed = bool(
            t.allclose(
                x.grad.to(device="cpu"),
                cpu_x.grad,
                atol=MPS_ATOL,
                rtol=MPS_RTOL,
            )
        )
        return {
            "hook_called": True,
            "gradient_finite": _finite(x.grad, t),
            "cpu_reference_passed": cpu_reference_passed,
        }

    return _record("autograd_hooks", autograd, torch, device=device)


def _jumprelu_probe(torch: Any, device: str) -> dict[str, Any]:
    def jumprelu(t: Any) -> dict[str, Any]:
        threshold = 0.5
        # 0.5009765625 is the next representable FP16 value above 0.5.
        source = [-0.5, 0.0, 0.5, 0.5009765625, 1.0]
        cpu_x = t.tensor(source, dtype=t.float16)
        mps_x = cpu_x.to(device=device)
        cpu_y = cpu_x * (cpu_x > threshold)
        mps_y = mps_x * (mps_x > t.tensor(threshold, device=device, dtype=t.float16))
        _sync(t)
        cpu_gate = (cpu_x > threshold).to(device="cpu")
        mps_gate = (mps_x > threshold).to(device="cpu")
        if not bool(t.equal(cpu_gate, mps_gate)):
            raise RuntimeError("strict JumpReLU gate differs between CPU and MPS")
        error = float((mps_y.to(device="cpu") - cpu_y).abs().max().item())
        if error > MPS_ATOL or not _finite(mps_y, t):
            raise RuntimeError(f"JumpReLU CPU comparison exceeded tolerance: {error}")
        if float(mps_y[2].item()) != 0.0:
            raise RuntimeError("JumpReLU threshold equality was treated as active")
        return {
            "strict_gate_equal": True,
            "equality_inactive": float(mps_y[2].item()) == 0.0,
            "max_abs_error": error,
            "cpu_reference_passed": True,
        }

    return _record("strict_jumprelu", jumprelu, torch, device=device)


class _TinyHookedEquivalent:
    """Small hookable transformer-like computation without TransformerLens."""

    def __init__(self, torch: Any, device: str) -> None:
        self.torch = torch
        self.device = device
        self.embedding = torch.nn.Linear(4, 4, bias=True).to(
            device=device, dtype=torch.float16
        )
        self.projection = torch.nn.Linear(4, 3, bias=True).to(
            device=device, dtype=torch.float16
        )
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.eye(4, device=device, dtype=torch.float16)
            )
            self.embedding.bias.fill_(0.25)
            self.projection.weight.fill_(0.125)
            self.projection.bias.zero_()

    def __call__(self, value: Any, hook: Callable[[Any], Any] | None = None) -> Any:
        activation = self.embedding(value)
        activation = self.torch.relu(activation)
        if hook is not None:
            activation = hook(activation)
        return self.projection(activation)


def _tiny_model_probe(torch: Any, device: str) -> dict[str, Any]:
    def tiny(t: Any) -> dict[str, Any]:
        model = _TinyHookedEquivalent(t, device)
        source = t.ones((1, 4), device=device, dtype=t.float16)
        baseline = model(source)
        captured: list[Any] = []

        def capture(activation: Any) -> Any:
            captured.append(activation.detach())
            return activation

        hooked = model(source, capture)
        if not captured:
            raise RuntimeError("tiny model forward hook was not called")
        baseline_activation = captured[0]
        conditions: dict[str, Any] = {}
        for alpha in (0.0, 0.5, 1.0):

            def replace_activation(_: Any, scale: float = alpha) -> Any:
                return (1.0 - scale) * baseline_activation

            edited = model(source, replace_activation)
            conditions[str(alpha)] = edited
        _sync(t)
        if not all(
            _same_device(item, device) and _finite(item, t)
            for item in (baseline, hooked, *conditions.values())
        ):
            raise RuntimeError("tiny hooked transformer produced invalid output")
        noop_error = float((conditions["0.0"] - hooked).abs().max().item())
        half_expected = baseline_activation * 0.5
        half_error = float(
            (conditions["0.5"] - model(source, lambda _: half_expected))
            .abs()
            .max()
            .item()
        )
        full_activation = model(source, lambda _: t.zeros_like(baseline_activation))
        full_error = float((conditions["1.0"] - full_activation).abs().max().item())
        if noop_error > MPS_ATOL or half_error > MPS_ATOL or full_error > MPS_ATOL:
            raise RuntimeError("tiny intervention mapping was not stable")
        cpu_model = _TinyHookedEquivalent(t, "cpu")
        cpu_source = t.ones((1, 4), device="cpu", dtype=t.float16)
        cpu_baseline = cpu_model(cpu_source)
        cpu_reference_passed = bool(
            t.allclose(
                baseline.to(device="cpu"),
                cpu_baseline,
                atol=MPS_ATOL,
                rtol=MPS_RTOL,
            )
        )
        return {
            "forward_hook_called": True,
            "intervention_alphas": [0.0, 0.5, 1.0],
            "noop_max_abs_error": noop_error,
            "half_mapping_max_abs_error": half_error,
            "full_mapping_max_abs_error": full_error,
            "output_shape": list(baseline.shape),
            "cpu_reference_passed": cpu_reference_passed,
        }

    return _record("tiny_hooked_transformer", tiny, torch, device=device)


def _sparse_boundary_probe(torch: Any, device: str) -> dict[str, Any]:
    def sparse(t: Any) -> dict[str, Any]:
        indices_cpu = t.tensor([[0, 1, 1], [0, 1, 2]], dtype=t.long)
        values = t.tensor([1.0, 2.0, 3.0], device=device, dtype=t.float16)
        dense = t.zeros((2, 3), device=device, dtype=t.float16)
        dense.index_put_((indices_cpu[0].to(device), indices_cpu[1].to(device)), values)
        replacement_passed = bool(_finite(dense, t) and dense.sum().item() == 6.0)
        if not replacement_passed:
            raise RuntimeError("dense MPS replacement for sparse metadata failed")
        native_supported: bool | None = None
        native_error: str | None = None
        try:
            native = t.sparse_coo_tensor(
                indices_cpu.to(device), values, size=(2, 3), device=device
            )
            native = native.coalesce()
            native_supported = _finite(native.values(), t)
        except Exception as error:
            native_supported = False
            native_error = _safe_error(error)
        if native_supported is False and not replacement_passed:
            raise RuntimeError(
                "native SparseMPS failed without a passing replacement boundary"
            )
        cpu_dense = t.zeros((2, 3), dtype=t.float16)
        cpu_dense.index_put_((indices_cpu[0], indices_cpu[1]), values.to(device="cpu"))
        return {
            "native_sparse_mps_supported": native_supported,
            "native_sparse_mps_error": native_error,
            "cpu_metadata_explicit": True,
            "replacement_boundary_passed": replacement_passed,
            "dense_scientific_device": device,
            "cpu_reference_passed": bool(
                t.allclose(
                    dense.to(device="cpu"),
                    cpu_dense,
                    atol=MPS_ATOL,
                    rtol=MPS_RTOL,
                )
            ),
        }

    return _record("sparse_metadata_boundary", sparse, torch, device=device)


def _graph_probe(torch: Any, device: str) -> dict[str, Any]:
    def graph(t: Any) -> dict[str, Any]:
        adjacency = t.zeros((16, 16), device=device, dtype=t.float16)
        rows = t.arange(1, 8, device=device)
        cols = rows - 1
        adjacency.index_put_((rows, cols), t.ones(7, device=device, dtype=t.float16))
        adjacency.index_put_(
            (t.tensor([15], device=device), t.tensor([7], device=device)),
            t.tensor([1], device=device, dtype=t.float16),
        )
        edge_count = int((adjacency != 0).sum().item())
        if edge_count <= 0 or not _finite(adjacency, t):
            raise RuntimeError("bounded graph adjacency was empty or non-finite")
        cpu_adjacency = t.zeros((16, 16), dtype=t.float16)
        cpu_adjacency.index_put_(
            (t.arange(1, 8), t.arange(0, 7)), t.ones(7, dtype=t.float16)
        )
        cpu_adjacency.index_put_(
            (t.tensor([15]), t.tensor([7])), t.tensor([1], dtype=t.float16)
        )
        return {
            "node_count": 16,
            "edge_count": edge_count,
            "bounded": True,
            "cpu_reference_passed": bool(
                t.equal(adjacency.to(device="cpu"), cpu_adjacency)
            ),
        }

    return _record("bounded_graph_construction", graph, torch, device=device)


def _disk_round_trip_probe(torch: Any, device: str) -> dict[str, Any]:
    def disk(t: Any) -> dict[str, Any]:
        from safetensors.torch import (  # type: ignore[import-not-found]
            load_file,
            save_file,
        )

        state = {"weight": t.arange(12, dtype=t.float16).reshape(3, 4)}
        with tempfile.NamedTemporaryFile(
            prefix="stage1a-mps-preflight-", suffix=".safetensors"
        ) as handle:
            save_file(state, handle.name)
            loaded = load_file(handle.name, device=device)
        value = loaded["weight"]
        _sync(t)
        if not _same_device(value, device) or not _finite(value, t):
            raise RuntimeError(
                "safetensors disk round trip returned an invalid MPS tensor"
            )
        error = float((value.to(device="cpu") - state["weight"]).abs().max().item())
        if error > MPS_ATOL:
            raise RuntimeError(f"safetensors round trip exceeded tolerance: {error}")
        upstream_helper_tested = False
        try:
            from circuit_tracer.utils.disk_offload import (  # type: ignore[import-not-found]
                disk_offload_module,
            )
        except ImportError:
            # The standalone safetensors check remains useful in a minimal probe
            # environment; the selected Stage 1A environment records this as true.
            pass
        else:
            module = t.nn.Linear(4, 3, device=device, dtype=t.float16)
            reload_handle = disk_offload_module(module)
            reload_handle(device=device)
            upstream_helper_tested = True
            if any(
                not _same_device(parameter, device) for parameter in module.parameters()
            ):
                raise RuntimeError(
                    "upstream disk offload helper restored the wrong device"
                )
        return {
            "round_trip_max_abs_error": error,
            "explicit_device": device,
            "upstream_disk_offload_helper_tested": upstream_helper_tested,
            "cpu_reference_passed": True,
        }

    return _record("disk_offload_safetensors", disk, torch, device=device)


def _torch_info() -> tuple[Any | None, dict[str, Any]]:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        return None, {"torch_importable": False, "torch_error": _safe_error(error)}
    backend = getattr(getattr(torch, "backends", None), "mps", None)
    built = bool(backend is not None and backend.is_built())
    available = bool(backend.is_available()) if backend is not None and built else False
    return torch, {
        "torch_importable": True,
        "torch_version": str(torch.__version__),
        "mps_built": built,
        "mps_available": available,
    }


def run_preflight() -> dict[str, Any]:
    """Run all bounded checks and return a JSON-serializable summary."""

    started = time.perf_counter()
    torch, info = _torch_info()
    architecture = platform.machine().lower()
    system = platform.system()
    python_ok = sys.version_info[:2] == (3, 11)
    arm64_ok = architecture in {"arm64", "aarch64"}
    torch_version = str(info.get("torch_version", "")).split("+", 1)[0]
    torch_2_6_ok = torch_version == "2.6.0"
    fallback = fallback_enabled()
    high_watermark = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
    guardrail_ok = high_watermark != "0.0"
    checks: dict[str, bool] = {
        "python_3_11": python_ok,
        "darwin": system == "Darwin",
        "native_arm64": arm64_ok,
        "torch_2_6_0": torch_2_6_ok,
        "fallback_disabled": not fallback,
        "memory_guardrail_preserved": guardrail_ok,
        "torch_importable": bool(info.get("torch_importable")),
        "mps_built": bool(info.get("mps_built", False)),
        "mps_available": bool(info.get("mps_available", False)),
    }
    operations: dict[str, Any] = {}
    if torch is not None and checks["mps_available"] and not fallback:
        operations["operators"] = _operation_probe(torch, MPS_DEVICE)
        operations["transfers"] = _transfer_probe(torch, MPS_DEVICE)
        operations["autograd_hooks"] = _autograd_probe(torch, MPS_DEVICE)
        operations["strict_jumprelu"] = _jumprelu_probe(torch, MPS_DEVICE)
        operations["tiny_hooked_transformer"] = _tiny_model_probe(torch, MPS_DEVICE)
        operations["sparse_metadata_boundary"] = _sparse_boundary_probe(
            torch, MPS_DEVICE
        )
        operations["bounded_graph_construction"] = _graph_probe(torch, MPS_DEVICE)
        operations["disk_offload_safetensors"] = _disk_round_trip_probe(
            torch, MPS_DEVICE
        )
        checks.update(
            {name: bool(record.get("passed")) for name, record in operations.items()}
        )
    else:
        reason = (
            "MPS unavailable"
            if not checks["mps_available"]
            else "MPS fallback is enabled"
        )
        for name in (
            "operators",
            "transfers",
            "autograd_hooks",
            "strict_jumprelu",
            "tiny_hooked_transformer",
            "sparse_metadata_boundary",
            "bounded_graph_construction",
            "disk_offload_safetensors",
        ):
            operations[name] = {"attempted": False, "passed": False, "error": reason}
            checks[name] = False
    all_checks = all(checks.values())
    status = (
        "passed"
        if all_checks
        else ("blocked" if not checks["mps_available"] else "failed")
    )
    return {
        "schema_version": 1,
        "probe": "stage1a_mps_fp16",
        "status": status,
        "verdict": (
            "mps_preflight_passed" if status == "passed" else "mps_preflight_not_passed"
        ),
        "probe_status": status,
        "environment": {
            "python": platform.python_version(),
            "system": system,
            "architecture": architecture,
            "platform": platform.platform(),
            "fallback_env_value_present": "PYTORCH_ENABLE_MPS_FALLBACK" in os.environ,
            "fallback_enabled": fallback,
            "high_watermark_override": high_watermark,
            **info,
            "torch_version": torch_version,
        },
        "checks": checks,
        "operations": operations,
        "tolerances": {
            "absolute": MPS_ATOL,
            "relative": MPS_RTOL,
            "finite_required": True,
        },
        "memory_samples": [_memory_snapshot(torch)],
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "large_assets_downloaded": False,
        "scientific_model_result": False,
    }


def _safe_output(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    allowed = RESULT_DIRECTORY.resolve()
    if candidate.name != "preflight_summary.json":
        raise ValueError("output filename must be preflight_summary.json")
    canonical = candidate.parent == allowed
    generated_root = GENERATED_RESULT_DIRECTORY.resolve()
    generated_relative = (
        candidate.relative_to(generated_root)
        if candidate.is_relative_to(generated_root)
        else None
    )
    internal_staging = (
        generated_relative is not None
        and len(generated_relative.parts) == 2
        and generated_relative.parts[0].startswith("preflight-")
        and len(generated_relative.parts[0]) > len("preflight-")
    )
    if not canonical and not internal_staging:
        raise ValueError(
            "output must remain in the canonical preflight directory or its "
            "isolated generated staging path"
        )
    return candidate


def write_summary(summary: dict[str, Any], output: Path) -> None:
    """Write a small JSON summary atomically within the allowlisted directory."""

    destination = _safe_output(output)
    encoded = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False).encode()
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise ValueError("preflight summary exceeds its bounded size")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--check",
        action="store_true",
        help="return a status code without writing output",
    )
    parser.add_argument(
        "--no-output", action="store_true", help="do not write a summary"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_preflight()
    if args.output is not None and not args.no_output and not args.check:
        write_summary(summary, args.output)
    print(f"Stage 1A MPS preflight: {summary['status']}")
    if summary["status"] == "passed":
        return 0
    return 2 if summary["status"] == "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
