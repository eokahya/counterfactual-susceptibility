#!/usr/bin/env python3
"""Verify pinned ``circuit-tracer`` semantics on the actually loaded assets."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from reproduce_attribution import (  # noqa: E402
    RuntimeBundle,
    Stage1ABlocked,
    _mapping,
    _peak_rss_bytes,
    _safe_output,
    _seed_runtime,
    load_runtime,
    load_yaml,
    repository_root,
)
from reproduce_intervention import (  # noqa: E402
    OFFICIAL_FEATURE,
    OFFICIAL_PROMPT,
    _intervention_config,
    _next_token_vector,
)


def _sample_record(
    *,
    layer: int,
    position: int,
    feature_id: int,
    preactivation: Any,
    threshold: Any,
    activation: Any,
) -> dict[str, Any]:
    z = float(preactivation[layer, position, feature_id].item())
    tau = float(threshold[layer, feature_id].item())
    a = float(activation[layer, position, feature_id].item())
    return {
        "layer": layer,
        "position": position,
        "feature_id": feature_id,
        "preactivation": z,
        "threshold": tau,
        "post_gate_activation": a,
        "active": z > tau,
        "signed_margin": z - tau,
    }


def _capture_manual_preactivations(bundle: RuntimeBundle, prompt: str) -> Any:
    """Independently project cached feature inputs with ``W_enc`` and ``b_enc``."""

    torch = bundle.torch
    model = bundle.model

    def names_filter(name: str) -> bool:
        return name.endswith(model.feature_input_hook)

    with torch.inference_mode():
        _, cache = model.run_with_cache(prompt, names_filter=names_filter)

    projected = []
    for layer in range(model.cfg.n_layers):
        name = f"blocks.{layer}.{model.feature_input_hook}"
        if name not in cache:
            raise Stage1ABlocked(f"feature-input hook was not captured: {name}")
        transcoder = model.transcoders[layer]
        inputs = cache[name]
        manual = torch.nn.functional.linear(
            inputs.to(transcoder.W_enc.dtype),
            transcoder.W_enc,
            transcoder.b_enc,
        ).squeeze(0)
        manual[model.zero_positions] = 0
        projected.append(manual)
    return torch.stack(projected)


def verify_runtime_semantics(
    bundle: RuntimeBundle,
    *,
    summary_output: str | None = None,
) -> dict[str, Any]:
    """Verify gates, raw thresholds/biases, inactive visibility, and no-op edits."""

    from cfsus.reproduction.artifacts import (
        make_artifact_envelope,
        write_json_atomic,
    )
    from cfsus.reproduction.runtime_helpers import (
        desired_activation,
        select_gate_samples,
    )

    config = bundle.config
    _intervention_config(config)
    artifacts = _mapping(config.get("artifacts"), "artifacts")
    output = _safe_output(
        summary_output or str(artifacts["semantics_summary"]), generated=False
    )
    torch = bundle.torch
    model = bundle.model
    seed = _seed_runtime(config, torch)

    started = time.perf_counter()
    with torch.inference_mode():
        _, preactivation = model.get_activations(
            OFFICIAL_PROMPT,
            sparse=False,
            apply_activation_function=False,
        )
        _, activation = model.get_activations(
            OFFICIAL_PROMPT,
            sparse=False,
            apply_activation_function=True,
        )

    if preactivation.shape != activation.shape or preactivation.ndim != 3:
        raise Stage1ABlocked(
            "activation and preactivation caches must align as "
            "[layer, position, feature]"
        )
    if not bool(torch.isfinite(preactivation).all().item()):
        raise Stage1ABlocked("preactivation cache contains non-finite values")
    if not bool(torch.isfinite(activation).all().item()):
        raise Stage1ABlocked("activation cache contains non-finite values")

    activation_names = {
        type(model.transcoders[layer].activation_function).__name__
        for layer in range(model.cfg.n_layers)
    }
    if activation_names != {"JumpReLU"}:
        raise Stage1ABlocked(
            f"expected JumpReLU at every layer, observed {sorted(activation_names)}"
        )
    thresholds = torch.stack(
        [
            model.transcoders[layer].activation_function.threshold.detach()
            for layer in range(model.cfg.n_layers)
        ]
    )
    if thresholds.shape != (preactivation.shape[0], preactivation.shape[2]):
        raise Stage1ABlocked("raw threshold tensors do not align with activation cache")
    if not bool(torch.isfinite(thresholds).all().item()):
        raise Stage1ABlocked("raw thresholds contain non-finite values")

    expected_layer_count = 26
    expected_d_model = 2304
    expected_d_transcoder = 16384
    if model.cfg.n_layers != expected_layer_count:
        raise Stage1ABlocked("loaded model/transcoder layer count is not 26")
    for layer_index in range(expected_layer_count):
        transcoder = model.transcoders[layer_index]
        expected_shapes = {
            "W_enc": (expected_d_transcoder, expected_d_model),
            "W_dec": (expected_d_transcoder, expected_d_model),
            "b_enc": (expected_d_transcoder,),
            "b_dec": (expected_d_model,),
            "threshold": (expected_d_transcoder,),
        }
        observed_tensors = {
            "W_enc": transcoder.W_enc,
            "W_dec": transcoder.W_dec,
            "b_enc": transcoder.b_enc,
            "b_dec": transcoder.b_dec,
            "threshold": transcoder.activation_function.threshold,
        }
        for tensor_name, expected_shape in expected_shapes.items():
            tensor = observed_tensors[tensor_name]
            if tuple(tensor.shape) != expected_shape:
                raise Stage1ABlocked(
                    f"layer {layer_index} {tensor_name} has an incompatible shape"
                )
            if tensor.dtype != torch.bfloat16:
                raise Stage1ABlocked(
                    f"layer {layer_index} {tensor_name} is not bfloat16"
                )

    expected = torch.where(
        preactivation > thresholds[:, None, :],
        preactivation,
        torch.zeros_like(preactivation),
    )
    gate_max_error = float(torch.max(torch.abs(expected - activation)).item())
    numerics = _mapping(config.get("numerics", {}), "numerics")
    gate_atol = float(numerics.get("gate_absolute_tolerance", 0.0))
    if gate_max_error > gate_atol:
        raise Stage1ABlocked(
            "loaded post-gate activations disagree with strict JumpReLU by "
            f"{gate_max_error:.6g}"
        )

    equality_max = 0.0
    for layer in range(model.cfg.n_layers):
        equality_output = model.transcoders[layer].activation_function(
            thresholds[layer]
        )
        equality_max = max(
            equality_max, float(torch.max(torch.abs(equality_output)).item())
        )
    if equality_max != 0.0:
        raise Stage1ABlocked(
            "loaded JumpReLU treats exact threshold equality as active"
        )

    manual_preactivation = _capture_manual_preactivations(bundle, OFFICIAL_PROMPT)
    projection_error = float(
        torch.max(torch.abs(manual_preactivation - preactivation)).item()
    )
    projection_atol = float(numerics.get("projection_absolute_tolerance", 0.0))
    if projection_error > projection_atol:
        raise Stage1ABlocked(
            "F.linear(input, W_enc, b_enc) disagrees with public preactivation "
            f"by {projection_error:.6g}"
        )

    layer, requested_position, official_feature_id = OFFICIAL_FEATURE
    position = int(preactivation.shape[1] - 1)
    layer_z = preactivation[layer, position].detach().float().cpu().tolist()
    layer_tau = thresholds[layer].detach().float().cpu().tolist()
    selection = select_gate_samples(layer_z, layer_tau)
    sample_ids = {
        "active": selection.active_feature_id,
        "inactive": selection.inactive_feature_id,
        "closest_margin": selection.closest_margin_feature_id,
        "official_intervention_source": official_feature_id,
    }
    samples = {
        label: _sample_record(
            layer=layer,
            position=position,
            feature_id=feature_id,
            preactivation=preactivation,
            threshold=thresholds,
            activation=activation,
        )
        for label, feature_id in sample_ids.items()
    }
    if samples["inactive"]["post_gate_activation"] != 0.0:
        raise Stage1ABlocked(
            "selected baseline-inactive feature has nonzero activation"
        )
    baseline_activation = float(
        activation[layer, requested_position, official_feature_id].item()
    )
    if not baseline_activation > 0.0:
        raise Stage1ABlocked(
            "official intervention source is inactive under pinned assets"
        )

    desired_noop = desired_activation(baseline_activation, 0.0)
    with torch.inference_mode():
        baseline_raw, _ = model.feature_intervention(
            OFFICIAL_PROMPT, [], return_activations=False
        )
        noop_raw, _ = model.feature_intervention(
            OFFICIAL_PROMPT,
            [(layer, requested_position, official_feature_id, desired_noop)],
            freeze_attention=True,
            constrained_layers=None,
            apply_activation_function=False,
            return_activations=False,
        )
        noop_repeat_raw, _ = model.feature_intervention(
            OFFICIAL_PROMPT,
            [(layer, requested_position, official_feature_id, desired_noop)],
            freeze_attention=True,
            constrained_layers=None,
            apply_activation_function=False,
            return_activations=False,
        )
    baseline_logits = _next_token_vector(baseline_raw)
    noop_logits = _next_token_vector(noop_raw)
    noop_repeat_logits = _next_token_vector(noop_repeat_raw)
    if not all(
        bool(torch.isfinite(values).all().item())
        for values in (baseline_logits, noop_logits, noop_repeat_logits)
    ):
        raise Stage1ABlocked("no-op semantic check produced non-finite logits")
    logit_atol = float(numerics.get("noop_absolute_tolerance", 2.0e-2))
    logit_rtol = float(numerics.get("noop_relative_tolerance", 2.0e-3))
    baseline_noop_error = float(
        torch.max(torch.abs(noop_logits - baseline_logits)).item()
    )
    repeat_error = float(torch.max(torch.abs(noop_repeat_logits - noop_logits)).item())
    if not bool(
        torch.allclose(noop_logits, baseline_logits, atol=logit_atol, rtol=logit_rtol)
    ):
        raise Stage1ABlocked("absolute-value no-op does not preserve baseline logits")
    if not bool(
        torch.allclose(
            noop_repeat_logits, noop_logits, atol=logit_atol, rtol=logit_rtol
        )
    ):
        raise Stage1ABlocked("absolute-value no-op was not deterministic")

    token_ids = [
        int(value)
        for value in model.ensure_tokenized(OFFICIAL_PROMPT).detach().cpu().tolist()
    ]
    first_transcoder = model.transcoders[0]
    payload = {
        "prompt": OFFICIAL_PROMPT,
        "token_ids": token_ids,
        "cache_shape": [int(value) for value in preactivation.shape],
        "cache_index_order": ["layer", "token_position", "feature_id"],
        "cache_flags": {
            "preactivation_apply_activation_function": False,
            "post_gate_apply_activation_function": True,
            "sparse": False,
        },
        "parameters": {
            "verified_all_layers": True,
            "layer_count": expected_layer_count,
            "d_model": expected_d_model,
            "d_transcoder": expected_d_transcoder,
            "W_enc_shape": [int(value) for value in first_transcoder.W_enc.shape],
            "W_dec_shape": [
                int(first_transcoder.d_transcoder),
                int(first_transcoder.d_model),
            ],
            "b_enc_shape": [int(value) for value in first_transcoder.b_enc.shape],
            "b_dec_shape": [int(value) for value in first_transcoder.b_dec.shape],
            "threshold_shape": [int(value) for value in thresholds.shape],
            "threshold_dtype": str(thresholds.dtype),
            "activation_function": "JumpReLU",
        },
        "preactivation_equation": {
            "formula": "F.linear(feature_input, W_enc, b_enc)",
            "b_enc_included": True,
            "b_dec_included": False,
            "maximum_absolute_projection_error": projection_error,
            "absolute_tolerance": projection_atol,
        },
        "gate_check": {
            "rule": "activation = preactivation if preactivation > threshold else 0",
            "strict_greater_than": True,
            "equality_inactive": True,
            "equality_probe_maximum_absolute_output": equality_max,
            "maximum_absolute_cache_discrepancy": gate_max_error,
            "absolute_tolerance": gate_atol,
            "samples": samples,
        },
        "intervention_value_check": {
            "upstream_argument": "absolute_desired_post_gate_activation",
            "project_mapping": "desired = (1 - alpha) * baseline_activation",
            "official_feature_baseline_activation": baseline_activation,
            "alpha": 0.0,
            "desired_noop_activation": desired_noop,
            "apply_activation_function_for_returned_cache": False,
            "delta_logic_still_uses_post_gate_activation": True,
            "baseline_noop_maximum_absolute_logit_error": baseline_noop_error,
            "noop_repeat_maximum_absolute_logit_error": repeat_error,
            "absolute_tolerance": logit_atol,
            "relative_tolerance": logit_rtol,
        },
        "timing": {
            "wall_seconds": time.perf_counter() - started,
            "process_peak_rss_bytes": _peak_rss_bytes(),
            "cuda_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else None
            ),
        },
        "seed": seed,
        "claim_boundary": (
            "Observability and API-semantics checks only; the inactive reference "
            "is not a susceptibility candidate or ranked scan result."
        ),
    }
    run_id = "stage1a-runtime-semantics"
    envelope = make_artifact_envelope(
        artifact_type="semantics_summary",
        run_id=run_id,
        status="completed",
        provenance=bundle.provenance,
        payload=payload,
    )
    write_json_atomic(output, envelope)
    return envelope


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=repository_root()
        / "configs/stage1a_gemma2_2b_official_reproduction.yaml",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--transcoder-snapshot", type=Path)
    parser.add_argument("--summary-output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_yaml(args.config.resolve())
        bundle = load_runtime(
            config,
            allow_download=args.allow_download,
            model_snapshot=args.model_snapshot,
            transcoder_snapshot=args.transcoder_snapshot,
        )
        verify_runtime_semantics(bundle, summary_output=args.summary_output)
    except Stage1ABlocked as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print("Stage 1A runtime semantics verification completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
