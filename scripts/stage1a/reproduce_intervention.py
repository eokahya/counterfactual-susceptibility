#!/usr/bin/env python3
"""Reproduce the pinned official Gemma feature-intervention example."""

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

OFFICIAL_PROMPT = "Hecho: Michael Jordan juega al"
OFFICIAL_FEATURE = (20, -1, 341)
OFFICIAL_ALPHAS = (0.0, 0.5, 1.0)


def _intervention_config(config: dict[str, Any]) -> dict[str, Any]:
    section = _mapping(
        config.get("intervention", config.get("official_intervention")),
        "intervention",
    )
    prompt = section.get("prompt", section.get("prompt_text"))
    if isinstance(prompt, dict):
        prompt = prompt.get("text")
    if prompt != OFFICIAL_PROMPT:
        raise Stage1ABlocked(
            "intervention prompt differs from the pinned official demo"
        )

    feature = section.get("feature", section.get("source_feature"))
    if isinstance(feature, dict):
        observed = (
            feature.get("layer"),
            feature.get("position", feature.get("pos")),
            feature.get("feature_id", feature.get("feature_idx")),
        )
    elif isinstance(feature, (list, tuple)):
        observed = tuple(feature)
    else:
        observed = None
    if observed != OFFICIAL_FEATURE:
        raise Stage1ABlocked(
            f"intervention feature must be the official {OFFICIAL_FEATURE!r}"
        )

    configured_alphas = section.get(
        "alphas", section.get("alpha_values", OFFICIAL_ALPHAS)
    )
    if (
        not isinstance(configured_alphas, (list, tuple))
        or tuple(configured_alphas) != OFFICIAL_ALPHAS
    ):
        raise Stage1ABlocked("intervention alphas must be exactly 0.0, 0.5, and 1.0")
    freeze_attention = section.get("freeze_attention", True)
    constrained_layers = section.get("constrained_layers")
    if freeze_attention is not True or constrained_layers is not None:
        raise Stage1ABlocked(
            "official intervention regime requires freeze_attention=true and "
            "constrained_layers=null"
        )
    return section


def _next_token_vector(logits: Any) -> Any:
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise Stage1ABlocked(
            "expected intervention logits [1, position, vocab], got "
            f"{tuple(logits.shape)}"
        )
    return logits[0, -1].float()


def _condition_rows(
    *,
    token_ids: list[int],
    baseline_logits: Any,
    baseline_probabilities: Any,
    logits: Any,
    probabilities: Any,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    rows = []
    for token_id in token_ids:
        rows.append(
            {
                "token_id": token_id,
                "token": str(tokenizer.convert_ids_to_tokens(token_id)),
                "logit": float(logits[token_id].item()),
                "probability": float(probabilities[token_id].item()),
                "signed_logit_change_from_baseline": float(
                    (logits[token_id] - baseline_logits[token_id]).item()
                ),
                "signed_probability_change_from_baseline": float(
                    (probabilities[token_id] - baseline_probabilities[token_id]).item()
                ),
            }
        )
    return rows


def reproduce_intervention(
    bundle: RuntimeBundle,
    *,
    summary_output: str | None = None,
) -> dict[str, Any]:
    """Run baseline, no-op, half-suppression, and official zero ablation."""

    from cfsus.reproduction.runtime_helpers import (
        desired_activation,
        deterministic_top_k,
    )

    config = bundle.config
    is_t4 = config.get("reproduction_class") == "hardware_adapted_fp16"
    section = _intervention_config(config)
    torch = bundle.torch
    model = bundle.model
    artifacts = _mapping(config.get("artifacts"), "artifacts")
    summary_path = _safe_output(
        summary_output or str(artifacts["intervention_summary"]), generated=False
    )

    seed = _seed_runtime(config, torch)

    started = time.perf_counter()
    with torch.inference_mode():
        _, post_gate = model.get_activations(
            OFFICIAL_PROMPT,
            sparse=False,
            apply_activation_function=True,
        )
    if post_gate.ndim != 3:
        raise Stage1ABlocked(
            "expected activation cache [layer, position, feature], got "
            f"{tuple(post_gate.shape)}"
        )
    layer, requested_position, feature_id = OFFICIAL_FEATURE
    resolved_position = int(post_gate.shape[1] - 1)
    baseline_activation = float(post_gate[layer, requested_position, feature_id].item())
    if not baseline_activation > 0.0:
        raise Stage1ABlocked(
            "official feature (20, -1, 341) is inactive; investigate asset mismatch"
        )

    desired_values = {
        alpha: desired_activation(baseline_activation, alpha)
        for alpha in OFFICIAL_ALPHAS
    }
    for alpha, desired in desired_values.items():
        if desired != (1.0 - alpha) * baseline_activation:
            raise Stage1ABlocked("intervention desired-value mapping is inconsistent")
    with torch.inference_mode():
        baseline_raw, _ = model.feature_intervention(
            OFFICIAL_PROMPT, [], return_activations=False
        )
        baseline_repeat_raw, _ = model.feature_intervention(
            OFFICIAL_PROMPT, [], return_activations=False
        )
        condition_raw = {
            alpha: model.feature_intervention(
                OFFICIAL_PROMPT,
                [(layer, requested_position, feature_id, desired)],
                freeze_attention=True,
                constrained_layers=None,
                apply_activation_function=True,
                sparse=False,
                return_activations=False,
            )[0]
            for alpha, desired in desired_values.items()
        }
        noop_repeat_raw, _ = model.feature_intervention(
            OFFICIAL_PROMPT,
            [(layer, requested_position, feature_id, desired_values[0.0])],
            freeze_attention=True,
            constrained_layers=None,
            return_activations=False,
        )
        ablation_repeat_raw, _ = model.feature_intervention(
            OFFICIAL_PROMPT,
            [(layer, requested_position, feature_id, desired_values[1.0])],
            freeze_attention=True,
            constrained_layers=None,
            return_activations=False,
        )

    baseline = _next_token_vector(baseline_raw)
    baseline_repeat = _next_token_vector(baseline_repeat_raw)
    conditions = {
        alpha: _next_token_vector(value) for alpha, value in condition_raw.items()
    }
    noop_repeat = _next_token_vector(noop_repeat_raw)
    ablation_repeat = _next_token_vector(ablation_repeat_raw)
    all_vectors = [
        baseline,
        baseline_repeat,
        *conditions.values(),
        noop_repeat,
        ablation_repeat,
    ]
    if not all(bool(torch.isfinite(vector).all().item()) for vector in all_vectors):
        raise Stage1ABlocked("intervention produced non-finite next-token logits")

    numerics = _mapping(config.get("numerics", {}), "numerics")
    atol = float(numerics.get("noop_absolute_tolerance", 2.0e-2))
    rtol = float(numerics.get("noop_relative_tolerance", 2.0e-3))
    determinism_atol = float(numerics.get("determinism_absolute_tolerance", atol))
    determinism_rtol = float(numerics.get("determinism_relative_tolerance", rtol))
    noop_max_error = float(torch.max(torch.abs(conditions[0.0] - baseline)).item())
    if not bool(torch.allclose(conditions[0.0], baseline, atol=atol, rtol=rtol)):
        raise Stage1ABlocked(
            f"baseline/no-op mismatch {noop_max_error:.6g} exceeds declared tolerance"
        )

    repeat_noop_error = float(
        torch.max(torch.abs(noop_repeat - conditions[0.0])).item()
    )
    repeat_ablation_error = float(
        torch.max(torch.abs(ablation_repeat - conditions[1.0])).item()
    )
    baseline_repeat_error = float(
        torch.max(torch.abs(baseline_repeat - baseline)).item()
    )
    if not bool(torch.allclose(noop_repeat, conditions[0.0], atol=atol, rtol=rtol)):
        raise Stage1ABlocked(
            "no-op intervention was not deterministic within tolerance"
        )
    if not bool(torch.allclose(ablation_repeat, conditions[1.0], atol=atol, rtol=rtol)):
        raise Stage1ABlocked("zero ablation was not deterministic within tolerance")
    if not bool(
        torch.allclose(
            baseline_repeat,
            baseline,
            atol=determinism_atol,
            rtol=determinism_rtol,
        )
    ):
        raise Stage1ABlocked("baseline forward was not deterministic within tolerance")

    top_k = int(section.get("top_k", 10))
    if top_k <= 0 or top_k > int(baseline.shape[0]):
        raise Stage1ABlocked("intervention top_k must be within the model vocabulary")
    selected: set[int] = set()
    for vector in [baseline, *conditions.values()]:
        selected.update(deterministic_top_k(vector.detach().cpu().tolist(), k=top_k))
    fixed_token_ids = sorted(selected)
    baseline_probabilities = torch.softmax(baseline, dim=-1)
    condition_probabilities = {
        alpha: torch.softmax(vector, dim=-1) for alpha, vector in conditions.items()
    }
    if not bool(torch.isfinite(baseline_probabilities).all().item()) or not all(
        bool(torch.isfinite(probabilities).all().item())
        for probabilities in condition_probabilities.values()
    ):
        raise Stage1ABlocked("intervention produced non-finite probabilities")
    condition_payload = {
        "baseline": _condition_rows(
            token_ids=fixed_token_ids,
            baseline_logits=baseline,
            baseline_probabilities=baseline_probabilities,
            logits=baseline,
            probabilities=baseline_probabilities,
            tokenizer=model.tokenizer,
        )
    }
    for alpha in OFFICIAL_ALPHAS:
        label = {0.0: "noop", 0.5: "half_suppression", 1.0: "full_ablation"}[alpha]
        condition_payload[label] = _condition_rows(
            token_ids=fixed_token_ids,
            baseline_logits=baseline,
            baseline_probabilities=baseline_probabilities,
            logits=conditions[alpha],
            probabilities=condition_probabilities[alpha],
            tokenizer=model.tokenizer,
        )

    token_ids = [
        int(value)
        for value in model.ensure_tokenized(OFFICIAL_PROMPT).detach().cpu().tolist()
    ]
    wall_seconds = time.perf_counter() - started
    payload = {
        "source_notebook": {
            "path": "demos/intervention_demo.ipynb",
            "code_cells_zero_based": [4, 8, 10, 12],
        },
        "prompt": OFFICIAL_PROMPT,
        "token_ids": token_ids,
        "feature": {
            "layer": layer,
            "requested_position": requested_position,
            "resolved_position": resolved_position,
            "feature_id": feature_id,
        },
        "baseline_activation": baseline_activation,
        "desired_values": [
            {
                "alpha": alpha,
                "desired_post_gate_activation": desired_values[alpha],
            }
            for alpha in OFFICIAL_ALPHAS
        ],
        "fixed_top_k_union_token_ids": fixed_token_ids,
        "conditions": condition_payload,
        "baseline_noop_comparison": {
            "maximum_absolute_logit_error": noop_max_error,
            "absolute_tolerance": atol,
            "relative_tolerance": rtol,
            "within_tolerance": True,
        },
        "determinism": {
            "baseline_repeat_maximum_absolute_logit_error": baseline_repeat_error,
            "noop_repeat_maximum_absolute_logit_error": repeat_noop_error,
            "ablation_repeat_maximum_absolute_logit_error": repeat_ablation_error,
            "absolute_tolerance": determinism_atol,
            "relative_tolerance": determinism_rtol,
            "within_tolerance": True,
        },
        "regime": {
            "model_setting": "underlying_language_model",
            "edit": "absolute_post_gate_decoder_coordinate",
            "freeze_attention": True,
            "constrained_layers": None,
            "apply_activation_function": True,
            "sparse_activation_cache": False,
            "return_activations": False,
        },
        "timing": {
            "wall_seconds": wall_seconds,
            "process_peak_rss_bytes": _peak_rss_bytes(),
            "cuda_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else None
            ),
        },
        "seed": seed,
        "nonfinite_count": 0,
        "claim_boundary": (
            "T4/FP16 hardware-adapted runtime/API reproduction using the pinned "
            "assets; native-BF16 reference reproduction remains pending."
            if is_t4
            else (
                "API and intervention-value reproduction only; behavioral changes "
                "do not establish semantic interpretation or Counterfactual "
                "Susceptibility."
            )
        ),
    }
    run_id = (
        "stage1a-t4-fp16-intervention" if is_t4 else "stage1a-official-intervention"
    )
    from cfsus.reproduction.artifacts import (
        make_artifact_envelope,
        write_json_atomic,
    )

    envelope = make_artifact_envelope(
        artifact_type="intervention_summary",
        run_id=run_id,
        status="completed",
        provenance=bundle.provenance,
        payload=payload,
    )
    write_json_atomic(summary_path, envelope)
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
        reproduce_intervention(bundle, summary_output=args.summary_output)
    except Stage1ABlocked as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print("Stage 1A official intervention reproduction completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
