#!/usr/bin/env python3
"""Evaluate deterministic counterfactual-susceptibility examples.

The script exercises only the public, backend-independent package API.  It does
not import a model backend, access the network, or require model weights.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "results" / "generated" / "verify_math.json"
EPSILON = 1e-12
TOLERANCE = 1e-9

# Make a source checkout directly runnable while still importing the public API.
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cfsus.susceptibility.pairwise import (  # noqa: E402
    activation_margin,
    classify_predicted_crossing,
    critical_suppression_fraction,
    pairwise_susceptibility,
    predicted_target_preactivation,
    suppression_response,
)


def _status_value(status: object) -> str:
    """Return a stable JSON value for the public status enum."""

    value = getattr(status, "value", status)
    return str(value)


def _require_close(actual: float | None, expected: float, label: str) -> None:
    if actual is None or not math.isclose(
        actual, expected, rel_tol=TOLERANCE, abs_tol=TOLERANCE
    ):
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


def _evaluate_example(
    *,
    name: str,
    preactivation: float,
    threshold: float,
    source_activation: float,
    local_response: float,
    alpha: float,
) -> dict[str, Any]:
    margin = activation_margin(preactivation, threshold, tolerance=TOLERANCE)
    q = suppression_response(source_activation, local_response)
    susceptibility = pairwise_susceptibility(
        q, margin, epsilon=EPSILON, tolerance=TOLERANCE
    )
    critical_alpha = critical_suppression_fraction(margin, q, tolerance=TOLERANCE)
    status = classify_predicted_crossing(margin, q, tolerance=TOLERANCE)
    predicted_preactivation = predicted_target_preactivation(preactivation, alpha, q)
    return {
        "name": name,
        "inputs": {
            "alpha": alpha,
            "local_response": local_response,
            "preactivation": preactivation,
            "source_activation": source_activation,
            "threshold": threshold,
        },
        "outputs": {
            "critical_suppression_fraction": critical_alpha,
            "margin": margin,
            "predicted_target_preactivation": predicted_preactivation,
            "q": q,
            "status": _status_value(status),
            "susceptibility": susceptibility,
        },
    }


def build_artifact() -> dict[str, Any]:
    """Compute examples and enforce independent scientific invariants."""

    examples = [
        _evaluate_example(
            name="inhibitory_source_crosses",
            preactivation=0.2,
            threshold=0.5,
            source_activation=2.0,
            local_response=-0.25,
            alpha=0.6,
        ),
        _evaluate_example(
            name="inhibitory_source_insufficient",
            preactivation=0.2,
            threshold=0.5,
            source_activation=1.0,
            local_response=-0.1,
            alpha=1.0,
        ),
        _evaluate_example(
            name="excitatory_source_moves_away",
            preactivation=0.2,
            threshold=0.5,
            source_activation=2.0,
            local_response=0.25,
            alpha=1.0,
        ),
        _evaluate_example(
            name="full_suppression_boundary",
            preactivation=0.2,
            threshold=0.5,
            source_activation=1.0,
            local_response=-0.3,
            alpha=1.0,
        ),
    ]
    outputs = {example["name"]: example["outputs"] for example in examples}

    crossing = outputs["inhibitory_source_crosses"]
    _require_close(crossing["margin"], 0.3, "crossing margin")
    _require_close(crossing["q"], 0.5, "crossing q")
    _require_close(crossing["critical_suppression_fraction"], 0.6, "crossing alpha*")
    _require_close(crossing["predicted_target_preactivation"], 0.5, "crossing z(alpha)")

    insufficient = outputs["inhibitory_source_insufficient"]
    _require_close(
        insufficient["critical_suppression_fraction"], 3.0, "insufficient alpha*"
    )
    _require_close(
        insufficient["predicted_target_preactivation"],
        0.3,
        "insufficient z(alpha)",
    )

    excitatory = outputs["excitatory_source_moves_away"]
    _require_close(excitatory["q"], -0.5, "excitatory q")
    if excitatory["critical_suppression_fraction"] is not None:
        raise RuntimeError("q <= 0 must not produce a critical suppression fraction")
    _require_close(
        excitatory["predicted_target_preactivation"],
        -0.3,
        "excitatory z(alpha)",
    )

    boundary = outputs["full_suppression_boundary"]
    _require_close(boundary["critical_suppression_fraction"], 1.0, "boundary alpha*")
    _require_close(boundary["predicted_target_preactivation"], 0.5, "boundary z(1)")

    if not (
        crossing["susceptibility"] > 1.0
        and crossing["critical_suppression_fraction"] < 1.0
    ):
        raise RuntimeError("away from tolerance, S > 1 must agree with alpha* < 1")

    return {
        "schema_version": 1,
        "generated_by": "scripts/verify_math.py",
        "deterministic": True,
        "epsilon": EPSILON,
        "tolerance": TOLERANCE,
        "units": (
            "preactivation, threshold, margin, q, and local response use compatible "
            "feature-coordinate units; alpha and susceptibility are dimensionless"
        ),
        "examples": examples,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON artifact path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = args.output.expanduser()
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    artifact = build_artifact()
    rendered = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(f"Wrote deterministic math verification: {output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
