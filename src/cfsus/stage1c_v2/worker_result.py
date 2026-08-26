"""Detached worker-result construction for the Stage 1C-v2 protocol.

The intervention worker owns mutable sweep lists while a model runtime is
alive.  This module is deliberately independent of that runtime: it creates
two recursively detached, value-equal sweep representations before callers
enter their cleanup/finally block.  The returned result contains no reference
to the worker's mutable containers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from cfsus.stage1c_v2.serialization import (
    SerializationError,
    detach_json,
    detached_sweep_copies,
)


def _point_count(sweep: Mapping[str, Any]) -> int:
    """Return one sweep's serialized point count, checking its consistency."""

    points = sweep.get("points")
    if not isinstance(points, list):
        raise SerializationError("sweep points must be a JSON array")
    count = sweep.get("point_count", len(points))
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise SerializationError("sweep point_count must be a non-negative integer")
    if count != len(points):
        raise SerializationError("sweep point_count differs from serialized points")
    return int(count)


def _replace_intervention_pairs(
    artifacts: Mapping[str, Any] | None,
    detached_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Copy an artifact map and install its independent intervention pairs."""

    copied: dict[str, Any]
    if artifacts is None:
        copied = {}
    else:
        candidate = detach_json(dict(artifacts))
        if not isinstance(candidate, dict):  # pragma: no cover - defensive
            raise SerializationError("intervention artifacts must be an object")
        copied = cast(dict[str, Any], candidate)

    intervention = copied.get("intervention_sweeps")
    if intervention is None:
        intervention = {}
    if not isinstance(intervention, dict):
        raise SerializationError("intervention_sweeps must be an object")
    intervention = cast(dict[str, Any], intervention)
    # ``detached_pairs`` is already independent from the top-level copy.  Do
    # not route it through the caller's original map.
    intervention["pairs"] = detached_pairs
    copied["intervention_sweeps"] = intervention
    return copied


def build_detached_worker_result(
    working_sweeps: Sequence[Mapping[str, Any]],
    *,
    intervention_artifacts: Mapping[str, Any] | None = None,
    canonical_source_suppression_api_calls: int | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Construct a worker result detached from ``working_sweeps``.

    ``working_sweeps`` is intentionally accepted as a sequence rather than a
    concrete list so tests can pass mutable list subclasses and verify that no
    alias crosses this boundary.  The returned top-level ``sweeps`` and
    ``intervention_artifacts.intervention_sweeps.pairs`` are value-equal but
    recursively disjoint containers.
    """

    top_level, artifact_pairs = detached_sweep_copies(working_sweeps)
    total_points = sum(_point_count(item) for item in top_level)
    calls = (
        total_points
        if canonical_source_suppression_api_calls is None
        else canonical_source_suppression_api_calls
    )
    if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
        raise SerializationError("API-call count must be a non-negative integer")
    if calls != total_points:
        raise SerializationError("API-call count differs from serialized point count")

    result: dict[str, Any] = dict(fields)
    result["canonical_source_suppression_api_calls"] = calls
    result["sweeps"] = top_level
    result["intervention_artifacts"] = _replace_intervention_pairs(
        intervention_artifacts, artifact_pairs
    )

    # A final strict round trip detaches any nested values supplied through
    # ``fields`` and verifies that the complete worker result is publication
    # safe.  Repeated JSON fields remain independent because each occurrence
    # is encoded and decoded separately.
    detached = detach_json(result)
    if not isinstance(detached, dict):  # pragma: no cover - defensive
        raise SerializationError("worker result must be a JSON object")
    top = detached.get("sweeps")
    artifacts = detached.get("intervention_artifacts")
    if not isinstance(top, list) or not isinstance(artifacts, dict):
        raise SerializationError("worker result sweep representations are missing")
    intervention = artifacts.get("intervention_sweeps")
    if not isinstance(intervention, dict) or not isinstance(
        intervention.get("pairs"), list
    ):
        raise SerializationError("worker artifact sweep representation is missing")
    if top != intervention["pairs"] or top is intervention["pairs"]:
        raise SerializationError("worker sweep representations are not detached")
    return cast(dict[str, Any], detached)


def construct_then_cleanup(
    working_sweeps: list[dict[str, Any]],
    *,
    intervention_artifacts: Mapping[str, Any] | None = None,
    cleanup: Callable[[], None] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Build evidence, then run cleanup that may clear the working list.

    This small helper models the worker's lifetime boundary and is useful for
    deterministic regression tests.  Cleanup runs even when it raises; the
    cleanup exception is propagated only after the working list has been
    cleared.  The detached result is returned on the normal path and remains
    valid after the list is cleared.
    """

    result = build_detached_worker_result(
        working_sweeps,
        intervention_artifacts=intervention_artifacts,
        **fields,
    )
    cleanup_error: BaseException | None = None
    try:
        if cleanup is not None:
            cleanup()
    except BaseException as error:  # pragma: no cover - exercised by tests
        cleanup_error = error
    finally:
        working_sweeps.clear()
    if cleanup_error is not None:
        raise cleanup_error
    return result


__all__ = [
    "build_detached_worker_result",
    "construct_then_cleanup",
]
