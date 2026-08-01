#!/usr/bin/env python3
"""Verify the Stage 1A Python runtime and pinned upstream installation.

This verifier is deliberately local-only. It reads installed distribution
metadata and tracked environment records, but it never imports the model stack,
contacts a package index, or prints installation/cache locations.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_UPSTREAM_COMMIT = "8f1e2438df612464e229e44c4a00ff637bf9379b"
EXPECTED_UPSTREAM_URL = "https://github.com/decoderesearch/circuit-tracer.git"
EXPECTED_DISTRIBUTION = "circuit-tracer"
EXPECTED_VERSION = "0.5.2"
EXPECTED_REQUIREMENT = (
    f"{EXPECTED_DISTRIBUTION} @ git+{EXPECTED_UPSTREAM_URL}@{EXPECTED_UPSTREAM_COMMIT}"
)
DEFAULT_ENVIRONMENT_SCHEMA = (
    REPOSITORY_ROOT / "environments" / "stage1a" / "environment-schema.json"
)
DEFAULT_LOCK = (
    REPOSITORY_ROOT
    / "environments"
    / "stage1a"
    / "requirements-lock-macos-arm64-py311.txt"
)


class EnvironmentVerificationError(ValueError):
    """Raised when a Stage 1A environment record violates an invariant."""


def _as_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EnvironmentVerificationError(f"{label} must be a JSON object")
    return value


def validate_direct_url(
    value: object,
    *,
    expected_url: str = EXPECTED_UPSTREAM_URL,
    expected_commit: str = EXPECTED_UPSTREAM_COMMIT,
) -> None:
    """Require a PEP 610 record for the exact audited Git commit."""

    record = _as_mapping(value, label="direct_url.json")
    if set(record) != {"url", "vcs_info"}:
        raise EnvironmentVerificationError(
            "direct_url.json must contain only url and vcs_info"
        )
    if record.get("url") != expected_url:
        raise EnvironmentVerificationError(
            "direct_url.json url is not the canonical audited upstream URL"
        )

    vcs_info = _as_mapping(record.get("vcs_info"), label="vcs_info")
    if set(vcs_info) != {"vcs", "commit_id", "requested_revision"}:
        raise EnvironmentVerificationError(
            "vcs_info must contain only vcs, commit_id, and requested_revision"
        )
    if vcs_info.get("vcs") != "git":
        raise EnvironmentVerificationError("direct_url.json vcs must be 'git'")
    for field in ("commit_id", "requested_revision"):
        if vcs_info.get(field) != expected_commit:
            raise EnvironmentVerificationError(
                f"direct_url.json {field} must equal the audited 40-character commit"
            )


def validate_environment_schema(value: object) -> None:
    """Validate stable identity fields in the observed environment record."""

    record = _as_mapping(value, label="environment schema")
    if record.get("schema_version") != 1:
        raise EnvironmentVerificationError("environment schema_version must be 1")
    if record.get("stage") != "stage1a":
        raise EnvironmentVerificationError("environment stage must be 'stage1a'")
    if record.get("observation_scope") != "macos-arm64-py311":
        raise EnvironmentVerificationError(
            "environment observation_scope must be 'macos-arm64-py311'"
        )

    lock = _as_mapping(record.get("lock"), label="environment lock")
    if lock.get("format") != "pip-freeze":
        raise EnvironmentVerificationError("environment lock format must be pip-freeze")
    if lock.get("provenance") != "observed":
        raise EnvironmentVerificationError(
            "the macOS Stage 1A lock must have observed provenance"
        )


def validate_lock_text(text: str) -> None:
    """Require exactly one immutable circuit-tracer VCS requirement."""

    requirements = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    upstream_lines = [
        line
        for line in requirements
        if line.split(" @ ", 1)[0] == EXPECTED_DISTRIBUTION
    ]
    if upstream_lines != [EXPECTED_REQUIREMENT]:
        raise EnvironmentVerificationError(
            "lock must contain exactly the canonical circuit-tracer commit requirement"
        )


def _requirement_lines(text: str) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def installed_freeze_lines() -> tuple[str, ...]:
    """Return the active interpreter's exact path-free pip freeze inventory."""

    try:
        completed = subprocess.run(
            (sys.executable, "-m", "pip", "freeze", "--all"),
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise EnvironmentVerificationError(
            "could not inspect installed packages"
        ) from exc
    lines = _requirement_lines(completed.stdout)
    if any("file://" in line or line.startswith("-e ") for line in lines):
        raise EnvironmentVerificationError(
            "installed package inventory contains a local or editable requirement"
        )
    return lines


def read_installed_direct_url(
    distribution_name: str = EXPECTED_DISTRIBUTION,
) -> tuple[str, object]:
    """Return installed version and parsed PEP 610 metadata without importing it."""

    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise EnvironmentVerificationError(
            f"required distribution {distribution_name!r} is not installed"
        ) from exc

    raw = distribution.read_text("direct_url.json")
    if raw is None:
        raise EnvironmentVerificationError(
            f"installed {distribution_name!r} has no direct_url.json"
        )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EnvironmentVerificationError(
            "installed direct_url.json is not valid JSON"
        ) from exc
    return distribution.version, value


def verify_environment(
    *,
    schema_path: Path | None = DEFAULT_ENVIRONMENT_SCHEMA,
    lock_path: Path | None = DEFAULT_LOCK,
) -> dict[str, Any]:
    """Verify the active interpreter, installed upstream, and tracked records."""

    errors: list[str] = []
    checks: dict[str, bool] = {}

    checks["python_3_11"] = sys.version_info[:2] == (3, 11)
    if not checks["python_3_11"]:
        errors.append("active interpreter must be Python 3.11")

    try:
        version, direct_url = read_installed_direct_url()
        checks["circuit_tracer_version"] = version == EXPECTED_VERSION
        if not checks["circuit_tracer_version"]:
            errors.append(
                f"installed circuit-tracer version must be {EXPECTED_VERSION}"
            )
        validate_direct_url(direct_url)
        checks["circuit_tracer_direct_url"] = True
    except EnvironmentVerificationError as exc:
        checks["circuit_tracer_direct_url"] = False
        errors.append(str(exc))

    checks["circuit_tracer_importable"] = (
        importlib.util.find_spec("circuit_tracer") is not None
    )
    if not checks["circuit_tracer_importable"]:
        errors.append("circuit_tracer module is not importable")

    if schema_path is not None:
        try:
            schema_value = json.loads(schema_path.read_text(encoding="utf-8"))
            validate_environment_schema(schema_value)
            checks["environment_schema"] = True
        except (OSError, json.JSONDecodeError, EnvironmentVerificationError) as exc:
            checks["environment_schema"] = False
            errors.append(f"environment schema invalid: {exc}")

    if lock_path is not None:
        try:
            lock_text = lock_path.read_text(encoding="utf-8")
            validate_lock_text(lock_text)
            checks["environment_lock"] = True
            checks["environment_lock_matches_installed"] = (
                _requirement_lines(lock_text) == installed_freeze_lines()
            )
            if not checks["environment_lock_matches_installed"]:
                errors.append(
                    "installed package inventory does not exactly match the lock"
                )
        except (OSError, EnvironmentVerificationError) as exc:
            checks["environment_lock"] = False
            checks["environment_lock_matches_installed"] = False
            errors.append(f"environment lock invalid: {exc}")

    return {
        "schema_version": 1,
        "valid": not errors,
        "checks": checks,
        "errors": errors,
        "expected": {
            "distribution": EXPECTED_DISTRIBUTION,
            "version": EXPECTED_VERSION,
            "upstream_commit": EXPECTED_UPSTREAM_COMMIT,
            "upstream_url": EXPECTED_UPSTREAM_URL,
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment-schema",
        type=Path,
        default=DEFAULT_ENVIRONMENT_SCHEMA,
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = verify_environment(
        schema_path=args.environment_schema,
        lock_path=args.lock,
    )
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
