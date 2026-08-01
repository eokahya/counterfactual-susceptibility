"""Deterministic, publication-safe Stage 1A artifact utilities.

The helpers in this module intentionally depend only on the Python standard
library.  Model-runtime objects must be reduced to ordinary JSON values before
they cross this boundary.  This keeps the lightweight package importable when
the optional empirical stack is absent and makes non-finite or sensitive data
fail closed before a committed artifact is written.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from cfsus.exceptions import ScientificInputError

SCHEMA_VERSION = 1
REDACTED = "[REDACTED]"

_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_ARTIFACT_TYPE_RE = re.compile(r"[a-z][a-z0-9_.-]{0,63}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ALLOWED_STATUSES = frozenset(
    {"blocked", "completed", "failed", "observed", "partial", "resolved"}
)
_REQUIRED_ENVELOPE_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "run_id",
        "status",
        "provenance",
        "payload",
        "warnings",
        "deviations",
    }
)

_SENSITIVE_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth_token",
        "authorization",
        "bearer_token",
        "cookie",
        "credentials",
        "github_token",
        "hf_token",
        "id_token",
        "password",
        "passwd",
        "private_token",
        "refresh_token",
        "secret",
    }
)
_SENSITIVE_KEY_SUFFIXES = (
    "_api_key",
    "_authorization",
    "_cookie",
    "_credentials",
    "_password",
    "_secret",
)
_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    re.compile(r"(?i)https?://[^/@\s:]+:[^/@\s]+@"),
)
_PRIVATE_PATH_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9_])(?:file://)?/Users/[^/\s\"']+(?:/[^\s\"']*)?"),
    re.compile(r"(?<![A-Za-z0-9_])(?:file://)?/home/[^/\s\"']+(?:/[^\s\"']*)?"),
    re.compile(
        r"(?i)(?<![A-Za-z0-9_])(?:file:///)?[A-Z]:\\Users\\[^\\\s\"']+"
        r"(?:\\[^\s\"']*)?"
    ),
    re.compile(r"(?i)(?:\$HOME|~)/(?:\.cache|Library/Caches)(?:/[^\s\"']*)?"),
)


class ArtifactValidationError(ScientificInputError):
    """Raised when an artifact is unsafe, ambiguous, or not schema-valid."""


def _path_label(path: tuple[str | int, ...]) -> str:
    label = "$"
    for component in path:
        if isinstance(component, int):
            label += f"[{component}]"
        else:
            label += f".{component}"
    return label


def _is_sensitive_key(key: str) -> bool:
    normalized = key.casefold().replace("-", "_")
    return normalized in _SENSITIVE_KEYS or normalized.endswith(_SENSITIVE_KEY_SUFFIXES)


def _unsafe_string_reason(value: str) -> str | None:
    for pattern in _SECRET_PATTERNS:
        if pattern.search(value):
            return "credential-like content"
    for pattern in _PRIVATE_PATH_PATTERNS:
        if pattern.search(value):
            return "a private home or cache path"
    return None


def validate_json_value(value: Any, *, _path: tuple[str | int, ...] = ()) -> None:
    """Validate that ``value`` is strict, finite JSON data.

    In particular, tuples, path objects, NumPy scalars, tensors, and mappings
    with non-string keys are rejected instead of being converted implicitly.
    This makes the serialization boundary explicit and reproducible.
    """

    label = _path_label(_path)
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArtifactValidationError(f"{label} must be finite, got {value!r}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_json_value(item, _path=(*_path, index))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ArtifactValidationError(
                    f"{label} has a non-string object key: {key!r}"
                )
            validate_json_value(item, _path=(*_path, key))
        return
    raise ArtifactValidationError(
        f"{label} has unsupported JSON type {type(value).__name__}"
    )


def redact_sensitive(value: Any) -> Any:
    """Return a JSON-compatible copy with secrets and private paths redacted.

    Sensitive mapping keys are retained with a constant marker so callers can
    diagnose that a field was present without preserving its value.  Strings
    containing credential patterns or machine-local home/cache paths are
    replaced in full, avoiding partial leakage through commands or error text.
    """

    validate_json_value(value)
    if isinstance(value, dict):
        return {
            key: REDACTED if _is_sensitive_key(key) else redact_sensitive(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, str) and _unsafe_string_reason(value) is not None:
        return REDACTED
    return value


def assert_publication_safe(value: Any, *, _path: tuple[str | int, ...] = ()) -> None:
    """Reject unredacted credentials and machine-local paths recursively."""

    validate_json_value(value, _path=_path)
    if isinstance(value, dict):
        for key, item in value.items():
            label = _path_label((*_path, key))
            if _is_sensitive_key(key) and item != REDACTED:
                raise ArtifactValidationError(
                    f"{label} is a sensitive field and must be removed or redacted"
                )
            assert_publication_safe(item, _path=(*_path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_publication_safe(item, _path=(*_path, index))
    elif isinstance(value, str):
        reason = _unsafe_string_reason(value)
        if reason is not None:
            raise ArtifactValidationError(
                f"{_path_label(_path)} contains {reason}; redact it before writing"
            )


def _require_non_empty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactValidationError(f"{name} must be a non-empty string")
    return value


def validate_artifact_envelope(value: Any, *, expected_type: str | None = None) -> None:
    """Validate the exact Stage 1A artifact envelope schema version 1."""

    validate_json_value(value)
    if not isinstance(value, dict):
        raise ArtifactValidationError("artifact envelope must be a JSON object")
    keys = frozenset(value)
    missing = sorted(_REQUIRED_ENVELOPE_KEYS - keys)
    unknown = sorted(keys - _REQUIRED_ENVELOPE_KEYS)
    if missing:
        raise ArtifactValidationError(
            f"artifact envelope is missing required keys: {', '.join(missing)}"
        )
    if unknown:
        raise ArtifactValidationError(
            f"artifact envelope has unknown keys: {', '.join(unknown)}"
        )

    schema_version = value["schema_version"]
    if isinstance(schema_version, bool) or schema_version != SCHEMA_VERSION:
        raise ArtifactValidationError(
            f"schema_version must be integer {SCHEMA_VERSION}"
        )
    artifact_type = _require_non_empty_string("artifact_type", value["artifact_type"])
    if _ARTIFACT_TYPE_RE.fullmatch(artifact_type) is None:
        raise ArtifactValidationError("artifact_type has invalid characters")
    if expected_type is not None and artifact_type != expected_type:
        raise ArtifactValidationError(
            f"expected artifact_type {expected_type!r}, got {artifact_type!r}"
        )
    run_id = _require_non_empty_string("run_id", value["run_id"])
    if _RUN_ID_RE.fullmatch(run_id) is None:
        raise ArtifactValidationError("run_id has invalid characters or length")
    status = _require_non_empty_string("status", value["status"])
    if status not in _ALLOWED_STATUSES:
        allowed = ", ".join(sorted(_ALLOWED_STATUSES))
        raise ArtifactValidationError(f"status must be one of: {allowed}")
    for name in ("provenance", "payload"):
        if not isinstance(value[name], dict):
            raise ArtifactValidationError(f"{name} must be a JSON object")
    for name in ("warnings", "deviations"):
        items = value[name]
        if not isinstance(items, list) or any(
            not isinstance(item, str) or not item.strip() for item in items
        ):
            raise ArtifactValidationError(f"{name} must be a list of non-empty strings")
    assert_publication_safe(value)


def make_artifact_envelope(
    *,
    artifact_type: str,
    run_id: str,
    status: str,
    provenance: Mapping[str, Any],
    payload: Mapping[str, Any],
    warnings: Sequence[str] = (),
    deviations: Sequence[str] = (),
) -> dict[str, Any]:
    """Construct and validate a Stage 1A schema-v1 artifact envelope."""

    envelope: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "run_id": run_id,
        "status": status,
        "provenance": dict(provenance),
        "payload": dict(payload),
        "warnings": list(warnings),
        "deviations": list(deviations),
    }
    validate_artifact_envelope(envelope, expected_type=artifact_type)
    return envelope


def deterministic_json_bytes(value: Any, *, publication_safe: bool = True) -> bytes:
    """Serialize strict JSON deterministically with a final newline."""

    validate_json_value(value)
    if publication_safe:
        assert_publication_safe(value)
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def deterministic_json_dumps(value: Any, *, publication_safe: bool = True) -> str:
    """Return the text form produced by :func:`deterministic_json_bytes`."""

    return deterministic_json_bytes(value, publication_safe=publication_safe).decode(
        "utf-8"
    )


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ArtifactValidationError(f"refusing to replace symlink: {path}")
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_json_atomic(
    path: str | Path, value: Any, *, publication_safe: bool = True
) -> str:
    """Atomically write deterministic JSON and return its SHA-256 digest."""

    content = deterministic_json_bytes(value, publication_safe=publication_safe)
    _atomic_write_bytes(Path(path), content)
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the streamed SHA-256 digest of one regular, non-symlink file."""

    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ArtifactValidationError("chunk_size must be a positive integer")
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ArtifactValidationError(
            f"checksum target must be a regular, non-symlink file: {candidate}"
        )
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_checksum_path(path: Path, root: Path) -> str:
    resolved_root = root.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ArtifactValidationError(
            f"checksum target escapes repository root: {path}"
        ) from error
    if path.is_symlink():
        raise ArtifactValidationError(f"checksum target must not be a symlink: {path}")
    relative_text = relative.as_posix()
    if not relative_text or "\n" in relative_text or "\r" in relative_text:
        raise ArtifactValidationError("checksum path is empty or contains a newline")
    assert_publication_safe(relative_text)
    return relative_text


def build_checksum_manifest(
    paths: Iterable[str | Path],
    *,
    root: str | Path,
    exclude: Iterable[str | Path] = (),
) -> str:
    """Build deterministic ``sha256sum`` text for repository-contained files."""

    root_path = Path(root)
    excluded: set[Path] = set()
    for item in exclude:
        excluded_path = Path(item)
        if not excluded_path.is_absolute():
            excluded_path = root_path / excluded_path
        excluded.add(excluded_path.resolve(strict=False))
    entries: dict[str, str] = {}
    for item in paths:
        path = Path(item)
        if not path.is_absolute():
            path = root_path / path
        if path.resolve(strict=False) in excluded:
            continue
        relative = _relative_checksum_path(path, root_path)
        if relative in entries:
            raise ArtifactValidationError(f"duplicate checksum path: {relative}")
        entries[relative] = sha256_file(path)
    return "".join(f"{entries[path]}  {path}\n" for path in sorted(entries))


def write_checksum_manifest_atomic(
    manifest_path: str | Path,
    paths: Iterable[str | Path],
    *,
    root: str | Path,
) -> str:
    """Atomically write a checksum manifest, always excluding the manifest itself."""

    root_path = Path(root)
    output = Path(manifest_path)
    if not output.is_absolute():
        output = root_path / output
    content = build_checksum_manifest(paths, root=root_path, exclude=(output,))
    encoded = content.encode("utf-8")
    _atomic_write_bytes(output, encoded)
    return hashlib.sha256(encoded).hexdigest()


def _parse_checksum_manifest(content: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line:
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise ArtifactValidationError(
                f"invalid checksum line {line_number}: expected two-space separator"
            ) from error
        if _SHA256_RE.fullmatch(digest) is None:
            raise ArtifactValidationError(
                f"invalid SHA-256 digest on checksum line {line_number}"
            )
        pure = PurePosixPath(relative)
        if (
            not relative
            or pure.is_absolute()
            or ".." in pure.parts
            or "\\" in relative
            or relative in seen
        ):
            raise ArtifactValidationError(
                f"unsafe or duplicate path on checksum line {line_number}"
            )
        assert_publication_safe(relative)
        seen.add(relative)
        entries.append((digest, relative))
    return entries


def verify_checksum_manifest(
    manifest_path: str | Path, *, root: str | Path
) -> tuple[str, ...]:
    """Verify every entry and return normalized paths in manifest order."""

    root_path = Path(root)
    manifest = Path(manifest_path)
    if not manifest.is_absolute():
        manifest = root_path / manifest
    if manifest.is_symlink():
        raise ArtifactValidationError(
            f"checksum manifest must not be a symlink: {manifest}"
        )
    entries = _parse_checksum_manifest(manifest.read_text(encoding="utf-8"))
    verified: list[str] = []
    for expected_digest, relative in entries:
        candidate = root_path / PurePosixPath(relative)
        normalized = _relative_checksum_path(candidate, root_path)
        if normalized != relative:
            raise ArtifactValidationError(
                f"checksum path is not normalized: {relative!r}"
            )
        actual_digest = sha256_file(candidate)
        if actual_digest != expected_digest:
            raise ArtifactValidationError(
                f"checksum mismatch for {relative}: expected {expected_digest}, "
                f"got {actual_digest}"
            )
        verified.append(relative)
    return tuple(verified)
