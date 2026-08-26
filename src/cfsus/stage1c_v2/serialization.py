"""Strict JSON detachment and durable new-file publication for Stage 1C-v2.

The v2 worker constructs evidence while model-runtime objects are still alive.
This module is the ownership boundary: only finite JSON primitives cross it,
and a serialize/parse round trip deliberately breaks every container alias.
Files are published by linking an fsynced temporary file into a previously
unused destination, so an existing destination can never be overwritten.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, NoReturn, TypeAlias, cast

DEFAULT_MAX_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_DEPTH = 64

JSONPrimitive: TypeAlias = bool | int | float | str | None
JSONValue: TypeAlias = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]


class SerializationError(ValueError):
    """Raised when a value or path cannot cross the publication boundary."""


_SENSITIVE_EXACT = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "bearer_token",
        "cookie",
        "credential",
        "credentials",
        "github_token",
        "hf_token",
        "id_token",
        "password",
        "passwd",
        "private_key",
        "private_path",
        "private_absolute_path",
        "refresh_token",
        "secret",
        "secret_key",
        "token",
    }
)
_SENSITIVE_SUFFIXES = (
    "_api_key",
    "_authorization",
    "_cookie",
    "_credential",
    "_password",
    "_private_key",
    "_secret",
    "_token",
)
_SECRET_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    re.compile(r"(?i)https?://[^/@\s:]+:[^/@\s]+@"),
)
_PRIVATE_PATH_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9_])(?:file://)?/Users/[^\s\"']+"),
    re.compile(r"(?<![A-Za-z0-9_])(?:file://)?/home/[^\s\"']+"),
    re.compile(r"(?<![A-Za-z0-9_])/(?:private/)?var/(?:folders|tmp)/[^\s\"']+"),
    re.compile(r"(?<![A-Za-z0-9_])(?:\$HOME|~)/(?:\.cache|Library/Caches)[^\s\"']*"),
    re.compile(r"(?i)(?<![A-Za-z0-9_])(?:file:///)?[A-Z]:\\Users\\[^\\\s\"']+"),
)


def _fail(message: str) -> NoReturn:
    raise SerializationError(message)


def _check_limit(value: int, limit: int, label: str) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        _fail(f"{label} must be a positive integer")
    if value > limit:
        _fail(f"{label} exceeds limit {limit}")


def _check_string(value: str, path: str) -> None:
    if any(
        ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F for character in value
    ):
        _fail(f"control character at {path}")
    if any(pattern.search(value) for pattern in _SECRET_PATTERNS):
        _fail(f"credential-like content at {path}")
    if any(pattern.search(value) for pattern in _PRIVATE_PATH_PATTERNS):
        _fail(f"private path at {path}")


def _is_sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
    return normalized in _SENSITIVE_EXACT or normalized.endswith(_SENSITIVE_SUFFIXES)


def _validate_value(
    value: Any,
    *,
    path: str,
    depth: int,
    maximum_depth: int,
    active: set[int],
) -> None:
    if depth > maximum_depth:
        _fail(f"JSON nesting exceeds depth {maximum_depth} at {path}")
    if value is None or type(value) is bool or type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            _fail(f"non-finite number at {path}")
        return
    if type(value) is str:
        _check_string(value, path)
        return

    if type(value) not in (list, dict):
        _fail(f"unsupported JSON type {type(value).__name__} at {path}")
    identity = id(value)
    if identity in active:
        _fail(f"cyclic JSON value at {path}")
    active.add(identity)
    try:
        if type(value) is list:
            _check_limit(len(value), DEFAULT_MAX_BYTES, f"array item count at {path}")
            for index, item in enumerate(value):
                _validate_value(
                    item,
                    path=f"{path}[{index}]",
                    depth=depth + 1,
                    maximum_depth=maximum_depth,
                    active=active,
                )
        else:
            mapping = cast(dict[Any, Any], value)
            _check_limit(
                len(mapping), DEFAULT_MAX_BYTES, f"object item count at {path}"
            )
            for key, item in mapping.items():
                if type(key) is not str:
                    _fail(f"non-string object key at {path}")
                _check_string(key, f"{path}.{key}")
                if _is_sensitive_key(key):
                    _fail(f"sensitive object key at {path}.{key}")
                _validate_value(
                    item,
                    path=f"{path}.{key}",
                    depth=depth + 1,
                    maximum_depth=maximum_depth,
                    active=active,
                )
    finally:
        active.remove(identity)


def _validate_limit(maximum_bytes: int, maximum_depth: int) -> None:
    _check_limit(1, maximum_bytes, "maximum_bytes")
    _check_limit(1, maximum_depth, "maximum_depth")


def _encode(value: Any, *, maximum_bytes: int, maximum_depth: int) -> bytes:
    _validate_limit(maximum_bytes, maximum_depth)
    try:
        _validate_value(
            value,
            path="$",
            depth=0,
            maximum_depth=maximum_depth,
            active=set(),
        )
    except RecursionError as error:
        raise SerializationError("JSON nesting exceeds interpreter depth") from error
    try:
        encoded = (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise SerializationError("value cannot be encoded as strict JSON") from error
    _check_limit(len(encoded), maximum_bytes, "encoded JSON")
    return encoded


def _reject_constant(value: str) -> NoReturn:
    _fail(f"non-finite JSON constant: {value}")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode(raw: bytes, *, maximum_bytes: int, maximum_depth: int) -> JSONValue:
    _validate_limit(maximum_bytes, maximum_depth)
    _check_limit(len(raw), maximum_bytes, "encoded JSON")
    try:
        decoded = raw.decode("utf-8")
        value = json.loads(
            decoded,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_pairs,
        )
    except SerializationError:
        raise
    except (UnicodeError, ValueError, RecursionError) as error:
        raise SerializationError("invalid strict UTF-8 JSON") from error
    try:
        _validate_value(
            value,
            path="$",
            depth=0,
            maximum_depth=maximum_depth,
            active=set(),
        )
    except RecursionError as error:
        raise SerializationError("JSON nesting exceeds interpreter depth") from error
    return cast(JSONValue, value)


def detach_json(
    value: Any,
    *,
    maximum_bytes: int = DEFAULT_MAX_BYTES,
    maximum_depth: int = DEFAULT_MAX_DEPTH,
) -> JSONValue:
    """Return a recursively detached, strict-JSON-safe copy of ``value``."""

    return _decode(
        _encode(
            value,
            maximum_bytes=maximum_bytes,
            maximum_depth=maximum_depth,
        ),
        maximum_bytes=maximum_bytes,
        maximum_depth=maximum_depth,
    )


def detached_sweep_copies(
    sweeps: Sequence[Any],
    *,
    maximum_bytes: int = DEFAULT_MAX_BYTES,
    maximum_depth: int = DEFAULT_MAX_DEPTH,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build two recursively independent, value-equal JSON sweep lists."""

    raw = _encode(
        list(sweeps),
        maximum_bytes=maximum_bytes,
        maximum_depth=maximum_depth,
    )
    first = _decode(raw, maximum_bytes=maximum_bytes, maximum_depth=maximum_depth)
    second = _decode(raw, maximum_bytes=maximum_bytes, maximum_depth=maximum_depth)
    if not isinstance(first, list) or not isinstance(second, list):  # pragma: no cover
        _fail("sweeps must encode as JSON arrays")
    if any(type(item) is not dict for item in (*first, *second)):
        _fail("every sweep must encode as a JSON object")
    return cast(list[dict[str, Any]], first), cast(list[dict[str, Any]], second)


def _path_parts(path: Path) -> tuple[Path, Path]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    parent = candidate.parent
    if not parent.exists():
        _fail(f"parent directory does not exist: {parent}")
    current = Path(parent.anchor)
    for component in parent.parts[1:]:
        current /= component
        try:
            info = current.lstat()
        except OSError as error:
            raise SerializationError(
                f"cannot inspect parent directory: {current}"
            ) from error
        if stat.S_ISLNK(info.st_mode):
            _fail(f"symlinked parent is forbidden: {current}")
        if not stat.S_ISDIR(info.st_mode):
            _fail(f"parent component is not a directory: {current}")
    try:
        info = parent.lstat()
    except OSError as error:
        raise SerializationError(
            f"cannot inspect parent directory: {parent}"
        ) from error
    if stat.S_ISLNK(info.st_mode):
        _fail(f"symlinked parent is forbidden: {parent}")
    if not stat.S_ISDIR(info.st_mode):
        _fail(f"parent is not a real directory: {parent}")
    try:
        resolved_parent = parent.resolve(strict=True)
    except OSError as error:
        raise SerializationError(
            f"cannot resolve parent directory: {parent}"
        ) from error
    return resolved_parent / candidate.name, resolved_parent


def _assert_new_destination(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise SerializationError(f"cannot inspect output path: {path}") from error
    if stat.S_ISLNK(info.st_mode):
        _fail(f"symlink output is forbidden: {path}")
    if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
        _fail(f"hardlinked output is forbidden: {path}")
    _fail(f"output path already exists: {path}")


def write_json_new(
    path: str | Path,
    value: Any,
    *,
    maximum_bytes: int = DEFAULT_MAX_BYTES,
    maximum_depth: int = DEFAULT_MAX_DEPTH,
) -> str:
    """Write strict JSON exactly once to a new path and return its SHA-256."""

    encoded = _encode(
        value,
        maximum_bytes=maximum_bytes,
        maximum_depth=maximum_depth,
    )
    candidate, parent = _path_parts(Path(path))
    _assert_new_destination(candidate)
    descriptor: int | None = None
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=parent,
            prefix=f".{candidate.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # link() fails if another writer won the new destination race; unlike
        # replace(), it can never overwrite an existing destination.
        os.link(temporary, candidate, follow_symlinks=False)
        os.unlink(temporary)
        temporary = None
        directory_fd = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as error:
        raise SerializationError(f"output path already exists: {candidate}") from error
    except (OSError, ValueError) as error:
        raise SerializationError(
            f"atomic JSON publication failed: {candidate}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            with suppress(FileNotFoundError):
                temporary.unlink()
    return hashlib.sha256(encoded).hexdigest()


def read_json_strict(
    path: str | Path,
    *,
    maximum_bytes: int = DEFAULT_MAX_BYTES,
    maximum_depth: int = DEFAULT_MAX_DEPTH,
) -> JSONValue:
    """Read one bounded single-link regular file as strict JSON."""

    candidate, parent = _path_parts(Path(path))
    del parent
    try:
        initial_info = candidate.lstat()
    except OSError as error:
        raise SerializationError(f"strict JSON read failed: {candidate}") from error
    if (
        stat.S_ISLNK(initial_info.st_mode)
        or not stat.S_ISREG(initial_info.st_mode)
        or initial_info.st_nlink != 1
    ):
        _fail(f"input is not a single-link regular file: {candidate}")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            candidate,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) != (initial_info.st_dev, initial_info.st_ino):
            _fail(f"input changed while opening: {candidate}")
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            _fail(f"input is not a single-link regular file: {candidate}")
        _check_limit(info.st_size, maximum_bytes, "input JSON")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            raw = stream.read(maximum_bytes + 1)
        if len(raw) > maximum_bytes:
            _fail(f"input JSON exceeds limit {maximum_bytes}")
        if len(raw) != info.st_size:
            _fail(f"input changed while reading: {candidate}")
        return _decode(raw, maximum_bytes=maximum_bytes, maximum_depth=maximum_depth)
    except SerializationError:
        raise
    except (OSError, ValueError) as error:
        raise SerializationError(f"strict JSON read failed: {candidate}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_DEPTH",
    "JSONPrimitive",
    "JSONValue",
    "SerializationError",
    "detach_json",
    "detached_sweep_copies",
    "read_json_strict",
    "write_json_new",
]
