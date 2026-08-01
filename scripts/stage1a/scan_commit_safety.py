#!/usr/bin/env python3
"""Scan prospective Git content for publication-safety violations.

The scanner reports only the category and line number of textual findings; it
never prints the matching credential-like value. Its patterns intentionally
target high-confidence token formats rather than generic words such as "token".
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAX_BYTES = 1_048_576
TEST_FIXTURE_ALLOW_MARKER = "commit-safety: allow-test-fixture"

FORBIDDEN_SUFFIXES = frozenset(
    {
        ".arrow",
        ".bin",
        ".ckpt",
        ".gguf",
        ".h5",
        ".hdf5",
        ".joblib",
        ".npy",
        ".npz",
        ".onnx",
        ".parquet",
        ".pickle",
        ".pkl",
        ".pt",
        ".pth",
        ".safetensors",
        ".tflite",
    }
)
SECRET_FILENAME_SUFFIXES = frozenset({".key", ".p12", ".pem", ".pfx"})
SECRET_FILENAMES = frozenset(
    {
        ".env",
        ".netrc",
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "secrets.json",
        "token",
        "token.txt",
    }
)
FORBIDDEN_PATH_PREFIXES = (
    (".cache",),
    (".mypy_cache",),
    (".pytest_cache",),
    (".ruff_cache",),
    (".ssh",),
    (".venv",),
    (".venv-stage1a",),
    ("checkpoints",),
    ("data", "raw"),
    ("datasets",),
    ("dist",),
    ("hf_cache",),
    ("models",),
    ("results", "cache"),
    ("results", "generated"),
    ("results", "raw"),
    ("transcoders",),
    ("upstream-clones",),
    ("wandb",),
)
FORBIDDEN_EXACT_PATHS = frozenset({"CODEX_STAGE_0_PROMPT(1).md"})

_SECRET_PATTERNS = (
    (
        "hugging_face_token",
        re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    ),
    (
        "github_token",
        re.compile(r"\b(?:github_pat_[A-Za-z0-9_]{20,}|gh[opusr]_[A-Za-z0-9]{30,})\b"),
    ),
    (
        "aws_access_key",
        re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ),
    (
        "private_key",
        re.compile(r"-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----"),
    ),
    (
        "authorization_bearer",
        re.compile(r"(?i)\bauthorization\s*:\s*bearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    ),
    (
        "credential_in_url",
        re.compile(r"https?://[^\s/:@]+:[^\s/@]+@[^\s/]+"),
    ),
)
_PRIVATE_PATH_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9_])/(?:Users|home)/[A-Za-z0-9._-]+/"),
    re.compile(r"(?i)(?<![A-Za-z0-9_])[A-Z]:\\Users\\[A-Za-z0-9._-]+\\"),
    re.compile(r"(?<![A-Za-z0-9_])/(?:private/)?var/folders/[^\s<>{}]+"),
)


@dataclass(frozen=True, order=True)
class SafetyFinding:
    """A redacted publication-safety finding."""

    path: str
    kind: str
    detail: str


def _run_git(root: Path, arguments: Sequence[str]) -> tuple[str, ...]:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"git {' '.join(arguments)} failed") from exc
    return tuple(
        item.decode("utf-8", errors="surrogateescape")
        for item in completed.stdout.split(b"\0")
        if item
    )


def _staged_blob(root: Path, relative_path: str) -> tuple[str, bytes] | None:
    """Return the index mode and blob bytes, never working-tree content."""

    try:
        entry = subprocess.run(
            ("git", "ls-files", "--stage", "--", relative_path),
            cwd=root,
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
        if not entry:
            return None
        mode = entry.split(None, 1)[0].decode("ascii")
        data = subprocess.run(
            ("git", "cat-file", "blob", f":{relative_path}"),
            cwd=root,
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError) as exc:
        raise RuntimeError("failed to read a staged Git blob") from exc
    return mode, data


def git_candidate_paths(root: Path, *, mode: str = "candidates") -> tuple[str, ...]:
    """Return deterministic paths from Git without following symlinks."""

    if mode == "tracked":
        paths = _run_git(root, ("ls-files", "-z"))
    elif mode == "staged":
        paths = _run_git(
            root,
            ("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"),
        )
    elif mode == "candidates":
        tracked = _run_git(root, ("ls-files", "-z"))
        untracked = _run_git(root, ("ls-files", "--others", "--exclude-standard", "-z"))
        paths = (*tracked, *untracked)
    else:
        raise ValueError("mode must be candidates, staged, or tracked")
    return tuple(sorted(set(paths)))


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _is_explicit_test_fixture(
    relative_path: str,
    text: str,
    offset: int,
) -> bool:
    pure_path = PurePosixPath(relative_path)
    if not pure_path.parts or pure_path.parts[0] != "tests":
        return False
    lines = text.splitlines()
    line_index = _line_number(text, offset) - 1
    candidates = lines[max(0, line_index - 1) : line_index + 1]
    return any(TEST_FIXTURE_ALLOW_MARKER in line for line in candidates)


def _text_findings(relative_path: str, text: str) -> list[SafetyFinding]:
    findings: list[SafetyFinding] = []
    for kind, pattern in _SECRET_PATTERNS:
        for match in pattern.finditer(text):
            if _is_explicit_test_fixture(relative_path, text, match.start()):
                continue
            line = _line_number(text, match.start())
            findings.append(
                SafetyFinding(
                    relative_path,
                    kind,
                    f"high-confidence secret pattern at line {line}",
                )
            )
    for pattern in _PRIVATE_PATH_PATTERNS:
        for match in pattern.finditer(text):
            if _is_explicit_test_fixture(relative_path, text, match.start()):
                continue
            line = _line_number(text, match.start())
            findings.append(
                SafetyFinding(
                    relative_path,
                    "private_absolute_path",
                    f"machine-local absolute path at line {line}",
                )
            )
    return findings


def _notebook_findings(relative_path: str, text: str) -> list[SafetyFinding]:
    try:
        notebook = json.loads(text)
    except json.JSONDecodeError:
        return [SafetyFinding(relative_path, "invalid_notebook", "not valid JSON")]
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        return [SafetyFinding(relative_path, "invalid_notebook", "missing cells list")]

    findings: list[SafetyFinding] = []
    for index, cell in enumerate(notebook["cells"]):
        if not isinstance(cell, dict):
            findings.append(
                SafetyFinding(
                    relative_path,
                    "invalid_notebook",
                    f"cell {index} is not an object",
                )
            )
            continue
        if cell.get("cell_type") == "code":
            outputs = cell.get("outputs", [])
            if outputs not in (None, []):
                findings.append(
                    SafetyFinding(
                        relative_path,
                        "notebook_output",
                        f"code cell {index} contains saved output",
                    )
                )
            if cell.get("execution_count") is not None:
                findings.append(
                    SafetyFinding(
                        relative_path,
                        "notebook_execution_count",
                        f"code cell {index} contains an execution count",
                    )
                )
        if cell.get("attachments") not in (None, {}):
            findings.append(
                SafetyFinding(
                    relative_path,
                    "notebook_attachment",
                    f"cell {index} contains an embedded attachment",
                )
            )
    return findings


def scan_paths(
    root: Path,
    paths: Iterable[str],
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    staged: bool = False,
) -> tuple[SafetyFinding, ...]:
    """Scan repository-relative working-tree or staged paths safely."""

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    root = root.resolve()
    findings: list[SafetyFinding] = []

    for relative_path in sorted(set(paths)):
        pure_path = PurePosixPath(relative_path)
        if pure_path.is_absolute() or ".." in pure_path.parts:
            findings.append(
                SafetyFinding(relative_path, "unsafe_path", "path escapes repository")
            )
            continue
        if relative_path in FORBIDDEN_EXACT_PATHS or any(
            pure_path.parts[: len(prefix)] == prefix
            for prefix in FORBIDDEN_PATH_PREFIXES
        ):
            findings.append(
                SafetyFinding(
                    relative_path,
                    "forbidden_path",
                    "path is reserved for ignored local or raw artifacts",
                )
            )
        candidate = root.joinpath(*pure_path.parts)
        if staged:
            staged_entry = _staged_blob(root, relative_path)
            if staged_entry is None:
                continue
            mode, data = staged_entry
            if mode == "120000":
                findings.append(
                    SafetyFinding(
                        relative_path, "symlink", "symlinks are not publishable"
                    )
                )
                continue
            if mode == "160000":
                findings.append(
                    SafetyFinding(
                        relative_path,
                        "non_file",
                        "Git submodules are not publishable artifacts",
                    )
                )
                continue
        else:
            if candidate.is_symlink():
                findings.append(
                    SafetyFinding(
                        relative_path, "symlink", "symlinks are not publishable"
                    )
                )
                continue
            if not candidate.exists():
                continue
            if not candidate.is_file():
                findings.append(
                    SafetyFinding(
                        relative_path, "non_file", "candidate is not a regular file"
                    )
                )
                continue
            data = candidate.read_bytes()

        suffix = pure_path.suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES or any(
            part.lower().endswith(".zarr") for part in pure_path.parts
        ):
            findings.append(
                SafetyFinding(
                    relative_path,
                    "forbidden_data_extension",
                    f"forbidden binary/model-data extension {suffix or '.zarr'}",
                )
            )
        if (
            pure_path.name in SECRET_FILENAMES or suffix in SECRET_FILENAME_SUFFIXES
        ) and pure_path.name != ".env.example":
            findings.append(
                SafetyFinding(
                    relative_path,
                    "secret_shaped_filename",
                    "filename is reserved for local credentials",
                )
            )

        size = len(data)
        if size > max_bytes:
            findings.append(
                SafetyFinding(
                    relative_path,
                    "large_file",
                    f"{size} bytes exceeds {max_bytes}-byte review limit",
                )
            )
            continue

        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        findings.extend(_text_findings(relative_path, text))
        if suffix == ".ipynb":
            findings.extend(_notebook_findings(relative_path, text))

    return tuple(sorted(set(findings)))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("candidates", "staged", "tracked"),
        default="candidates",
    )
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument(
        "--path",
        action="append",
        dest="paths",
        help="Scan this repository-relative path; may be repeated.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = (
        tuple(args.paths)
        if args.paths is not None
        else git_candidate_paths(REPOSITORY_ROOT, mode=args.mode)
    )
    findings = scan_paths(
        REPOSITORY_ROOT,
        paths,
        max_bytes=args.max_bytes,
        staged=args.mode == "staged" and args.paths is None,
    )
    report = {
        "schema_version": 1,
        "valid": not findings,
        "mode": args.mode,
        "scanned_paths": len(paths),
        "findings": [asdict(finding) for finding in findings],
    }
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
