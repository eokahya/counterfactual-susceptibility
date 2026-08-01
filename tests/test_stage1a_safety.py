from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

STAGE1A_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "stage1a"
if str(STAGE1A_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STAGE1A_SCRIPTS))

from scan_commit_safety import scan_paths  # noqa: E402
from validate_artifacts import (  # noqa: E402
    regenerate_checksums,
    validate_present_artifacts,
    verify_checksums,
)
from verify_environment import (  # noqa: E402
    EXPECTED_REQUIREMENT,
    EXPECTED_UPSTREAM_COMMIT,
    EXPECTED_UPSTREAM_URL,
    EnvironmentVerificationError,
    validate_direct_url,
    validate_environment_schema,
    validate_lock_text,
)

from cfsus.reproduction.artifacts import (  # noqa: E402
    ArtifactValidationError,
    make_artifact_envelope,
    write_json_atomic,
)


def _direct_url() -> dict[str, object]:
    return {
        "url": EXPECTED_UPSTREAM_URL,
        "vcs_info": {
            "vcs": "git",
            "commit_id": EXPECTED_UPSTREAM_COMMIT,
            "requested_revision": EXPECTED_UPSTREAM_COMMIT,
        },
    }


def _artifact(artifact_type: str) -> dict[str, object]:
    return make_artifact_envelope(
        artifact_type=artifact_type,
        run_id="stage1a-test",
        status="observed",
        provenance={"upstream_commit": EXPECTED_UPSTREAM_COMMIT},
        payload={"finite_value": 1.0},
    )


def test_direct_url_requires_exact_audited_git_identity() -> None:
    validate_direct_url(_direct_url())

    mutable = _direct_url()
    vcs_info = dict(cast(dict[str, object], mutable["vcs_info"]))
    vcs_info["requested_revision"] = "main"
    mutable["vcs_info"] = vcs_info
    with pytest.raises(EnvironmentVerificationError, match="requested_revision"):
        validate_direct_url(mutable)

    mirror = _direct_url()
    mirror["url"] = "https://example.invalid/circuit-tracer.git"
    with pytest.raises(EnvironmentVerificationError, match="canonical"):
        validate_direct_url(mirror)


def test_lock_and_environment_schema_require_observed_py311_record() -> None:
    validate_lock_text(f"# observed lock\n{EXPECTED_REQUIREMENT}\n")
    validate_environment_schema(
        {
            "schema_version": 1,
            "stage": "stage1a",
            "observation_scope": "macos-arm64-py311",
            "lock": {"format": "pip-freeze", "provenance": "observed"},
        }
    )

    with pytest.raises(EnvironmentVerificationError, match="canonical"):
        validate_lock_text(
            "circuit-tracer @ git+https://github.com/decoderesearch/"
            "circuit-tracer.git@main\n"
        )


def test_artifact_validation_and_checksum_round_trip(tmp_path: Path) -> None:
    artifact_directory = tmp_path / "results" / "stage1a"
    artifact_directory.mkdir(parents=True)
    artifact_path = artifact_directory / "environment_manifest.json"
    checksum_path = artifact_directory / "checksums.sha256"
    write_json_atomic(artifact_path, _artifact("environment_manifest"))

    assert validate_present_artifacts(artifact_directory) == (
        "environment_manifest.json",
    )
    regenerate_checksums(
        artifact_directory,
        checksum_path,
        checksum_root=tmp_path,
    )
    assert verify_checksums(
        artifact_directory,
        checksum_path,
        checksum_root=tmp_path,
    ) == ("results/stage1a/environment_manifest.json",)

    artifact_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="checksum mismatch"):
        verify_checksums(
            artifact_directory,
            checksum_path,
            checksum_root=tmp_path,
        )


def test_checksum_manifest_must_cover_every_present_file(tmp_path: Path) -> None:
    artifact_directory = tmp_path / "results" / "stage1a"
    artifact_directory.mkdir(parents=True)
    checksum_path = artifact_directory / "checksums.sha256"
    write_json_atomic(
        artifact_directory / "environment_manifest.json",
        _artifact("environment_manifest"),
    )
    regenerate_checksums(
        artifact_directory,
        checksum_path,
        checksum_root=tmp_path,
    )
    write_json_atomic(
        artifact_directory / "asset_manifest.json",
        _artifact("asset_manifest"),
    )

    with pytest.raises(ArtifactValidationError, match="incomplete"):
        verify_checksums(
            artifact_directory,
            checksum_path,
            checksum_root=tmp_path,
        )


def test_filename_enforces_artifact_type(tmp_path: Path) -> None:
    artifact_directory = tmp_path / "results" / "stage1a"
    artifact_directory.mkdir(parents=True)
    write_json_atomic(
        artifact_directory / "environment_manifest.json",
        _artifact("asset_manifest"),
    )

    with pytest.raises(ArtifactValidationError, match="expected artifact_type"):
        validate_present_artifacts(artifact_directory)


def test_artifact_validation_rejects_duplicate_keys_and_unrelated_files(
    tmp_path: Path,
) -> None:
    artifact_directory = tmp_path / "results" / "stage1a"
    artifact_directory.mkdir(parents=True)
    duplicate = artifact_directory / "environment_manifest.json"
    duplicate.write_text(
        '{"schema_version": 1, "schema_version": 1}\n',
        encoding="utf-8",
    )
    with pytest.raises(ArtifactValidationError, match="duplicate JSON key"):
        validate_present_artifacts(artifact_directory)

    duplicate.unlink()
    unrelated = artifact_directory / "captured-output.txt"
    unrelated.write_text("must not be committed here\n", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="unsupported file"):
        regenerate_checksums(
            artifact_directory,
            artifact_directory / "checksums.sha256",
            checksum_root=tmp_path,
        )


def test_commit_scan_detects_secrets_without_reporting_the_value(
    tmp_path: Path,
) -> None:
    token = "hf_" + ("a" * 24)
    path = tmp_path / "unsafe.txt"
    path.write_text(f"credential={token}\n", encoding="utf-8")

    findings = scan_paths(tmp_path, ("unsafe.txt",))

    assert [finding.kind for finding in findings] == ["hugging_face_token"]
    assert all(token not in finding.detail for finding in findings)


def test_staged_scan_reads_index_blob_not_working_tree(tmp_path: Path) -> None:
    subprocess.run(("git", "init"), cwd=tmp_path, check=True, capture_output=True)
    path = tmp_path / "artifact.txt"
    path.write_text("safe\n", encoding="utf-8")
    subprocess.run(
        ("git", "add", "artifact.txt"), cwd=tmp_path, check=True, capture_output=True
    )
    token = "hf_" + ("c" * 24)
    path.write_text(f"credential={token}\n", encoding="utf-8")
    assert scan_paths(tmp_path, ("artifact.txt",), staged=True) == ()

    subprocess.run(
        ("git", "add", "artifact.txt"), cwd=tmp_path, check=True, capture_output=True
    )
    path.write_text("safe again\n", encoding="utf-8")
    findings = scan_paths(tmp_path, ("artifact.txt",), staged=True)
    assert [finding.kind for finding in findings] == ["hugging_face_token"]
    assert all(token not in finding.detail for finding in findings)


def test_commit_scan_detects_private_paths_and_forbidden_data(tmp_path: Path) -> None:
    private_path = tmp_path / "manifest.json"
    private_path.write_text(
        json.dumps({"cache": "file://" + "/" + "Users/alice/Library/Caches/hf"}),
        encoding="utf-8",
    )
    weight_path = tmp_path / "weights.safetensors"
    weight_path.write_bytes(b"not-real-weights")
    raw_path = tmp_path / "results" / "raw" / "graph.json"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text("{}\n", encoding="utf-8")

    findings = scan_paths(
        tmp_path,
        ("manifest.json", "results/raw/graph.json", "weights.safetensors"),
    )
    kinds = {finding.kind for finding in findings}

    assert "private_absolute_path" in kinds
    assert "forbidden_data_extension" in kinds
    assert "forbidden_path" in kinds


def test_commit_scan_detects_symlinks_and_large_files(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("safe\n", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    large = tmp_path / "large.txt"
    large.write_bytes(b"a" * 33)

    findings = scan_paths(
        tmp_path,
        ("link.txt", "large.txt"),
        max_bytes=32,
    )
    kinds = {finding.kind for finding in findings}

    assert "symlink" in kinds
    assert "large_file" in kinds


def test_commit_scan_rejects_saved_notebook_state(tmp_path: Path) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [{"output_type": "stream", "text": ["result\n"]}],
                "source": ["print('result')\n"],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = tmp_path / "executed.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")

    kinds = {finding.kind for finding in scan_paths(tmp_path, ("executed.ipynb",))}

    assert "notebook_output" in kinds
    assert "notebook_execution_count" in kinds


def test_commit_scan_avoids_generic_security_false_positives(tmp_path: Path) -> None:
    path = tmp_path / "safe.md"
    path.write_text(
        "Set the HF_TOKEN environment variable; never print it.\n"
        "Use /Users/<name>/ only as a documented placeholder.\n"
        f"Pinned source: {EXPECTED_UPSTREAM_URL}\n",
        encoding="utf-8",
    )

    assert scan_paths(tmp_path, ("safe.md",)) == ()


def test_commit_scan_allows_only_explicit_test_fixtures(tmp_path: Path) -> None:
    token = "hf_" + ("b" * 24)
    fixture = tmp_path / "tests" / "fixture.py"
    fixture.parent.mkdir()
    fixture.write_text(
        "# commit-safety: allow-test-fixture\n" + repr(token) + "\n",
        encoding="utf-8",
    )
    source = tmp_path / "src" / "unsafe.py"
    source.parent.mkdir()
    source.write_text(
        "# commit-safety: allow-test-fixture\n" + repr(token) + "\n",
        encoding="utf-8",
    )

    assert scan_paths(tmp_path, ("tests/fixture.py",)) == ()
    findings = scan_paths(tmp_path, ("src/unsafe.py",))
    assert [finding.kind for finding in findings] == ["hugging_face_token"]
