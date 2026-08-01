from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import pytest

from cfsus.reproduction.artifacts import (
    REDACTED,
    ArtifactValidationError,
    assert_publication_safe,
    build_checksum_manifest,
    deterministic_json_bytes,
    make_artifact_envelope,
    redact_sensitive,
    sha256_file,
    validate_artifact_envelope,
    verify_checksum_manifest,
    write_checksum_manifest_atomic,
    write_json_atomic,
)


def _valid_envelope() -> dict[str, object]:
    return make_artifact_envelope(
        artifact_type="environment_manifest",
        run_id="stage1a-20260801-001",
        status="observed",
        provenance={
            "code_commit": "7aacf30d888f96a29a1cfc82d035fca489ed0c17",
            "upstream_commit": "8f1e2438df612464e229e44c4a00ff637bf9379b",
        },
        payload={"python": "3.11.13", "mps_available": True},
        warnings=["No empirical model run was performed."],
    )


def test_valid_schema_v1_envelope_round_trips_deterministically() -> None:
    envelope = _valid_envelope()

    validate_artifact_envelope(envelope, expected_type="environment_manifest")
    first = deterministic_json_bytes(envelope)
    reordered = {key: envelope[key] for key in reversed(tuple(envelope))}
    second = deterministic_json_bytes(reordered)

    assert first == second
    assert first.endswith(b"\n")
    assert b'"schema_version": 1' in first


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"schema_version": True}, "schema_version"),
        ({"schema_version": 2}, "schema_version"),
        ({"artifact_type": "Environment Manifest"}, "artifact_type"),
        ({"run_id": "../escape"}, "run_id"),
        ({"status": "success"}, "status"),
        ({"warnings": [""]}, "warnings"),
    ],
)
def test_envelope_rejects_invalid_schema_fields(
    mutation: dict[str, object], message: str
) -> None:
    envelope = _valid_envelope()
    envelope.update(mutation)

    with pytest.raises(ArtifactValidationError, match=message):
        validate_artifact_envelope(envelope)


def test_envelope_rejects_missing_and_unknown_keys() -> None:
    missing = _valid_envelope()
    del missing["payload"]
    with pytest.raises(ArtifactValidationError, match="missing required keys: payload"):
        validate_artifact_envelope(missing)

    unknown = _valid_envelope()
    unknown["extra"] = None
    with pytest.raises(ArtifactValidationError, match="unknown keys: extra"):
        validate_artifact_envelope(unknown)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_serialization_rejects_nested_non_finite_values(value: float) -> None:
    envelope = _valid_envelope()
    payload = envelope["payload"]
    assert isinstance(payload, dict)
    payload["nested"] = [{"value": value}]

    with pytest.raises(ArtifactValidationError, match=r"payload\.nested\[0\]\.value"):
        deterministic_json_bytes(envelope)


def test_serialization_rejects_implicit_non_json_conversions() -> None:
    with pytest.raises(ArtifactValidationError, match="unsupported JSON type tuple"):
        deterministic_json_bytes({"values": (1, 2, 3)})
    with pytest.raises(ArtifactValidationError, match="non-string object key"):
        deterministic_json_bytes({1: "value"})


def test_redaction_is_recursive_without_redacting_scientific_token_ids() -> None:
    unsafe = {
        "token_ids": [1, 2, 3],
        "nested": {
            # commit-safety: allow-test-fixture
            "hf_token": "hf_abcdefghijklmnopqrstuvwxyz",
            # commit-safety: allow-test-fixture
            "command": "load /Users/researcher/.cache/huggingface/model",
        },
        # commit-safety: allow-test-fixture
        "log": "Authorization: Bearer abcdefghijklmnop",
    }

    redacted = redact_sensitive(unsafe)

    assert redacted["token_ids"] == [1, 2, 3]
    assert redacted["nested"]["hf_token"] == REDACTED
    assert redacted["nested"]["command"] == REDACTED
    assert redacted["log"] == REDACTED
    assert_publication_safe(redacted)
    rendered = deterministic_json_bytes(redacted)
    assert b"researcher" not in rendered
    # commit-safety: allow-test-fixture
    assert b"hf_abcdefghijklmnopqrstuvwxyz" not in rendered


@pytest.mark.parametrize(
    "unsafe",
    [
        # commit-safety: allow-test-fixture
        {"password": "correct horse battery staple"},
        # commit-safety: allow-test-fixture
        {"message": "Bearer abcdefghijklmnop"},
        # commit-safety: allow-test-fixture
        {"path": "file:///home/alice/.cache/huggingface/hub"},
        # commit-safety: allow-test-fixture
        {"path": r"C:\Users\alice\.cache\weights"},
        # commit-safety: allow-test-fixture
        {"key": "-----BEGIN OPENSSH PRIVATE KEY-----"},
    ],
)
def test_publication_safety_rejects_secrets_and_private_paths(
    unsafe: dict[str, str],
) -> None:
    with pytest.raises(ArtifactValidationError):
        assert_publication_safe(unsafe)


def test_publication_safety_allows_public_ids_shas_and_redacted_fields() -> None:
    value = {
        "repository": "google/gemma-2-2b",
        "revision": "8f1e2438df612464e229e44c4a00ff637bf9379b",
        "token": " Texas",
        "token_ids": [1, 2],
        "access_token": REDACTED,
    }

    assert_publication_safe(value)


def test_atomic_json_write_returns_digest_and_writes_complete_document(
    tmp_path: Path,
) -> None:
    output = tmp_path / "nested" / "environment.json"
    envelope = _valid_envelope()

    digest = write_json_atomic(output, envelope)

    content = deterministic_json_bytes(envelope)
    assert output.read_bytes() == content
    assert digest == hashlib.sha256(content).hexdigest()
    assert not tuple(output.parent.glob(f".{output.name}.*.tmp"))


def test_atomic_write_failure_preserves_existing_file_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "artifact.json"
    output.write_text("previous\n", encoding="utf-8")

    def fail_replace(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated"):
        write_json_atomic(output, _valid_envelope())
    assert output.read_text(encoding="utf-8") == "previous\n"
    assert not tuple(tmp_path.glob(f".{output.name}.*.tmp"))


def test_streamed_sha256_matches_known_vector_and_validates_chunk_size(
    tmp_path: Path,
) -> None:
    target = tmp_path / "abc.txt"
    target.write_bytes(b"abc")

    assert sha256_file(target, chunk_size=1) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    with pytest.raises(ArtifactValidationError, match="positive integer"):
        sha256_file(target, chunk_size=0)


def test_checksum_manifest_is_sorted_excludes_itself_and_verifies(
    tmp_path: Path,
) -> None:
    first = tmp_path / "z.json"
    second = tmp_path / "nested" / "a.json"
    second.parent.mkdir()
    first.write_text("z\n", encoding="utf-8")
    second.write_text("a\n", encoding="utf-8")
    manifest = tmp_path / "checksums.sha256"
    manifest.write_text("stale\n", encoding="utf-8")

    write_checksum_manifest_atomic(
        "checksums.sha256",
        ("z.json", "checksums.sha256", "nested/a.json"),
        root=tmp_path,
    )

    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert [line.split("  ", 1)[1] for line in lines] == [
        "nested/a.json",
        "z.json",
    ]
    assert verify_checksum_manifest("checksums.sha256", root=tmp_path) == (
        "nested/a.json",
        "z.json",
    )


def test_checksum_verification_detects_mutation(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    target.write_text("before\n", encoding="utf-8")
    manifest = tmp_path / "checksums.sha256"
    write_checksum_manifest_atomic(manifest, (target,), root=tmp_path)
    target.write_text("after\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="checksum mismatch"):
        verify_checksum_manifest(manifest, root=tmp_path)


def test_checksum_generation_rejects_escape_duplicate_and_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifact.json"
    target.write_text("{}\n", encoding="utf-8")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    try:
        with pytest.raises(ArtifactValidationError, match="escapes repository"):
            build_checksum_manifest((outside,), root=tmp_path)
        with pytest.raises(ArtifactValidationError, match="duplicate"):
            build_checksum_manifest((target, target), root=tmp_path)

        link = tmp_path / "link.json"
        os.symlink(target, link)
        with pytest.raises(ArtifactValidationError, match="symlink"):
            build_checksum_manifest((link,), root=tmp_path)
    finally:
        outside.unlink(missing_ok=True)


def test_checksum_verifier_rejects_unsafe_manifest_path(tmp_path: Path) -> None:
    manifest = tmp_path / "checksums.sha256"
    manifest.write_text(f"{'0' * 64}  ../outside.json\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="unsafe or duplicate"):
        verify_checksum_manifest(manifest, root=tmp_path)
