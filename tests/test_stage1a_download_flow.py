from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

STAGE1A_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "stage1a"
if str(STAGE1A_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STAGE1A_SCRIPTS))

import reproduce_attribution as runtime  # noqa: E402
import resolve_assets as resolver  # noqa: E402
import run_stage1a as orchestrator  # noqa: E402

from cfsus.reproduction import artifacts as artifact_helpers  # noqa: E402


def _stub_metadata(spec: resolver.AssetSpec) -> dict[str, object]:
    return {
        "repo_id": spec.repo_id,
        "verified_revision": spec.revision,
        "files": [],
    }


def _stub_resolution_network(
    monkeypatch: pytest.MonkeyPatch,
    *,
    access_granted: bool,
) -> None:
    monkeypatch.setattr(resolver, "_safe_repo_metadata", _stub_metadata)
    monkeypatch.setattr(resolver, "_comparison", lambda assets: {})
    monkeypatch.setattr(resolver, "_available_token", lambda: None)
    monkeypatch.setattr(
        resolver,
        "_probe_exact_config",
        lambda token: {
            "repo_id": resolver.MODEL.repo_id,
            "revision": resolver.MODEL.revision,
            "filename": "config.json",
            "authentication_present": False,
            "http_status": 200 if access_granted else 403,
            "access_granted": access_granted,
            "response_body_recorded": False,
        },
    )


def test_gated_model_probe_prevents_every_requested_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_resolution_network(monkeypatch, access_granted=False)

    def unexpected_download(targets: object, token: object) -> list[dict[str, Any]]:
        raise AssertionError("download must not start after a denied model probe")

    monkeypatch.setattr(resolver, "_download_selected", unexpected_download)

    manifest = resolver.resolve_assets(("model", "transcoder"))

    assert manifest["status"] == "blocked"
    results = manifest["payload"]["download_results"]
    assert [result["target"] for result in results] == ["model", "transcoder"]
    assert {result["status"] for result in results} == {"skipped"}
    assert (
        manifest["payload"]["metadata_range_audit"]["complete_weight_files_downloaded"]
        is False
    )


def test_failed_model_download_short_circuits_transcoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    fake_hub = ModuleType("huggingface_hub")

    def snapshot_download(**kwargs: Any) -> None:
        calls.append(str(kwargs["repo_id"]))
        raise RuntimeError("simulated model download failure")

    fake_hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    results = resolver._download_selected(("model", "transcoder"), None)

    assert calls == [resolver.MODEL.repo_id]
    assert [result["status"] for result in results] == ["failed", "skipped"]
    assert results[1]["reason"] == "earlier_requested_download_failed"


def test_successful_requested_downloads_are_recorded_truthfully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_resolution_network(monkeypatch, access_granted=True)
    monkeypatch.setattr(
        resolver,
        "_download_selected",
        lambda targets, token: [
            {
                "target": target,
                "repo_id": (
                    resolver.MODEL.repo_id
                    if target == "model"
                    else resolver.TRANSCODER.repo_id
                ),
                "revision": (
                    resolver.MODEL.revision
                    if target == "model"
                    else resolver.TRANSCODER.revision
                ),
                "status": "completed",
            }
            for target in targets
        ],
    )

    manifest = resolver.resolve_assets(("model", "transcoder"))
    audit = manifest["payload"]["metadata_range_audit"]

    assert manifest["status"] == "resolved"
    assert audit["complete_weight_files_downloaded"] is True
    assert audit["completed_snapshot_targets"] == ["model", "transcoder"]
    assert audit["incomplete_snapshot_targets"] == []


def test_resolver_cli_returns_nonzero_for_logically_blocked_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resolver,
        "resolve_assets",
        lambda downloads: {"status": "blocked"},
    )

    assert resolver.main([]) == 2


def test_orchestrator_requires_strict_scientific_payload_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "artifacts": {
            "environment_manifest": "results/stage1a/environment_manifest.json",
            "checksums": "results/stage1a/checksums.sha256",
        }
    }
    calls: list[bool] = []
    monkeypatch.setattr(orchestrator, "load_yaml", lambda path: config)
    monkeypatch.setattr(orchestrator, "validate_official_config", lambda value: None)
    monkeypatch.setattr(
        orchestrator,
        "_write_metadata_artifacts",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(orchestrator, "load_runtime", lambda *args, **kwargs: object())
    monkeypatch.setattr(orchestrator, "collect_report", lambda **kwargs: {})
    monkeypatch.setattr(artifact_helpers, "write_json_atomic", lambda *args: None)
    monkeypatch.setattr(orchestrator, "verify_runtime_semantics", lambda bundle: None)
    monkeypatch.setattr(orchestrator, "reproduce_intervention", lambda bundle: None)
    monkeypatch.setattr(orchestrator, "reproduce_attribution", lambda bundle: None)
    monkeypatch.setattr(
        orchestrator,
        "validate_present_artifacts",
        lambda path, *, strict_payloads: calls.append(strict_payloads),
    )
    monkeypatch.setattr(orchestrator, "regenerate_checksums", lambda *args: None)
    monkeypatch.setattr(orchestrator, "verify_checksums", lambda *args: None)
    monkeypatch.setattr(orchestrator, "_release_runtime", lambda bundle: True)

    assert orchestrator.main([]) == 0
    assert calls == [True]


def test_snapshot_file_verification_accepts_exact_hash_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    content = b'{"model_type":"gemma2"}\n'
    path.write_bytes(content)
    inventory = {
        "config.json": {
            "size_bytes": len(content),
            "lfs_sha256": hashlib.sha256(content).hexdigest(),
            "git_blob_id": None,
        }
    }

    assert (
        runtime._verify_snapshot_files(
            snapshot=tmp_path,
            inventory=inventory,
            required_files=("config.json",),
            role="model",
        )
        == 1
    )

    path.write_bytes(content.replace(b"gemma2", b"gemma3"))
    with pytest.raises(runtime.Stage1ABlocked, match="hash mismatch"):
        runtime._verify_snapshot_files(
            snapshot=tmp_path,
            inventory=inventory,
            required_files=("config.json",),
            role="model",
        )


def test_snapshot_verification_rejects_missing_metadata_and_extra_loader_file(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")

    with pytest.raises(runtime.Stage1ABlocked, match="omits required model files"):
        runtime._verify_snapshot_files(
            snapshot=tmp_path,
            inventory={},
            required_files=("config.json",),
            role="model",
        )

    extra = tmp_path / "model.safetensors"
    extra.write_bytes(b"unverified loader-precedence candidate")
    with pytest.raises(runtime.Stage1ABlocked, match="unmanifested entry"):
        runtime._reject_unmanifested_snapshot_entries(
            snapshot=tmp_path,
            inventory={"config.json": {}},
            role="model",
        )
