from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cfsus.backends.nnsight_plt import NNSightPLTMeasurementBackend
from cfsus.types import BackendOperation, CapabilityStatus

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STAGE1B_SCRIPTS = REPOSITORY_ROOT / "scripts/stage1b"
if str(STAGE1B_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STAGE1B_SCRIPTS))

from run_stage1b_measurement_primitives import (  # noqa: E402
    CREDENTIAL_VARIABLES,
    _safe_process_tail,
    safe_worker_environment,
)


def test_safe_worker_environment_strips_credentials_fallback_and_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("GH_TOKEN", "secret")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    environment = safe_worker_environment(Path("/frozen/source"))
    assert not CREDENTIAL_VARIABLES.intersection(environment)
    assert "PYTORCH_ENABLE_MPS_FALLBACK" not in environment
    assert not any(key.startswith("GIT_") for key in environment)
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["PYTHONPATH"] == "/frozen/source"


def test_backend_capabilities_remain_unverified_until_empirical_gate() -> None:
    backend = NNSightPLTMeasurementBackend.__new__(NNSightPLTMeasurementBackend)
    backend._measurement_primitives_validated = False
    report = backend.capability_report()
    assert (
        report.status_for(BackendOperation.LOCAL_JACOBIAN)
        is CapabilityStatus.UNVERIFIED
    )
    assert (
        report.status_for(BackendOperation.REPLACEMENT_SOURCE_SUPPRESSION)
        is CapabilityStatus.UNSUPPORTED
    )
    assert (
        report.status_for(BackendOperation.DIRECT_EFFECT)
        is CapabilityStatus.UNSUPPORTED
    )
    backend.mark_measurement_primitives_validated()
    assert (
        backend.capability_report().status_for(BackendOperation.LOCAL_JACOBIAN)
        is CapabilityStatus.SUPPORTED
    )


def test_process_tail_redacts_paths_and_credentials() -> None:
    rendered = _safe_process_tail(
        'File "/Users/example/private/run.py", line 7\n'
        "RuntimeError: failed at /private/tmp/run/output\n"
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz"
    )
    assert "/Users/" not in rendered
    assert "/private/tmp" not in rendered
    assert "abcdefghijklmnopqrstuvwxyz" not in rendered
