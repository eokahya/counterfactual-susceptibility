from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "stage1a" / "probe_stage1a_mps.py"
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))

import probe_stage1a_mps as probe  # noqa: E402


def test_mps_config_contains_fixed_pins_and_runtime_class() -> None:
    config = (
        REPOSITORY_ROOT / "configs" / "stage1a_gemma2_2b_mps_fp16_reproduction.yaml"
    ).read_text(encoding="utf-8")

    for pin in (
        "d965e43c34a2ba408b8ae35b13b5651bf269beed",
        "8f1e2438df612464e229e44c4a00ff637bf9379b",
        "c5ebcd40d208330abc697524c919956e692655cf",
        "bd5773156dea09893636c801df1237d0410307d2",
    ):
        assert pin in config
    assert "device: mps" in config
    assert "dtype: float16" in config
    assert "offload: disk" in config
    assert "trigger: mps_out_of_memory_only" in config
    assert "official_bf16_reproduction: false" in config
    assert "t4_fp16_reproduction: false" in config


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({}, False),
        ({"PYTORCH_ENABLE_MPS_FALLBACK": "1"}, True),
        ({"PYTORCH_ENABLE_MPS_FALLBACK": "false"}, False),
    ],
)
def test_fallback_detection_is_strict_and_side_effect_free(
    environment: dict[str, str], expected: bool
) -> None:
    assert probe.fallback_enabled(environment) is expected


def test_mps_device_match_accepts_indexed_mps_but_rejects_cpu() -> None:
    assert probe._same_device(SimpleNamespace(device="mps"), "mps")
    assert probe._same_device(SimpleNamespace(device="mps:0"), "mps")
    assert not probe._same_device(SimpleNamespace(device="cpu"), "mps")
    assert not probe._same_device(SimpleNamespace(device="cuda:0"), "mps")


def test_output_path_is_allowlisted(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="canonical preflight directory"):
        probe.write_summary({}, tmp_path / "preflight_summary.json")


def test_generated_preflight_staging_path_is_narrowly_allowlisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generated = tmp_path / "results" / "generated" / "stage1a_mps_fp16"
    monkeypatch.setattr(probe, "GENERATED_RESULT_DIRECTORY", generated)
    output = generated / "preflight-abc123" / "preflight_summary.json"
    probe.write_summary({"status": "passed"}, output)
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "passed"}

    with pytest.raises(ValueError, match="isolated generated staging"):
        probe.write_summary(
            {}, generated / "preflight-abc123" / "nested" / "preflight_summary.json"
        )


def test_summary_is_bounded_json_and_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(probe, "RESULT_DIRECTORY", tmp_path)
    monkeypatch.setattr(probe, "DEFAULT_OUTPUT", tmp_path / "preflight_summary.json")
    summary = {"schema_version": 1, "status": "blocked", "checks": {"mps": False}}
    probe.write_summary(summary, tmp_path / "preflight_summary.json")
    assert json.loads((tmp_path / "preflight_summary.json").read_text()) == summary


def test_run_preflight_never_enables_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    summary = probe.run_preflight()
    assert summary["environment"]["fallback_enabled"] is False
    assert summary["large_assets_downloaded"] is False
    assert summary["scientific_model_result"] is False
    assert set(summary["checks"]) >= {
        "python_3_11",
        "darwin",
        "native_arm64",
        "torch_2_6_0",
        "fallback_disabled",
        "mps_built",
        "mps_available",
    }


def test_passed_preflight_matches_validator_contract() -> None:
    summary = probe.run_preflight()
    if summary["status"] != "passed":
        pytest.skip("selected test environment does not provide passing MPS")
    assert summary["probe_status"] == "passed"
    assert summary["environment"]["system"] == "Darwin"
    assert summary["environment"]["torch_version"] == "2.6.0"
    expected_categories = {
        "operators",
        "transfers",
        "autograd_hooks",
        "strict_jumprelu",
        "tiny_hooked_transformer",
        "sparse_metadata_boundary",
        "bounded_graph_construction",
        "disk_offload_safetensors",
    }
    assert set(summary["operations"]) == expected_categories
    for operation in summary["operations"].values():
        assert operation["attempted"] is True
        assert operation["passed"] is True
        assert operation["device"] == "mps"
        assert operation["dtype"] == "float16"
        assert operation["cpu_reference_passed"] is True
    assert summary["operations"]["strict_jumprelu"]["strict_gate_equal"] is True
    assert summary["operations"]["strict_jumprelu"]["equality_inactive"] is True
    sparse = summary["operations"]["sparse_metadata_boundary"]
    assert sparse["cpu_metadata_explicit"] is True
    assert sparse["replacement_boundary_passed"] is True
    assert sparse["dense_scientific_device"] == "mps"
    assert (
        summary["operations"]["disk_offload_safetensors"][
            "upstream_disk_offload_helper_tested"
        ]
        is True
    )
    primitives = summary["operations"]["operators"]["operations"]
    expected_primitives = {
        "matmul",
        "einsum",
        "softmax",
        "layernorm",
        "topk",
        "gather",
        "scatter",
        "index_add",
        "index_put",
        "where",
        "sort",
        "unique",
        "searchsorted",
    }
    assert set(primitives) == expected_primitives
    for primitive in primitives.values():
        assert primitive["attempted"] is True
        assert primitive["passed"] is True
        assert primitive["device"] == "mps"
        assert primitive["dtype"] == "float16"
        assert primitive["cpu_reference_passed"] is True


def test_run_preflight_records_blocked_mps_without_cpu_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    summary = probe.run_preflight()
    if not summary["checks"]["mps_available"]:
        assert summary["status"] == "blocked"
        assert all(not value["attempted"] for value in summary["operations"].values())


def test_cli_no_output_does_not_create_artifact(tmp_path: Path) -> None:
    output = tmp_path / "preflight_summary.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output), "--no-output"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in {0, 1, 2}
    assert not output.exists()


def test_high_watermark_override_is_not_set_by_probe() -> None:
    assert (
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ
        or os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] != "0.0"
    )
