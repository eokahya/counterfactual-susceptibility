"""Static and synthetic execution-gate tests for Stage 1C-v3."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY_ROOT / "scripts" / "stage1c_v3"


def _source(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8")


def _module(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prediction_worker_has_no_intervention_import_or_call() -> None:
    source = _source("run_stage1c_v3_prediction_worker.py")
    tree = ast.parse(source)
    imported: list[str] = []
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
    assert not any("intervention" in item.casefold() for item in imported)
    assert not any("intervention" in item.casefold() for item in calls)
    assert "stage1c_first" not in source
    assert "capital of France" not in source


def test_prediction_token_positions_are_derived_not_embedded() -> None:
    source = _source("run_stage1c_v3_prediction_worker.py")
    assert "selected_positions_for_token_ids(token_ids)" in source
    assert "range(1, len(token_ids))" not in source
    assert "expected_token_ids" not in source
    assert "The capital of Norway is" not in source


def test_intervention_worker_has_one_attempt_and_detached_boundary() -> None:
    source = _source("run_stage1c_v3_intervention_worker.py")
    assert (
        "stage1c_v3_preregistered_prospective_prediction/prediction_manifest.json"
        in source
    )
    assert "canonical_attempts" in source
    assert "scientific_retry_count=0" in source
    assert "build_detached_worker_result" in source
    assert "sweeps.clear()" in source
    assert "stage1c_first" not in source
    assert "capital of France" not in source


def test_supervisor_uses_only_v3_worker_paths_and_offline_child_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source("run_stage1c_v3.py")
    assert "run_stage1c_v3_prediction_worker.py" in source
    assert "run_stage1c_v3_intervention_worker.py" in source
    assert "stage1c_first" not in source
    module = _module("stage1c_v3_supervisor", "run_stage1c_v3.py")
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    environment = module.safe_worker_environment(Path("/synthetic/src"))
    assert "HF_TOKEN" not in environment
    assert "GIT_CONFIG_COUNT" not in environment
    assert "PYTORCH_ENABLE_MPS_FALLBACK" not in environment
    assert environment["PYTHONPATH"] == "/synthetic/src"


def test_supervisor_tail_is_strict_json_safe() -> None:
    module = _module("stage1c_v3_supervisor_tail", "run_stage1c_v3.py")
    rendered = module._safe_process_tail(
        "first line\n\x1b[31mcolored\x1b[0m\tmessage\r\nlast line"
    )
    assert not any(
        ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F
        for character in rendered
    )
    assert "first line" in rendered
    assert "last line" in rendered


def test_v1_config_identity_is_rejected_by_v3_preflight() -> None:
    module = _module("stage1c_v3_preflight_identity", "preflight_stage1c_v3.py")
    invalid = {
        "experiment_class": "stage1c_first_prospective_prediction",
        "branch": "stage-1c-first-prospective-prediction",
        "base_commit": "efbf70a7e462e640a0e1819a93f3b92727bbd193",
        "prompt": {"id": "capital_france_v1", "text": "The capital of France is"},
    }
    with pytest.raises(RuntimeError):
        module._config_identity(invalid)


def test_preflight_environment_rejects_credentials_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("stage1c_v3_preflight", "preflight_stage1c_v3.py")
    monkeypatch.setenv("HF_TOKEN", "synthetic-secret")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    with pytest.raises(RuntimeError):
        module.collect_preflight(
            REPOSITORY_ROOT
            / "configs/stage1c_v3_preregistered_prospective_prediction.yaml",
            Path("/synthetic/cache"),
            "prediction",
        )


def test_v3_config_keeps_exact_prompt_and_dynamic_position_contract() -> None:
    from cfsus.stage1c_v3.config import (
        load_stage1c_v3_config,
        selected_positions_for_token_ids,
    )

    config = load_stage1c_v3_config()
    assert config["prompt"]["id"] == "capital_norway_preregistered_v3"
    assert config["prompt"]["text"] == "The capital of Norway is"
    token_ids = [2, 818, 5279, 529, 32649, 563]
    assert selected_positions_for_token_ids(token_ids) == [1, 2, 3, 4, 5]
    assert config["scanner"]["selected_positions"] == [1, 2, 3, 4, 5]


def test_prediction_and_intervention_workers_freeze_the_same_protocol_files() -> None:
    prediction = _module(
        "stage1c_v3_prediction_worker_hashes",
        "run_stage1c_v3_prediction_worker.py",
    )
    intervention = _module(
        "stage1c_v3_intervention_worker_hashes",
        "run_stage1c_v3_intervention_worker.py",
    )
    validator = _module(
        "stage1c_v3_validator_hashes", "validate_stage1c_v3_artifacts.py"
    )
    expected = {
        path: prediction._sha256(REPOSITORY_ROOT / path)
        for path in validator.PROTOCOL_PATHS
    }
    assert prediction._protocol_hashes() == expected
    assert intervention._protocol_hashes() == expected


def test_workers_normalize_torch_version_to_builtin_string() -> None:
    """Strict JSON publication must not receive TorchVersion subclasses."""

    for filename in (
        "run_stage1c_v3_prediction_worker.py",
        "run_stage1c_v3_intervention_worker.py",
    ):
        tree = ast.parse((SCRIPTS / filename).read_text(encoding="utf-8"))
        torch_values: list[ast.expr] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "torch":
                    torch_values.append(value)
        assert len(torch_values) == 1
        value = torch_values[0]
        assert isinstance(value, ast.Call)
        assert isinstance(value.func, ast.Name)
        assert value.func.id == "str"
        assert len(value.args) == 1
        assert ast.unparse(value.args[0]) == "torch.__version__"


def test_preflight_package_version_lookup_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("stage1c_v3_preflight_versions", "preflight_stage1c_v3.py")
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "synthetic")
    assert module._package_versions() == {
        name: "synthetic" for name in module.EXPECTED_VERSIONS
    }


def test_prediction_git_gate_requires_clean_protocol_descendant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("stage1c_v3_preflight_git", "preflight_stage1c_v3.py")
    head = "d" * 40

    def fake_git(*arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return head
        if arguments == ("branch", "--show-current"):
            return cast(str, module.V3_BRANCH)
        if arguments == (
            "rev-parse",
            f"refs/remotes/origin/{module.V3_BRANCH}",
        ):
            return head
        if arguments == (
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{u}",
        ):
            return f"origin/{module.V3_BRANCH}"
        if len(arguments) == 2 and arguments[0] == "rev-parse":
            prefix = "refs/remotes/origin/"
            if arguments[1].startswith(prefix):
                name = arguments[1][len(prefix) :]
                if name in module.PROTECTED_ORIGIN_REFS:
                    return cast(str, module.PROTECTED_ORIGIN_REFS[name])
        if arguments == ("merge-base", "--is-ancestor", module.V3_BASE_COMMIT, head):
            return ""
        if arguments == (
            "ls-tree",
            module.HISTORICAL_MANIFEST_FREEZE_COMMIT,
            "--",
            module.HISTORICAL_MANIFEST_PATH.as_posix(),
        ):
            return (
                f"100644 blob {module.HISTORICAL_MANIFEST_GIT_BLOB_SHA1}\t"
                f"{module.HISTORICAL_MANIFEST_PATH.as_posix()}"
            )
        raise AssertionError(arguments)

    monkeypatch.setattr(module, "_git", fake_git)
    monkeypatch.setattr(module, "_status_paths", lambda: ((), ()))
    result = module.verify_git("prediction")
    assert result["head"] == head
    assert result["working_tree_clean"] is True
    monkeypatch.setattr(module, "_status_paths", lambda: (("tracked.py",), ()))
    with pytest.raises(RuntimeError, match="clean committed"):
        module.verify_git("prediction")
