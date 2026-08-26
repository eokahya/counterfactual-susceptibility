from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from cfsus.reproduction.artifacts import ArtifactValidationError
from cfsus.reproduction.small_model_mps_bf16 import assert_fallback_disabled
from cfsus.stage1c.config import BASE_COMMIT, BRANCH, load_stage1c_config
from cfsus.stage1c.vjp import TargetBatchVJPContext

ROOT = Path(__file__).resolve().parents[1]
PREDICTION_WORKER = ROOT / "scripts/stage1c/run_stage1c_prediction_worker.py"


def _ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_stage1c_config_freezes_identity_regime_schedule_and_graph_prohibition() -> (
    None
):
    config = load_stage1c_config(
        ROOT / "configs/stage1c_first_prospective_prediction.yaml"
    )
    assert config["branch"] == BRANCH
    assert config["base_commit"] == BASE_COMMIT
    assert config["phase"] == "prediction_only_open"
    assert config["intervention"] == {
        "source_count": 1,
        "mapping": "desired=(1-alpha)*baseline",
        "freeze_attention": True,
        "constrained_layers": None,
        "target_clamp_allowed": False,
        "canonical_attempts": 1,
    }
    assert config["schedule"]["coarse_alphas"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert config["schedule"]["maximum_bisection_steps"] == 8
    assert config["responses"]["graph_edge_input"] == "forbidden"
    assert config["source_pool"]["raw_graph_input"] == "forbidden"


def test_prediction_worker_has_no_intervention_import_or_call() -> None:
    tree = _ast(PREDICTION_WORKER)
    imported_modules: list[str] = []
    called_attributes: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")
        elif isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            called_attributes.append(node.func.attr)
    assert not any("intervention" in module for module in imported_modules)
    assert "feature_intervention" not in called_attributes


def test_prediction_worker_enforces_exact_precommit_before_optional_dependencies() -> (
    None
):
    source = PREDICTION_WORKER.read_text(encoding="utf-8")
    assert 'git_start["branch"] != BRANCH' in source
    assert 'git_start["head"] != BASE_COMMIT' in source
    assert "prediction worker must start from the exact Stage 1C base" in source


def test_prediction_worker_is_offline_and_strips_credentials_and_mps_fallback() -> None:
    source = PREDICTION_WORKER.read_text(encoding="utf-8")
    assert '"HF_HUB_OFFLINE": "1"' in source
    assert '"TRANSFORMERS_OFFLINE": "1"' in source
    assert "PYTORCH_ENABLE_MPS_FALLBACK" in source
    assert "HF_TOKEN" in source
    assert "GH_TOKEN" in source

    with pytest.raises(ArtifactValidationError, match="must be absent or false"):
        assert_fallback_disabled({"PYTORCH_ENABLE_MPS_FALLBACK": "1"})
    assert_fallback_disabled({"PYTORCH_ENABLE_MPS_FALLBACK": "0"})


def test_prediction_module_does_not_accept_or_read_graph_objects() -> None:
    prediction_source = (ROOT / "src/cfsus/stage1c/prediction.py").read_text(
        encoding="utf-8"
    )
    vjp_source = (ROOT / "src/cfsus/stage1c/vjp.py").read_text(encoding="utf-8")
    assert "adjacency" not in prediction_source
    assert "raw_graph" not in prediction_source
    assert "adjacency" not in vjp_source
    signature = inspect.signature(TargetBatchVJPContext.compute)
    assert "graph" not in signature.parameters
    assert "adjacency" not in signature.parameters


def test_prediction_manifest_protocol_is_outcome_independent() -> None:
    source = (ROOT / "src/cfsus/stage1c/prediction.py").read_text(encoding="utf-8")
    assert "target_active" not in source
    assert "observed_critical" not in source
    assert "intervention_sweeps" not in source
