"""Nonempty synthetic worker -> assembler -> standalone-validator coverage."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/stage1c_v3"))
sys.path.insert(0, str(ROOT / "tests"))

from test_stage1c_v3_artifact_security import (  # noqa: E402
    _fixture,
    _supervisor,
    assembler,
    validator,
)


def test_nonempty_synthetic_worker_assembler_standalone_validator(
    tmp_path: Path,
) -> None:
    prediction, prediction_worker, intervention_worker = _fixture()
    top = intervention_worker["sweeps"]
    nested = intervention_worker["intervention_artifacts"]["intervention_sweeps"][
        "pairs"
    ]
    assert top and top == nested and top is not nested
    assert all(item["points"] for item in top)

    artifacts = assembler.records(
        prediction,
        prediction_worker,
        intervention_worker,
        prediction_supervisor=_supervisor(),
        intervention_supervisor=_supervisor(),
        execution="d" * 40,
    )
    bundle = tmp_path.resolve() / "bundle"
    assembler.write_bundle(artifacts, bundle)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage1c_v3/validate_stage1c_v3_artifacts.py"),
            "--bundle",
            str(bundle),
            "--execution-commit",
            "d" * 40,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert '"status": "passed"' in completed.stdout
    validated = validator.validate_bundle(bundle, "d" * 40)
    assert validated["point_count"] > 0
    assert validated["point_count"] == validated["api_call_count"]
