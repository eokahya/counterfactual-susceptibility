"""Focused Stage 1C-v4 production-contract tests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from cfsus.stage1c_v3.execution_journal import CanonicalExecutionJournal
from cfsus.stage1c_v3.quantization_audit import audit_frozen_quantization
from cfsus.stage1c_v3.serialization import SerializationError

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "results/stage1c_v3_preregistered_prospective_prediction"
    / "prediction_manifest.json"
)
EXPECTED_SHA256 = "b2c489317852a2f54d50db783abc17dfdc08590353b0473dbab01ec3d04574cc"


def test_frozen_quantization_audit_covers_every_requested_alpha() -> None:
    raw = MANIFEST.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA256
    result = audit_frozen_quantization(json.loads(raw), manifest_bytes=raw)
    assert result["pair_count"] == 28
    assert result["scientific_intervention_calls"] == 0
    assert result["scientific_schedule_changed"] is False
    assert result["requested_point_count"] == sum(
        pair["requested_point_count"] for pair in result["pairs"]
    )
    assert result["distinct_applied_point_count"] == sum(
        pair["distinct_applied_point_count"] for pair in result["pairs"]
    )
    assert all(pair["has_distinct_nonzero_suppression"] for pair in result["pairs"])


def test_journal_fails_closed_when_calls_lack_completed_points(
    tmp_path: Path,
) -> None:
    pair_id = "a" * 64
    journal = CanonicalExecutionJournal(
        tmp_path / "journal.jsonl",
        tmp_path / "attempt.lock",
        frozen_pair_ids=(pair_id,),
        pre_intervention_commit="b" * 40,
        prediction_manifest_sha256="c" * 64,
    )
    try:
        journal.before_source_suppression({"pair_id": pair_id}, 1)
        with pytest.raises(SerializationError, match="not exact"):
            journal.verify_complete(expected_point_count=1)
    finally:
        journal.close()


def test_journal_rereads_durable_records_fail_closed(tmp_path: Path) -> None:
    pair_id = "a" * 64
    journal_path = tmp_path / "journal.jsonl"
    journal = CanonicalExecutionJournal(
        journal_path,
        tmp_path / "attempt.lock",
        frozen_pair_ids=(pair_id,),
        pre_intervention_commit="b" * 40,
        prediction_manifest_sha256="c" * 64,
    )
    try:
        journal.before_source_suppression({"pair_id": pair_id}, 1)
        journal.append_completed_point(
            {
                "pair_id": pair_id,
                "source_suppression_api_call_index": 1,
            }
        )
        with journal_path.open("ab") as stream:
            stream.write(b"{}\n")
            stream.flush()
            os.fsync(stream.fileno())
        with pytest.raises(SerializationError, match="record count"):
            journal.verify_complete(expected_point_count=1)
    finally:
        journal.close()
