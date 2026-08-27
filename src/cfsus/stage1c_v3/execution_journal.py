"""Durable ignored-local evidence for the Stage 1C-v4 execution boundary."""

from __future__ import annotations

import fcntl
import json
import os
import stat
from pathlib import Path
from typing import Any

from cfsus.stage1c_v3.serialization import (
    SerializationError,
    detach_json,
    write_json_new,
)


class CanonicalExecutionJournal:
    """Append call intents and completed points, fsyncing every record."""

    def __init__(
        self,
        journal_path: Path,
        attempt_lock_path: Path | None,
        *,
        frozen_pair_ids: tuple[str, ...],
        pre_intervention_commit: str,
        prediction_manifest_sha256: str,
        experiment_class: str | None = None,
        attempt_boundary: str = (
            "first_instrumented_source_suppression_api_call_on_frozen_pair"
        ),
        attempt_lock_artifact_type: str = ("stage1c_v4_local_scientific_attempt_lock"),
    ) -> None:
        if not frozen_pair_ids or len(set(frozen_pair_ids)) != len(frozen_pair_ids):
            raise SerializationError("frozen pair IDs must be nonempty and unique")
        self.journal_path = self._new_local_path(journal_path, "point journal")
        self.attempt_lock_path = (
            None
            if attempt_lock_path is None
            else self._new_local_path(attempt_lock_path, "attempt lock")
        )
        self.frozen_pair_ids = frozenset(frozen_pair_ids)
        self.pre_intervention_commit = pre_intervention_commit
        self.prediction_manifest_sha256 = prediction_manifest_sha256
        if (
            (experiment_class is not None and not experiment_class.strip())
            or not attempt_boundary.strip()
            or not attempt_lock_artifact_type.strip()
        ):
            raise SerializationError("journal experiment identity must be nonempty")
        self.experiment_class = experiment_class
        self.attempt_boundary = attempt_boundary
        self.attempt_lock_artifact_type = attempt_lock_artifact_type
        self._descriptor: int | None = None
        self._started = False
        self._started_calls: list[tuple[int, str]] = []
        self._completed_calls: list[tuple[int, str]] = []

    @staticmethod
    def _new_local_path(path: Path, label: str) -> Path:
        candidate = path.resolve(strict=False)
        parent = candidate.parent
        if not parent.is_dir() or parent.is_symlink():
            raise SerializationError(f"{label} parent must be a real directory")
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            return candidate
        if stat.S_ISLNK(info.st_mode):
            raise SerializationError(f"{label} must not be a symlink")
        raise SerializationError(f"{label} already exists")

    def _append(self, value: dict[str, Any]) -> None:
        detached = detach_json(value)
        if not isinstance(detached, dict):  # pragma: no cover - defensive
            raise SerializationError("journal record must be an object")
        encoded = (
            json.dumps(
                detached,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        if self._descriptor is None:
            flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_NOFOLLOW", 0)
            try:
                self._descriptor = os.open(self.journal_path, flags, 0o600)
            except OSError as error:
                raise SerializationError("cannot create append-only journal") from error
        try:
            written = os.write(self._descriptor, encoded)
            if written != len(encoded):
                raise SerializationError("short append-only journal write")
            os.fsync(self._descriptor)
            full_sync = getattr(fcntl, "F_FULLFSYNC", None)
            if full_sync is not None:
                fcntl.fcntl(self._descriptor, full_sync)
        except OSError as error:
            raise SerializationError("append-only journal fsync failed") from error

    def before_source_suppression(self, pair: dict[str, Any], call_index: int) -> None:
        pair_id = pair.get("pair_id")
        if not isinstance(pair_id, str) or pair_id not in self.frozen_pair_ids:
            raise SerializationError(
                "scientific attempt recorder received a non-frozen pair"
            )
        expected = len(self._started_calls) + 1
        if call_index != expected:
            raise SerializationError("source-suppression call index is not contiguous")
        if not self._started:
            if self.attempt_lock_path is not None:
                write_json_new(
                    self.attempt_lock_path,
                    {
                        "schema_version": 4,
                        "artifact_type": self.attempt_lock_artifact_type,
                        "scientific_attempt_started": True,
                        "boundary": self.attempt_boundary,
                        "canonical_attempts": 1,
                        "scientific_retries": 0,
                        "first_pair_id": pair_id,
                        "first_call_index": call_index,
                        "pre_intervention_commit": self.pre_intervention_commit,
                        "prediction_manifest_sha256": self.prediction_manifest_sha256,
                        **(
                            {"experiment_class": self.experiment_class}
                            if self.experiment_class is not None
                            else {}
                        ),
                    },
                )
            self._started = True
        self._append(
            {
                "record_type": "source_suppression_call_started",
                "call_index": call_index,
                "pair_id": pair_id,
            }
        )
        self._started_calls.append((call_index, pair_id))

    def append_completed_point(self, point: dict[str, Any]) -> None:
        call_index = point.get("source_suppression_api_call_index")
        pair_id = point.get("pair_id")
        if not isinstance(call_index, int) or isinstance(call_index, bool):
            raise SerializationError("completed point lacks a call index")
        if not isinstance(pair_id, str):
            raise SerializationError("completed point lacks a pair ID")
        expected = len(self._completed_calls) + 1
        if call_index != expected or (call_index, pair_id) not in self._started_calls:
            raise SerializationError("completed point does not match a started call")
        self._append(
            {
                "record_type": "point_completed",
                "call_index": call_index,
                "pair_id": pair_id,
                "point": point,
            }
        )
        self._completed_calls.append((call_index, pair_id))

    def verify_complete(self, *, expected_point_count: int) -> None:
        if (
            not self._started
            or expected_point_count <= 0
            or len(self._started_calls) != expected_point_count
            or self._completed_calls != self._started_calls
        ):
            raise SerializationError(
                "journal calls and completed point records are not exact"
            )
        if self._descriptor is None:
            raise SerializationError("journal descriptor is not open")
        os.fsync(self._descriptor)
        try:
            lines = self.journal_path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise SerializationError("cannot reread the durable journal") from error
        if len(lines) != expected_point_count * 2:
            raise SerializationError("durable journal record count differs")
        try:
            records = [json.loads(line) for line in lines]
        except (json.JSONDecodeError, UnicodeError) as error:
            raise SerializationError("durable journal is not strict JSONL") from error
        observed: list[tuple[int, str]] = []
        for expected_index, (started, completed) in enumerate(
            zip(records[::2], records[1::2], strict=True), start=1
        ):
            if (
                not isinstance(started, dict)
                or started.get("record_type") != "source_suppression_call_started"
                or started.get("call_index") != expected_index
                or not isinstance(started.get("pair_id"), str)
                or set(started) != {"record_type", "call_index", "pair_id"}
            ):
                raise SerializationError("durable journal call record differs")
            if (
                not isinstance(completed, dict)
                or completed.get("record_type") != "point_completed"
                or completed.get("call_index") != expected_index
                or completed.get("pair_id") != started["pair_id"]
                or not isinstance(completed.get("point"), dict)
                or completed["point"].get("source_suppression_api_call_index")
                != expected_index
                or completed["point"].get("pair_id") != started["pair_id"]
                or set(completed) != {"record_type", "call_index", "pair_id", "point"}
            ):
                raise SerializationError("durable journal point record differs")
            observed.append((expected_index, started["pair_id"]))
        if observed != self._started_calls or observed != self._completed_calls:
            raise SerializationError("durable journal order differs from execution")

    def close(self) -> None:
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None

    def __enter__(self) -> CanonicalExecutionJournal:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["CanonicalExecutionJournal"]
